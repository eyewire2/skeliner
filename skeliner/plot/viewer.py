"""
Interactive mesh & skeleton viewer for skeliner.

Usage:
    skeliner view                        # empty viewer, drag & drop files
    skeliner view path/to/mesh.obj       # pre-load a mesh

Launches a local web server with a Three.js viewer. Supports:
  - Mesh files: .obj, .ply, .stl (always in nm)
  - Skeleton files: .swc (always in µm → converted to nm),
                    .npz (unit from metadata)

State files are written to /tmp/skeliner_view/<port>/ for
communication with external tools (e.g. Claude).
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import sys
import threading
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

_STATE_DIR = Path(os.environ.get("SKELINER_VIEW_STATE_DIR", "/tmp/skeliner_view"))


def _ensure_state_dir():
    _STATE_DIR.mkdir(parents=True, exist_ok=True)


# ── Mesh helpers ──────────────────────────────────────────────────────

def _mesh_to_buffers(mesh: trimesh.Trimesh) -> dict[str, Any]:
    verts = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    centroid = verts.mean(axis=0)
    verts_centered = verts - centroid
    return {
        "vertices": verts_centered.ravel().tolist(),
        "faces": faces.ravel().tolist(),
        "nVertices": len(verts),
        "nFaces": len(faces),
        "centroid": centroid.tolist(),
    }


# ── Skeleton helpers ──────────────────────────────────────────────────

def _skeleton_to_buffers(skel, centroid: np.ndarray) -> dict[str, Any]:
    """Convert a Skeleton to line-segment buffers for Three.js.

    Coordinates are shifted by the same centroid as the mesh so they
    overlay correctly.
    """
    nodes = np.asarray(skel.nodes, dtype=np.float32) - centroid
    edges = np.asarray(skel.edges, dtype=np.int32)
    radii = np.asarray(skel.r, dtype=np.float32)

    # Line segments: pairs of (x,y,z) for each edge
    positions = np.empty((len(edges) * 2, 3), dtype=np.float32)
    positions[0::2] = nodes[edges[:, 0]]
    positions[1::2] = nodes[edges[:, 1]]

    # Per-edge radius (average of endpoints)
    edge_radii = (radii[edges[:, 0]] + radii[edges[:, 1]]) / 2.0

    return {
        "nodes": nodes.ravel().tolist(),
        "edges": edges.ravel().tolist(),
        "positions": positions.ravel().tolist(),
        "radii": radii.tolist(),
        "edgeRadii": edge_radii.tolist(),
        "nNodes": len(nodes),
        "nEdges": len(edges),
    }


def _load_skeleton_as_nm(path: Path) -> Any:
    """Load a skeleton file and convert to nm."""
    from skeliner.io import load_swc, load_npz

    suffix = path.suffix.lower()
    if suffix == ".swc":
        skel = load_swc(path)
        # Check if unit is in meta; if not, infer from coordinate magnitude
        unit = skel.meta.get("unit", None)
        if unit is None:
            # Heuristic: if max coordinate > 10000, likely nm already
            if skel.nodes.max() > 10000:
                unit = "nm"
            else:
                unit = "um"
        if unit in ("um", "µm", "μm", "micron", "micrometer"):
            skel.nodes *= 1000.0
            for k in skel.radii:
                skel.radii[k] *= 1000.0
            skel.soma.center *= 1000.0
            skel.soma.axes *= 1000.0
        return skel
    elif suffix == ".npz":
        skel = load_npz(path)
        unit = skel.meta.get("unit", "nm")
        if unit in ("um", "µm", "μm", "micron", "micrometer"):
            skel.nodes *= 1000.0
            for k in skel.radii:
                skel.radii[k] *= 1000.0
            skel.soma.center *= 1000.0
            skel.soma.axes *= 1000.0
        return skel
    else:
        raise ValueError(f"Unsupported skeleton format: {suffix}")


def _is_l2_graph(path: Path) -> bool:
    """Check if an npz file is an L2 graph (not a skeliner skeleton)."""
    with np.load(path, allow_pickle=False) as data:
        return "graph_nodes" in data and "graph_edges" in data


def _l2_graph_to_buffers(path: Path, centroid: np.ndarray) -> dict[str, Any]:
    """Load an L2 supervoxel graph and convert to skeleton-like buffers."""
    with np.load(path, allow_pickle=False) as data:
        nodes = np.asarray(data["graph_nodes"], dtype=np.float32) - centroid
        edges = np.asarray(data["graph_edges"], dtype=np.int32)
        radii = np.zeros(len(nodes), dtype=np.float32)

        if "max_dt_nm" in data:
            n_l2 = len(data["max_dt_nm"])
            l2_radii = np.asarray(data["max_dt_nm"], dtype=np.float32)
            orig_edges = np.asarray(data["edges"], dtype=np.int32)
            n_edges = len(orig_edges)
            radii[:n_l2] = l2_radii
            # boundary_src [N..N+M-1] inherits from src L2 node
            radii[n_l2:n_l2 + n_edges] = l2_radii[orig_edges[:, 0]]
            # boundary_dst [N+M..N+2M-1] inherits from dst L2 node
            radii[n_l2 + n_edges:] = l2_radii[orig_edges[:, 1]]

    positions = np.empty((len(edges) * 2, 3), dtype=np.float32)
    positions[0::2] = nodes[edges[:, 0]]
    positions[1::2] = nodes[edges[:, 1]]
    edge_radii = (radii[edges[:, 0]] + radii[edges[:, 1]]) / 2.0

    return {
        "nodes": nodes.ravel().tolist(),
        "edges": edges.ravel().tolist(),
        "positions": positions.ravel().tolist(),
        "radii": radii.tolist(),
        "edgeRadii": edge_radii.tolist(),
        "nNodes": len(nodes),
        "nEdges": len(edges),
    }




def _get_viewer_html() -> str:
    html_path = Path(__file__).parent / "viewer.html"
    return html_path.read_text(encoding="utf-8")


# ── Server ────────────────────────────────────────────────────────────

def _create_app(mesh_path: str | Path | None = None, port: int = 8777):
    """Create the Starlette app."""
    from starlette.applications import Starlette
    from starlette.responses import HTMLResponse, JSONResponse
    from starlette.routing import Route, WebSocketRoute
    from starlette.websockets import WebSocket

    # ── Shared mutable state ──────────────────────────────────────────
    # These are modified by upload/remove endpoints
    mesh_state: dict[str, Any] = {
        "mesh": None,           # trimesh object
        "buffers": None,        # JSON-serialisable mesh data
        "path": None,           # source file path
        "centroid": np.zeros(3, dtype=np.float32),
        "organelle_mask": None, # cached from detect_organelles
        "fusion_clusters": None, # cached from detect_fusions
        "hole_loops": None, # cached from detect_holes
    }
    # Multiple skeletons, keyed by filename
    skeleton_states: dict[str, dict[str, Any]] = {}
    SKEL_COLORS = [
        [1.0, 0.4, 0.1],   # orange
        [0.2, 0.6, 1.0],   # blue
        [0.1, 0.9, 0.4],   # green
        [0.9, 0.2, 0.8],   # magenta
        [1.0, 0.9, 0.1],   # yellow
    ]

    # Pre-load mesh if path given
    if mesh_path is not None:
        mesh_path = Path(mesh_path)
        mesh = trimesh.load_mesh(str(mesh_path), process=False)
        print(f"Loaded mesh: {len(mesh.vertices):,} vertices, {len(mesh.faces):,} faces")

        buffers = _mesh_to_buffers(mesh)

        mesh_state["mesh"] = mesh
        mesh_state["buffers"] = buffers
        mesh_state["path"] = str(mesh_path.resolve())
        mesh_state["centroid"] = np.asarray(buffers["centroid"], dtype=np.float32)

    # ── State files ───────────────────────────────────────────────────
    _ensure_state_dir()
    port_dir = _STATE_DIR / str(port)
    port_dir.mkdir(parents=True, exist_ok=True)
    state_path = port_dir / "state.json"
    annotations_path = port_dir / "annotations.json"
    camera_cmd_path = port_dir / "camera.json"

    if not annotations_path.exists():
        annotations_path.write_text("{}", encoding="utf-8")
    if not camera_cmd_path.exists():
        camera_cmd_path.write_text("{}", encoding="utf-8")

    # Write initial state
    initial = {}
    if mesh_state["mesh"] is not None:
        initial["mesh"] = {
            "path": mesh_state["path"],
            "nVertices": len(mesh_state["mesh"].vertices),
            "nFaces": len(mesh_state["mesh"].faces),
        }
    state_path.write_text(json.dumps(initial, indent=2), encoding="utf-8")

    connected_clients: list[WebSocket] = []
    _undo_stack: list[trimesh.Trimesh] = []
    _UNDO_LIMIT = 10

    # ── Broadcast helper ──────────────────────────────────────────────
    async def broadcast(msg: dict):
        data = json.dumps(msg)
        for ws in connected_clients:
            try:
                await ws.send_text(data)
            except Exception:
                pass

    # ── Log-capturing executor helper ─────────────────────────────────

    class _LogTee(io.TextIOBase):
        """Wraps stdout: writes to original AND broadcasts lines via WS."""

        def __init__(self, original, loop, broadcast_fn):
            self._original = original
            self._loop = loop
            self._broadcast = broadcast_fn

        def write(self, s):
            self._original.write(s)
            self._original.flush()
            for line in s.splitlines():
                text = line.strip()
                if text:
                    asyncio.run_coroutine_threadsafe(
                        self._broadcast({"type": "log", "text": text}),
                        self._loop,
                    )
            return len(s)

        def flush(self):
            self._original.flush()

    async def _run_with_log(func, *args, **kwargs):
        """Run *func* in executor, streaming its stdout to WS clients."""
        loop = asyncio.get_event_loop()

        def _wrapper():
            old = sys.stdout
            sys.stdout = _LogTee(old, loop, broadcast)
            try:
                return func(*args, **kwargs)
            finally:
                sys.stdout = old

        result = await loop.run_in_executor(None, _wrapper)
        await broadcast({"type": "log_end"})
        return result

    # ── Routes ────────────────────────────────────────────────────────

    async def index(request):
        return HTMLResponse(_get_viewer_html())

    async def get_mesh(request):
        if mesh_state["buffers"] is None:
            return JSONResponse(None)
        return JSONResponse(mesh_state["buffers"])

    async def get_skeletons(request):
        """Return all loaded skeletons."""
        result = {}
        for name, state in skeleton_states.items():
            result[name] = state["buffers"]
        return JSONResponse(result if result else None)

    async def get_state(request):
        if state_path.exists():
            return JSONResponse(json.loads(state_path.read_text(encoding="utf-8")))
        return JSONResponse({})

    async def get_annotations(request):
        if annotations_path.exists():
            return JSONResponse(json.loads(annotations_path.read_text(encoding="utf-8")))
        return JSONResponse({})

    async def get_loaded(request):
        """Return what's currently loaded."""
        result = {"mesh": None, "skeletons": {}}
        if mesh_state["path"]:
            result["mesh"] = {
                "path": mesh_state["path"],
                "nVertices": len(mesh_state["mesh"].vertices),
                "nFaces": len(mesh_state["mesh"].faces),
            }
        for name, state in skeleton_states.items():
            result["skeletons"][name] = {
                "path": state["path"],
                "nNodes": state["buffers"]["nNodes"],
                "nEdges": state["buffers"]["nEdges"],
                "color": state["color"],
            }
        return JSONResponse(result)

    async def upload_file(request):
        """Handle file upload (mesh or skeleton)."""
        try:
            form = await request.form()
        except Exception:
            # Client disconnected during upload (e.g. drag-drop retry)
            return JSONResponse({"ok": False, "error": "Upload interrupted"}, status_code=499)
        upload = form["file"]
        filename = upload.filename
        content = await upload.read()
        suffix = Path(filename).suffix.lower()

        # Save to temp
        tmp_path = port_dir / filename
        tmp_path.write_bytes(content)

        try:
            if suffix in (".obj", ".ply", ".stl"):
                mesh = trimesh.load_mesh(str(tmp_path), process=False)
                print(f"Uploaded mesh: {len(mesh.vertices):,} verts, {len(mesh.faces):,} faces")
                buffers = _mesh_to_buffers(mesh)

                mesh_state["mesh"] = mesh
                mesh_state["buffers"] = buffers
                mesh_state["path"] = str(tmp_path.resolve())
                mesh_state["centroid"] = np.asarray(buffers["centroid"], dtype=np.float32)

                # Update state file
                current = {}
                if state_path.exists():
                    current = json.loads(state_path.read_text(encoding="utf-8"))
                current["mesh"] = {
                    "path": mesh_state["path"],
                    "nVertices": len(mesh.vertices),
                    "nFaces": len(mesh.faces),
                }
                state_path.write_text(json.dumps(current, indent=2), encoding="utf-8")

                await broadcast({"type": "mesh_loaded", "payload": buffers})
                # Re-send all skeletons (centroid changed)
                for sname, sstate in skeleton_states.items():
                    if sstate.get("l2_graph"):
                        sstate["buffers"] = _l2_graph_to_buffers(
                            Path(sstate["path"]), mesh_state["centroid"]
                        )
                    else:
                        sstate["buffers"] = _skeleton_to_buffers(
                            sstate["skeleton"], mesh_state["centroid"]
                        )
                    sstate["buffers"]["color"] = sstate["color"]
                    await broadcast({"type": "skeleton_loaded", "payload": {
                        "name": sname, **sstate["buffers"]
                    }})

                return JSONResponse({"ok": True, "type": "mesh", "name": filename})

            elif suffix == ".npz" and _is_l2_graph(tmp_path):
                buffers = _l2_graph_to_buffers(tmp_path, mesh_state["centroid"])
                print(f"Uploaded L2 graph: {buffers['nNodes']:,} nodes, {buffers['nEdges']:,} edges")

                color = SKEL_COLORS[len(skeleton_states) % len(SKEL_COLORS)]
                buffers["color"] = color

                skeleton_states[filename] = {
                    "skeleton": None,
                    "path": str(tmp_path.resolve()),
                    "buffers": buffers,
                    "color": color,
                    "l2_graph": True,
                }

                await broadcast({"type": "skeleton_loaded", "payload": {
                    "name": filename, **buffers
                }})
                return JSONResponse({"ok": True, "type": "skeleton", "name": filename})

            elif suffix in (".swc", ".npz"):
                skel = _load_skeleton_as_nm(tmp_path)
                print(f"Uploaded skeleton: {len(skel.nodes):,} nodes, {len(skel.edges):,} edges")

                color = SKEL_COLORS[len(skeleton_states) % len(SKEL_COLORS)]
                buffers = _skeleton_to_buffers(skel, mesh_state["centroid"])
                buffers["color"] = color

                skeleton_states[filename] = {
                    "skeleton": skel,
                    "path": str(tmp_path.resolve()),
                    "buffers": buffers,
                    "color": color,
                }

                await broadcast({"type": "skeleton_loaded", "payload": {
                    "name": filename, **buffers
                }})
                return JSONResponse({"ok": True, "type": "skeleton", "name": filename})

            else:
                return JSONResponse({"ok": False, "error": f"Unsupported format: {suffix}"}, status_code=400)

        except Exception as e:
            return JSONResponse({"ok": False, "error": str(e)}, status_code=400)

    async def remove_item(request):
        """Remove mesh or skeleton."""
        body = await request.json()
        item_type = body.get("type")

        if item_type == "mesh":
            mesh_state["mesh"] = None
            mesh_state["buffers"] = None
            mesh_state["path"] = None
            mesh_state["centroid"] = np.zeros(3, dtype=np.float32)
            for sname, sstate in skeleton_states.items():
                if sstate.get("l2_graph"):
                    sstate["buffers"] = _l2_graph_to_buffers(
                        Path(sstate["path"]), mesh_state["centroid"]
                    )
                else:
                    sstate["buffers"] = _skeleton_to_buffers(
                        sstate["skeleton"], mesh_state["centroid"]
                    )
                sstate["buffers"]["color"] = sstate["color"]
                await broadcast({"type": "skeleton_loaded", "payload": {
                    "name": sname, **sstate["buffers"]
                }})
            await broadcast({"type": "mesh_removed"})
            return JSONResponse({"ok": True})

        elif item_type == "skeleton":
            name = body.get("name")
            if name and name in skeleton_states:
                del skeleton_states[name]
                await broadcast({"type": "skeleton_removed", "payload": {"name": name}})
            else:
                # Remove all skeletons
                skeleton_states.clear()
                await broadcast({"type": "all_skeletons_removed"})
            return JSONResponse({"ok": True})

        return JSONResponse({"ok": False, "error": "Unknown type"}, status_code=400)

    async def update_annotations(request):
        body = await request.json()
        annotations_path.write_text(json.dumps(body), encoding="utf-8")
        return JSONResponse({"ok": True})

    async def post_state(request):
        body = await request.json()
        current = {}
        if state_path.exists():
            current = json.loads(state_path.read_text(encoding="utf-8"))
        current.update(body)
        state_path.write_text(json.dumps(current, indent=2), encoding="utf-8")
        return JSONResponse({"ok": True})

    async def post_selection(request):
        body = await request.json()
        current = {}
        if state_path.exists():
            current = json.loads(state_path.read_text(encoding="utf-8"))
        current["selection"] = body
        state_path.write_text(json.dumps(current, indent=2), encoding="utf-8")
        return JSONResponse({"ok": True})

    async def detect_organelles(request):
        """Run organelle detection."""
        if mesh_state["mesh"] is None:
            return JSONResponse({"ok": False, "error": "No mesh loaded"}, status_code=400)

        mesh = mesh_state["mesh"]
        det_type = request.query_params.get("type", "all")

        highlights = []
        # Start from cached mask if available, else empty
        cached = mesh_state.get("organelle_mask")
        if cached is not None and len(cached) == len(mesh.faces):
            combined = cached.copy()
        else:
            combined = np.zeros(len(mesh.faces), dtype=bool)

        if det_type in ("pocket", "surface"):
            from skeliner.pre import find_pocket_organelles
            mask = await _run_with_log(find_pocket_organelles, mesh, verbose=True)
            combined |= mask
            faces = [int(fi) for fi in np.where(mask)[0]]
            if faces:
                highlights.append({
                    "faces": faces,
                    "color": [1, 0.15, 0.15],
                    "label": f"organelle:pocket ({len(faces):,})",
                })
        elif det_type == "isolated":
            from skeliner.pre import find_isolated_organelles
            mask = await _run_with_log(find_isolated_organelles, mesh, verbose=True)
            combined |= mask
            faces = [int(fi) for fi in np.where(mask)[0]]
            if faces:
                highlights.append({
                    "faces": faces,
                    "color": [0.8, 0.4, 0.1],
                    "label": f"organelle:isolated ({len(faces):,})",
                })
        else:
            from skeliner.pre import find_organelles
            surface, isolated = await _run_with_log(find_organelles, mesh, verbose=True)
            combined = surface | isolated
            sf = [int(fi) for fi in np.where(surface)[0]]
            iso = [int(fi) for fi in np.where(isolated)[0]]
            if sf:
                highlights.append({
                    "faces": sf,
                    "color": [1, 0.15, 0.15],
                    "label": f"organelle:pocket ({len(sf):,})",
                })
            if iso:
                highlights.append({
                    "faces": iso,
                    "color": [0.8, 0.4, 0.1],
                    "label": f"organelle:isolated ({len(iso):,})",
                })

        mesh_state["organelle_mask"] = combined

        ann = {}
        if annotations_path.exists():
            ann = json.loads(annotations_path.read_text(encoding="utf-8"))
        if "highlights" not in ann:
            ann["highlights"] = []
        ann["highlights"].extend(highlights)
        annotations_path.write_text(json.dumps(ann), encoding="utf-8")

        n_faces = sum(len(h["faces"]) for h in highlights)
        return JSONResponse({"ok": True, "nFaces": n_faces})

    async def detect_fragments(request):
        """Run fragment detection and write results to annotations."""
        if mesh_state["mesh"] is None:
            return JSONResponse(
                {"ok": False, "error": "No mesh loaded"}, status_code=400
            )

        from skeliner.pre import find_fragments

        mesh = mesh_state["mesh"]
        mask = await _run_with_log(find_fragments, mesh, verbose=True)
        mesh_state["fragment_mask"] = mask

        faces = [int(fi) for fi in np.where(mask)[0]]
        if faces:
            ann = {}
            if annotations_path.exists():
                ann = json.loads(annotations_path.read_text(encoding="utf-8"))
            if "highlights" not in ann:
                ann["highlights"] = []
            ann["highlights"].append({
                "faces": faces,
                "color": [0.2, 0.8, 0.8],
                "label": f"fragments ({len(faces):,})",
            })
            annotations_path.write_text(json.dumps(ann), encoding="utf-8")

        return JSONResponse({"ok": True, "nFaces": len(faces)})

    async def check_fusion(request):
        """Analyze highlighted faces for fusion signals."""
        if mesh_state["mesh"] is None:
            return JSONResponse(
                {"ok": False, "error": "No mesh loaded"}, status_code=400
            )

        body = await request.json()
        query_faces = set(body.get("faces", []))
        if not query_faces:
            return JSONResponse({"ok": False, "error": "No faces"})

        mesh = mesh_state["mesh"]

        def _run():
            from collections import Counter

            areas = mesh.area_faces
            zero_faces = set(
                int(i) for i in np.where(areas < 1e-6)[0]
            )

            # Build edge-to-face, excluding zero-area
            edge_to_face = {}
            for fi, f in enumerate(mesh.faces):
                if fi in zero_faces:
                    continue
                for i in range(3):
                    a, b = int(f[i]), int(f[(i + 1) % 3])
                    e = (min(a, b), max(a, b))
                    edge_to_face.setdefault(e, []).append(fi)

            # Collect all edges and vertices in the query region
            region_edges = set()
            region_verts = set()
            for fi in query_faces:
                f = mesh.faces[fi]
                for v in f:
                    region_verts.add(int(v))
                for i in range(3):
                    a, b = int(f[i]), int(f[(i + 1) % 3])
                    region_edges.add((min(a, b), max(a, b)))

            # Non-manifold edges in region
            nm_edges = []
            for e in region_edges:
                faces_on_e = edge_to_face.get(e, [])
                if len(faces_on_e) > 2:
                    nm_edges.append((e, len(faces_on_e)))

            # Duplicate faces in region
            face_tuples = {
                fi: tuple(sorted(int(v) for v in mesh.faces[fi]))
                for fi in query_faces
            }
            # Also check all faces sharing vertices with region
            all_nearby = set()
            for fi, f in enumerate(mesh.faces):
                if fi in zero_faces:
                    continue
                if any(int(v) in region_verts for v in f):
                    all_nearby.add(fi)

            all_tuples = {}
            for fi in all_nearby:
                all_tuples[fi] = tuple(
                    sorted(int(v) for v in mesh.faces[fi])
                )
            tuple_count = Counter(all_tuples.values())
            dup_faces = [
                fi
                for fi in query_faces
                if face_tuples.get(fi) in tuple_count
                and tuple_count[face_tuples[fi]] > 1
            ]

            # Faces with >3 neighbors
            high_nb = []
            for fi in query_faces:
                f = mesh.faces[fi]
                nb = set()
                for i in range(3):
                    a, b = int(f[i]), int(f[(i + 1) % 3])
                    e = (min(a, b), max(a, b))
                    for nfi in edge_to_face.get(e, []):
                        if nfi != fi:
                            nb.add(nfi)
                if len(nb) > 3:
                    high_nb.append(fi)

            # All fusion faces: union of signals
            fusion = set(high_nb) | set(dup_faces)
            # Also add faces at non-manifold edges
            for e, _ in nm_edges:
                for fi in edge_to_face.get(e, []):
                    if fi in query_faces:
                        fusion.add(fi)

            return {
                "nm_edges": len(nm_edges),
                "duplicate_faces": len(dup_faces),
                "high_neighbor_faces": len(high_nb),
                "fusion_faces": sorted(fusion),
            }

        result = await _run_with_log(_run)
        return JSONResponse({"ok": True, **result})

    async def detect_fusions(request):
        """Run fusion detection and write results to annotations."""
        if mesh_state["mesh"] is None:
            return JSONResponse(
                {"ok": False, "error": "No mesh loaded"}, status_code=400
            )

        from skeliner.pre import find_fusions

        mesh = mesh_state["mesh"]
        clusters = await _run_with_log(find_fusions, mesh, verbose=True)
        mesh_state["fusion_clusters"] = clusters

        ann = {}
        if annotations_path.exists():
            ann = json.loads(annotations_path.read_text(encoding="utf-8"))
        if "highlights" not in ann:
            ann["highlights"] = []

        colors = [
            [0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0],
            [1.0, 1.0, 0.0], [1.0, 0.5, 0.0], [0.5, 0.0, 1.0],
        ]
        for i, cluster in enumerate(clusters):
            ann["highlights"].append({
                "faces": cluster,
                "color": colors[i % len(colors)],
                "label": f"fusion {i} ({len(cluster)}f)",
            })

        annotations_path.write_text(json.dumps(ann), encoding="utf-8")
        return JSONResponse({
            "ok": True,
            "nClusters": len(clusters),
        })

    async def detect_rims(request):
        """Run rim detection and write results as edge annotations."""
        if mesh_state["mesh"] is None:
            return JSONResponse(
                {"ok": False, "error": "No mesh loaded"}, status_code=400
            )

        from skeliner.pre import find_rims

        mesh = mesh_state["mesh"]
        centroid = mesh_state["centroid"]

        def _run():
            rims = find_rims(mesh, verbose=True)
            verts = np.asarray(mesh.vertices, dtype=np.float32)
            colors = [
                [0.2, 1.0, 0.6], [0.1, 0.8, 0.9], [0.9, 1.0, 0.2],
                [1.0, 0.5, 0.8], [0.5, 1.0, 0.4], [0.3, 0.7, 1.0],
            ]
            edge_groups = []
            for i, rim_edges in enumerate(rims):
                color = colors[i % len(colors)]
                segments = []
                for e in rim_edges:
                    a = (verts[e[0]] - centroid).tolist()
                    b = (verts[e[1]] - centroid).tolist()
                    segments.append([a, b])
                edge_groups.append({
                    "segments": segments,
                    "color": color,
                    "label": f"rim {i} ({len(rim_edges)}e)",
                })
            return edge_groups, len(rims)

        edge_groups, n_rims = await _run_with_log(_run)

        ann = {}
        if annotations_path.exists():
            ann = json.loads(annotations_path.read_text(encoding="utf-8"))
        if "edge_groups" not in ann:
            ann["edge_groups"] = []
        ann["edge_groups"].extend(edge_groups)
        annotations_path.write_text(json.dumps(ann), encoding="utf-8")
        return JSONResponse({"ok": True, "nFaces": n_rims})

    async def _apply_new_mesh(new_mesh):
        """Replace the current mesh with a modified one and broadcast."""
        # Save current mesh for undo
        old = mesh_state["mesh"]
        if old is not None:
            _undo_stack.append(old)
            if len(_undo_stack) > _UNDO_LIMIT:
                _undo_stack.pop(0)

        mesh_state["mesh"] = new_mesh
        mesh_state["organelle_mask"] = None  # invalidate caches
        mesh_state["fusion_clusters"] = None
        mesh_state["hole_loops"] = None
        buffers = _mesh_to_buffers(new_mesh)
        mesh_state["buffers"] = buffers
        mesh_state["centroid"] = np.asarray(buffers["centroid"], dtype=np.float32)

        # Clear annotations (they reference old face indices)
        annotations_path.write_text("{}", encoding="utf-8")

        buffers["keepCamera"] = True
        await broadcast({"type": "mesh_loaded", "payload": buffers})

        # Re-send skeletons (centroid may have changed)
        for sname, sstate in skeleton_states.items():
            if sstate.get("l2_graph"):
                sstate["buffers"] = _l2_graph_to_buffers(
                    Path(sstate["path"]), mesh_state["centroid"]
                )
            else:
                sstate["buffers"] = _skeleton_to_buffers(
                    sstate["skeleton"], mesh_state["centroid"]
                )
            sstate["buffers"]["color"] = sstate["color"]
            await broadcast({"type": "skeleton_loaded", "payload": {
                "name": sname, **sstate["buffers"]
            }})

        # Update state file
        current = {}
        if state_path.exists():
            current = json.loads(state_path.read_text(encoding="utf-8"))
        current["mesh"] = {
            "path": mesh_state["path"],
            "nVertices": len(new_mesh.vertices),
            "nFaces": len(new_mesh.faces),
        }
        state_path.write_text(json.dumps(current, indent=2), encoding="utf-8")

    async def do_remove_organelles(request):
        """Remove organelles from the mesh.

        Reuses the cached mask from detect_organelles if available,
        otherwise runs full detection.
        """
        if mesh_state["mesh"] is None:
            return JSONResponse(
                {"ok": False, "error": "No mesh loaded"}, status_code=400
            )

        mesh = mesh_state["mesh"]
        n_before = len(mesh.faces)
        cached = mesh_state.get("organelle_mask")

        if cached is not None and len(cached) == len(mesh.faces) and cached.any():
            # Use cached detection result
            print(f"Using cached organelle mask ({int(cached.sum()):,} faces)")
            keep = ~cached
            new_mesh = mesh.submesh([np.where(keep)[0]], append=True)
            new_mesh.remove_unreferenced_vertices()
        else:
            # No cache — run full detection + removal
            from skeliner.pre import remove_organelles as _remove_organelles
            new_mesh = await _run_with_log(_remove_organelles, mesh, verbose=True)

        n_after = len(new_mesh.faces)

        await _apply_new_mesh(new_mesh)
        print(f"Remove organelles: {n_before:,} → {n_after:,} faces ({n_before - n_after:,} removed)")
        return JSONResponse({
            "ok": True,
            "facesBefore": n_before,
            "facesAfter": n_after,
            "facesRemoved": n_before - n_after,
        })

    async def do_remove_fusions(request):
        """Remove fusions from the mesh."""
        if mesh_state["mesh"] is None:
            return JSONResponse(
                {"ok": False, "error": "No mesh loaded"}, status_code=400
            )
        from skeliner.pre import remove_fusions as _remove_fusions

        mesh = mesh_state["mesh"]
        n_before = len(mesh.faces)
        cached = mesh_state.get("fusion_clusters")
        new_mesh = await _run_with_log(
            _remove_fusions, mesh,
            fusion_clusters=cached,
            verbose=True,
        )
        n_after = len(new_mesh.faces)

        await _apply_new_mesh(new_mesh)
        print(f"Remove fusions: {n_before:,} → {n_after:,} faces ({n_before - n_after:,} removed)")
        return JSONResponse({
            "ok": True,
            "facesBefore": n_before,
            "facesAfter": n_after,
            "facesRemoved": n_before - n_after,
        })

    async def do_remove_fragments(request):
        """Remove fragments (islands and fins) from the mesh."""
        if mesh_state["mesh"] is None:
            return JSONResponse(
                {"ok": False, "error": "No mesh loaded"}, status_code=400
            )
        from skeliner.pre import remove_fragments as _remove_fragments

        mesh = mesh_state["mesh"]
        n_before = len(mesh.faces)
        cached = mesh_state.get("fragment_mask")

        if cached is not None and len(cached) == len(mesh.faces) and cached.any():
            print(f"Using cached fragment mask ({int(cached.sum()):,} faces)")
            new_mesh = _remove_fragments(mesh, _precomputed=cached)
        else:
            new_mesh = await _run_with_log(_remove_fragments, mesh, verbose=True)
        n_after = len(new_mesh.faces)

        await _apply_new_mesh(new_mesh)
        print(f"Remove fragments: {n_before:,} → {n_after:,} faces ({n_before - n_after:,} removed)")
        return JSONResponse({
            "ok": True,
            "facesBefore": n_before,
            "facesAfter": n_after,
            "facesRemoved": n_before - n_after,
        })

    async def do_fill_holes(request):
        """Fill holes in the mesh."""
        if mesh_state["mesh"] is None:
            return JSONResponse(
                {"ok": False, "error": "No mesh loaded"}, status_code=400
            )
        from skeliner.pre import fill_holes

        method = request.query_params.get("method", "advancing_front")
        dome_factor = float(request.query_params.get("dome_factor", "0.5"))

        mesh = mesh_state["mesh"]
        n_before = len(mesh.faces)
        cached_holes = mesh_state.get("hole_loops")
        new_mesh = await _run_with_log(
            fill_holes, mesh,
            holes=cached_holes,
            method=method, dome_factor=dome_factor, verbose=True,
        )
        n_after = len(new_mesh.faces)

        await _apply_new_mesh(new_mesh)
        print(f"Fill holes: {n_before:,} → {n_after:,} faces ({n_after - n_before:,} added)")
        return JSONResponse({
            "ok": True,
            "facesBefore": n_before,
            "facesAfter": n_after,
        })

    async def do_merge_selected(request):
        """Merge two components by stitching boundary loops of selected faces."""
        if mesh_state["mesh"] is None:
            return JSONResponse(
                {"ok": False, "error": "No mesh loaded"}, status_code=400
            )

        body = await request.json()
        face_indices = body.get("faces", [])
        if not face_indices:
            return JSONResponse(
                {"ok": False, "error": "No faces selected"}, status_code=400
            )

        from skeliner.pre import merge_selected_faces

        mesh = mesh_state["mesh"]
        n_before = len(mesh.faces)
        new_mesh = await _run_with_log(
            merge_selected_faces, mesh, face_indices, verbose=True
        )
        n_after = len(new_mesh.faces)

        await _apply_new_mesh(new_mesh)
        return JSONResponse({
            "ok": True,
            "facesBefore": n_before,
            "facesAfter": n_after,
            "facesRemoved": len(face_indices),
            "facesStitched": n_after - n_before + len(face_indices),
        })

    async def undo_mesh(request):
        """Revert to the previous mesh state."""
        if not _undo_stack:
            return JSONResponse(
                {"ok": False, "error": "Nothing to undo"}, status_code=400
            )
        prev_mesh = _undo_stack.pop()
        # Apply without pushing to undo stack
        mesh_state["mesh"] = prev_mesh
        buffers = _mesh_to_buffers(prev_mesh)
        mesh_state["buffers"] = buffers
        mesh_state["centroid"] = np.asarray(buffers["centroid"], dtype=np.float32)
        annotations_path.write_text("{}", encoding="utf-8")
        buffers["keepCamera"] = True
        await broadcast({"type": "mesh_loaded", "payload": buffers})

        for sname, sstate in skeleton_states.items():
            if sstate.get("l2_graph"):
                sstate["buffers"] = _l2_graph_to_buffers(
                    Path(sstate["path"]), mesh_state["centroid"]
                )
            else:
                sstate["buffers"] = _skeleton_to_buffers(
                    sstate["skeleton"], mesh_state["centroid"]
                )
            sstate["buffers"]["color"] = sstate["color"]
            await broadcast({"type": "skeleton_loaded", "payload": {
                "name": sname, **sstate["buffers"]
            }})

        current = {}
        if state_path.exists():
            current = json.loads(state_path.read_text(encoding="utf-8"))
        current["mesh"] = {
            "path": mesh_state["path"],
            "nVertices": len(prev_mesh.vertices),
            "nFaces": len(prev_mesh.faces),
        }
        state_path.write_text(json.dumps(current, indent=2), encoding="utf-8")

        print(f"Undo: restored mesh with {len(prev_mesh.faces):,} faces ({len(_undo_stack)} steps remaining)")
        return JSONResponse({
            "ok": True,
            "nFaces": len(prev_mesh.faces),
            "undoRemaining": len(_undo_stack),
        })

    async def export_mesh(request):
        """Export the current mesh as a downloadable OBJ file."""
        from starlette.responses import Response

        if mesh_state["mesh"] is None:
            return JSONResponse(
                {"ok": False, "error": "No mesh loaded"}, status_code=400
            )

        mesh = mesh_state["mesh"]
        fmt = request.query_params.get("format", "obj")
        loop = asyncio.get_event_loop()
        data = await loop.run_in_executor(
            None, lambda: mesh.export(file_type=fmt)
        )
        if isinstance(data, str):
            data = data.encode("utf-8")

        stem = Path(mesh_state["path"]).stem if mesh_state["path"] else "mesh"
        filename = f"{stem}_cleaned"
        return Response(
            content=data,
            media_type="application/octet-stream",
            headers={"Content-Disposition": f'attachment; filename="{filename}.{fmt}"'},
        )

    async def export_skeleton(request):
        """Export a skeleton as a downloadable SWC or NPZ file."""
        from starlette.responses import Response
        import tempfile

        name = request.query_params.get("name")
        fmt = request.query_params.get("format", "swc")
        if not name or name not in skeleton_states:
            return JSONResponse(
                {"ok": False, "error": "No skeleton found"}, status_code=400
            )
        sstate = skeleton_states[name]
        skel = sstate.get("skeleton")
        if skel is None:
            return JSONResponse(
                {"ok": False, "error": "Skeleton has no exportable data"},
                status_code=400,
            )

        from skeliner.io import to_swc, to_npz

        loop = asyncio.get_event_loop()
        tmp = Path(tempfile.mktemp(suffix=f".{fmt}"))
        if fmt == "npz":
            await loop.run_in_executor(None, lambda: to_npz(skel, tmp))
        else:
            await loop.run_in_executor(None, lambda: to_swc(skel, tmp))
        content = tmp.read_bytes()
        tmp.unlink(missing_ok=True)

        stem = Path(name).stem if name else "skeleton"
        return Response(
            content=content,
            media_type="application/octet-stream",
            headers={"Content-Disposition": f'attachment; filename="{stem}.{fmt}"'},
        )

    async def detect_holes(request):
        """Run hole detection and write results to annotations."""
        if mesh_state["mesh"] is None:
            return JSONResponse(
                {"ok": False, "error": "No mesh loaded"}, status_code=400
            )

        from skeliner.pre import find_holes

        mesh = mesh_state["mesh"]
        centroid = mesh_state["centroid"]

        def _run():
            loops = find_holes(mesh, verbose=True)
            mesh_state["hole_loops"] = loops
            verts = np.asarray(mesh.vertices, dtype=np.float32)
            colors = [
                [0.2, 0.6, 1.0], [0.1, 0.9, 0.4], [0.9, 0.2, 0.8],
                [1.0, 0.9, 0.1], [1.0, 0.4, 0.1], [0.4, 0.9, 0.9],
            ]
            edge_groups = []
            for i, loop in enumerate(loops):
                color = colors[i % len(colors)]
                segments = []
                for j in range(len(loop)):
                    a = (verts[loop[j]] - centroid).tolist()
                    b = (verts[loop[(j + 1) % len(loop)]] - centroid).tolist()
                    segments.append([a, b])
                edge_groups.append({
                    "segments": segments,
                    "color": color,
                    "label": f"hole {i} ({len(loop)}v)",
                })
            return edge_groups, len(loops)

        edge_groups, n_holes = await _run_with_log(_run)

        ann = {}
        if annotations_path.exists():
            ann = json.loads(annotations_path.read_text(encoding="utf-8"))
        if "edge_groups" not in ann:
            ann["edge_groups"] = []
        ann["edge_groups"].extend(edge_groups)
        annotations_path.write_text(json.dumps(ann), encoding="utf-8")
        return JSONResponse({"ok": True, "nHoles": n_holes})

    async def run_skeletonize(request):
        """Run skeletonization on the current mesh."""
        if mesh_state["mesh"] is None:
            return JSONResponse(
                {"ok": False, "error": "No mesh loaded"}, status_code=400
            )

        from skeliner.skeletonize import skeletonize

        body = await request.json()
        params = body.get("params", {})
        mesh = mesh_state["mesh"]

        skel = await _run_with_log(skeletonize, mesh, verbose=True, **params)
        print(f"Skeletonized: {len(skel.nodes)} nodes, {len(skel.edges)} edges")

        # Add as a skeleton layer
        skel_name = "skeleton"
        color = SKEL_COLORS[len(skeleton_states) % len(SKEL_COLORS)]
        buffers = _skeleton_to_buffers(skel, mesh_state["centroid"])
        buffers["color"] = color

        skeleton_states[skel_name] = {
            "skeleton": skel,
            "path": "",
            "buffers": buffers,
            "color": color,
        }

        await broadcast({"type": "skeleton_loaded", "payload": {
            "name": skel_name, **buffers
        }})
        return JSONResponse({
            "ok": True,
            "nNodes": len(skel.nodes),
            "nEdges": len(skel.edges),
        })

    async def shortest_path_endpoint(request):
        """Compute shortest path between two faces or two skeleton nodes."""
        import heapq

        body = await request.json()
        path_type = body.get("type")

        if path_type == "mesh":
            if mesh_state["mesh"] is None:
                return JSONResponse(
                    {"ok": False, "error": "No mesh loaded"}, status_code=400
                )
            face1 = body.get("face1")
            face2 = body.get("face2")
            mode = body.get("mode", "edge")  # "edge" or "vertex"
            if face1 is None or face2 is None:
                return JSONResponse(
                    {"ok": False, "error": "Need face1 and face2"}, status_code=400
                )

            mesh = mesh_state["mesh"]

            def _run_mesh_path():
                from scipy.sparse import csr_matrix
                from scipy.sparse.csgraph import dijkstra

                n_faces = len(mesh.faces)
                if face1 < 0 or face1 >= n_faces or face2 < 0 or face2 >= n_faces:
                    return None, 0
                if face1 == face2:
                    return [face1], 0.0

                print(f"[skeliner.path] Building {mode} adjacency for {n_faces:,} faces...")
                centroids = mesh.triangles_center

                if mode == "vertex":
                    # Sparse face-vertex incidence matrix M, then
                    # M @ M.T gives vertex-adjacency (crosses fusions).
                    n_verts = len(mesh.vertices)
                    rows = np.repeat(np.arange(n_faces), 3)
                    cols = mesh.faces.ravel().astype(np.int32)
                    M = csr_matrix(
                        (np.ones(len(rows), dtype=np.int8), (rows, cols)),
                        shape=(n_faces, n_verts),
                    )
                    adj = M @ M.T
                    adj.setdiag(0)
                    adj.eliminate_zeros()
                    fi, fj = adj.nonzero()
                else:
                    # Edge-based: faces sharing an edge.
                    adj_pairs = mesh.face_adjacency
                    fi = np.concatenate([adj_pairs[:, 0], adj_pairs[:, 1]])
                    fj = np.concatenate([adj_pairs[:, 1], adj_pairs[:, 0]])

                dists = np.linalg.norm(
                    centroids[fi] - centroids[fj], axis=1
                ).astype(np.float64)
                graph = csr_matrix(
                    (dists, (fi, fj)), shape=(n_faces, n_faces)
                )

                print(f"[skeliner.path] Running Dijkstra face {face1} → {face2}...")
                dist_arr, predecessors = dijkstra(
                    graph, directed=False, indices=face1,
                    return_predecessors=True,
                )

                if np.isinf(dist_arr[face2]):
                    return None, 0

                path = []
                cur = face2
                while cur != face1 and cur >= 0:
                    path.append(int(cur))
                    cur = int(predecessors[cur])
                if cur < 0:
                    return None, 0
                path.append(int(face1))
                path.reverse()
                print(f"[skeliner.path] Found path: {len(path)} faces, length={dist_arr[face2]:.1f}")
                return path, float(dist_arr[face2])

            path, length = await _run_with_log(_run_mesh_path)

            if path is None:
                return JSONResponse(
                    {"ok": False, "error": "No path found (disconnected components?)"}
                )

            return JSONResponse({
                "ok": True,
                "type": "mesh",
                "path": path,
                "length": length,
                "nFaces": len(path),
            })

        elif path_type == "skeleton":
            skel_name = body.get("skelName")
            node1 = body.get("node1")
            node2 = body.get("node2")
            if skel_name not in skeleton_states:
                return JSONResponse(
                    {"ok": False, "error": "Skeleton not found"}, status_code=400
                )

            buffers = skeleton_states[skel_name]["buffers"]

            def _run_skel_path():
                nodes = np.array(buffers["nodes"], dtype=np.float32).reshape(-1, 3)
                edges = np.array(buffers["edges"], dtype=np.int32).reshape(-1, 2)
                n_nodes = len(nodes)

                if node1 < 0 or node1 >= n_nodes or node2 < 0 or node2 >= n_nodes:
                    return None, None, 0
                if node1 == node2:
                    return [node1], [], 0.0

                print(f"[skeliner.path] Building skeleton adjacency for {n_nodes:,} nodes...")
                adj = [[] for _ in range(n_nodes)]
                for ei, (a, b) in enumerate(edges):
                    d = float(np.linalg.norm(nodes[a] - nodes[b]))
                    adj[int(a)].append((int(b), d, ei))
                    adj[int(b)].append((int(a), d, ei))

                print(f"[skeliner.path] Running Dijkstra node {node1} → {node2}...")
                dist_arr = [float("inf")] * n_nodes
                dist_arr[node1] = 0
                prev = [(-1, -1)] * n_nodes
                pq = [(0.0, node1)]
                visited = [False] * n_nodes

                while pq:
                    d, u = heapq.heappop(pq)
                    if visited[u]:
                        continue
                    visited[u] = True
                    if u == node2:
                        break
                    for v, w, ei in adj[u]:
                        if not visited[v]:
                            nd = d + w
                            if nd < dist_arr[v]:
                                dist_arr[v] = nd
                                prev[v] = (u, ei)
                                heapq.heappush(pq, (nd, v))

                if dist_arr[node2] == float("inf"):
                    return None, None, 0

                path_nodes = []
                path_edges = []
                cur = node2
                while cur != node1:
                    path_nodes.append(int(cur))
                    pn, pe = prev[cur]
                    path_edges.append(int(pe))
                    cur = pn
                path_nodes.append(int(node1))
                path_nodes.reverse()
                path_edges.reverse()

                segments = []
                for ei in path_edges:
                    a, b = int(edges[ei][0]), int(edges[ei][1])
                    segments.append([nodes[a].tolist(), nodes[b].tolist()])

                print(f"[skeliner.path] Found path: {len(path_nodes)} nodes, length={dist_arr[node2]:.1f}")
                return path_nodes, segments, float(dist_arr[node2])

            path_nodes, segments, length = await _run_with_log(_run_skel_path)

            if path_nodes is None:
                return JSONResponse({"ok": False, "error": "No path found"})

            return JSONResponse({
                "ok": True,
                "type": "skeleton",
                "pathNodes": path_nodes,
                "segments": segments,
                "length": length,
                "nNodes": len(path_nodes),
            })

        return JSONResponse(
            {"ok": False, "error": "Unknown path type"}, status_code=400
        )

    async def ws_endpoint(ws: WebSocket):
        await ws.accept()
        connected_clients.append(ws)
        try:
            while True:
                data = await ws.receive_text()
                msg = json.loads(data)

                if msg.get("type") == "state_update":
                    current = {}
                    if state_path.exists():
                        current = json.loads(state_path.read_text(encoding="utf-8"))
                    current.update(msg["payload"])
                    state_path.write_text(
                        json.dumps(current, indent=2), encoding="utf-8"
                    )
                elif msg.get("type") == "selection":
                    current = {}
                    if state_path.exists():
                        current = json.loads(state_path.read_text(encoding="utf-8"))
                    current["selection"] = msg["payload"]
                    state_path.write_text(
                        json.dumps(current, indent=2), encoding="utf-8"
                    )
                elif msg.get("type") == "node_selection":
                    current = {}
                    if state_path.exists():
                        current = json.loads(state_path.read_text(encoding="utf-8"))
                    current["nodeSelection"] = msg["payload"]
                    state_path.write_text(
                        json.dumps(current, indent=2), encoding="utf-8"
                    )
                elif msg.get("type") == "manual_highlight":
                    current = {}
                    if state_path.exists():
                        current = json.loads(state_path.read_text(encoding="utf-8"))
                    current["manualHighlight"] = msg["payload"].get("manualHighlight", [])
                    state_path.write_text(
                        json.dumps(current, indent=2), encoding="utf-8"
                    )
                elif msg.get("type") == "save_highlight":
                    # Append user-selected faces as a named annotation
                    ann = {}
                    if annotations_path.exists():
                        ann = json.loads(
                            annotations_path.read_text(encoding="utf-8")
                        )
                    if "highlights" not in ann:
                        ann["highlights"] = []
                    p = msg["payload"]
                    ann["highlights"].append({
                        "faces": p.get("faces", []),
                        "color": p.get("color", [1, 0.8, 0]),
                        "label": p.get("label", "selection"),
                    })
                    annotations_path.write_text(
                        json.dumps(ann), encoding="utf-8"
                    )
        except Exception:
            pass
        finally:
            connected_clients.remove(ws)

    # ── File watcher ──────────────────────────────────────────────────

    html_path = Path(__file__).parent / "viewer.html"

    async def file_watcher():
        last_ann_mtime = 0.0
        last_cam_mtime = 0.0
        last_html_mtime = html_path.stat().st_mtime if html_path.exists() else 0.0
        while True:
            await asyncio.sleep(0.5)
            try:
                mtime = annotations_path.stat().st_mtime
                if mtime > last_ann_mtime:
                    last_ann_mtime = mtime
                    content = annotations_path.read_text(encoding="utf-8")
                    await broadcast({
                        "type": "annotations",
                        "payload": json.loads(content),
                    })
            except FileNotFoundError:
                pass
            try:
                mtime = camera_cmd_path.stat().st_mtime
                if mtime > last_cam_mtime:
                    last_cam_mtime = mtime
                    content = camera_cmd_path.read_text(encoding="utf-8")
                    parsed = json.loads(content)
                    if parsed:
                        await broadcast({
                            "type": "camera_command",
                            "payload": parsed,
                        })
            except FileNotFoundError:
                pass
            try:
                mtime = html_path.stat().st_mtime
                if mtime > last_html_mtime:
                    last_html_mtime = mtime
                    await broadcast({"type": "reload"})
            except FileNotFoundError:
                pass

    async def on_startup():
        asyncio.create_task(file_watcher())

    app = Starlette(
        routes=[
            Route("/", index),
            Route("/mesh", get_mesh),
            Route("/skeletons", get_skeletons),
            Route("/loaded", get_loaded),
            Route("/state", get_state, methods=["GET"]),
            Route("/annotations", get_annotations, methods=["GET"]),
            Route("/update_annotations", update_annotations, methods=["POST"]),
            Route("/upload", upload_file, methods=["POST"]),
            Route("/remove", remove_item, methods=["POST"]),
            Route("/update_state", post_state, methods=["POST"]),
            Route("/update_selection", post_selection, methods=["POST"]),
            Route("/detect_organelles", detect_organelles, methods=["POST"]),
            Route("/check_fusion", check_fusion, methods=["POST"]),
            Route("/detect_fusions", detect_fusions, methods=["POST"]),
            Route("/detect_rims", detect_rims, methods=["POST"]),
            Route("/detect_holes", detect_holes, methods=["POST"]),
            Route("/remove_organelles", do_remove_organelles, methods=["POST"]),
            Route("/remove_fusions", do_remove_fusions, methods=["POST"]),
            Route("/detect_fragments", detect_fragments, methods=["POST"]),
            Route("/remove_fragments", do_remove_fragments, methods=["POST"]),
            Route("/fill_holes", do_fill_holes, methods=["POST"]),
            Route("/merge_selected", do_merge_selected, methods=["POST"]),
            Route("/undo", undo_mesh, methods=["POST"]),
            Route("/export_mesh", export_mesh, methods=["GET"]),
            Route("/export_skeleton", export_skeleton, methods=["GET"]),
            Route("/skeletonize", run_skeletonize, methods=["POST"]),
            Route("/shortest_path", shortest_path_endpoint, methods=["POST"]),
            WebSocketRoute("/ws", ws_endpoint),
        ],
        on_startup=[on_startup],
    )

    return app


def view(
    mesh_path: str | Path | None = None,
    *,
    host: str = "127.0.0.1",
    port: int = 8777,
    no_browser: bool = False,
):
    """Launch the interactive viewer."""
    try:
        import uvicorn
    except ImportError:
        raise ImportError(
            "The viewer requires uvicorn. Install with:\n"
            "  pip install uvicorn[standard]"
        )

    app = _create_app(mesh_path, port=port)

    port_dir = _STATE_DIR / str(port)
    url = f"http://{host}:{port}"
    print(f"\nSkeliner Viewer")
    print(f"  URL:          {url}")
    print(f"  State file:   {port_dir / 'state.json'}")
    print(f"  Annotations:  {port_dir / 'annotations.json'}")
    print(f"  Camera cmd:   {port_dir / 'camera.json'}")
    print()

    if not no_browser:
        def _open():
            import webbrowser
            webbrowser.open(url)
        threading.Timer(1.5, _open).start()

    uvicorn.run(app, host=host, port=port, log_level="warning")
