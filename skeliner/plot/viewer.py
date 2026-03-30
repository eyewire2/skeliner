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

def _mesh_to_buffers(
    mesh: trimesh.Trimesh, *, centroid: np.ndarray | None = None
) -> dict[str, Any]:
    verts = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    if centroid is None:
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
    from skeliner.io import load_skeleton_swc, load_skeleton_npz

    suffix = path.suffix.lower()
    if suffix == ".swc":
        skel = load_skeleton_swc(path)
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
        skel = load_skeleton_npz(path)
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


def _is_organelle_data(path: Path) -> bool:
    """Check if an npz file contains organelle-related data."""
    try:
        with np.load(path, allow_pickle=False) as data:
            known_keys = {"pocket", "isolated", "outward_dots", "face_comp", "main_ci"}
            return bool(known_keys & set(data.files))
    except Exception:
        return False


def _is_soma_data(path: Path) -> bool:
    """Check if an npz file contains standalone soma data."""
    try:
        with np.load(path, allow_pickle=False) as data:
            return "center" in data and "axes" in data and "R" in data
    except Exception:
        return False


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
        "offsets": None, # cached from detect_offsets
        "mesh_stats": None, # cached from compute_mesh_stats
        "organelles": None, # dict(pocket, isolated, expanded) bool masks, or None
        "fusion_clusters": None, # cached from detect_fusions
        "soma": None, # cached from detect_soma
        "disconnected": None, # cached from detect_disconnected
        "gap_clusters": None, # cached from detect_gaps
        "hole_loops": None, # cached from detect_holes
    }

    def _organelle_mask(org: dict | None) -> np.ndarray | None:
        """Combined bool mask from organelles dict."""
        if org is None:
            return None
        return org["pocket"] | org["isolated"] | org["expanded"]

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

    async def _log(text: str):
        """Print to terminal AND broadcast to browser."""
        print(text)
        await broadcast({"type": "log", "text": text})

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

                # ── Clear previous session data ──────────────────────
                # Remove old mesh/skeleton files (keep the new upload)
                for old_file in port_dir.iterdir():
                    if old_file == tmp_path:
                        continue
                    if old_file.suffix in (".obj", ".ply", ".stl", ".npz", ".swc"):
                        old_file.unlink(missing_ok=True)
                # Reset skeletons and annotations
                skeleton_states.clear()
                mesh_state["soma"] = None
                mesh_state["organelles"] = None
                mesh_state["mesh_stats"] = None
                annotations_path.write_text("{}", encoding="utf-8")
                await broadcast({"type": "all_skeletons_removed"})

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

            elif suffix == ".npz" and _is_organelle_data(tmp_path):
                from skeliner.io import load_organelles_npz
                data = load_organelles_npz(tmp_path)
                loaded = []

                pocket, isolated, expanded = data["pocket"], data["isolated"], data["expanded"]
                mesh_state["organelles"] = {
                    "pocket": pocket, "isolated": isolated, "expanded": expanded,
                }
                loaded.append(f"pocket={int(pocket.sum()):,}")
                loaded.append(f"isolated={int(isolated.sum()):,}")
                if expanded.any():
                    loaded.append(f"expanded={int(expanded.sum()):,}")

                # Visualize
                ann = {}
                if annotations_path.exists():
                    ann = json.loads(annotations_path.read_text(encoding="utf-8"))
                if "highlights" not in ann:
                    ann["highlights"] = []
                if pocket.any():
                    ann["highlights"].append({
                        "faces": np.where(pocket)[0].tolist(),
                        "color": [1, 0.15, 0.15],
                        "label": f"organelle:pocket ({int(pocket.sum()):,})",
                    })
                if isolated.any():
                    ann["highlights"].append({
                        "faces": np.where(isolated)[0].tolist(),
                        "color": [0.15, 0.8, 0.15],
                        "label": f"organelle:isolated ({int(isolated.sum()):,})",
                    })
                if expanded.any():
                    ann["highlights"].append({
                        "faces": np.where(expanded)[0].tolist(),
                        "color": [1, 0.6, 0.15],
                        "label": f"organelle:expanded ({int(expanded.sum()):,})",
                    })
                annotations_path.write_text(json.dumps(ann), encoding="utf-8")

                # Mesh stats
                if "outward_dots" in data:
                    face_comp = data["face_comp"]
                    main_ci = int(data["main_ci"])
                    mesh_state["mesh_stats"] = (
                        data["outward_dots"], face_comp, main_ci, face_comp == main_ci
                    )
                    loaded.append("mesh_stats")

                print(f"Loaded organelle npz: {', '.join(loaded)}")
                await broadcast({"type": "annotations_updated"})
                return JSONResponse({
                    "ok": True, "type": "organelles", "name": filename,
                    "loaded": loaded,
                })

            elif suffix == ".npz" and _is_soma_data(tmp_path):
                from skeliner.io import load_soma_npz

                soma = load_soma_npz(tmp_path)
                mesh_state["soma"] = soma
                n_verts = len(soma.verts) if soma.verts is not None else 0
                print(
                    f"Loaded soma: center=[{soma.center[0]:.0f}, "
                    f"{soma.center[1]:.0f}, {soma.center[2]:.0f}], "
                    f"axes=[{soma.axes[0]:.0f}, {soma.axes[1]:.0f}, "
                    f"{soma.axes[2]:.0f}], {n_verts:,} verts"
                )

                # Annotate soma (faces + ellipsoid wireframe)
                mesh = mesh_state.get("mesh")
                if mesh is not None and soma.verts is not None:
                    soma_vset = set(int(v) for v in soma.verts)
                    soma_faces = [
                        int(fi) for fi in range(len(mesh.faces))
                        if sum(1 for v in mesh.faces[fi] if int(v) in soma_vset) >= 2
                    ]
                    ann = {}
                    if annotations_path.exists():
                        ann = json.loads(annotations_path.read_text(encoding="utf-8"))
                    if "highlights" not in ann:
                        ann["highlights"] = []
                    ann["highlights"].append({
                        "faces": soma_faces,
                        "color": [0.9, 0.5, 0.9],
                        "label": f"soma ({len(soma_faces):,}f, {n_verts:,}v)",
                    })
                    centroid = mesh_state["centroid"]
                    if "ellipsoids" not in ann:
                        ann["ellipsoids"] = []
                    ann["ellipsoids"].append({
                        "center": (soma.center - centroid).tolist(),
                        "axes": soma.axes.tolist(),
                        "R": soma.R.tolist(),
                        "color": [0.9, 0.5, 0.9],
                    })
                    annotations_path.write_text(json.dumps(ann), encoding="utf-8")
                    await broadcast({"type": "annotations_updated"})

                return JSONResponse({
                    "ok": True, "type": "soma", "name": filename,
                })

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

            elif suffix == ".json":
                # Annotation JSON — merge with deduplication
                incoming = json.loads(content.decode("utf-8"))
                current = {}
                if annotations_path.exists():
                    current = json.loads(annotations_path.read_text(encoding="utf-8"))
                if "highlights" not in current:
                    current["highlights"] = []

                # Deduplicate: skip incoming highlights whose label
                # already exists with the same face count
                existing = {
                    (h.get("label", ""), len(h.get("faces", []))): True
                    for h in current["highlights"]
                }
                added = 0
                for h in incoming.get("highlights", []):
                    key = (h.get("label", ""), len(h.get("faces", [])))
                    if key not in existing:
                        current["highlights"].append(h)
                        existing[key] = True
                        added += 1

                # Copy non-highlight keys (skip duplicates)
                for k, v in incoming.items():
                    if k != "highlights" and k not in current:
                        current[k] = v

                annotations_path.write_text(json.dumps(current), encoding="utf-8")
                print(
                    f"Loaded annotations: {added} new highlights "
                    f"(total {len(current['highlights'])})"
                )
                await broadcast({"type": "annotations_updated"})
                return JSONResponse({
                    "ok": True, "type": "annotations", "name": filename,
                    "added": added, "total": len(current["highlights"]),
                })

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

    async def detect_offsets(request):
        """Run offset detection and write results to annotations."""
        if mesh_state["mesh"] is None:
            return JSONResponse(
                {"ok": False, "error": "No mesh loaded"}, status_code=400
            )

        from skeliner.pre import find_offsets

        mesh = mesh_state["mesh"]
        await _log("[skeliner.pre] Detecting offsets...")
        offsets = await _run_with_log(find_offsets, mesh, verbose=True)
        mesh_state["offsets"] = offsets

        if not offsets:
            await _log("[skeliner.pre] No offsets found")
            return JSONResponse({"ok": True, "nOffsets": 0})

        ann = {}
        if annotations_path.exists():
            ann = json.loads(annotations_path.read_text(encoding="utf-8"))
        if "highlights" not in ann:
            ann["highlights"] = []
        if "edge_groups" not in ann:
            ann["edge_groups"] = []

        # Clear previous offset annotations
        ann["highlights"] = [
            h for h in ann["highlights"]
            if not h.get("label", "").startswith("offset ")
        ]
        ann["edge_groups"] = [
            eg for eg in ann["edge_groups"]
            if not eg.get("label", "").startswith("offset ")
        ]

        colors = [
            [1.0, 0.3, 0.3], [0.3, 1.0, 0.3], [0.3, 0.3, 1.0],
            [1.0, 1.0, 0.3], [1.0, 0.3, 1.0], [0.3, 1.0, 1.0],
        ]

        for i, o in enumerate(offsets):
            dx, dy = o["offset"]
            color = colors[i % len(colors)]
            ann["highlights"].append({
                "faces": o["cap_faces"].tolist(),
                "color": color,
                "label": (
                    f"offset {i}: z={o['z_floor']:.0f}→{o['z_ceil']:.0f} "
                    f"d=({dx:.0f},{dy:.0f}) "
                    f"err={o['match_error']:.0f} "
                    f"{len(o['shifted_verts']):,}v"
                ),
            })

            # Add a direction vector: red (floor) → green (ceil)
            # Coordinates must be centroid-relative for the viewer
            centroid = mesh_state["centroid"]
            start = (o["floor_center"] - centroid).tolist()
            end = (o["ceil_center"] - centroid).tolist()
            midpt = [(s + e) / 2 for s, e in zip(start, end)]
            ann["edge_groups"].append({
                "segments": [[start, midpt]],
                "color": [1.0, 0.2, 0.2],
                "label": f"offset {i} from",
            })
            ann["edge_groups"].append({
                "segments": [[midpt, end]],
                "color": [0.2, 1.0, 0.2],
                "label": f"offset {i} to",
            })

        annotations_path.write_text(json.dumps(ann), encoding="utf-8")
        return JSONResponse({"ok": True, "nOffsets": len(offsets)})

    async def do_remove_offsets(request):
        """Remove detected offsets from the mesh."""
        if mesh_state["mesh"] is None:
            return JSONResponse(
                {"ok": False, "error": "No mesh loaded"}, status_code=400
            )

        from skeliner.pre import remove_offsets

        mesh = mesh_state["mesh"]
        cached = mesh_state.get("offsets")

        def _do_remove():
            return remove_offsets(mesh, offsets=cached, verbose=True)

        new_mesh = await _run_with_log(_do_remove)
        await _apply_new_mesh(new_mesh)

        # Clear cached offsets
        mesh_state["offsets"] = None
        mesh_state["mesh_stats"] = None

        return JSONResponse({"ok": True})

    async def save_offsets(request):
        """Save detected offsets to a file."""
        cached = mesh_state.get("offsets")
        if not cached:
            return JSONResponse(
                {"ok": False, "error": "No offsets detected"}, status_code=400
            )
        import pickle
        save_path = port_dir / "offsets.pkl"
        with open(save_path, "wb") as f:
            pickle.dump(cached, f)
        return JSONResponse({"ok": True, "path": str(save_path), "n": len(cached)})

    async def load_offsets(request):
        """Load previously saved offsets and apply annotations."""
        import pickle
        save_path = port_dir / "offsets.pkl"
        if not save_path.exists():
            return JSONResponse(
                {"ok": False, "error": "No saved offsets found"}, status_code=404
            )
        with open(save_path, "rb") as f:
            offsets = pickle.load(f)
        mesh_state["offsets"] = offsets

        # Rebuild annotations
        ann = {}
        if annotations_path.exists():
            ann = json.loads(annotations_path.read_text(encoding="utf-8"))
        if "highlights" not in ann:
            ann["highlights"] = []
        if "edge_groups" not in ann:
            ann["edge_groups"] = []
        ann["highlights"] = [
            h for h in ann["highlights"]
            if not h.get("label", "").startswith("offset ")
        ]
        ann["edge_groups"] = [
            eg for eg in ann["edge_groups"]
            if not eg.get("label", "").startswith("offset ")
        ]

        colors = [
            [1.0, 0.3, 0.3], [0.3, 1.0, 0.3], [0.3, 0.3, 1.0],
            [1.0, 1.0, 0.3], [1.0, 0.3, 1.0], [0.3, 1.0, 1.0],
        ]
        mesh = mesh_state["mesh"]
        centroid = mesh_state["centroid"]
        for i, o in enumerate(offsets):
            dx, dy = o["offset"]
            color = colors[i % len(colors)]
            ann["highlights"].append({
                "faces": o["cap_faces"].tolist(),
                "color": color,
                "label": (
                    f"offset {i}: z={o['z_floor']:.0f}→{o['z_ceil']:.0f} "
                    f"d=({dx:.0f},{dy:.0f}) "
                    f"err={o['match_error']:.0f} "
                    f"{len(o['shifted_verts']):,}v"
                ),
            })
            start = (o["floor_center"] - centroid).tolist()
            end = (o["ceil_center"] - centroid).tolist()
            midpt = [(s + e) / 2 for s, e in zip(start, end)]
            ann["edge_groups"].append({
                "segments": [[start, midpt]],
                "color": [1.0, 0.2, 0.2],
                "label": f"offset {i} from",
            })
            ann["edge_groups"].append({
                "segments": [[midpt, end]],
                "color": [0.2, 1.0, 0.2],
                "label": f"offset {i} to",
            })

        annotations_path.write_text(json.dumps(ann), encoding="utf-8")
        return JSONResponse({"ok": True, "nOffsets": len(offsets)})

    async def detect_organelles(request):
        """Run organelle detection."""
        if mesh_state["mesh"] is None:
            return JSONResponse({"ok": False, "error": "No mesh loaded"}, status_code=400)

        mesh = mesh_state["mesh"]
        det_type = request.query_params.get("type", "all")

        highlights = []
        nF = len(mesh.faces)
        cached = mesh_state.get("organelles")
        if cached is not None and len(cached["pocket"]) == nF:
            org = {k: v.copy() for k, v in cached.items()}
        else:
            org = {
                "pocket": np.zeros(nF, dtype=bool),
                "isolated": np.zeros(nF, dtype=bool),
                "expanded": np.zeros(nF, dtype=bool),
            }

        from skeliner.pre import (
            compute_mesh_stats, find_pocket_organelles,
            find_isolated_organelles, find_organelles,
        )

        # Precompute once, reuse for all detection types
        precomputed = mesh_state.get("mesh_stats")
        if precomputed is None or len(precomputed[0]) != len(mesh.faces):
            precomputed = await _run_with_log(
                compute_mesh_stats, mesh, None, 5.0, True
            )
            mesh_state["mesh_stats"] = precomputed

        if det_type in ("pocket", "surface"):
            mask = await _run_with_log(
                find_pocket_organelles, mesh, verbose=True,
                mesh_stats=precomputed,
            )
            org["pocket"] |= mask
            faces = [int(fi) for fi in np.where(mask)[0]]
            if faces:
                highlights.append({
                    "faces": faces,
                    "color": [1, 0.15, 0.15],
                    "label": f"organelle:pocket ({len(faces):,})",
                })
        elif det_type == "isolated":
            mask = await _run_with_log(
                find_isolated_organelles, mesh, verbose=True,
                mesh_stats=precomputed,
            )
            org["isolated"] |= mask
            faces = [int(fi) for fi in np.where(mask)[0]]
            if faces:
                highlights.append({
                    "faces": faces,
                    "color": [0.15, 0.8, 0.15],
                    "label": f"organelle:isolated ({len(faces):,})",
                })
        else:
            surface, iso_mask = await _run_with_log(
                find_organelles, mesh, verbose=True,
                mesh_stats=precomputed,
            )
            org["pocket"] = surface
            org["isolated"] = iso_mask
            org["expanded"] = np.zeros(nF, dtype=bool)
            sf = [int(fi) for fi in np.where(surface)[0]]
            iso = [int(fi) for fi in np.where(iso_mask)[0]]
            if sf:
                highlights.append({
                    "faces": sf,
                    "color": [1, 0.15, 0.15],
                    "label": f"organelle:pocket ({len(sf):,})",
                })
            if iso:
                highlights.append({
                    "faces": iso,
                    "color": [0.15, 0.8, 0.15],
                    "label": f"organelle:isolated ({len(iso):,})",
                })

        mesh_state["organelles"] = org

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

    async def detect_disconnected(request):
        """Run disconnected-component detection and write results to annotations."""
        if mesh_state["mesh"] is None:
            return JSONResponse(
                {"ok": False, "error": "No mesh loaded"}, status_code=400
            )

        from skeliner.pre import find_disconnected, find_soma

        mesh = mesh_state["mesh"]
        soma = mesh_state.get("soma")
        if soma is None:
            await _log("[skeliner.pre] Detecting soma first...")
            soma = await _run_with_log(
                find_soma, mesh,
                organelles=_organelle_mask(mesh_state.get("organelles")),
                mesh_stats=mesh_state.get("mesh_stats"),
                verbose=True,
            )
            mesh_state["soma"] = soma
        await _log("[skeliner.pre] Detecting disconnected components...")
        components = await _run_with_log(
            find_disconnected, mesh, verbose=True, soma=soma,
            organelles=mesh_state.get("organelles"),
            mesh_stats=mesh_state.get("mesh_stats"),
        )
        mesh_state["disconnected"] = components

        if not components:
            await _log("[skeliner.pre] No disconnected components found")
            return JSONResponse({"ok": False, "nComponents": 0})

        ann = {}
        if annotations_path.exists():
            ann = json.loads(annotations_path.read_text(encoding="utf-8"))
        if "highlights" not in ann:
            ann["highlights"] = []

        colors = [
            [1.0, 0.3, 0.3], [0.3, 1.0, 0.3], [0.3, 0.3, 1.0],
            [1.0, 1.0, 0.3], [1.0, 0.3, 1.0], [0.3, 1.0, 1.0],
        ]
        for i, faces in enumerate(components):
            ann["highlights"].append({
                "faces": faces,
                "color": colors[i % len(colors)],
                "label": f"disconnected {i} ({len(faces):,}f)",
            })

        annotations_path.write_text(json.dumps(ann), encoding="utf-8")
        return JSONResponse({"ok": True, "nComponents": len(components)})

    async def detect_soma(request):
        """Run soma detection and write result to annotations.

        Accepts optional JSON body ``{"method": "new"}`` or
        ``{"method": "ring_cutoff"}``.  Defaults to the new per-tip
        neurite-exclusion method.
        """
        if mesh_state["mesh"] is None:
            return JSONResponse(
                {"ok": False, "error": "No mesh loaded"}, status_code=400
            )

        # Parse method choice from request body (POST with JSON)
        method = "new"
        try:
            body = await request.json()
            method = body.get("method", "new")
        except Exception:
            pass

        if method == "ring_cutoff":
            from skeliner.pre import find_soma_via_ring_cutoff as _find
            label_prefix = "soma (ring_cutoff)"
            color = [0.9, 0.7, 0.3]
        elif method == "alt":
            from skeliner.pre import find_soma_alt as _find
            label_prefix = "soma (alt)"
            color = [0.3, 0.9, 0.7]
        elif method == "z_contour":
            from skeliner.pre import find_soma_via_z_contour as _find
            label_prefix = "soma (z_contour)"
            color = [0.85, 0.65, 0.13]
        else:
            from skeliner.pre import find_soma as _find
            label_prefix = "soma"
            color = [0.9, 0.5, 0.9]

        mesh = mesh_state["mesh"]
        if method == "z_contour":
            soma = await _run_with_log(
                _find, mesh, verbose=True,
            )
        else:
            soma = await _run_with_log(
                _find, mesh,
                organelles=_organelle_mask(mesh_state.get("organelles")),
                mesh_stats=mesh_state.get("mesh_stats"),
                verbose=True,
            )
        if soma is None:
            return JSONResponse({"ok": False, "error": "No soma found"})

        mesh_state["soma"] = soma

        # Highlight soma surface vertices via their faces
        # Include faces where at least 2 of 3 vertices are soma verts
        soma_vset = set(int(v) for v in soma.verts)
        soma_faces = [
            int(fi) for fi in range(len(mesh.faces))
            if sum(1 for v in mesh.faces[fi] if int(v) in soma_vset) >= 2
        ]

        ann = {}
        if annotations_path.exists():
            ann = json.loads(annotations_path.read_text(encoding="utf-8"))
        if "highlights" not in ann:
            ann["highlights"] = []
        ann["highlights"].append({
            "faces": soma_faces,
            "color": color,
            "label": f"{label_prefix} ({len(soma_faces):,}f, {len(soma.verts):,}v)",
        })

        # Wireframe ellipsoid (coordinates shifted by mesh centroid)
        centroid = mesh_state["centroid"]
        if "ellipsoids" not in ann:
            ann["ellipsoids"] = []
        ann["ellipsoids"].append({
            "center": (soma.center - centroid).tolist(),
            "axes": soma.axes.tolist(),
            "R": soma.R.tolist(),
            "color": color,
        })

        annotations_path.write_text(json.dumps(ann), encoding="utf-8")

        return JSONResponse({
            "ok": True,
            "method": method,
            "nFaces": len(soma_faces),
            "nVerts": len(soma.verts),
        })

    async def detect_gaps(request):
        """Run gap detection and write results to annotations."""
        if mesh_state["mesh"] is None:
            return JSONResponse(
                {"ok": False, "error": "No mesh loaded"}, status_code=400
            )

        from skeliner.pre import find_gaps, find_soma

        mesh = mesh_state["mesh"]

        # Use cached soma or compute it
        soma = mesh_state.get("soma")
        if soma is None:
            await _log("[skeliner.pre] Detecting soma first...")
            soma = await _run_with_log(
                find_soma, mesh,
                organelles=_organelle_mask(mesh_state.get("organelles")),
                mesh_stats=mesh_state.get("mesh_stats"),
                verbose=True,
            )
            mesh_state["soma"] = soma

        cached_disc = mesh_state.get("disconnected")
        await _log("[skeliner.pre] Detecting gaps...")
        gaps = await _run_with_log(
            find_gaps, mesh, verbose=True,
            soma=soma,
            disconnected=cached_disc,
            mesh_stats=mesh_state.get("mesh_stats"),
        )
        mesh_state["gap_clusters"] = gaps

        ann = {}
        if annotations_path.exists():
            ann = json.loads(annotations_path.read_text(encoding="utf-8"))
        if "highlights" not in ann:
            ann["highlights"] = []

        colors = [
            [1.0, 0.3, 0.3], [0.3, 1.0, 0.3], [0.3, 0.3, 1.0],
            [1.0, 1.0, 0.3], [1.0, 0.3, 1.0], [0.3, 1.0, 1.0],
        ]
        for i, (faces_a, faces_b, dist, da, db) in enumerate(gaps):
            label_a = "main" if da == -1 else f"disc {da}"
            label_b = "main" if db == -1 else f"disc {db}"
            color = colors[i % len(colors)]
            ann["highlights"].append({
                "faces": faces_a + faces_b,
                "color": color,
                "label": f"gap {i}: {label_a} ({len(faces_a)}f) ↔ {label_b} ({len(faces_b)}f), {dist:.0f}nm",
            })

        annotations_path.write_text(json.dumps(ann), encoding="utf-8")
        return JSONResponse({
            "ok": True,
            "nGaps": len(gaps),
        })

    async def do_remove_gaps(request):
        """Bridge all detected gaps."""
        if mesh_state["mesh"] is None:
            return JSONResponse(
                {"ok": False, "error": "No mesh loaded"}, status_code=400
            )

        from skeliner.pre import remove_gaps

        mesh = mesh_state["mesh"]
        n_before = len(mesh.faces)

        soma = mesh_state.get("soma")
        cached_gaps = mesh_state.get("gap_clusters")

        new_mesh = await _run_with_log(
            remove_gaps, mesh, verbose=True,
            soma=soma,
            gaps=cached_gaps,
        )
        n_after = len(new_mesh.faces)
        n_added = n_after - n_before

        # Stitch faces were appended — extend cached masks so indices stay valid
        if n_added > 0:
            pad = np.zeros(n_added, dtype=bool)
            if mesh_state.get("organelles") is not None:
                org = mesh_state["organelles"]
                for k in org:
                    org[k] = np.concatenate([org[k], pad])
            cached_stats = mesh_state.get("mesh_stats")
            if cached_stats is not None:
                od, fc, mc, mm = cached_stats
                # New stitch faces: assign to main component, outward_dot=1 (external)
                mesh_state["mesh_stats"] = (
                    np.concatenate([od, np.ones(n_added, dtype=od.dtype)]),
                    np.concatenate([fc, np.full(n_added, mc, dtype=fc.dtype)]),
                    mc,
                    np.concatenate([mm, np.ones(n_added, dtype=mm.dtype)]),
                )

        _clear_annotations("gap ", "disconnected ")
        await _apply_new_mesh(new_mesh)
        mesh_state["gap_clusters"] = None
        await _log(f"Remove gaps: {n_after:,} faces after bridging")

        return JSONResponse({
            "ok": True,
            "nFaces": n_after,
        })

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

        from skeliner.pre import find_rims, compute_mesh_stats

        mesh = mesh_state["mesh"]
        centroid = mesh_state["centroid"]

        # Reuse cached precomputed data
        precomputed = mesh_state.get("mesh_stats")
        if precomputed is None or len(precomputed[0]) != len(mesh.faces):
            precomputed = compute_mesh_stats(mesh, None, 5.0, True)
            mesh_state["mesh_stats"] = precomputed

        def _run():
            rims = find_rims(mesh, verbose=True, mesh_stats=precomputed)
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

    def _clear_annotations(*prefixes: str) -> None:
        """Remove highlight annotations whose label starts with any prefix."""
        if not annotations_path.exists():
            return
        try:
            ann = json.loads(annotations_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, ValueError):
            return
        if "highlights" not in ann:
            return
        ann["highlights"] = [
            h for h in ann["highlights"]
            if not any(h.get("label", "").startswith(p) for p in prefixes)
        ]
        annotations_path.write_text(
            json.dumps(ann, separators=(",", ":")), encoding="utf-8"
        )

    async def _apply_new_mesh(new_mesh):
        """Replace the current mesh with a modified one and broadcast."""
        # Save current mesh for undo
        old = mesh_state["mesh"]
        if old is not None:
            _undo_stack.append(old)
            if len(_undo_stack) > _UNDO_LIMIT:
                _undo_stack.pop(0)

        mesh_state["mesh"] = new_mesh
        # Organelle mask/precompute and soma survive — _rebuild_mesh
        # preserves all face/vertex indices (degenerate faces only).
        mesh_state["fusion_clusters"] = None
        mesh_state["disconnected"] = None
        mesh_state["hole_loops"] = None
        # Keep the original centroid so the camera doesn't shift
        original_centroid = mesh_state["centroid"]
        buffers = _mesh_to_buffers(new_mesh, centroid=original_centroid)
        mesh_state["buffers"] = buffers

        # Both vertex and face indices are stable — all annotations
        # survive mesh changes without remapping.

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
        cached_mask = _organelle_mask(mesh_state.get("organelles"))

        from skeliner.pre import _rebuild_mesh

        def _do_remove():
            if cached_mask is not None and len(cached_mask) == len(mesh.faces) and cached_mask.any():
                print(f"[skeliner.pre] Using cached organelle mask ({int(cached_mask.sum()):,} faces)")
                return _rebuild_mesh(mesh, ~cached_mask)
            else:
                reason = (
                    "no cached mask" if cached_mask is None
                    else f"length mismatch ({len(cached_mask)} vs {len(mesh.faces)})" if len(cached_mask) != len(mesh.faces)
                    else "mask is empty"
                )
                print(f"[skeliner.pre] No cached organelle mask ({reason}), running detection")
                from skeliner.pre import remove_organelles as _remove_organelles
                return _remove_organelles(mesh, verbose=True)

        new_mesh = await _run_with_log(_do_remove)

        n_after = n_before - int((new_mesh.faces == 0).all(axis=1).sum()) if len(new_mesh.faces) == n_before else len(new_mesh.faces)

        _clear_annotations("organelle:")
        await _apply_new_mesh(new_mesh)
        n_degen = int(np.all(new_mesh.faces == 0, axis=1).sum())
        await _log(f"Remove organelles: {n_before:,} faces, {n_degen:,} degenerated")
        return JSONResponse({
            "ok": True,
            "facesBefore": n_before,
            "facesRemoved": n_degen,
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

        _clear_annotations("fusion ")
        await _apply_new_mesh(new_mesh)
        n_degen = int(np.all(new_mesh.faces == 0, axis=1).sum())
        await _log(f"Remove fusions: {n_degen:,} faces degenerated")
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

        def _do_remove():
            if cached is not None and len(cached) == len(mesh.faces) and cached.any():
                print(f"[skeliner.pre] Using cached fragment mask ({int(cached.sum()):,} faces)")
                return _remove_fragments(mesh, fragments=cached, verbose=True)
            else:
                return _remove_fragments(mesh, verbose=True)

        new_mesh = await _run_with_log(_do_remove)

        _clear_annotations("fragments ")
        await _apply_new_mesh(new_mesh)
        n_degen = int(np.all(new_mesh.faces == 0, axis=1).sum())
        await _log(f"Remove fragments: {n_degen:,} faces degenerated")
        return JSONResponse({
            "ok": True,
            "facesBefore": n_before,
            "facesRemoved": n_degen,
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

        _clear_annotations("hole ")
        await _apply_new_mesh(new_mesh)
        await _log(f"Fill holes: {n_before:,} → {n_after:,} faces ({n_after - n_before:,} added)")
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

    async def edit_vertices(request):
        """Apply vertex position edits from the transform gizmo."""
        if mesh_state["mesh"] is None:
            return JSONResponse(
                {"ok": False, "error": "No mesh loaded"}, status_code=400
            )

        body = await request.json()
        face_edits = body.get("faceEdits", [])
        if not face_edits:
            return JSONResponse(
                {"ok": False, "error": "No edits provided"}, status_code=400
            )

        mesh = mesh_state["mesh"]
        verts = np.array(mesh.vertices, dtype=np.float64)  # copy
        faces = np.asarray(mesh.faces, dtype=np.int32)
        centroid = mesh_state["centroid"]

        # Push undo before modifying
        old = mesh_state["mesh"]
        if old is not None:
            _undo_stack.append(old)
            if len(_undo_stack) > _UNDO_LIMIT:
                _undo_stack.pop(0)

        # Apply edits: each faceEdit has {face: int, verts: [[x,y,z]*3]}
        # The frontend positions are centroid-subtracted, so add centroid back
        for edit in face_edits:
            fi = edit["face"]
            new_verts = edit["verts"]
            for vi in range(3):
                vert_idx = faces[fi, vi]
                verts[vert_idx] = [
                    new_verts[vi][0] + centroid[0],
                    new_verts[vi][1] + centroid[1],
                    new_verts[vi][2] + centroid[2],
                ]

        new_mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        mesh_state["mesh"] = new_mesh

        # Keep the original centroid
        original_centroid = mesh_state["centroid"]
        buffers = _mesh_to_buffers(new_mesh, centroid=original_centroid)
        mesh_state["buffers"] = buffers
        buffers["keepCamera"] = True
        await broadcast({"type": "mesh_loaded", "payload": buffers})

        n_faces_edited = len(face_edits)
        print(f"Edit vertices: {n_faces_edited} faces modified")

        return JSONResponse({"ok": True, "facesEdited": n_faces_edited})

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

    async def do_align_offsets(request):
        """Align offset layers: detect and correct Z-plane offsets."""
        if mesh_state["mesh"] is None:
            return JSONResponse(
                {"ok": False, "error": "No mesh loaded"}, status_code=400
            )

        from skeliner.pre import remove_offsets

        mesh = mesh_state["mesh"]

        def _run():
            return remove_offsets(mesh, verbose=True)

        new_mesh = await _run_with_log(_run)
        await _apply_new_mesh(new_mesh)
        mesh_state["offsets"] = None
        mesh_state["mesh_stats"] = None

        return JSONResponse({"ok": True})

    async def do_break_at_soma(request):
        """Break mesh at soma: classify components, expand soma + organelles."""
        if mesh_state["mesh"] is None:
            return JSONResponse(
                {"ok": False, "error": "No mesh loaded"}, status_code=400
            )
        soma = mesh_state.get("soma")
        if soma is None:
            return JSONResponse(
                {"ok": False, "error": "Run soma detection first"}, status_code=400
            )
        org_dict = mesh_state.get("organelles")
        if org_dict is None:
            return JSONResponse(
                {"ok": False, "error": "Run organelle detection first"},
                status_code=400,
            )
        org_mask = _organelle_mask(org_dict)

        from skeliner.pre import break_at_soma

        mesh = mesh_state["mesh"]

        def _run():
            return break_at_soma(mesh, soma, org_mask, verbose=True)

        new_soma, new_org, neurites = await _run_with_log(_run)

        mesh_state["soma"] = new_soma
        mesh_state["organelles"] = {
            "pocket": org_dict["pocket"],
            "isolated": org_dict["isolated"],
            "expanded": new_org & ~(org_dict["pocket"] | org_dict["isolated"]),
        }

        # Build annotations
        centroid = mesh_state["centroid"]
        faces = mesh.faces

        soma_vset = set(int(v) for v in new_soma.verts)
        soma_faces = [
            int(fi)
            for fi in range(len(faces))
            if sum(1 for v in faces[fi] if int(v) in soma_vset) >= 2
        ]

        ann = {}
        if annotations_path.exists():
            ann = json.loads(annotations_path.read_text(encoding="utf-8"))

        # Replace all highlights and ellipsoids with break_at_soma results
        highlights = []
        highlights.append(
            {
                "faces": soma_faces,
                "color": [0.9, 0.5, 0.9],
                "label": f"soma ({len(soma_faces):,}f, {len(new_soma.verts):,}v)",
            }
        )

        org_only = new_org & ~np.array(
            [
                sum(1 for v in faces[fi] if int(v) in soma_vset) >= 2
                for fi in range(len(faces))
            ],
            dtype=bool,
        )
        highlights.append(
            {
                "faces": np.where(org_only)[0].tolist(),
                "color": [1.0, 0.8, 0.0],
                "label": f"organelles ({int(org_only.sum()):,}f)",
            }
        )

        neurite_colors = [
            [0.2, 0.6, 1.0],
            [0.3, 1.0, 0.3],
            [1.0, 0.4, 0.1],
            [0.0, 0.9, 0.9],
            [1.0, 0.2, 0.6],
        ]
        for i, nf in enumerate(neurites):
            c = neurite_colors[i % len(neurite_colors)]
            highlights.append(
                {
                    "faces": nf.tolist(),
                    "color": c,
                    "label": f"neurite {i} ({len(nf):,}f)",
                }
            )

        ann["highlights"] = highlights
        ann["ellipsoids"] = [
            {
                "center": (new_soma.center - centroid).tolist(),
                "axes": new_soma.axes.tolist(),
                "R": new_soma.R.tolist(),
                "color": [0.9, 0.5, 0.9],
            }
        ]

        annotations_path.write_text(json.dumps(ann), encoding="utf-8")

        return JSONResponse(
            {
                "ok": True,
                "nNeurites": len(neurites),
                "somaVerts": len(new_soma.verts),
                "orgFaces": int(new_org.sum()),
            }
        )

    async def do_compact_mesh(request):
        """Compact mesh: remove degenerate faces, reindex vertices, remap annotations."""
        if mesh_state["mesh"] is None:
            return JSONResponse(
                {"ok": False, "error": "No mesh loaded"}, status_code=400
            )

        from skeliner.pre import compact_mesh

        mesh = mesh_state["mesh"]
        soma = mesh_state.get("soma")
        n_faces_before = len(mesh.faces)
        n_verts_before = len(mesh.vertices)

        def _run():
            return compact_mesh(mesh, soma=soma, verbose=True)

        clean, vert_map, remapped_soma = await _run_with_log(_run)

        # Build face map: old face index → new face index (or -1)
        good = ~np.all(mesh.faces == mesh.faces[:, :1], axis=1)
        face_map = np.full(n_faces_before, -1, dtype=np.int64)
        face_map[good] = np.arange(int(good.sum()), dtype=np.int64)

        # Compute new centroid and the delta from old
        old_centroid = mesh_state["centroid"].copy()
        buffers = _mesh_to_buffers(clean)
        new_centroid = np.asarray(buffers["centroid"], dtype=np.float32)
        delta = (old_centroid - new_centroid).tolist()

        # Remap annotations
        if annotations_path.exists():
            ann = json.loads(annotations_path.read_text(encoding="utf-8"))

            # Remap face-indexed highlights
            if "highlights" in ann:
                new_highlights = []
                for h in ann["highlights"]:
                    old_faces = np.array(h.get("faces", []), dtype=np.int64)
                    valid = old_faces < n_faces_before
                    mapped = face_map[old_faces[valid]]
                    new_faces = mapped[mapped >= 0].tolist()
                    if new_faces:
                        h = dict(h)
                        h["faces"] = new_faces
                        new_highlights.append(h)
                ann["highlights"] = new_highlights

            # Shift ellipsoid centers by centroid delta
            for ell in ann.get("ellipsoids", []):
                c = ell["center"]
                ell["center"] = [c[0] + delta[0], c[1] + delta[1], c[2] + delta[2]]

            # Shift edge group segments by centroid delta
            for eg in ann.get("edge_groups", []):
                for seg in eg.get("segments", []):
                    for pt in seg:
                        pt[0] += delta[0]
                        pt[1] += delta[1]
                        pt[2] += delta[2]

            annotations_path.write_text(json.dumps(ann), encoding="utf-8")

        # Update state — compact is destructive, clear all cached face-indexed data
        mesh_state["mesh"] = clean
        mesh_state["soma"] = remapped_soma
        mesh_state["organelles"] = None
        mesh_state["mesh_stats"] = None
        mesh_state["fusion_clusters"] = None
        mesh_state["disconnected"] = None
        mesh_state["gap_clusters"] = None
        mesh_state["hole_loops"] = None

        mesh_state["buffers"] = buffers
        mesh_state["centroid"] = new_centroid
        buffers["keepCamera"] = True
        await broadcast({"type": "mesh_loaded", "payload": buffers})
        await broadcast({"type": "annotations_updated"})

        # Re-send skeletons
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

        print(
            f"Compact: {n_verts_before:,} → {len(clean.vertices):,} verts, "
            f"{n_faces_before:,} → {len(clean.faces):,} faces"
        )
        return JSONResponse({
            "ok": True,
            "vertsBefore": n_verts_before,
            "vertsAfter": len(clean.vertices),
            "facesBefore": n_faces_before,
            "facesAfter": len(clean.faces),
        })

    async def export_mesh(request):
        """Export the current mesh as a downloadable file."""
        from starlette.responses import Response
        from skeliner.io import save_mesh
        import tempfile

        if mesh_state["mesh"] is None:
            return JSONResponse(
                {"ok": False, "error": "No mesh loaded"}, status_code=400
            )

        mesh = mesh_state["mesh"]
        fmt = request.query_params.get("format", "obj")
        prefix = request.query_params.get("prefix", "")
        tmp = Path(tempfile.mktemp(suffix=f".{fmt}"))
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: save_mesh(mesh, tmp))
        content = tmp.read_bytes()
        tmp.unlink(missing_ok=True)
        return Response(
            content=content,
            media_type="application/octet-stream",
            headers={"Content-Disposition": f'attachment; filename="{prefix}mesh_cleaned.{fmt}"'},
        )

    async def export_organelles(request):
        """Save organelle masks (pocket, isolated, expanded) + mesh stats."""
        from starlette.responses import Response
        from skeliner.io import save_organelles_npz
        import tempfile

        org = mesh_state.get("organelles")
        if org is None:
            return JSONResponse(
                {"ok": False, "error": "No organelle masks"}, status_code=400
            )

        prefix = request.query_params.get("prefix", "")
        tmp = Path(tempfile.mktemp(suffix=".npz"))
        save_organelles_npz(
            org["pocket"], org["isolated"],
            expanded=org["expanded"],
            mesh_stats=mesh_state.get("mesh_stats"),
            path=tmp,
        )
        content = tmp.read_bytes()
        tmp.unlink(missing_ok=True)
        return Response(
            content=content,
            media_type="application/octet-stream",
            headers={"Content-Disposition": f'attachment; filename="{prefix}organelles.npz"'},
        )

    async def export_soma(request):
        """Export the cached soma as a downloadable NPZ file."""
        from starlette.responses import Response
        import tempfile

        soma = mesh_state.get("soma")
        if soma is None:
            return JSONResponse(
                {"ok": False, "error": "No soma detected"}, status_code=400
            )

        from skeliner.io import save_soma_npz

        prefix = request.query_params.get("prefix", "")
        tmp = Path(tempfile.mktemp(suffix=".npz"))
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: save_soma_npz(soma, tmp))
        content = tmp.read_bytes()
        tmp.unlink(missing_ok=True)
        return Response(
            content=content,
            media_type="application/octet-stream",
            headers={"Content-Disposition": f'attachment; filename="{prefix}soma.npz"'},
        )

    async def export_annotations(request):
        """Export annotations as a downloadable JSON file."""
        from starlette.responses import Response

        if not annotations_path.exists():
            return JSONResponse(
                {"ok": False, "error": "No annotations"}, status_code=400
            )
        content = annotations_path.read_bytes()
        prefix = request.query_params.get("prefix", "")
        return Response(
            content=content,
            media_type="application/json",
            headers={"Content-Disposition": f'attachment; filename="{prefix}annotations.json"'},
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

        from skeliner.io import save_skeleton_swc, save_skeleton_npz

        loop = asyncio.get_event_loop()
        tmp = Path(tempfile.mktemp(suffix=f".{fmt}"))
        if fmt == "npz":
            await loop.run_in_executor(None, lambda: save_skeleton_npz(skel, tmp))
        else:
            await loop.run_in_executor(None, lambda: save_skeleton_swc(skel, tmp))
        content = tmp.read_bytes()
        tmp.unlink(missing_ok=True)

        prefix = request.query_params.get("prefix", "")
        return Response(
            content=content,
            media_type="application/octet-stream",
            headers={"Content-Disposition": f'attachment; filename="{prefix}skeleton.{fmt}"'},
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
            Route("/detect_offsets", detect_offsets, methods=["POST"]),
            Route("/save_offsets", save_offsets, methods=["POST"]),
            Route("/load_offsets", load_offsets, methods=["POST"]),
            Route("/remove_offsets", do_remove_offsets, methods=["POST"]),
            Route("/detect_organelles", detect_organelles, methods=["POST"]),
            Route("/check_fusion", check_fusion, methods=["POST"]),
            Route("/detect_fusions", detect_fusions, methods=["POST"]),
            Route("/detect_rims", detect_rims, methods=["POST"]),
            Route("/detect_holes", detect_holes, methods=["POST"]),
            Route("/remove_organelles", do_remove_organelles, methods=["POST"]),
            Route("/remove_fusions", do_remove_fusions, methods=["POST"]),
            Route("/detect_fragments", detect_fragments, methods=["POST"]),
            Route("/detect_disconnected", detect_disconnected, methods=["POST"]),
            Route("/detect_soma", detect_soma, methods=["POST"]),
            Route("/detect_gaps", detect_gaps, methods=["POST"]),
            Route("/remove_gaps", do_remove_gaps, methods=["POST"]),
            Route("/remove_fragments", do_remove_fragments, methods=["POST"]),
            Route("/fill_holes", do_fill_holes, methods=["POST"]),
            Route("/merge_selected", do_merge_selected, methods=["POST"]),
            Route("/edit_vertices", edit_vertices, methods=["POST"]),
            Route("/undo", undo_mesh, methods=["POST"]),
            Route("/align_offsets", do_align_offsets, methods=["POST"]),
            Route("/break_at_soma", do_break_at_soma, methods=["POST"]),
            Route("/compact_mesh", do_compact_mesh, methods=["POST"]),
            Route("/export_mesh", export_mesh, methods=["GET"]),
            Route("/export_skeleton", export_skeleton, methods=["GET"]),
            Route("/export_organelles", export_organelles, methods=["GET"]),
            Route("/export_soma", export_soma, methods=["GET"]),
            Route("/export_annotations", export_annotations, methods=["GET"]),
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
