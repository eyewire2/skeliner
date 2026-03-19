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
import json
import os
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

    # ── Broadcast helper ──────────────────────────────────────────────
    async def broadcast(msg: dict):
        data = json.dumps(msg)
        for ws in connected_clients:
            try:
                await ws.send_text(data)
            except Exception:
                pass

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
        form = await request.form()
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
        """Run organelle detection matching remove_organelles() exactly."""
        if mesh_state["mesh"] is None:
            return JSONResponse({"ok": False, "error": "No mesh loaded"}, status_code=400)

        from skeliner.pre import _outward_dot, _filter_small_clusters
        import igraph as ig

        mesh = mesh_state["mesh"]

        def _run():
            # Step 1 – outward dot scoring
            median_edge = float(np.median(mesh.edges_unique_length))
            radius = 5.0 * median_edge
            outward_dots = _outward_dot(mesh, radius=radius)
            organelle = _filter_small_clusters(
                mesh, outward_dots < 0, min_cluster_size=5
            )

            # Step 3 – flag internal disconnected fragments
            edge_set = set()
            for face in mesh.faces:
                for i in range(3):
                    a, b = int(face[i]), int(face[(i + 1) % 3])
                    edge_set.add((min(a, b), max(a, b)))
            g = ig.Graph(
                n=len(mesh.vertices),
                edges=list(edge_set),
                directed=False,
            )
            comps = g.connected_components()
            main_ci = max(range(len(comps)), key=lambda i: len(comps[i]))

            if len(comps) > 1:
                vert_comp = np.full(len(mesh.vertices), -1, dtype=np.intp)
                for ci, cl in enumerate(comps):
                    for v in cl:
                        vert_comp[v] = ci
                face_comp = vert_comp[mesh.faces[:, 0]]
                for ci in range(len(comps)):
                    if ci == main_ci:
                        continue
                    comp_face_idx = np.where(face_comp == ci)[0]
                    if len(comp_face_idx) == 0:
                        continue
                    if outward_dots[comp_face_idx].mean() < 0:
                        organelle[comp_face_idx] = True

            return [int(fi) for fi in np.where(organelle)[0]]

        loop = asyncio.get_event_loop()
        faces = await loop.run_in_executor(None, _run)

        ann = {"highlights": [
            {"faces": faces, "color": [1, 0.15, 0.15], "label": "organelle"},
        ]}
        annotations_path.write_text(json.dumps(ann), encoding="utf-8")
        return JSONResponse({"ok": True, "nFaces": len(faces)})

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
            loops = find_holes(mesh)
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

        ev_loop = asyncio.get_event_loop()
        edge_groups, n_holes = await ev_loop.run_in_executor(None, _run)

        ann = {"edge_groups": edge_groups}
        annotations_path.write_text(json.dumps(ann), encoding="utf-8")
        return JSONResponse({"ok": True, "nHoles": n_holes})

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
            Route("/detect_holes", detect_holes, methods=["POST"]),
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
