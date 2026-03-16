"""
Interactive mesh viewer for skeliner.

Usage:
    skeliner view path/to/mesh.obj

Launches a local web server with a Three.js viewer. Camera state and
visible face/vertex IDs are written to a JSON state file that can be
read (and written) by external tools (e.g. Claude).
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


def _compute_face_winding(mesh: trimesh.Trimesh, offset: float = 5.0) -> np.ndarray:
    try:
        import igl
    except ImportError:
        return np.zeros(len(mesh.faces), dtype=np.float32)

    V = np.asarray(mesh.vertices, dtype=np.float64)
    F = np.asarray(mesh.faces, dtype=np.int32)
    centroids = mesh.triangles_center
    normals = mesh.face_normals

    query_out = centroids + offset * normals
    query_in = centroids - offset * normals

    wn_out = igl.fast_winding_number(V, F, query_out).astype(np.float32)
    wn_in = igl.fast_winding_number(V, F, query_in).astype(np.float32)

    score = (wn_out + wn_in) / 2.0
    return score


def _get_viewer_html() -> str:
    html_path = Path(__file__).parent / "viewer.html"
    return html_path.read_text(encoding="utf-8")


def _create_app(mesh_path: str | Path, port: int = 8777):
    """Create the Starlette app."""
    from starlette.applications import Starlette
    from starlette.responses import HTMLResponse, JSONResponse
    from starlette.routing import Route, WebSocketRoute
    from starlette.websockets import WebSocket

    mesh_path = Path(mesh_path)
    mesh = trimesh.load_mesh(str(mesh_path), process=False)

    print(f"Loaded mesh: {len(mesh.vertices):,} vertices, {len(mesh.faces):,} faces")
    print("Computing face winding numbers...")
    face_wn = _compute_face_winding(mesh)
    print(f"Winding number range: [{face_wn.min():.3f}, {face_wn.max():.3f}]")

    buffers = _mesh_to_buffers(mesh)
    buffers["faceWindingNumbers"] = face_wn.tolist()

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

    # Write initial state with mesh metadata
    state_path.write_text(json.dumps({
        "mesh": {
            "path": str(mesh_path.resolve()),
            "nVertices": len(mesh.vertices),
            "nFaces": len(mesh.faces),
        },
    }, indent=2), encoding="utf-8")

    connected_clients: list[WebSocket] = []

    # ── Routes ────────────────────────────────────────────────────────

    async def index(request):
        return HTMLResponse(_get_viewer_html())

    async def get_mesh(request):
        return JSONResponse(buffers)

    async def get_state(request):
        if state_path.exists():
            return JSONResponse(json.loads(state_path.read_text(encoding="utf-8")))
        return JSONResponse({})

    async def get_annotations(request):
        if annotations_path.exists():
            return JSONResponse(json.loads(annotations_path.read_text(encoding="utf-8")))
        return JSONResponse({})

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

    async def detect_avocados(request):
        """Run avocado detection and write results to annotations."""
        import asyncio
        from skeliner.pre import _outward_dot, _filter_small_clusters

        def _run():
            median_edge = float(np.median(mesh.edges_unique_length))
            radius = 5.0 * median_edge
            dots = _outward_dot(mesh, radius=radius)
            avocado = _filter_small_clusters(mesh, dots < 0, min_cluster_size=5)
            return [int(fi) for fi in np.where(avocado)[0]]

        loop = asyncio.get_event_loop()
        faces = await loop.run_in_executor(None, _run)

        ann = {"highlights": [
            {"faces": faces, "color": [1, 0.15, 0.15], "label": "avocado"},
        ]}
        annotations_path.write_text(json.dumps(ann), encoding="utf-8")
        return JSONResponse({"ok": True, "nFaces": len(faces)})

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
                elif msg.get("type") == "manual_highlight":
                    current = {}
                    if state_path.exists():
                        current = json.loads(state_path.read_text(encoding="utf-8"))
                    current["manualHighlight"] = msg["payload"].get("manualHighlight", [])
                    state_path.write_text(
                        json.dumps(current, indent=2), encoding="utf-8"
                    )
        except Exception:
            pass
        finally:
            connected_clients.remove(ws)

    # ── File watcher (annotations + camera commands → push to browser) ─

    html_path = Path(__file__).parent / "viewer.html"

    async def file_watcher():
        last_ann_mtime = 0.0
        last_cam_mtime = 0.0
        last_html_mtime = html_path.stat().st_mtime if html_path.exists() else 0.0
        while True:
            await asyncio.sleep(0.5)
            # Watch annotations
            try:
                mtime = annotations_path.stat().st_mtime
                if mtime > last_ann_mtime:
                    last_ann_mtime = mtime
                    content = annotations_path.read_text(encoding="utf-8")
                    for ws in connected_clients:
                        try:
                            await ws.send_text(json.dumps({
                                "type": "annotations",
                                "payload": json.loads(content),
                            }))
                        except Exception:
                            pass
            except FileNotFoundError:
                pass
            # Watch camera commands
            try:
                mtime = camera_cmd_path.stat().st_mtime
                if mtime > last_cam_mtime:
                    last_cam_mtime = mtime
                    content = camera_cmd_path.read_text(encoding="utf-8")
                    parsed = json.loads(content)
                    if parsed:
                        for ws in connected_clients:
                            try:
                                await ws.send_text(json.dumps({
                                    "type": "camera_command",
                                    "payload": parsed,
                                }))
                            except Exception:
                                pass
            except FileNotFoundError:
                pass
            # Watch viewer.html for hot reload
            try:
                mtime = html_path.stat().st_mtime
                if mtime > last_html_mtime:
                    last_html_mtime = mtime
                    for ws in connected_clients:
                        try:
                            await ws.send_text(json.dumps({"type": "reload"}))
                        except Exception:
                            pass
            except FileNotFoundError:
                pass

    async def on_startup():
        asyncio.create_task(file_watcher())

    app = Starlette(
        routes=[
            Route("/", index),
            Route("/mesh", get_mesh),
            Route("/state", get_state, methods=["GET"]),
            Route("/update_state", post_state, methods=["POST"]),
            Route("/update_selection", post_selection, methods=["POST"]),
            Route("/detect_avocados", detect_avocados, methods=["POST"]),
            WebSocketRoute("/ws", ws_endpoint),
        ],
        on_startup=[on_startup],
    )

    return app


def view(
    mesh_path: str | Path,
    *,
    host: str = "127.0.0.1",
    port: int = 8777,
    no_browser: bool = False,
):
    """Launch the interactive mesh viewer."""
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
