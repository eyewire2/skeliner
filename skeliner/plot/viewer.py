"""
Interactive mesh viewer for skeliner.

Usage:
    skeliner view path/to/mesh.obj

Launches a local web server with a Three.js viewer. Camera state and
visible face/vertex IDs are written to a JSON state file that can be
read (and written) by external tools (e.g. Claude).

Architecture:
    Browser (Three.js + WebGL)  <-- WebSocket -->  Python (FastAPI + uvicorn)
                                                        |
                                                   state.json  <-->  Claude
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
import webbrowser
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

# ---------------------------------------------------------------------------
# State file path (next to the mesh, or in /tmp)
# ---------------------------------------------------------------------------
_STATE_DIR = Path(os.environ.get("SKELINER_VIEW_STATE_DIR", "/tmp/skeliner_view"))


def _ensure_state_dir():
    _STATE_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Mesh → JSON-serialisable buffers
# ---------------------------------------------------------------------------

def _mesh_to_buffers(mesh: trimesh.Trimesh) -> dict[str, Any]:
    """Convert trimesh to flat arrays for the browser."""
    verts = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    # Centre the mesh at origin for easier navigation
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
    """Per-face winding number (overlap score). Returns (nFaces,) float32."""
    try:
        import igl
    except ImportError:
        # No igl → return zeros (viewer still works, just no heatmap)
        return np.zeros(len(mesh.faces), dtype=np.float32)

    V = np.asarray(mesh.vertices, dtype=np.float64)
    F = np.asarray(mesh.faces, dtype=np.int32)
    centroids = mesh.triangles_center
    normals = mesh.face_normals

    query_out = centroids + offset * normals
    query_in = centroids - offset * normals

    wn_out = igl.fast_winding_number(V, F, query_out).astype(np.float32)
    wn_in = igl.fast_winding_number(V, F, query_in).astype(np.float32)

    # Overlap score: average of inner/outer winding numbers
    score = (wn_out + wn_in) / 2.0
    return score


# ---------------------------------------------------------------------------
# HTML template (self-contained, Three.js from CDN)
# ---------------------------------------------------------------------------

def _get_viewer_html() -> str:
    html_path = Path(__file__).parent / "viewer.html"
    return html_path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

def _create_app(mesh_path: str | Path):
    """Create the FastAPI app."""
    try:
        from fastapi import FastAPI, WebSocket as FastAPIWebSocket
        from fastapi.responses import HTMLResponse
    except ImportError:
        raise ImportError(
            "The viewer requires fastapi and uvicorn. Install with:\n"
            "  pip install fastapi uvicorn[standard]"
        )

    mesh_path = Path(mesh_path)
    mesh = trimesh.load_mesh(str(mesh_path), process=False)

    print(f"Loaded mesh: {len(mesh.vertices):,} vertices, {len(mesh.faces):,} faces")
    print("Computing face winding numbers...")
    face_wn = _compute_face_winding(mesh)
    print(f"Winding number range: [{face_wn.min():.3f}, {face_wn.max():.3f}]")

    buffers = _mesh_to_buffers(mesh)
    buffers["faceWindingNumbers"] = face_wn.tolist()

    _ensure_state_dir()
    state_path = _STATE_DIR / "state.json"
    annotations_path = _STATE_DIR / "annotations.json"

    # Write initial empty annotation file
    if not annotations_path.exists():
        annotations_path.write_text("{}", encoding="utf-8")

    app = FastAPI()
    connected_clients: list[FastAPIWebSocket] = []

    @app.get("/")
    async def index():
        return HTMLResponse(_get_viewer_html())

    @app.get("/mesh")
    async def get_mesh():
        return buffers

    @app.get("/state")
    async def get_state():
        if state_path.exists():
            return json.loads(state_path.read_text(encoding="utf-8"))
        return {}

    @app.get("/annotations")
    async def get_annotations():
        if annotations_path.exists():
            return json.loads(annotations_path.read_text(encoding="utf-8"))
        return {}

    @app.websocket("/ws")
    async def websocket_endpoint(ws: FastAPIWebSocket):
        await ws.accept()
        connected_clients.append(ws)
        try:
            while True:
                data = await ws.receive_text()
                msg = json.loads(data)

                if msg.get("type") == "state_update":
                    # Browser sends camera + visible faces
                    state_path.write_text(
                        json.dumps(msg["payload"], indent=2),
                        encoding="utf-8",
                    )
                elif msg.get("type") == "selection":
                    # Browser sends selected face/vertex
                    current = {}
                    if state_path.exists():
                        current = json.loads(state_path.read_text(encoding="utf-8"))
                    current["selection"] = msg["payload"]
                    state_path.write_text(
                        json.dumps(current, indent=2),
                        encoding="utf-8",
                    )
        except Exception:
            pass
        finally:
            connected_clients.remove(ws)

    # Background task: watch annotations.json for changes from Claude
    @app.on_event("startup")
    async def watch_annotations():
        async def _watcher():
            last_mtime = 0.0
            while True:
                await asyncio.sleep(0.5)
                try:
                    mtime = annotations_path.stat().st_mtime
                    if mtime > last_mtime:
                        last_mtime = mtime
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

        asyncio.create_task(_watcher())

    return app


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def view(mesh_path: str | Path, *, host: str = "127.0.0.1", port: int = 8777):
    """Launch the interactive mesh viewer.

    Parameters
    ----------
    mesh_path
        Path to a mesh file (.obj, .ply, etc.)
    host
        Server bind address.
    port
        Server port.
    """
    try:
        import uvicorn
    except ImportError:
        raise ImportError(
            "The viewer requires uvicorn. Install with:\n"
            "  pip install uvicorn[standard]"
        )

    app = _create_app(mesh_path)

    url = f"http://{host}:{port}"
    print(f"\nSkeliner Viewer")
    print(f"  URL:          {url}")
    print(f"  State file:   {_STATE_DIR / 'state.json'}")
    print(f"  Annotations:  {_STATE_DIR / 'annotations.json'}")
    print(f"\nOpen in VSCode: Cmd+Shift+P → 'Simple Browser: Show' → {url}")
    print()

    # Open browser after a short delay
    threading.Timer(1.5, lambda: webbrowser.open(url)).start()

    uvicorn.run(app, host=host, port=port, log_level="warning")
