"""Tests for skeliner.plot.viewer — the interactive server.

The viewer mirrors an operation's terminal output into the browser over
the WebSocket, which is the only sign a long-running step is moving.
"""

import asyncio
import contextlib
import io
import threading

import numpy as np
import pytest
import trimesh

from skeliner.dataclass import Discarded, Neurites, Organelles, Soma
from skeliner.plot.viewer import _LogTee, _create_app
from skeliner.skeletonize import _timed


def _tee_output(emit):
    """Run *emit* through a ``_LogTee``.

    Returns what it broadcast and what reached the wrapped stream.
    """
    sent = []
    original = io.StringIO()

    async def broadcast(msg):
        sent.append(msg["text"])

    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    try:
        tee = _LogTee(original, loop, broadcast)
        with contextlib.redirect_stdout(tee):
            emit()
        tee.finish()
        # queued in order, so this one completing means all of them did
        asyncio.run_coroutine_threadsafe(asyncio.sleep(0), loop).result(5)
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5)
        loop.close()
    return sent, original.getvalue()


# ── _LogTee ───────────────────────────────────────────────────────────
#
# Driven through the real ``_timed`` helper rather than hand-written
# strings, so these cannot drift from what the pipeline actually prints.


def test_timed_stage_reaches_the_browser_as_one_line():
    """A stage is a label printed before the work and an elapsed time
    printed after, on the same line.  The browser must get the label
    immediately — it is the only sign a long stage is running — and then
    the finished line, never a bare "1.99 s" on its own."""

    def emit():
        with _timed("↳  build surface graph", verbose=True):
            pass

    sent, _ = _tee_output(emit)
    assert len(sent) == 2, sent
    assert sent[0].strip().startswith("↳  build surface graph")
    assert sent[0].endswith("…"), "the label must arrive before the timing"
    assert sent[1].startswith(sent[0]), "the finished line completes the label"
    assert sent[1].endswith(" s")


def test_timed_sub_messages_are_their_own_lines():
    def emit():
        with _timed("↳  skeletonize neurites", verbose=True) as log:
            log("neurite 0: 12 verts → 3 nodes")

    sent, _ = _tee_output(emit)
    assert len(sent) == 3, sent
    assert sent[2].strip() == "└─ neurite 0: 12 verts → 3 nodes"


def test_whole_lines_are_sent_once_each_and_reach_the_terminal():
    def emit():
        print("alpha")
        print("beta")

    sent, raw = _tee_output(emit)
    assert sent == ["alpha", "beta"]
    assert raw == "alpha\nbeta\n", "the terminal must still get everything"


# ── Reassign routes ───────────────────────────────────────────────────
#
# Driven through the real ASGI app, so the guards and the preview/apply
# handshake are exercised the way the page uses them.


def _tube_with_components():
    """A tube split by a soma band, with components already derived."""
    from skeliner import pre

    mesh = trimesh.creation.cylinder(radius=50.0, height=1000.0, sections=16)
    mesh = mesh.subdivide().subdivide()
    z = mesh.vertices[mesh.faces].mean(axis=1)[:, 2]
    band = np.flatnonzero(np.abs(z) <= 100.0)
    soma_verts = np.unique(np.asarray(mesh.faces)[band])
    soma = Soma.fit(mesh.vertices[soma_verts], verts=soma_verts)
    empty = np.zeros(len(mesh.faces), dtype=bool)
    org = Organelles(pocket=empty.copy(), isolated=empty.copy(), expanded=empty.copy())
    arbor = np.flatnonzero(~pre.soma_face_mask(mesh.faces, soma_verts))
    neurites = Neurites(pre._face_components_fast(mesh.faces, arbor))
    return mesh, soma, org, neurites


@pytest.fixture
def reassign_client(tmp_path, monkeypatch):
    """A viewer serving a tube whose components are already derived."""
    from starlette.testclient import TestClient

    from skeliner.plot import viewer as viewer_mod

    monkeypatch.setattr(viewer_mod, "_STATE_DIR", tmp_path, raising=False)
    mesh, soma, org, neurites = _tube_with_components()
    app = _create_app(preload_mesh=mesh, port=8912)

    # Reach the app's private state the way no caller normally would; the
    # alternative is re-running detection, which these tests are not about.
    state = None
    for route in app.routes:
        fn = getattr(route, "endpoint", None)
        for cell in getattr(fn, "__closure__", None) or ():
            val = cell.cell_contents
            if isinstance(val, dict) and "pending_reassignment" in val:
                state = val
        if state is not None:
            break
    assert state is not None, "could not reach mesh_state"

    with TestClient(app) as client:
        state["mesh"] = mesh
        state["soma"] = soma
        state["organelles"] = org
        state["neurites"] = neurites
        state["discarded"] = Discarded([])
        yield client, state, neurites


def test_reassign_refuses_before_components_exist(reassign_client):
    client, state, _ = reassign_client
    state["neurites"] = Neurites([])
    r = client.post("/reassign_preview", json={"faces": [0], "to": "soma"})
    assert r.status_code == 400
    assert "break_up_mesh" in r.json()["error"]


@pytest.mark.parametrize(
    "body, expected",
    [
        ({"faces": [], "to": "soma"}, "No faces selected"),
        ({"faces": [0], "to": "bogus"}, "to must be"),
        ({"faces": [10**9], "to": "soma"}, "face ids must lie"),
    ],
)
def test_reassign_preview_rejects_bad_input(reassign_client, body, expected):
    client, _, _ = reassign_client
    r = client.post("/reassign_preview", json=body)
    assert r.status_code == 400
    assert expected in r.json()["error"]


def test_apply_without_a_preview_is_refused(reassign_client):
    client, _, _ = reassign_client
    r = client.post("/reassign_apply")
    assert r.status_code == 400
    assert "preview first" in r.json()["error"]


def test_preview_reports_the_fringe_and_leaves_state_alone(reassign_client):
    client, state, neurites = reassign_client
    sel = neurites[0][:64].tolist()
    before = len(state["soma"].verts)

    r = client.post("/reassign_preview", json={"faces": sel, "to": "soma"})
    assert r.status_code == 200
    body = r.json()
    # the ≥2-of-3 rule drags neighbours along, so more leaves than was picked
    assert set(sel) <= set(body["leaving"])
    assert len(body["leaving"]) > len(sel)
    assert body["entering"] == []
    assert len(state["soma"].verts) == before, "preview must not commit"


def test_apply_commits_exactly_what_the_preview_promised(reassign_client):
    client, state, neurites = reassign_client
    sel = neurites[0][:64].tolist()

    preview = client.post("/reassign_preview", json={"faces": sel, "to": "soma"}).json()
    applied = client.post("/reassign_apply").json()

    assert applied["ok"]
    assert applied["nNeurites"] == preview["nNeurites"]
    assert applied["nDiscarded"] == preview["nDiscarded"]
    assert set(np.unique(np.asarray(state["mesh"].faces)[sel]).tolist()) <= set(
        state["soma"].verts.tolist()
    )


def test_cancel_retires_the_preview(reassign_client):
    client, _, neurites = reassign_client
    client.post(
        "/reassign_preview", json={"faces": neurites[0][:64].tolist(), "to": "soma"}
    )
    assert client.post("/reassign_cancel").json()["ok"]
    assert client.post("/reassign_apply").status_code == 400


def test_compacting_the_mesh_retires_the_preview(reassign_client):
    """Face ids do not survive compaction, so applying afterwards would
    reassign whatever now sits at those indices."""
    client, _, neurites = reassign_client
    client.post(
        "/reassign_preview", json={"faces": neurites[0][:64].tolist(), "to": "soma"}
    )
    assert client.post("/compact_mesh").status_code == 200
    assert client.post("/reassign_apply").status_code == 400


def test_organelle_target_writes_the_manual_mask(reassign_client):
    client, state, neurites = reassign_client
    sel = neurites[0][:32].tolist()
    client.post("/reassign_preview", json={"faces": sel, "to": "organelle"})
    client.post("/reassign_apply")
    assert state["organelles"].manual[sel].all()
    assert not state["organelles"].pocket.any()
