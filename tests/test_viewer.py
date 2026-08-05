"""Tests for skeliner.plot.viewer — the interactive server.

The viewer mirrors an operation's terminal output into the browser over
the WebSocket, which is the only sign a long-running step is moving.
"""

import asyncio
import contextlib
import copy
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


# ── /bin ──────────────────────────────────────────────────────────────
#
# A node *is* the set of mesh vertices it owns, so Edit Partition renders a
# selected node as that surface.  A bin is small enough to fetch per click,
# which is why the partition is never shipped in bulk.


@pytest.fixture
def bin_client(tmp_path, monkeypatch):
    """A viewer with a mesh and a skeleton layer built from it.

    Subdivided three times so the tube is long enough in mesh edges to bin
    into several nodes; two subdivisions give four, which is too few for the
    neighbour and non-overlap checks to say anything.
    """
    from starlette.testclient import TestClient

    from skeliner import skeletonize
    from skeliner.plot import viewer as viewer_mod

    monkeypatch.setattr(viewer_mod, "_STATE_DIR", tmp_path, raising=False)
    mesh = trimesh.creation.cylinder(radius=40.0, height=1200.0, sections=24)
    mesh = mesh.subdivide().subdivide().subdivide()
    skel = skeletonize(mesh, verbose=False)
    assert len(skel.nodes) >= 6, "fixture must produce enough bins to test with"

    app = _create_app(preload_mesh=mesh, port=8913)
    with TestClient(app) as client:
        path = tmp_path / "skeleton.npz"
        skel.to_npz(path)
        client.post("/upload", files={"file": ("skeleton.npz", path.read_bytes())})
        name = next(iter(client.get("/skeletons").json()))
        yield client, name, mesh, skel


def test_a_bin_is_the_surface_its_node_owns(bin_client):
    client, name, mesh, skel = bin_client
    r = client.post("/bin", json={"name": name, "node": 3})
    assert r.status_code == 200
    body = r.json()

    assert body["node"] == 3
    assert body["nVerts"] == len(skel.node2verts[3])
    assert body["radius"] == pytest.approx(float(skel.r[3]))
    assert len(body["faces"]) > 0

    # every returned face really is a >=2-of-3 majority of that bin
    owned = set(np.asarray(skel.node2verts[3]).tolist())
    for f in body["faces"]:
        assert sum(v in owned for v in mesh.faces[f]) >= 2


def test_clicking_a_face_finds_the_bin_that_owns_it(bin_client):
    """The inverse lookup: what is in front of you is a strip of surface,
    not a node id."""
    client, name, _, _ = bin_client
    faces = client.post("/bin", json={"name": name, "node": 4}).json()["faces"]
    back = client.post("/bin", json={"name": name, "face": faces[0]}).json()
    assert back["node"] == 4


def test_bins_do_not_overlap(bin_client):
    client, name, _, skel = bin_client
    seen = set()
    for node in range(1, min(8, len(skel.nodes))):
        faces = client.post("/bin", json={"name": name, "node": node}).json()["faces"]
        assert not (seen & set(faces)), f"node {node} shares faces with an earlier bin"
        seen |= set(faces)


def test_node_0_is_reported_as_not_editable(bin_client):
    """Node 0's "bin" is ``soma.verts``, assigned wholesale by the soma
    stitch.  It belongs to Edit Mesh, not Edit Partition."""
    client, name, _, _ = bin_client
    assert (
        client.post("/bin", json={"name": name, "node": 0}).json()["editable"] is False
    )
    assert (
        client.post("/bin", json={"name": name, "node": 1}).json()["editable"] is True
    )


def test_a_face_owned_by_no_bin_is_refused(bin_client, tmp_path):
    """Soma, organelle and discarded surface is display-only here — saying
    so beats silently returning an empty selection.

    A bare tube has no unowned surface, so one is made: emptying a bin
    leaves its faces owned by nobody, which is exactly the shape of the
    real case.
    """
    from skeliner import dx

    client, _, mesh, skel = bin_client
    holed = copy.deepcopy(skel)
    orphaned = np.asarray(holed.node2verts[2], dtype=np.int64)
    holed.node2verts[2] = np.empty(0, dtype=np.int64)
    for v in orphaned:
        holed.vert2node.pop(int(v), None)

    path = tmp_path / "holed.npz"
    holed.to_npz(path)
    client.post("/upload", files={"file": ("holed.npz", path.read_bytes())})

    owner = dx.face_owner(holed, mesh)
    orphans = np.flatnonzero(owner < 0)
    assert orphans.size > 0, "emptying a bin must leave surface unowned"

    r = client.post("/bin", json={"name": "holed.npz", "face": int(orphans[0])})
    assert r.status_code == 400
    assert "no bin" in r.json()["error"]


@pytest.mark.parametrize(
    "body, expected",
    [
        ({}, "No such skeleton"),
        ({"name": "nope", "node": 0}, "No such skeleton"),
        ({"name": "skeleton.npz"}, "Pass either node or face"),
        ({"name": "skeleton.npz", "node": 10**9}, "node id out of range"),
        ({"name": "skeleton.npz", "face": 10**9}, "face id out of range"),
    ],
)
def test_bin_rejects_bad_input(bin_client, body, expected):
    client, name, _, _ = bin_client
    if body.get("name") == "skeleton.npz":
        body = {**body, "name": name}
    r = client.post("/bin", json=body)
    assert r.status_code == 400
    assert expected in r.json()["error"]


# ── skeleton invalidation ─────────────────────────────────────────────
#
# `node2verts` / `vert2node` index *the mesh the skeleton was built from*.
# After a face removal or a compaction they name different vertices, with
# no shape mismatch to catch it — so a skeleton kept across a mesh edit is
# not stale, it is wrong.  Staleness is derived by comparing the bound mesh
# by identity rather than set by a flag, which is what these pin down.


def _mesh_state_of(app):
    """Reach the app's private mesh_state, as the reassign fixture does."""
    for route in app.routes:
        fn = getattr(route, "endpoint", None)
        for cell in getattr(fn, "__closure__", None) or ():
            val = cell.cell_contents
            if isinstance(val, dict) and "pending_reassignment" in val:
                return val
    raise AssertionError("could not reach mesh_state")


def test_a_skeleton_built_from_another_mesh_is_refused(bin_client):
    client, name, mesh, _ = bin_client
    assert client.post("/bin", json={"name": name, "node": 3}).status_code == 200

    _mesh_state_of(client.app)["mesh"] = mesh.copy()  # same geometry, new object

    r = client.post("/bin", json={"name": name, "node": 3})
    assert r.status_code == 409
    assert "different mesh" in r.json()["error"]


def test_a_stale_skeleton_cannot_be_exported(bin_client):
    """The one irreversible step: exported radii would belong to surface
    that is no longer there, and nothing downstream could tell."""
    client, name, mesh, _ = bin_client
    assert client.get(f"/export_skeleton?name={name}").status_code == 200

    _mesh_state_of(client.app)["mesh"] = mesh.copy()

    r = client.get(f"/export_skeleton?name={name}")
    assert r.status_code == 409
    assert "Re-skeletonize" in r.json()["error"]


@pytest.mark.parametrize("route, payload", [("/remove_selected", "faces")])
def test_a_real_preprocessing_action_invalidates_the_skeleton(
    bin_client, route, payload
):
    """The synthetic swaps above prove the rule; this proves it is wired to
    the routes a user actually clicks, which is where a missed site shows."""
    client, name, _, _ = bin_client
    faces = client.post("/bin", json={"name": name, "node": 3}).json()["faces"]

    r = client.post(route, json={payload: faces[:5]})
    assert r.status_code == 200, r.text

    assert client.post("/bin", json={"name": name, "node": 3}).status_code == 409
    assert client.get(f"/export_skeleton?name={name}").status_code == 409


def test_undo_after_a_real_edit_restores_currency(bin_client):
    client, name, _, _ = bin_client
    faces = client.post("/bin", json={"name": name, "node": 3}).json()["faces"]
    client.post("/remove_selected", json={"faces": faces[:5]})
    assert client.post("/bin", json={"name": name, "node": 3}).status_code == 409

    assert client.post("/undo").status_code == 200
    assert client.post("/bin", json={"name": name, "node": 3}).status_code == 200


def test_returning_to_the_original_mesh_makes_it_current_again(bin_client):
    """Comparing by identity gets undo right for free: restoring the very
    mesh a skeleton was built from restores the same object."""
    client, name, mesh, _ = bin_client
    state = _mesh_state_of(client.app)
    original = state["mesh"]

    state["mesh"] = mesh.copy()
    assert client.post("/bin", json={"name": name, "node": 3}).status_code == 409

    state["mesh"] = original
    assert client.post("/bin", json={"name": name, "node": 3}).status_code == 200


def test_the_owner_cache_follows_the_skeleton(bin_client):
    """The cache is keyed on the mesh *and* the skeleton it was built from,
    so swapping the skeleton under an unchanged mesh recomputes.

    Driven by an actual swap, because a cache that is only ever read proves
    nothing.
    """
    client, name, mesh, skel = bin_client
    before = client.post("/bin", json={"name": name, "node": 3}).json()["faces"]
    assert before, "nothing to invalidate"

    # a different skeleton object: node 3 now owns what node 4 owned
    swapped = copy.deepcopy(skel)
    swapped.node2verts[3], swapped.node2verts[4] = (
        swapped.node2verts[4],
        swapped.node2verts[3],
    )
    swapped.vert2node = {}
    for nid, vs in enumerate(swapped.node2verts):
        for v in np.asarray(vs):
            swapped.vert2node[int(v)] = nid

    for route in client.app.routes:
        fn = getattr(route, "endpoint", None)
        for cell in getattr(fn, "__closure__", None) or ():
            val = cell.cell_contents
            if isinstance(val, dict) and name in val:
                val[name]["skeleton"] = swapped

    after = client.post("/bin", json={"name": name, "node": 3}).json()["faces"]
    assert after != before, "the cache served an answer for the old skeleton"
    assert after == sorted(
        int(f) for f in np.flatnonzero(_dx().face_owner(swapped, mesh) == 3)
    )


def _dx():
    from skeliner import dx

    return dx
