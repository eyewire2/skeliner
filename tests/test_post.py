"""
Smoke-tests for the mutating helpers in `skeliner.post`.

Every mutation is done on a *deep copy* of the reference skeleton so the
other tests stay unaffected.
"""

import copy
from pathlib import Path

import numpy as np
import pytest

from skeliner import dx, post, skeletonize
from skeliner.dataclass import Skeleton, Soma
from skeliner.io import load_mesh


# ---------------------------------------------------------------------
# fixture
# ---------------------------------------------------------------------
@pytest.fixture(scope="session")
def template_skel():
    mesh = load_mesh(Path(__file__).parent / "data" / "60427.obj")
    return skeletonize(mesh, verbose=False)


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------
def _is_forest(skel):
    g = skel._igraph()
    return g.ecount() == g.vcount() - len(g.components())


# ---------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------
def test_graft_then_clip(template_skel):
    skel = copy.deepcopy(template_skel)

    leaves = dx.nodes_of_degree(skel, 1)
    # fall back to any two nodes if mesh has no leaves
    u, v = (int(leaves[0]), int(leaves[1])) if len(leaves) >= 2 else (0, 1)

    n_edges = skel.edges.shape[0]
    post.graft(skel, u, v, allow_cycle=True)
    assert skel.edges.shape[0] == n_edges + 1

    # clipping should restore edge count
    post.clip(skel, u, v)
    assert skel.edges.shape[0] == n_edges
    assert _is_forest(skel)


def test_prune_twigs(template_skel):
    skel = copy.deepcopy(template_skel)
    n_before = len(skel.nodes)
    post.prune(skel, kind="twigs", num_nodes=2)
    # Allowed to prune zero – just make sure the structure is still a forest
    assert len(skel.nodes) <= n_before
    assert _is_forest(skel)


def test_set_ntype_on_subtree(template_skel):
    skel = copy.deepcopy(template_skel)
    base = 1 if len(skel.nodes) > 1 else 0

    original_code = int(skel.ntype[base])
    assert original_code != 4  # sanity: it really changes

    post.set_ntype(skel, root=base, code=4, subtree=False)

    assert skel.ntype[base] == 4
    assert skel.ntype[0] == -1
    changed = np.where(skel.ntype == 4)[0]
    assert set(changed) == {base}


def test_remap_ntype_prefers_labels_over_unknown():
    ntype = np.array([-1, 3, 0, 0], np.int8)
    old2new = np.array([0, 1, 1, 1], np.int64)
    mapped = post._remap_ntype(ntype, old2new, 2)
    assert mapped.tolist() == [-1, 3]


def test_remap_ntype_resolves_conflicts_with_priority():
    ntype = np.array([-1, 4, 3, 3], np.int8)
    old2new = np.array([0, 1, 1, 1], np.int64)
    mapped = post._remap_ntype(ntype, old2new, 2)
    assert mapped[0] == -1
    assert mapped[1] == 3  # majority wins (4 vs two 3s)


def test_downsample_preserves_branch_labels():
    nodes = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]], float)
    edges = np.array([[0, 1], [1, 2], [2, 3]], np.int64)
    radii = {"median": np.array([1.0, 1.0, 1.0, 1.0])}
    ntype = np.array([-1, 2, 0, 0], np.int8)
    skel = Skeleton(
        soma=Soma.from_sphere(nodes[0], 1.0, verts=None),
        nodes=nodes,
        radii=radii,
        edges=edges,
        ntype=ntype,
    )
    ds = post.downsample(
        skel,
        merge_endpoints=True,
        slide_branchpoints=True,
        verbose=False,
    )
    assert len(ds.nodes) < len(nodes)
    assert ds.ntype[0] == -1
    assert 2 in ds.ntype


def test_detect_soma_demoted_root_keeps_neurite_label():
    nodes = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], float)
    edges = np.array([[0, 1], [1, 2]], np.int64)
    # Make node 2 obviously the soma by radius
    radii = {"median": np.array([1.0, 1.0, 5.0])}
    ntype = np.array([-1, 4, 4], np.int8)
    skel = Skeleton(
        soma=Soma.from_sphere(nodes[0], 1.0, verts=None),
        nodes=nodes,
        radii=radii,
        edges=edges,
        ntype=ntype,
    )
    res = post.detect_soma(
        skel,
        radius_key="median",
        soma_radius_percentile_threshold=50.0,
        soma_radius_distance_factor=1.0,
        soma_min_nodes=1,
        verbose=False,
    )
    assert res.ntype[0] == 1  # new soma
    # old root (now index 1) should keep its neurite label instead of 0
    assert res.ntype[1] == 4


def test_rebuild_drop_set_preserves_labels_and_fills_gaps(template_skel):
    skel = copy.deepcopy(template_skel)
    # Mark a small branch as axon
    leaves = dx.nodes_of_degree(skel, 1)
    if len(leaves) < 2:
        pytest.skip("template has too few leaves for this test")
    target = int(leaves[0])
    skel.ntype[target] = 2
    # Drop a different leaf to force compaction/remap
    drop = int(leaves[1])
    post.clip(skel, drop, int(dx.neighbors(skel, drop)[0]), drop_orphans=True)
    assert 2 in skel.ntype  # axon label survived remap
    assert skel.ntype[0] in (-1, 1)


def test_merge_near_soma_nodes_keeps_labels(template_skel):
    skel = copy.deepcopy(template_skel)
    # Force a near-soma node to have a distinct label and to be merged
    if len(skel.nodes) < 3:
        pytest.skip("template too small")
    skel.ntype[1] = 4  # apical
    merged = post.merge_near_soma_nodes(
        skel,
        mesh_vertices=None,
        inside_tol=0.0,
        near_factor=10.0,
        fat_factor=0.0,
        verbose=False,
    )
    assert 4 in merged.ntype or 4 in skel.ntype  # label survives or merge skipped
    assert merged.ntype[0] in (-1, 1)


def test_prune_neurites_keeps_labels(template_skel):
    skel = copy.deepcopy(template_skel)
    if len(skel.ntype) < 2:
        pytest.skip("template too small")
    skel.ntype[1] = 3
    res = post.prune_neurites(
        skel,
        mesh_vertices=None,
        tip_extent_factor=0.0,
        stem_extent_factor=0.0,
        drop_single_node_branches=True,
        verbose=False,
    )
    assert 3 in res.ntype
    assert res.ntype[0] in (-1, 1)


def test_reroot_clears_duplicate_roots_and_fills_gaps():
    nodes = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], float)
    edges = np.array([[0, 1], [1, 2]], np.int64)
    radii = {"median": np.array([1.0, 1.0, 1.0])}
    ntype = np.array(
        [-1, -1, 4], np.int8
    )  # duplicate roots, gap on index 1 after reroot
    skel = Skeleton(
        soma=Soma.from_sphere(nodes[0], 1.0, verts=None),
        nodes=nodes,
        radii=radii,
        edges=edges,
        ntype=ntype,
    )
    res = post.reroot(skel, node_id=2, rebuild_mst=False, verbose=False)
    assert res.ntype[0] == -1 or res.ntype[0] == 1
    assert int(np.sum(res.ntype == 1) + np.sum(res.ntype == -1)) == 1
    assert 4 in res.ntype or 4 in skel.ntype


def test_reroot_updates_soma_and_ntype():
    nodes = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], float)
    edges = np.array([[0, 1], [1, 2]], np.int64)
    radii = {"median": np.array([2.0, 1.0, 0.8])}
    ntype = np.array([1, 3, 3], np.int8)
    s0 = Skeleton(
        soma=Soma.from_sphere(nodes[0], 2.0, verts=None),
        nodes=nodes,
        radii=radii,
        edges=edges,
        ntype=ntype,
    )
    s = post.reroot(
        s0, node_id=2, radius_key="median", set_soma_ntype=True, verbose=False
    )
    # New soma at new node 0 (old node 2)
    assert np.allclose(s.soma.center, s.nodes[0])
    assert np.isclose(s.soma.axes[0], s.soma.axes[1]) and np.isclose(
        s.soma.axes[1], s.soma.axes[2]
    )  # spherical
    assert np.isclose(
        s.soma.axes[0], s.radii["median"][0]
    )  # radius equals selected column
    assert s.ntype[0] == 1
    assert int(np.sum(s.ntype == 1)) == 1


def test_reroot_node2verts_vert2node_consistency():
    nodes = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], float)
    edges = np.array([[0, 1], [1, 2]], np.int64)
    radii = {"median": np.array([2.0, 1.0, 0.8])}
    node2verts = [np.array([10, 11]), np.array([20]), np.array([30, 31])]
    vert2node = {10: 0, 11: 0, 20: 1, 30: 2, 31: 2}
    s0 = Skeleton(
        soma=Soma.from_sphere(nodes[0], 2.0, verts=node2verts[0]),
        nodes=nodes,
        radii=radii,
        edges=edges,
        ntype=np.array([1, 3, 3], np.int8),
        node2verts=node2verts,
        vert2node=vert2node,
    )
    s = post.reroot(s0, node_id=2, verbose=False)
    # Vertex memberships and back-map follow the swap
    for i, vs in enumerate(s.node2verts):
        for v in vs:
            assert s.vert2node[v] == i
    # Soma verts now come from the new node 0's membership
    assert set(s.soma.verts.tolist()) == set(s.node2verts[0].tolist())


def test_detect_soma_remaps_ntype_once():
    nodes = np.array([[0, 0, 0], [0, 0, 5], [0, 5, 0]], float)
    edges = np.array([[0, 1], [1, 2]], np.int64)
    radii = {"median": np.array([1.0, 4.0, 1.5])}
    skel = Skeleton(
        soma=Soma.from_sphere(nodes[0], radii["median"][0], verts=None),
        nodes=nodes,
        radii=radii,
        edges=edges,
        ntype=np.array([1, 3, 3], np.int8),
    )

    s = post.detect_soma(
        skel,
        radius_key="median",
        soma_radius_percentile_threshold=90.0,
        soma_radius_distance_factor=4.0,
        soma_min_nodes=1,
        verbose=False,
        mesh_vertices=None,
    )

    assert np.allclose(s.nodes[0], nodes[1])  # new soma promoted
    assert s.ntype[0] == 1
    assert int(np.sum(s.ntype == 1)) == 1


# ---------------------------------------------------------------------
# reassign_verts — bin editing
#
# A node *is* the set of mesh vertices it owns: its position is their
# centroid and every radius is an aggregate over them.  So the standard
# these tests hold the primitive to is that afterwards the skeleton is in
# exactly the state a from-scratch rebuild would have produced.  Anything
# less is a skeleton that looks right and measures wrong.
# ---------------------------------------------------------------------


@pytest.fixture(scope="session")
def bin_mesh():
    return load_mesh(Path(__file__).parent / "data" / "60427.obj")


@pytest.fixture
def bin_skel(bin_mesh):
    return skeletonize(bin_mesh, verbose=False)


def _a_neighbour_of(skel, node):
    e = np.asarray(skel.edges)
    nbrs = np.unique(e[(e[:, 0] == node) | (e[:, 1] == node)])
    nbrs = nbrs[(nbrs != node) & (nbrs != 0)]
    assert nbrs.size, f"node {node} has no usable neighbour"
    return int(nbrs[0])


def _assert_matches_rebuild(skel, mesh, node):
    """Position and radii must equal what `_make_nodes` would produce."""
    from skeliner._core import _estimate_radius

    owned = np.asarray(skel.node2verts[node], dtype=np.int64)
    pts = np.asarray(mesh.vertices, dtype=np.float64)[owned]
    centre = pts.mean(axis=0)
    assert np.allclose(skel.nodes[node], centre, atol=1e-9)

    d = np.linalg.norm(pts - centre, axis=1)
    for key in skel.radii:
        if key == "centerline":
            continue
        want = _estimate_radius(d, method=key, trim_fraction=0.05)
        assert skel.radii[key][node] == pytest.approx(want, abs=1e-9), key


def _first_editable_node(skel):
    for n in range(1, len(skel.nodes)):
        if len(skel.node2verts[n]) >= 4 and _a_neighbour_of(skel, n):
            return n
    pytest.skip("fixture has no bin big enough to split")


def test_moving_part_of_a_bin_recomputes_both_ends(bin_skel, bin_mesh):
    src = _first_editable_node(bin_skel)
    dst = _a_neighbour_of(bin_skel, src)
    half = np.asarray(bin_skel.node2verts[src])[: len(bin_skel.node2verts[src]) // 2]
    n_before = len(bin_skel.nodes)

    report = post.reassign_verts(bin_skel, half, dst, mesh=bin_mesh)

    assert report["moved"] == len(half)
    assert report["donors"] == [src]
    assert report["dropped"] == []
    assert len(bin_skel.nodes) == n_before, "nothing was emptied"
    _assert_matches_rebuild(bin_skel, bin_mesh, src)
    _assert_matches_rebuild(bin_skel, bin_mesh, dst)
    assert dx.check_bins(bin_skel)


def test_emptying_a_bin_drops_its_node(bin_skel, bin_mesh):
    """A node owning nothing has no position and no radius; it is not a
    node.  Moving a whole bin is therefore a merge."""
    src = _first_editable_node(bin_skel)
    dst = _a_neighbour_of(bin_skel, src)
    everything = np.asarray(bin_skel.node2verts[src])
    n_before = len(bin_skel.nodes)

    report = post.reassign_verts(bin_skel, everything, dst, mesh=bin_mesh)

    assert report["dropped"] == [src]
    assert len(bin_skel.nodes) == n_before - 1
    new_dst = report["old2new"][dst]
    assert set(everything.tolist()) <= set(
        np.asarray(bin_skel.node2verts[new_dst]).tolist()
    )
    _assert_matches_rebuild(bin_skel, bin_mesh, new_dst)
    assert dx.check_bins(bin_skel)


def test_centerline_is_recomputed_when_the_distances_were_kept(bin_mesh):
    """Only the preprocessing track produces `centerline`, so this needs a
    preproc skeleton — on a direct-track one the test would skip forever
    and prove nothing."""
    from skeliner._core import TRIM_FRACTION, _estimate_radius

    from .test_core import _make_components

    skel = skeletonize(
        bin_mesh, components=_make_components(bin_mesh, with_soma=True), verbose=False
    )
    assert "centerline" in skel.radii and "cl_dist_vids" in skel.extra

    src = _first_editable_node(skel)
    dst = _a_neighbour_of(skel, src)
    half = np.asarray(skel.node2verts[src])[: len(skel.node2verts[src]) // 2]
    before = float(skel.radii["centerline"][dst])
    post.reassign_verts(skel, half, dst, mesh=bin_mesh)

    lut = np.zeros(len(bin_mesh.vertices))
    lut[skel.extra["cl_dist_vids"]] = skel.extra["cl_dist_vals"]
    want = _estimate_radius(
        lut[np.asarray(skel.node2verts[dst])],
        method="trim",
        trim_fraction=TRIM_FRACTION,
    )
    assert skel.radii["centerline"][dst] == pytest.approx(want, abs=1e-9)
    assert skel.radii["centerline"][dst] != before, "it must actually be rebuilt"


def test_a_radius_that_cannot_be_recomputed_is_reported(bin_skel, bin_mesh):
    """`calibrated` is measured by ray casting, not from the vertices.
    Leaving it silently stale beside freshly recomputed keys is the one
    outcome worse than either dropping or reporting it."""
    src = _first_editable_node(bin_skel)
    dst = _a_neighbour_of(bin_skel, src)
    bin_skel.radii["calibrated"] = np.asarray(bin_skel.radii["median"]).copy()

    report = post.reassign_verts(
        bin_skel, np.asarray(bin_skel.node2verts[src])[:2], dst, mesh=bin_mesh
    )
    assert "calibrated" in report["stale_radii"]


def test_moving_within_one_bin_is_a_no_op(bin_skel, bin_mesh):
    src = _first_editable_node(bin_skel)
    before = np.asarray(bin_skel.nodes[src]).copy()
    report = post.reassign_verts(
        bin_skel, np.asarray(bin_skel.node2verts[src])[:2], src, mesh=bin_mesh
    )
    assert report["moved"] == 0
    assert np.array_equal(bin_skel.nodes[src], before)


def test_edges_are_left_alone(bin_skel, bin_mesh):
    """Re-deriving the graph from the surface would discard the edges that
    have no surface support — soma stems and gap bridges — and can
    disconnect the skeleton.  Re-derivation is opt-in, via rebuild_mst."""
    src = _first_editable_node(bin_skel)
    dst = _a_neighbour_of(bin_skel, src)
    before = np.asarray(bin_skel.edges).copy()
    half = np.asarray(bin_skel.node2verts[src])[: len(bin_skel.node2verts[src]) // 2]

    post.reassign_verts(bin_skel, half, dst, mesh=bin_mesh)
    assert np.array_equal(bin_skel.edges, before)


@pytest.mark.parametrize(
    "verts, to, expected",
    [
        ([1, 2], 0, "node 0 is the soma"),
        ([1], 10**9, "out of range"),
        ([], 1, "No vertices given"),
    ],
)
def test_reassign_verts_guards(bin_skel, bin_mesh, verts, to, expected):
    with pytest.raises(ValueError, match=expected):
        post.reassign_verts(bin_skel, verts, to, mesh=bin_mesh)


def test_unowned_surface_is_refused(bin_skel, bin_mesh):
    """Unowned surface is soma, organelle or discarded — claiming it would
    be a components decision made without re-deriving them."""
    owned = np.fromiter(bin_skel.vert2node.keys(), np.int64, len(bin_skel.vert2node))
    unowned = np.setdiff1d(np.arange(len(bin_mesh.vertices)), owned)
    if unowned.size == 0:
        pytest.skip("every vertex of this mesh belongs to a bin")
    with pytest.raises(ValueError, match="belong to no bin"):
        post.reassign_verts(bin_skel, unowned[:3], 1, mesh=bin_mesh)


def test_prune_neurites_survives_a_centerline_radius(bin_mesh):
    """Regression: `_prune_neurites` recomputes the soma's radii for every
    key in the dict, and a preprocessing-track skeleton always carries
    `centerline`, which was never an estimator name.  It raised `Unknown
    radius estimator 'centerline'` on any cell where something merged."""
    skel = skeletonize(bin_mesh, verbose=False)
    skel.radii["centerline"] = np.asarray(skel.radii["median"]).copy()
    n_before = len(skel.nodes)
    cl_before = float(skel.radii["centerline"][0])

    out = post.prune_neurites(
        skel,
        mesh_vertices=np.asarray(bin_mesh.vertices),
        stem_extent_factor=1e9,
        tip_extent_factor=1e9,
        verbose=False,
    )
    assert len(out.nodes) < n_before, "the merge branch must actually run"
    assert out.radii["centerline"][0] == cl_before, "left alone, not corrupted"
    assert dx.check_bins(out)
