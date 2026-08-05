"""
Smoke-check every public diagnostic helper in `skeliner.dx`.

No hard-coded biology – we just compare against ground-truth values
computed on-the-fly with igraph so the test is independent of the mesh
content.
"""

import copy
from pathlib import Path

import numpy as np
import pytest

from skeliner import dx, post, skeletonize
from skeliner.io import load_mesh


# ---------------------------------------------------------------------
# shared fixture: skeleton of the reference mesh
# ---------------------------------------------------------------------
@pytest.fixture(scope="session")
def mesh():
    return load_mesh(Path(__file__).parent / "data" / "60427.obj")


@pytest.fixture(scope="session")
def skel(mesh):
    return skeletonize(mesh, verbose=False)


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------
def _igraph(skel):
    return skel._igraph()


# ---------------------------------------------------------------------
# individual tests
# ---------------------------------------------------------------------
def test_check_connectivity(skel):
    assert dx.check_connectivity(skel)


def test_connectivity_deprecated_alias_warns(skel):
    with pytest.warns(DeprecationWarning, match="check_connectivity"):
        assert dx.connectivity(skel)


def test_check_acyclicity(skel):
    assert dx.check_acyclicity(skel) is True
    # check_acyclicity(..., return_cycles=True) must return a boolean when acyclic
    assert dx.check_acyclicity(skel, return_cycles=True) is True


def test_acyclicity_deprecated_alias_warns(skel):
    with pytest.warns(DeprecationWarning, match="check_acyclicity"):
        assert dx.acyclicity(skel, return_cycles=True) is True


# ---------------------------------------------------------------------
# check_bins
#
# A wrong node2verts / vert2node is the one corruption that is invisible:
# the skeleton looks fine, exports fine, and carries radii belonging to
# the wrong surface.  These tests are what make it loud.
# ---------------------------------------------------------------------
class TestCheckBins:
    def test_a_fresh_skeleton_passes(self, skel):
        assert dx.check_bins(skel) is True

    def test_report_shape(self, skel):
        r = dx.check_bins(skel, return_report=True)
        assert r["ok"] is True
        assert r["n_nodes"] == len(skel.nodes)
        assert r["duplicated"].size == 0
        assert r["mismatched"].size == 0

    def test_a_wrong_vert2node_entry_fails(self, skel):
        bad = copy.deepcopy(skel)
        vid = int(bad.node2verts[5][0])
        bad.vert2node[vid] = 999_999
        assert dx.check_bins(bad) is False
        assert dx.check_bins(bad, return_report=True)["mismatched"].tolist() == [vid]

    def test_a_vertex_missing_from_vert2node_fails(self, skel):
        bad = copy.deepcopy(skel)
        del bad.vert2node[int(bad.node2verts[5][0])]
        assert dx.check_bins(bad) is False

    def test_two_arbor_bins_sharing_a_vertex_fails(self, skel):
        bad = copy.deepcopy(skel)
        stolen = bad.node2verts[7][:3]
        bad.node2verts[6] = np.concatenate([bad.node2verts[6], stolen])
        assert dx.check_bins(bad) is False
        assert dx.check_bins(bad, return_report=True)["duplicated"].size == 3

    def test_node_0_may_overlap_an_arbor_bin(self, skel):
        """Structural, not a defect.

        ``node2verts[0]`` is ``soma.verts`` wholesale while neurites are
        binned over the face-based arbor, so under the >=2-of-3 rule a
        boundary vertex belongs to both.  A check that failed on this
        would fire on every skeleton the pipeline produces.
        """
        ok = copy.deepcopy(skel)
        shared = ok.node2verts[9][:5]
        ok.node2verts[0] = np.concatenate([ok.node2verts[0], shared])
        assert dx.check_bins(ok) is True
        assert dx.check_bins(ok, return_report=True)["soma_overlap"] >= 5

    def test_disowned_surface_is_legal(self, skel):
        """Coverage is not an invariant: ``prune`` drops a twig's vertices
        on the floor, and soma / organelle / discarded surface is never
        owned in the first place."""
        pruned = copy.deepcopy(skel)
        post.prune(pruned, kind="twigs", num_nodes=3)
        assert dx.check_bins(pruned) is True

    def test_length_mismatch_is_caught(self, skel):
        bad = copy.deepcopy(skel)
        bad.node2verts = bad.node2verts[:-1]
        assert dx.check_bins(bad) is False
        assert "length" in dx.check_bins(bad, return_report=True)["reason"]

    def test_no_mesh_data_is_vacuously_ok(self, skel):
        bare = copy.deepcopy(skel)
        bare.node2verts = None
        bare.vert2node = None
        assert dx.check_bins(bare) is True

    def test_half_the_mesh_data_is_not(self, skel):
        bare = copy.deepcopy(skel)
        bare.vert2node = None
        assert dx.check_bins(bare) is False

    def test_fragmented_bins_are_reported_not_failed(self, skel, mesh):
        """The binning's reunite pass is capped at 8 rounds, so real cells
        keep a few multi-piece bins.  Reporting them is useful; failing on
        them would fire on the pipeline's own output."""
        r = dx.check_bins(skel, mesh=mesh, return_report=True)
        assert r["ok"] is True
        assert isinstance(r["fragmented"], dict)
        assert all(v > 1 for v in r["fragmented"].values())

    def test_face_owner_uses_the_same_rule_as_the_soma(self, skel, mesh):
        """Bins own vertices; everything that looks at a mesh works in
        faces.  The bridge is the >=2-of-3 majority rule, the same one
        ``pre.soma_face_mask`` uses."""
        owner = dx.face_owner(skel, mesh)
        assert owner.shape == (len(mesh.faces),)

        faces = np.asarray(mesh.faces)
        for fi in range(0, len(faces), max(1, len(faces) // 200)):
            owners = [skel.vert2node.get(int(v), -1) for v in faces[fi]]
            counts = {o: owners.count(o) for o in set(owners)}
            want = next((o for o, c in counts.items() if c >= 2), -1)
            assert owner[fi] == want, f"face {fi}: {owners}"

    def test_bin_faces_partition_the_owned_surface(self, skel, mesh):
        owner = dx.face_owner(skel, mesh)
        seen = np.zeros(len(mesh.faces), dtype=bool)
        for node in range(len(skel.nodes)):
            f = dx.bin_faces(skel, mesh, node, owner=owner)
            assert not seen[f].any(), f"node {node} claims a face already claimed"
            seen[f] = True
        assert seen.sum() == int((owner >= 0).sum())

    def test_bin_faces_accepts_a_cached_owner(self, skel, mesh):
        owner = dx.face_owner(skel, mesh)
        assert np.array_equal(
            dx.bin_faces(skel, mesh, 2), dx.bin_faces(skel, mesh, 2, owner=owner)
        )

    def test_face_owner_without_mesh_data(self, skel, mesh):
        bare = copy.deepcopy(skel)
        bare.vert2node = None
        assert (dx.face_owner(bare, mesh) == -1).all()

    def test_a_bin_split_across_the_surface_is_detected(self, skel, mesh):
        bad = copy.deepcopy(skel)
        # hand node 3 a vertex from the far end of the arbor: same bin,
        # two patches, no shared surface between them
        far = int(bad.node2verts[len(bad.node2verts) - 1][0])
        bad.node2verts[3] = np.append(bad.node2verts[3], far)
        bad.node2verts[len(bad.node2verts) - 1] = bad.node2verts[
            len(bad.node2verts) - 1
        ][1:]
        bad.vert2node[far] = 3
        r = dx.check_bins(bad, mesh=mesh, return_report=True)
        assert r["ok"] is True, "still a valid partition — just an ugly one"
        assert r["fragmented"].get(3, 1) > 1


# ---------------------------------------------------------------------
# edge_support
#
# Which node pairs the mesh joins, against which ones the tree carries.
# This is a classifier for the edge-editing verbs — restore (the surface
# supports it) versus graft (it does not) — and nothing more.  It is
# deliberately not a defect report: see the docstring, and the entry
# 2026-08-05-skeleton-as-derived-state in the labbook for the measurement
# that ruled that out.
# ---------------------------------------------------------------------
@pytest.fixture(scope="session")
def torus():
    """A mesh whose surface graph closes a cycle the tree has to cut.

    A tube gives a path, so its ``G`` and ``T`` are identical and there is
    nothing to say.  A ring is the smallest thing with a dropped edge.
    """
    import trimesh

    return trimesh.creation.torus(
        major_radius=300.0, minor_radius=60.0, major_sections=64, minor_sections=16
    )


@pytest.fixture(scope="session")
def torus_skel(torus):
    return skeletonize(torus, verbose=False)


@pytest.fixture
def tangled(skel):
    """A bin owning surface in two places, so several pairs are dropped.

    Half of one bin is handed to a distant node, which is what a densely
    packed arbor looks like from the graph's side: bins that share surface
    while sitting far apart in the tree.
    """
    bad = copy.deepcopy(skel)
    far = len(bad.nodes) - 1
    take = np.asarray(bad.node2verts[3])[: len(bad.node2verts[3]) // 2]
    bad.node2verts[3] = np.setdiff1d(np.asarray(bad.node2verts[3]), take)
    bad.node2verts[far] = np.union1d(np.asarray(bad.node2verts[far]), take)
    for v in take:
        bad.vert2node[int(v)] = far
    return bad


class TestEdgeSupport:
    def test_a_tree_shaped_arbor_drops_nothing(self, skel, mesh):
        """The common case: every adjacency the surface has is in the tree,
        so no pair is a restore and every graft is honestly a graft."""
        rep = dx.edge_support(skel, mesh)
        assert rep["dropped"] == []
        assert rep["n_tree"] == len({tuple(sorted(map(int, e))) for e in skel.edges})

    def test_a_ring_drops_the_one_edge_that_closes_it(self, torus_skel, torus):
        rep = dx.edge_support(torus_skel, torus)
        assert len(rep["dropped"]) == 1

    def test_a_dropped_pair_is_real_surface_adjacency(self, torus_skel, torus):
        """Not an invention: the two bins really do share mesh edges, which
        is the whole difference between a restore and a graft."""
        u, v = dx.edge_support(torus_skel, torus)["dropped"][0]
        v2n = torus_skel.vert2node
        touching = [
            (a, b)
            for a, b in np.asarray(torus.edges_unique)
            if {v2n.get(int(a), -1), v2n.get(int(b), -1)} == {u, v}
        ]
        assert touching, "a dropped pair names bins that do not touch"

    def test_dropped_and_the_tree_are_disjoint_and_canonical(self, tangled, mesh):
        rep = dx.edge_support(tangled, mesh)
        tree = {tuple(sorted(map(int, e))) for e in tangled.edges}
        assert rep["dropped"]
        for u, v in rep["dropped"]:
            assert (u, v) not in tree
            assert u < v

    def test_unsupported_are_tree_edges_with_no_surface(self, skel, mesh):
        """The soma stems and the ``bridge_gaps`` bridges — the edges a
        re-span of ``G`` would silently delete."""
        rep = dx.edge_support(skel, mesh)
        tree = {tuple(sorted(map(int, e))) for e in skel.edges}
        assert rep["unsupported"], "reference skeleton has soma stems"
        for u, v in rep["unsupported"]:
            assert (u, v) in tree
        assert not set(rep["unsupported"]) & set(rep["dropped"])

    def test_clipping_makes_the_cut_edge_a_dropped_pair(self, torus_skel, torus):
        """The two halves of the classification meet: cut a supported tree
        edge and it moves from the tree into the surface-supported set, so
        putting it back is offered as a restore rather than as a graft."""
        cut = copy.deepcopy(torus_skel)
        u, v = (int(x) for x in cut.edges[len(cut.edges) // 2])
        post.clip(cut, u, v)
        rep = dx.edge_support(cut, torus)
        assert (min(u, v), max(u, v)) in rep["dropped"]

    def test_a_skeleton_without_bins_is_refused(self, skel, mesh):
        bare = copy.deepcopy(skel)
        bare.vert2node = None
        with pytest.raises(ValueError, match="vert2node"):
            dx.edge_support(bare, mesh)


def test_degree_and_neighbors_match_igraph(skel):
    g = _igraph(skel)
    degrees_ref = g.degree()
    # vector query
    assert np.array_equal(
        [dx.degree(skel, node_id=n) for n in range(len(skel.nodes))], degrees_ref
    )
    # scalar query + neighbors
    nid = 0  # arbitrary but deterministic
    assert dx.degree(skel, nid) == degrees_ref[nid]
    assert set(dx.neighbors(skel, nid)) == set(g.neighbors(nid))


def test_nodes_of_degree(skel):
    g = _igraph(skel)
    deg = np.asarray(g.degree())
    for k in (0, 1, 2, 3):  # 0 included on purpose – should be empty
        expected = {int(i) for i in np.where(deg == k)[0] if i != 0}
        got = set(dx.nodes_of_degree(skel, k))
        assert got == expected


def test_branches_and_twigs_lengths(skel):
    # We do not assume k actually exists – just assert path lengths.
    for k in (1, 2, 3):
        for path in dx.branches_of_length(skel, k):
            assert len(path) == k
        for twig in dx.twigs_of_length(skel, k):
            assert len(twig) == k


def test_suspicious_tips_are_leaves(skel):
    tips = dx.suspicious_tips(skel)  # may be empty
    if not tips:
        return
    g = _igraph(skel)
    deg = np.asarray(g.degree())
    assert all(deg[t] == 1 and t != 0 for t in tips)


def test_distance_point_queries(skel):
    unit = skel.meta.get("unit", "nm")
    soma = skel.nodes[0]
    # distance to a node should be zero irrespective of units
    assert dx.distance(skel, soma, point_unit=unit) == pytest.approx(0.0, abs=1e-9)

    # take edge midpoint and move it away along a perpendicular direction
    u, v = map(int, skel.edges[0])
    edge_vec = skel.nodes[v] - skel.nodes[u]
    mid = 0.5 * (skel.nodes[u] + skel.nodes[v])

    # robust perpendicular
    perp = None
    for axis in np.eye(3):
        candidate = np.cross(edge_vec, axis)
        norm = np.linalg.norm(candidate)
        if norm > 1e-9:
            perp = candidate / norm
            break
    if perp is None:  # degenerate edge, fall back to arbitrary axis
        perp = np.array([1.0, 0.0, 0.0])

    offset_nm = perp * 500.0  # 500 nm away from the edge
    point_nm = mid + offset_nm

    radius_key = skel.recommend_radius()[0]
    radii = np.asarray(skel.radii[radius_key], dtype=float)

    def brute_centerline(point_nm_space: np.ndarray) -> float:
        """Distance to the centreline (no radii)."""
        d_nodes = np.linalg.norm(skel.nodes - point_nm_space, axis=1).min()
        d_edges = np.inf
        for a, b in skel.edges:
            d_edges = min(
                d_edges,
                dx._point_segment_distance(
                    point_nm_space, skel.nodes[a], skel.nodes[b]
                ),
            )
        return min(d_nodes, d_edges)

    def brute_surface(point_nm_space: np.ndarray) -> float:
        """Distance to the capsule envelope (clamped to zero inside)."""
        d_nodes = float(
            np.min(np.linalg.norm(skel.nodes - point_nm_space, axis=1) - radii)
        )
        d_edges = np.inf
        for a, b in skel.edges:
            d_edges = min(
                d_edges,
                dx._point_segment_capsule_distance(
                    point_nm_space,
                    skel.nodes[a],
                    skel.nodes[b],
                    radii[a],
                    radii[b],
                ),
            )
        return max(min(d_nodes, d_edges), 0.0)

    expected_center_nm = brute_centerline(point_nm)
    expected_surface_nm = brute_surface(point_nm)

    # --- centreline mode -------------------------------------------------
    d_center_nm = dx.distance(skel, point_nm, point_unit="nm", mode="centerline")
    assert d_center_nm == pytest.approx(expected_center_nm, rel=1e-6)

    point_um = point_nm * 1e-3
    d_center_um = dx.distance(skel, point_um, point_unit="um", mode="centerline")
    assert d_center_um == pytest.approx(expected_center_nm * 1e-3, rel=1e-6)

    # --- surface mode ----------------------------------------------------
    d_surface_nm = dx.distance(skel, point_nm, point_unit="nm", mode="surface")
    assert d_surface_nm == pytest.approx(expected_surface_nm, rel=1e-6)

    point_um_surface = dx.distance(skel, point_um, point_unit="um", mode="surface")
    assert point_um_surface == pytest.approx(expected_surface_nm * 1e-3, rel=1e-6)

    # vectorised query mixes modes
    arr_nm = np.vstack([point_nm, mid])
    distances_surface = dx.distance(skel, arr_nm, point_unit="nm", mode="surface")
    assert distances_surface.shape == (2,)
    assert distances_surface[0] == pytest.approx(expected_surface_nm, rel=1e-6)
    assert distances_surface[1] == pytest.approx(0.0, abs=1e-9)

    distances_center = dx.distance(skel, arr_nm, point_unit="nm", mode="centerline")
    assert distances_center.shape == (2,)
    assert distances_center[0] == pytest.approx(expected_center_nm, rel=1e-6)
    assert distances_center[1] == pytest.approx(0.0, abs=1e-9)
