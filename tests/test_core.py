"""
Core pipeline smoke-test.

Runs `skeletonize()` on the reference mesh and checks a handful of
topological / numerical invariants so that regressions blow up early.
"""

from pathlib import Path

import igraph as ig
import numpy as np
import pytest

from skeliner import skeletonize
from skeliner.dataclass import (
    Discarded,
    MeshComponents,
    Neurites,
    Organelles,
    Soma,
)
from skeliner.io import load_mesh
from skeliner.skeletonize import (
    _neighbour_groups,
    _split_branch_band,
    _surface_graph,
)


def _assert_skeleton_valid(skel, *, expect_soma: bool = False):
    """A few cheap invariants that *must* hold for every result."""
    # ----- basic shapes -------------------------------------------------
    assert skel.nodes.ndim == 2 and skel.nodes.shape[1] == 3
    assert skel.edges.ndim == 2 and skel.edges.shape[1] == 2
    assert skel.nodes.shape[0] > 0, "no nodes produced"
    assert (skel.r > 0).all(), "non-positive radii"

    # ----- edges are sorted & acyclic (forest) --------------------------
    assert (skel.edges[:, 0] < skel.edges[:, 1]).all(), "edges not sorted"

    g = skel._igraph()
    n_components = len(g.components())
    # for every forest: |E| = |V| − #components
    assert skel.edges.shape[0] == skel.nodes.shape[0] - n_components

    # ----- node 0 type --------------------------------------------------
    if expect_soma:
        assert skel.ntype[0] == 1, "node 0 should be soma"
    else:
        assert skel.ntype[0] == -1, "node 0 should be root"


@pytest.fixture(scope="session")
def reference_mesh():
    data_dir = Path(__file__).parent / "data"
    mesh_path = data_dir / "60427.obj"
    return load_mesh(mesh_path)


# ----- direct track -------------------------------------------------------


def test_skeletonize_smoke(reference_mesh):
    skel = skeletonize(reference_mesh, verbose=False)
    _assert_skeleton_valid(skel)


def test_skeletonize_soma_init_guess_axis_mode(reference_mesh):
    """soma_init_guess='z-min' should behave like the default."""
    skel = skeletonize(
        reference_mesh,
        soma_init_guess="z-min",
        verbose=False,
    )
    _assert_skeleton_valid(skel)


# ----- preprocessing track ------------------------------------------------


def _make_components(mesh, *, with_soma: bool):
    """Build synthetic MeshComponents from a mesh.

    Uses only the main connected component's faces (like
    real ``break_up_mesh`` would), since the preproc track
    expects each neurite to be surface-connected.
    """
    nF = len(mesh.faces)
    # find the largest connected vertex component
    gsurf = _surface_graph(mesh)
    comps = list(gsurf.components())
    main = set(max(comps, key=len))
    main_faces = np.array(
        [fi for fi in range(nF) if all(int(v) in main for v in mesh.faces[fi])],
        dtype=np.int64,
    )

    org = Organelles(
        pocket=np.zeros(nF, bool),
        isolated=np.zeros(nF, bool),
        expanded=np.zeros(nF, bool),
    )
    neurites = Neurites([main_faces])
    discarded = Discarded([])

    if with_soma:
        center = mesh.vertices.mean(axis=0)
        r = float(mesh.edges_unique_length.mean()) * 5
        verts = np.where(np.linalg.norm(mesh.vertices - center, axis=1) < r)[0].astype(
            np.int64
        )
        soma = Soma.from_sphere(center, r, verts=verts)
    else:
        soma = None

    return MeshComponents(
        soma=soma,
        organelles=org,
        neurites=neurites,
        discarded=discarded,
    )


def test_preproc_track_no_soma(reference_mesh):
    """Preprocessing track without soma → valid forest."""
    comp = _make_components(reference_mesh, with_soma=False)
    skel = skeletonize(reference_mesh, components=comp, verbose=False)
    _assert_skeleton_valid(skel, expect_soma=False)


def test_preproc_track_with_soma(reference_mesh):
    """Preprocessing track with soma → valid tree rooted at soma."""
    comp = _make_components(reference_mesh, with_soma=True)
    skel = skeletonize(reference_mesh, components=comp, verbose=False)
    _assert_skeleton_valid(skel, expect_soma=True)
    # soma center should be close to the mesh centroid
    mesh_center = reference_mesh.vertices.mean(axis=0)
    assert np.linalg.norm(skel.soma.center - mesh_center) < 1e3


# ----- verbose timing breakdown -------------------------------------------


def test_preproc_track_reports_each_stage_as_it_finishes(reference_mesh, capsys):
    """One number for the whole neurite loop says nothing about which
    stage is slow, and `_timed` holds sub-messages until its block ends,
    so the stages are top-level steps that print when they complete."""
    comp = _make_components(reference_mesh, with_soma=False)
    skeletonize(reference_mesh, components=comp, verbose=True)
    out = capsys.readouterr().out

    stages = [ln for ln in out.splitlines() if ln.lstrip().startswith("↳")]
    named = {ln.split("…")[0].split("↳")[1].strip() for ln in stages}
    assert "bin vertices by geodesic distance" in named, named
    assert "split bins that wrap a branch point" in named, named
    assert "skeletonize neurites" not in named, "the opaque wrapper is gone"
    for ln in stages:
        assert ln.rstrip().endswith(" s"), ln


def test_no_breakdown_when_quiet(reference_mesh, capsys):
    comp = _make_components(reference_mesh, with_soma=False)
    skeletonize(reference_mesh, components=comp, verbose=False)
    assert capsys.readouterr().out == ""


# ----- branch-band split --------------------------------------------------
#
# A geodesic shell that lands on a branch point wraps the parent tube and
# both children in one connected band.  It passes the ring test, so it
# becomes a bin, and its node is the mean of points on two diverging tubes
# — which lands between them instead of inside one.  `_neighbour_groups`
# counts how many separate neighbourhoods a bin touches (two for a
# cross-section, three at a branch) and `_split_branch_band` cuts on that.


def _ring(start, n):
    """Vertex ids of a cycle plus its edges, as (ids, edges)."""
    ids = list(range(start, start + n))
    edges = [(ids[i], ids[(i + 1) % n]) for i in range(n)]
    return ids, edges


def _pants():
    """Three rings around a fourth: the parent, the band, and two children.

    ``band`` touches all three of the others, so it is the bin a branch
    point produces.  Returns the graph, the band's vertex ids, and the
    ownership map.
    """
    n = 8
    parent, e_parent = _ring(0, n)
    band, e_band = _ring(n, 2 * n)  # wide enough to feed two children
    childA, e_a = _ring(3 * n, n)
    childB, e_b = _ring(4 * n, n)

    edges = e_parent + e_band + e_a + e_b
    # parent sits under the first half of the band, the children over the
    # second half, one on each quarter
    for i in range(n):
        edges.append((parent[i], band[i]))
    for i in range(n):
        edges.append((childA[i], band[n + (i % (n // 2))]))
        edges.append((childB[i], band[n + n // 2 + (i % (n // 2))]))

    owner = {}
    for bin_id, verts in enumerate((parent, band, childA, childB)):
        for v in verts:
            owner[v] = (bin_id, 0)
    g = ig.Graph(n=5 * n, edges=edges)
    return g, np.asarray(band, dtype=np.int64), owner


def test_branch_band_sees_three_neighbourhoods():
    g, band, owner = _pants()
    assert len(_neighbour_groups(band, g, owner, (1, 0))) == 3


def test_two_patches_of_one_bin_are_one_neighbourhood():
    """Dipping into the same neighbour twice is one tube, not two.

    Counting vertex patches instead of bins made a plain band look like a
    branch point and cut it into slivers.
    """
    g, band, owner = _pants()
    # merge both children into a single bin: the band now touches two bins
    # (parent, child) but along three separate patches
    for v in list(owner):
        if owner[v] == (3, 0):
            owner[v] = (2, 0)
    assert len(_neighbour_groups(band, g, owner, (1, 0))) == 2


def test_end_ring_sees_one_neighbourhood():
    """The parent ring touches only the band: nothing to split."""
    g, _, owner = _pants()
    parent = np.arange(0, 8, dtype=np.int64)
    assert len(_neighbour_groups(parent, g, owner, (0, 0))) == 1


def test_split_yields_one_connected_piece_per_tube():
    g, band, owner = _pants()
    groups = _neighbour_groups(band, g, owner, (1, 0))
    parts = _split_branch_band(band, groups, g)
    assert len(parts) == 3

    allv = np.concatenate(parts).tolist()
    assert sorted(allv) == sorted(band.tolist())
    assert len(set(allv)) == len(allv), "a vertex landed in two pieces"
    for part in parts:
        # a disconnected piece leaves its node with no place to sit
        sub = g.induced_subgraph([int(v) for v in part])
        assert len(part) > 0 and len(sub.components()) == 1


def test_split_is_deterministic():
    g, band, owner = _pants()
    groups = _neighbour_groups(band, g, owner, (1, 0))
    first = [p.tolist() for p in _split_branch_band(band, groups, g)]
    second = [p.tolist() for p in _split_branch_band(band, groups, g)]
    assert first == second
