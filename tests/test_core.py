"""
Core pipeline smoke-test.

Runs `skeletonize()` on the reference mesh and checks a handful of
topological / numerical invariants so that regressions blow up early.
"""

from pathlib import Path

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
    from skeliner.skeletonize import _surface_graph

    nF = len(mesh.faces)
    # find the largest connected vertex component
    gsurf = _surface_graph(mesh)
    comps = list(gsurf.components())
    main = set(max(comps, key=len))
    main_faces = np.array(
        [
            fi
            for fi in range(nF)
            if all(int(v) in main for v in mesh.faces[fi])
        ],
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
        verts = np.where(
            np.linalg.norm(
                mesh.vertices - center, axis=1
            )
            < r
        )[0].astype(np.int64)
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
    skel = skeletonize(
        reference_mesh, components=comp, verbose=False
    )
    _assert_skeleton_valid(skel, expect_soma=False)


def test_preproc_track_with_soma(reference_mesh):
    """Preprocessing track with soma → valid tree rooted at soma."""
    comp = _make_components(reference_mesh, with_soma=True)
    skel = skeletonize(
        reference_mesh, components=comp, verbose=False
    )
    _assert_skeleton_valid(skel, expect_soma=True)
    # soma center should be close to the mesh centroid
    mesh_center = reference_mesh.vertices.mean(axis=0)
    assert np.linalg.norm(skel.soma.center - mesh_center) < 1e3
