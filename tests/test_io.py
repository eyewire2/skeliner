"""
IO round-trip smoke tests.

* load `.obj`
* run skeletonize
* save to SWC & NPZ
* reload and compare a few coarse features
"""

from pathlib import Path

import numpy as np
import pytest

from skeliner import Skeleton, Soma, dx, skeletonize
from skeliner.dataclass import MeshStats, Organelles
from skeliner.io import (
    load_mesh,
    load_skeleton_npz,
    load_soma_npz,
    load_skeleton_swc,
    save_soma_npz,
    save_organelles_npz,
    load_organelles_npz,
    save_mesh_stats_npz,
    load_mesh_stats_npz,
)

SAMPLES_DIR = Path(__file__).parent / "data"

SAMPLE_SWCS = [
    "60427.swc",
]


@pytest.fixture(scope="session")
def reference_mesh():
    mesh_path = SAMPLES_DIR / "60427.obj"
    return load_mesh(mesh_path)


def test_io_roundtrip(reference_mesh, tmp_path):
    skel = skeletonize(reference_mesh, verbose=False)

    # warm up KD-tree cache (and verify it exists)
    dx.distance(skel, skel.nodes[0], point_unit=skel.meta.get("unit", "nm"))
    assert skel._nodes_kdtree is not None

    # --- write ---------------------------------------------------------
    swc_path = tmp_path / "60427_test.swc"
    npz_path = tmp_path / "60427_test.npz"
    skel.to_swc(swc_path)
    skel.to_npz(npz_path)

    assert swc_path.exists()
    assert npz_path.exists()

    # --- read back -----------------------------------------------------
    skel_from_swc = load_skeleton_swc(swc_path)
    skel_from_npz = load_skeleton_npz(npz_path)
    assert skel_from_npz._nodes_kdtree is not None
    assert skel_from_npz._node_neighbors is not None

    # --- very coarse equivalence checks --------------------------------
    # (Exact float equality is not expected; topology & sizes should match.)
    assert skel_from_swc.nodes.shape[0] == skel.nodes.shape[0]
    assert skel_from_npz.edges.shape == skel.edges.shape
    assert np.isclose(
        skel_from_npz.soma.equiv_radius,
        skel.soma.equiv_radius,
        rtol=1e-4,
    )


# ------------------------------------------------------------------------
#  Helper
# ------------------------------------------------------------------------
def _edges_equal(a: np.ndarray, b: np.ndarray) -> bool:
    """
    True iff two undirected edge lists connect the same pairs of
    vertices (row order may differ).
    """
    a = np.sort(a, axis=1)
    b = np.sort(b, axis=1)
    if a.shape != b.shape:
        return False
    # sort rows to make order irrelevant
    a = a[np.lexsort(a.T[::-1])]
    b = b[np.lexsort(b.T[::-1])]
    return np.array_equal(a, b)  # exact – they are integers


# ------------------------------------------------------------------------
#  Parametrised smoke-test
# ------------------------------------------------------------------------
@pytest.mark.parametrize("fname", SAMPLE_SWCS)
def test_swc_roundtrip_exact(fname: str, tmp_path: Path):
    src_path = SAMPLES_DIR / fname
    assert src_path.exists(), f"missing sample file {src_path}"

    # 1 · load the reference skeleton
    skel_ref = load_skeleton_swc(src_path)

    # 2 · write it back
    out_path = tmp_path / fname
    skel_ref.to_swc(out_path)
    assert out_path.exists()

    # 3 · read what we just wrote
    skel_rt = load_skeleton_swc(out_path)

    # 4 · compare ­­­—­­ geometry ------------------------------------------------
    assert np.allclose(skel_rt.nodes, skel_ref.nodes, rtol=1e-6, atol=0.0), (
        "XYZ coordinates changed"
    )

    for k in skel_ref.radii:
        assert np.allclose(skel_rt.radii[k], skel_ref.radii[k], rtol=1e-6, atol=0.0), (
            f"radius column '{k}' changed"
        )

    # topology ----------------------------------------------------------
    assert _edges_equal(skel_rt.edges, skel_ref.edges), "edge list changed"

    # node-type labels --------------------------------------------------
    assert np.array_equal(skel_rt.ntype, skel_ref.ntype), "ntype vector changed"

    # soma geometry (allow rounding) ------------------------------------
    assert np.isclose(
        skel_rt.soma.equiv_radius, skel_ref.soma.equiv_radius, rtol=1e-6
    ), "soma radius changed"

    assert np.allclose(skel_rt.soma.center, skel_ref.soma.center, rtol=1e-6), (
        "soma center changed"
    )


def test_skeleton_roundtrip(tmp_path, reference_mesh):
    skel0 = skeletonize(reference_mesh, verbose=False)
    out = tmp_path / "rt.swc"
    skel0.to_swc(out)
    skel1 = load_skeleton_swc(out)

    assert np.allclose(skel0.nodes, skel1.nodes, rtol=1e-6)
    assert _edges_equal(skel0.edges, skel1.edges)
    assert np.allclose(skel0.r, skel1.r, rtol=1e-6)


# ------------------------------------------------------------------------
#  Soma NPZ round-trips
# ------------------------------------------------------------------------
def test_soma_npz_roundtrip_with_verts(tmp_path):
    """Full ellipsoid soma with verts survives save/load."""
    R = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=np.float64)
    soma = Soma(
        center=[10, 20, 30],
        axes=[5, 4, 3],
        R=R,
        verts=np.array([0, 7, 42, 999], dtype=np.int64),
    )

    path = tmp_path / "soma_verts"
    soma.to_npz(path)

    loaded = Soma.from_npz(path.with_suffix(".npz"))
    assert np.allclose(loaded.center, soma.center)
    assert np.allclose(loaded.axes, soma.axes)
    assert np.allclose(loaded.R, soma.R)
    assert np.array_equal(loaded.verts, soma.verts)


def test_soma_npz_roundtrip_no_verts(tmp_path):
    """Soma without verts round-trips cleanly (verts stays None)."""
    soma = Soma(center=[0, 0, 0], axes=[1, 1, 1], R=np.eye(3))

    path = tmp_path / "soma_noverts.npz"
    soma.to_npz(path)

    loaded = Soma.from_npz(path)
    assert loaded.verts is None
    assert np.allclose(loaded.center, soma.center)
    assert np.allclose(loaded.axes, soma.axes)


def test_soma_npz_functional_api(tmp_path):
    """save_soma_npz / load_soma_npz work directly."""
    soma = Soma.from_sphere([1, 2, 3], radius=5.0, verts=np.arange(10))

    path = tmp_path / "soma_func"
    save_soma_npz(soma, path)

    loaded = load_soma_npz(path.with_suffix(".npz"))
    assert np.isclose(loaded.equiv_radius, soma.equiv_radius)
    assert np.array_equal(loaded.verts, soma.verts)


def test_soma_npz_derived_fields(tmp_path):
    """Derived field _W is recomputed on load (not stored)."""
    soma = Soma(center=[1, 2, 3], axes=[6, 4, 2], R=np.eye(3))
    path = tmp_path / "soma_derived.npz"
    soma.to_npz(path)

    loaded = Soma.from_npz(path)
    # _W should be recomputed via __post_init__
    pt = np.array([[2, 3, 4]], dtype=np.float64)
    assert np.allclose(soma.contains(pt), loaded.contains(pt))


def test_soma_npz_nucleus_roundtrip(tmp_path):
    """Full nucleus dict survives round-trip."""
    nucleus = {
        "center": np.array([100.0, 200.0, 300.0]),
        "peak_r": 1500.0,
        "z_range": (280.0, 340.0),
        "slices": np.array(
            [
                [280, 101, 201, 1200],
                [290, 102, 202, 1400],
                [300, 100, 200, 1500],
                [310, 99, 199, 1300],
                [340, 98, 198, 1000],
            ],
            dtype=np.float64,
        ),
    }
    soma = Soma(center=[10, 20, 30], axes=[5, 4, 3], R=np.eye(3), nucleus=nucleus)
    path = tmp_path / "soma_nuc.npz"
    save_soma_npz(soma, path)

    loaded = load_soma_npz(path)
    assert np.allclose(loaded.nucleus["center"], nucleus["center"])
    assert loaded.nucleus["peak_r"] == nucleus["peak_r"]
    assert loaded.nucleus["z_range"] == nucleus["z_range"]
    assert np.array_equal(loaded.nucleus["slices"], nucleus["slices"])


def test_soma_npz_nucleus_none(tmp_path):
    """Soma without nucleus loads as None (backward compat)."""
    soma = Soma(center=[0, 0, 0], axes=[1, 1, 1], R=np.eye(3))
    path = tmp_path / "soma_no_nuc.npz"
    save_soma_npz(soma, path)

    loaded = load_soma_npz(path)
    assert loaded.nucleus is None


# ------------------------------------------------------------------------
#  Skeleton classmethod round-trips
# ------------------------------------------------------------------------
def test_skeleton_from_swc():
    """Skeleton.from_swc matches load_skeleton_swc."""
    src = SAMPLES_DIR / "60427.swc"
    skel_func = load_skeleton_swc(src)
    skel_cls = Skeleton.from_swc(src)

    assert np.allclose(skel_cls.nodes, skel_func.nodes)
    assert _edges_equal(skel_cls.edges, skel_func.edges)
    assert np.array_equal(skel_cls.ntype, skel_func.ntype)


def test_skeleton_from_npz(tmp_path, reference_mesh):
    """Skeleton.from_npz matches load_skeleton_npz."""
    skel = skeletonize(reference_mesh, verbose=False)
    path = tmp_path / "skel_cls.npz"
    skel.to_npz(path)

    loaded = Skeleton.from_npz(path)
    assert loaded.nodes.shape == skel.nodes.shape
    assert loaded.edges.shape == skel.edges.shape
    assert np.allclose(loaded.soma.center, skel.soma.center)
    assert np.allclose(loaded.soma.axes, skel.soma.axes)


# ------------------------------------------------------------------------
#  Organelles NPZ round-trips
# ------------------------------------------------------------------------
def test_organelles_npz_full_roundtrip(tmp_path):
    """All masks survive round-trip."""
    nF = 1000
    pocket = np.zeros(nF, dtype=bool)
    pocket[:100] = True
    isolated = np.zeros(nF, dtype=bool)
    isolated[200:250] = True
    expanded = np.zeros(nF, dtype=bool)
    expanded[500:520] = True

    org = Organelles(pocket=pocket, isolated=isolated, expanded=expanded)

    path = tmp_path / "org_full.npz"
    save_organelles_npz(org, path)

    loaded = load_organelles_npz(path)
    assert np.array_equal(loaded.pocket, org.pocket)
    assert np.array_equal(loaded.isolated, org.isolated)
    assert np.array_equal(loaded.expanded, org.expanded)
    # .mask property
    assert np.array_equal(loaded.mask, pocket | isolated | expanded)


def test_organelles_npz_backward_compat(tmp_path):
    """Old files with 'expanded' missing or with embedded mesh_stats still load."""
    nF = 100
    pocket = np.ones(nF, dtype=bool)
    isolated = np.zeros(nF, dtype=bool)
    # Simulate old format: no expanded key, with leftover mesh_stats fields
    path = tmp_path / "org_old.npz"
    face_comp = np.zeros(nF, dtype=np.int64)
    np.savez_compressed(
        path,
        pocket=pocket,
        isolated=isolated,
        outward_dots=np.zeros(nF),
        face_comp=face_comp,
        main_ci=np.array(0),
    )

    loaded = load_organelles_npz(path)
    assert np.array_equal(loaded.pocket, pocket)
    assert np.array_equal(loaded.expanded, np.zeros(nF, dtype=bool))


# ------------------------------------------------------------------------
#  MeshStats NPZ round-trip
# ------------------------------------------------------------------------
def test_mesh_stats_npz_roundtrip(tmp_path):
    """outward_dots / face_comp / main_ci survive a round-trip."""
    nF = 500
    rng = np.random.default_rng(42)
    stats = MeshStats(
        outward_dots=rng.uniform(-1, 1, nF).astype(np.float32),
        face_comp=rng.integers(0, 5, nF).astype(np.int64),
        main_ci=2,
    )

    path = tmp_path / "mesh_stats.npz"
    save_mesh_stats_npz(stats, path)

    loaded = load_mesh_stats_npz(path)
    assert np.allclose(loaded.outward_dots, stats.outward_dots)
    assert np.array_equal(loaded.face_comp, stats.face_comp)
    assert loaded.main_ci == stats.main_ci


def test_mesh_stats_npz_face_count_validation(tmp_path):
    """load_mesh_stats_npz raises when face count disagrees with mesh."""
    import trimesh

    nF = 50
    stats = MeshStats(
        outward_dots=np.zeros(nF, dtype=np.float32),
        face_comp=np.zeros(nF, dtype=np.int64),
        main_ci=0,
    )
    path = tmp_path / "mesh_stats.npz"
    save_mesh_stats_npz(stats, path)

    # Build a tetrahedron mesh (4 faces) — does not match nF=50
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    faces = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=np.int64)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)

    with pytest.raises(ValueError, match="stale"):
        load_mesh_stats_npz(path, mesh=mesh)

    # Without the mesh argument it should still load
    loaded = load_mesh_stats_npz(path)
    assert loaded.main_ci == 0
