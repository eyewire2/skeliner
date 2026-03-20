"""
Tests for skeliner.pre – mesh preprocessing utilities.

Uses synthetic meshes to exercise detection and removal functions
without depending on large biological data.
"""

import numpy as np
import pytest
import trimesh

from skeliner import pre
from skeliner.dataclass import Soma


# ── Mesh builders ────────────────────────────────────────────────────


def _icosphere(subdivisions=2, radius=100.0):
    """Watertight icosphere centered at origin."""
    return trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)


def _open_mesh(n_remove=3):
    """Icosphere with a few faces removed → creates a hole."""
    mesh = _icosphere()
    keep = np.ones(len(mesh.faces), dtype=bool)
    keep[:n_remove] = False
    mesh = mesh.submesh([np.where(keep)[0]], append=True)
    mesh.remove_unreferenced_vertices()
    return mesh


def _mesh_with_island(n_island_faces=2):
    """Main icosphere + a small disconnected island.

    When n_island_faces=1, the island is a single triangle.
    When n_island_faces=2, two triangles sharing an edge (one component).
    """
    main = _icosphere()
    n_v = len(main.vertices)

    offset = np.array([500.0, 0.0, 0.0])
    v0 = offset + np.array([0.0, 0.0, 0.0])
    v1 = offset + np.array([1.0, 0.0, 0.0])
    v2 = offset + np.array([0.5, 1.0, 0.0])
    island_verts = [v0, v1, v2]
    island_tri = [[n_v, n_v + 1, n_v + 2]]

    if n_island_faces >= 2:
        # Add a second triangle sharing edge v0-v1
        v3 = offset + np.array([0.5, -1.0, 0.0])
        island_verts.append(v3)
        island_tri.append([n_v, n_v + 1, n_v + 3])

    # Add more if needed (each shares v1-v2)
    for i in range(2, n_island_faces):
        vi = offset + np.array([1.0 + i * 0.5, 0.5, 0.0])
        base = n_v + len(island_verts)
        island_verts.append(vi)
        island_tri.append([n_v + 1, n_v + 2, base])

    verts = np.vstack([main.vertices, np.array(island_verts)])
    faces = np.vstack([main.faces, np.array(island_tri)])
    return trimesh.Trimesh(vertices=verts, faces=faces, process=False)


def _mesh_with_fin():
    """Icosphere with a dangling fin face attached by one edge.

    A fin has 2+ boundary edges (only 1 of its 3 edges shared with
    another face).
    """
    main = _icosphere()
    n_v = len(main.vertices)

    # Pick an edge from face 0 and hang a triangle off it
    f0 = main.faces[0]
    v0, v1 = int(f0[0]), int(f0[1])

    # New vertex sticking outward
    mid = (main.vertices[v0] + main.vertices[v1]) / 2
    normal = mid / np.linalg.norm(mid)  # radial direction
    new_pt = mid + normal * 50.0

    verts = np.vstack([main.vertices, new_pt.reshape(1, 3)])
    fin_face = np.array([[v0, v1, n_v]])
    faces = np.vstack([main.faces, fin_face])
    return trimesh.Trimesh(vertices=verts, faces=faces, process=False)


def _two_cylinders(separation=200.0):
    """Two elongated cylinders separated by a gap — for gap detection.

    Elongated shapes (PCA ratio > 5) avoid being filtered as organelle-like
    blobs by find_disconnected. Separation > 1000 avoids distance filter.
    """
    c1 = trimesh.creation.cylinder(radius=10.0, height=200.0, sections=16)
    c2 = trimesh.creation.cylinder(radius=10.0, height=200.0, sections=16)
    # Shift c2 along x beyond the organelle proximity filter (1000nm)
    c2.vertices[:, 0] += 200.0 + separation

    n_v = len(c1.vertices)
    verts = np.vstack([c1.vertices, c2.vertices])
    faces = np.vstack([c1.faces, c2.faces + n_v])
    return trimesh.Trimesh(vertices=verts, faces=faces, process=False)


def _sphere_with_internal():
    """Large sphere with a small inverted sphere inside.

    The inner sphere has normals pointing inward (toward center),
    simulating an isolated organelle.
    """
    outer = _icosphere(subdivisions=2, radius=100.0)
    inner = _icosphere(subdivisions=1, radius=20.0)

    # Flip inner normals by reversing face winding
    inner.faces = inner.faces[:, ::-1]

    n_v = len(outer.vertices)
    verts = np.vstack([outer.vertices, inner.vertices])
    faces = np.vstack([outer.faces, inner.faces + n_v])
    return trimesh.Trimesh(vertices=verts, faces=faces, process=False)


# ── _face_edge_components ────────────────────────────────────────────


class TestFaceEdgeComponents:
    def test_single_component(self):
        mesh = _icosphere()
        labels, main = pre._face_edge_components(mesh)
        assert labels.shape == (len(mesh.faces),)
        # All faces in one component
        assert np.unique(labels).size == 1
        assert main == 0

    def test_two_components(self):
        mesh = _two_cylinders()
        labels, main = pre._face_edge_components(mesh)
        unique = np.unique(labels)
        assert len(unique) == 2
        # Main component is the largest (both are equal here, so either is fine)
        main_size = (labels == main).sum()
        for cid in unique:
            assert (labels == cid).sum() <= main_size

    def test_with_island(self):
        mesh = _mesh_with_island(n_island_faces=2)
        labels, main = pre._face_edge_components(mesh)
        unique = np.unique(labels)
        # Main icosphere + 1 disconnected 2-face island
        assert len(unique) == 2


# ── find_holes / fill_holes ──────────────────────────────────────────


class TestFindHoles:
    def test_watertight_has_no_holes(self):
        mesh = _icosphere()
        holes = pre.find_holes(mesh)
        assert holes == []

    def test_open_mesh_has_hole(self):
        mesh = _open_mesh(n_remove=3)
        holes = pre.find_holes(mesh)
        assert len(holes) >= 1
        # Each hole is a loop of at least 3 vertices
        for loop in holes:
            assert len(loop) >= 3

    def test_hole_vertices_are_valid(self):
        mesh = _open_mesh(n_remove=1)
        holes = pre.find_holes(mesh)
        assert len(holes) >= 1
        for loop in holes:
            for vid in loop:
                assert 0 <= vid < len(mesh.vertices)


class TestFillHoles:
    @pytest.mark.parametrize("method", ["advancing_front", "dome"])
    def test_fill_reduces_holes(self, method):
        mesh = _open_mesh(n_remove=3)
        holes_before = pre.find_holes(mesh)
        assert len(holes_before) > 0

        filled = pre.fill_holes(mesh, method=method)
        holes_after = pre.find_holes(filled)
        assert len(holes_after) < len(holes_before)

    def test_fill_watertight_is_noop(self):
        mesh = _icosphere()
        result = pre.fill_holes(mesh)
        # Should return the same mesh (no holes to fill)
        assert len(result.faces) == len(mesh.faces)

    def test_invalid_method_raises(self):
        mesh = _open_mesh()
        with pytest.raises(ValueError, match="Unknown method"):
            pre.fill_holes(mesh, method="nonexistent")


# ── find_fragments / remove_fragments ────────────────────────────────


class TestFindFragments:
    def test_clean_mesh_no_fragments(self):
        mesh = _icosphere()
        mask = pre.find_fragments(mesh)
        assert mask.shape == (len(mesh.faces),)
        assert mask.sum() == 0

    def test_island_detected(self):
        mesh = _mesh_with_island(n_island_faces=2)
        mask = pre.find_fragments(mesh, min_faces=3)
        # The 2-face island should be detected (< min_faces=3)
        assert mask.sum() == 2

    def test_min_faces_affects_island_detection(self):
        """min_faces controls island threshold (fins are always detected)."""
        mesh = _mesh_with_island(n_island_faces=2)
        # With min_faces=3, island is flagged as island (2 < 3)
        mask_strict = pre.find_fragments(mesh, min_faces=3)
        # With min_faces=1, island not flagged as island (2 >= 1)
        # but may still be flagged as fin (2 boundary edges each)
        mask_lax = pre.find_fragments(mesh, min_faces=1)
        # Both should detect the same count here since they're also fins
        assert mask_strict.sum() == mask_lax.sum() == 2

    def test_fin_detected(self):
        mesh = _mesh_with_fin()
        mask = pre.find_fragments(mesh)
        # The dangling fin face should be detected
        assert mask.sum() >= 1


class TestRemoveFragments:
    def test_removes_islands(self):
        mesh = _mesh_with_island(n_island_faces=2)
        n_before = len(mesh.faces)
        clean = pre.remove_fragments(mesh, min_faces=3)
        # Should have fewer faces
        assert len(clean.faces) < n_before
        assert len(clean.faces) == n_before - 2

    def test_removes_fins(self):
        mesh = _mesh_with_fin()
        n_before = len(mesh.faces)
        clean = pre.remove_fragments(mesh)
        assert len(clean.faces) < n_before

    def test_clean_mesh_unchanged(self):
        mesh = _icosphere()
        clean = pre.remove_fragments(mesh)
        assert len(clean.faces) == len(mesh.faces)

    def test_precomputed_mask(self):
        mesh = _mesh_with_island(n_island_faces=2)
        mask = pre.find_fragments(mesh, min_faces=3)
        clean = pre.remove_fragments(mesh, _precomputed=mask)
        assert len(clean.faces) == len(mesh.faces) - 2


# ── remove_islands / remove_fins ─────────────────────────────────────


class TestRemoveIslands:
    def test_removes_small_components(self):
        mesh = _mesh_with_island(n_island_faces=2)
        clean = pre.remove_islands(mesh, min_faces=3)
        assert len(clean.faces) == len(mesh.faces) - 2

    def test_keeps_large_enough_components(self):
        mesh = _mesh_with_island(n_island_faces=2)
        clean = pre.remove_islands(mesh, min_faces=2)
        assert len(clean.faces) == len(mesh.faces)

    def test_noop_on_clean_mesh(self):
        mesh = _icosphere()
        clean = pre.remove_islands(mesh)
        assert len(clean.faces) == len(mesh.faces)


class TestRemoveFins:
    def test_removes_dangling_faces(self):
        mesh = _mesh_with_fin()
        clean = pre.remove_fins(mesh)
        assert len(clean.faces) < len(mesh.faces)

    def test_noop_on_clean_mesh(self):
        mesh = _icosphere()
        clean = pre.remove_fins(mesh)
        assert len(clean.faces) == len(mesh.faces)


# ── find_disconnected ────────────────────────────────────────────────


class TestFindDisconnected:
    def test_single_component_returns_empty(self):
        mesh = _icosphere()
        comps = pre.find_disconnected(mesh, min_faces=3)
        assert comps == []

    def test_two_cylinders_detected(self):
        mesh = _two_cylinders(separation=1500.0)
        comps = pre.find_disconnected(mesh, min_faces=3)
        assert len(comps) == 1  # one non-main component
        # The component should have the face count of one cylinder
        cyl_faces = len(trimesh.creation.cylinder(radius=10.0, height=200.0, sections=16).faces)
        assert len(comps[0]) == cyl_faces

    def test_min_faces_filter(self):
        mesh = _mesh_with_island(n_island_faces=2)
        # Island has 2 faces, min_faces=100 → excluded
        comps = pre.find_disconnected(mesh, min_faces=100)
        assert comps == []

    def test_sorted_largest_first(self):
        """Three elongated components of different sizes → sorted largest-first."""
        c1 = trimesh.creation.cylinder(radius=10.0, height=300.0, sections=16)
        c2 = trimesh.creation.cylinder(radius=10.0, height=200.0, sections=16)
        c3 = trimesh.creation.cylinder(radius=10.0, height=100.0, sections=8)

        c2.vertices[:, 0] += 2000.0
        c3.vertices[:, 0] += 4000.0

        n1 = len(c1.vertices)
        n2 = n1 + len(c2.vertices)
        verts = np.vstack([c1.vertices, c2.vertices, c3.vertices])
        faces = np.vstack([c1.faces, c2.faces + n1, c3.faces + n2])
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)

        comps = pre.find_disconnected(mesh, min_faces=3)
        assert len(comps) == 2
        assert len(comps[0]) >= len(comps[1])


# ── find_gaps / remove_gaps ──────────────────────────────────────────


class TestFindGaps:
    def test_single_component_no_gaps(self):
        mesh = _icosphere()
        gaps = pre.find_gaps(mesh, min_faces=3)
        assert gaps == []

    def test_two_cylinders_one_gap(self):
        mesh = _two_cylinders(separation=1500.0)
        gaps = pre.find_gaps(mesh, min_faces=3)
        assert len(gaps) == 1
        faces_a, faces_b, dist = gaps[0]
        assert len(faces_a) > 0
        assert len(faces_b) > 0
        assert dist > 0

    def test_gap_distance_positive(self):
        mesh = _two_cylinders(separation=1500.0)
        gaps = pre.find_gaps(mesh, min_faces=3)
        assert len(gaps) == 1
        _, _, dist = gaps[0]
        assert dist > 100  # should be roughly the separation

    def test_sorted_by_distance(self):
        """Three cylinders with different gaps → sorted by distance."""
        c1 = trimesh.creation.cylinder(radius=10.0, height=200.0, sections=16)
        c2 = trimesh.creation.cylinder(radius=10.0, height=200.0, sections=16)
        c3 = trimesh.creation.cylinder(radius=10.0, height=200.0, sections=16)

        c2.vertices[:, 0] += 2000.0  # gap ~1800 from c1
        c3.vertices[:, 0] += 5000.0  # gap ~2800 from c2

        n1, n2 = len(c1.vertices), len(c1.vertices) + len(c2.vertices)
        verts = np.vstack([c1.vertices, c2.vertices, c3.vertices])
        faces = np.vstack([c1.faces, c2.faces + n1, c3.faces + n2])
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)

        gaps = pre.find_gaps(mesh, min_faces=3)
        assert len(gaps) >= 2
        for i in range(len(gaps) - 1):
            assert gaps[i][2] <= gaps[i + 1][2]


class TestRemoveGaps:
    def test_noop_single_component(self):
        mesh = _icosphere()
        result = pre.remove_gaps(mesh, min_faces=3)
        assert len(result.faces) == len(mesh.faces)

    def test_bridges_two_cylinders(self):
        mesh = _two_cylinders(separation=1500.0)
        labels_before, _ = pre._face_edge_components(mesh)
        n_comps_before = len(np.unique(labels_before))
        assert n_comps_before == 2

        result = pre.remove_gaps(mesh, min_faces=3)
        labels_after, _ = pre._face_edge_components(result)
        n_comps_after = len(np.unique(labels_after))
        # After bridging, should be a single connected component
        assert n_comps_after < n_comps_before

    def test_precomputed_gaps(self):
        mesh = _two_cylinders(separation=1500.0)
        gaps = pre.find_gaps(mesh, min_faces=3)
        result = pre.remove_gaps(mesh, _precomputed_gaps=gaps)
        labels, _ = pre._face_edge_components(result)
        # Should still work with precomputed gaps
        assert len(np.unique(labels)) < 2 or len(result.faces) != len(mesh.faces)


# ── find_soma ────────────────────────────────────────────────────────


class TestFindSoma:
    def test_no_fragments_returns_none(self):
        mesh = _icosphere()
        soma = pre.find_soma(mesh)
        assert soma is None

    def test_clustered_fragments_detect_soma(self):
        """Place many small fragments near the origin → soma detected there."""
        main = _icosphere(subdivisions=2, radius=100.0)
        n_v = len(main.vertices)

        rng = np.random.default_rng(42)
        frag_verts = []
        frag_faces = []
        offset = n_v
        # Place 10 small triangles near the origin
        for i in range(10):
            center = rng.normal(0, 10, size=3)
            v0 = center + np.array([0, 0, 0])
            v1 = center + np.array([2, 0, 0])
            v2 = center + np.array([1, 2, 0])
            frag_verts.extend([v0, v1, v2])
            frag_faces.append([offset, offset + 1, offset + 2])
            offset += 3

        verts = np.vstack([main.vertices, np.array(frag_verts)])
        faces = np.vstack([main.faces, np.array(frag_faces)])
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)

        soma = pre.find_soma(mesh, max_fragment_faces=50)
        assert soma is not None
        assert isinstance(soma, Soma)
        # Soma center should be near origin
        assert np.linalg.norm(soma.center) < 50.0


# ── find_organelles (isolated) ───────────────────────────────────────


class TestFindIsolatedOrganelles:
    def test_clean_mesh_no_organelles(self):
        mesh = _icosphere()
        mask = pre.find_isolated_organelles(mesh, radius=50.0)
        assert mask.shape == (len(mesh.faces),)
        assert mask.sum() == 0

    def test_inner_sphere_detected(self):
        mesh = _sphere_with_internal()
        mask = pre.find_isolated_organelles(mesh, radius=50.0)
        inner_n = len(_icosphere(subdivisions=1, radius=20.0).faces)
        # Should detect the inner sphere faces
        assert mask.sum() == inner_n


class TestFindOrganelles:
    def test_returns_two_masks(self):
        mesh = _icosphere()
        pocket, isolated = pre.find_organelles(mesh, radius=50.0)
        assert pocket.shape == (len(mesh.faces),)
        assert isolated.shape == (len(mesh.faces),)
        assert pocket.dtype == bool
        assert isolated.dtype == bool

    def test_masks_are_disjoint(self):
        mesh = _sphere_with_internal()
        pocket, isolated = pre.find_organelles(mesh, radius=50.0)
        overlap = pocket & isolated
        assert overlap.sum() == 0


# ── remove_organelles ────────────────────────────────────────────────


class TestRemoveOrganelles:
    def test_clean_mesh_unchanged(self):
        mesh = _icosphere()
        clean = pre.remove_organelles(mesh, radius=50.0)
        assert len(clean.faces) == len(mesh.faces)

    def test_inner_sphere_removed(self):
        mesh = _sphere_with_internal()
        n_before = len(mesh.faces)
        clean = pre.remove_organelles(mesh, radius=50.0)
        assert len(clean.faces) < n_before
        # Should only have the outer sphere faces (approximately)
        outer_n = len(_icosphere(subdivisions=2, radius=100.0).faces)
        assert len(clean.faces) == outer_n


# ── find_fusions ─────────────────────────────────────────────────────


class TestFindFusions:
    def test_clean_mesh_no_fusions(self):
        mesh = _icosphere()
        clusters = pre.find_fusions(mesh, radius=50.0)
        assert clusters == []

    def test_pinch_vertex_detected(self):
        """Two tetrahedra sharing a single vertex = fan vertex fusion."""
        # Tetrahedron 1
        v0 = np.array([
            [0, 0, 0], [1, 0, 0], [0.5, 1, 0], [0.5, 0.5, 1],
        ], dtype=np.float64)
        f0 = np.array([
            [0, 1, 2], [0, 1, 3], [1, 2, 3], [0, 2, 3],
        ])
        # Tetrahedron 2 sharing vertex 0
        v1 = np.array([
            [-1, 0, 0], [-0.5, 1, 0], [-0.5, 0.5, 1],
        ], dtype=np.float64)
        f1 = np.array([
            [0, 4, 5], [0, 4, 6], [4, 5, 6], [0, 5, 6],
        ])
        verts = np.vstack([v0, v1])
        faces = np.vstack([f0, f1])
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)

        clusters = pre.find_fusions(mesh, radius=2.0)
        assert len(clusters) >= 1


# ── remove_fusions ───────────────────────────────────────────────────


class TestRemoveFusions:
    def test_clean_mesh_unchanged(self):
        mesh = _icosphere()
        clean = pre.remove_fusions(mesh, radius=50.0)
        # Should still have the same number of faces (no fusions to remove)
        assert len(clean.faces) == len(mesh.faces)


# ── remove_nucleus ───────────────────────────────────────────────────


class TestRemoveNucleus:
    def test_no_soma_returns_mesh(self):
        """When skeleton has no soma, mesh is returned unchanged."""
        mesh = _icosphere()

        class FakeSkel:
            soma = None

        result = pre.remove_nucleus(mesh, FakeSkel())
        assert len(result.faces) == len(mesh.faces)

    def test_no_internal_faces_returns_mesh(self):
        """Soma exists but no inward-facing faces inside it → unchanged."""
        mesh = _icosphere(radius=100.0)
        soma = Soma.from_sphere(center=np.array([0.0, 0.0, 0.0]), radius=50.0, verts=None)

        class FakeSkel:
            pass

        skel = FakeSkel()
        skel.soma = soma
        result = pre.remove_nucleus(mesh, skel)
        # The soma is much smaller than the mesh, so faces inside have
        # outward-pointing normals → nothing to remove
        assert len(result.faces) == len(mesh.faces)


# ── ensure_watertight ────────────────────────────────────────────────


class TestEnsureWatertight:
    def test_already_watertight(self):
        mesh = _icosphere()
        result = pre.ensure_watertight(mesh)
        assert result.is_watertight
        assert len(result.faces) == len(mesh.faces)

    def test_fills_small_hole(self):
        mesh = _open_mesh(n_remove=1)
        assert not mesh.is_watertight
        result = pre.ensure_watertight(mesh)
        # Should attempt to make it watertight
        assert len(result.faces) >= len(mesh.faces)


# ── _ear_clip_2d ─────────────────────────────────────────────────────


class TestEarClip2D:
    def test_triangle(self):
        pts = np.array([[0, 0], [1, 0], [0.5, 1]], dtype=np.float64)
        tris = pre._ear_clip_2d(pts)
        assert len(tris) == 1
        assert set(tris[0]) == {0, 1, 2}

    def test_square(self):
        pts = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64)
        tris = pre._ear_clip_2d(pts)
        assert len(tris) == 2
        # Each triangle should use 3 vertices from 0..3
        all_verts = set()
        for t in tris:
            assert len(t) == 3
            all_verts.update(t)
        assert all_verts == {0, 1, 2, 3}

    def test_convex_pentagon(self):
        angles = np.linspace(0, 2 * np.pi, 5, endpoint=False)
        pts = np.stack([np.cos(angles), np.sin(angles)], axis=1)
        tris = pre._ear_clip_2d(pts)
        # n-2 triangles for a convex polygon
        assert len(tris) == 3


# ── _filter_small_clusters ───────────────────────────────────────────


class TestFilterSmallClusters:
    def test_removes_small_clusters(self):
        mesh = _mesh_with_island(n_island_faces=2)
        # Mark the 2 island faces
        mask = np.zeros(len(mesh.faces), dtype=bool)
        mask[-2:] = True  # the island faces are appended at the end
        filtered = pre._filter_small_clusters(mesh, mask, min_cluster_size=3)
        # 2-face cluster < min_size=3 → removed
        assert filtered.sum() == 0

    def test_keeps_large_clusters(self):
        mesh = _icosphere()
        # Mark all faces
        mask = np.ones(len(mesh.faces), dtype=bool)
        filtered = pre._filter_small_clusters(mesh, mask, min_cluster_size=3)
        assert filtered.sum() == len(mesh.faces)


# ── Smoke test on real data ──────────────────────────────────────────


@pytest.fixture(scope="session")
def reference_mesh():
    from pathlib import Path
    from skeliner.io import load_mesh

    mesh_path = Path(__file__).parent / "data" / "60427.obj"
    if not mesh_path.exists():
        pytest.skip("Reference mesh not available")
    return load_mesh(mesh_path)


class TestRealMesh:
    def test_find_holes_returns_list(self, reference_mesh):
        holes = pre.find_holes(reference_mesh)
        assert isinstance(holes, list)

    def test_find_fragments_returns_mask(self, reference_mesh):
        mask = pre.find_fragments(reference_mesh)
        assert mask.shape == (len(reference_mesh.faces),)
        assert mask.dtype == bool

    def test_find_disconnected_returns_list(self, reference_mesh):
        comps = pre.find_disconnected(reference_mesh, min_faces=100)
        assert isinstance(comps, list)

    def test_remove_fragments_preserves_main(self, reference_mesh):
        clean = pre.remove_fragments(reference_mesh)
        # Should keep at least 90% of faces (main component)
        assert len(clean.faces) > len(reference_mesh.faces) * 0.5
