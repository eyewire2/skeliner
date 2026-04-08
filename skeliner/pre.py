"""skeliner.pre – mesh preprocessing utilities."""

import warnings
from collections import defaultdict, deque

import igraph as ig
import numpy as np
import trimesh
from scipy.spatial import KDTree

from skeliner.dataclass import (
    Discarded,
    MeshComponents,
    MeshStats,
    Neurites,
    Organelles,
    Soma,
)

__all__ = [
    "break_up_mesh",
    "compact_mesh",
    "compute_mesh_stats",
    "fill_holes",
    "find_disconnected",
    "find_gaps",
    "find_holes",
    "find_nucleus_center",
    "find_soma_via_neurite_exclusion",
    "find_soma_via_ring_cutoff",
    "find_soma_via_z_contour",
    "preprocess",
    "PreprocessResult",
    "remove_fins",
    "remove_fragments",
    "remove_fusions",
    "remove_islands",
    "remove_organelles",
]


def _non_degenerate(faces: np.ndarray) -> np.ndarray:
    """Boolean mask of faces with 3 distinct vertices."""
    return (
        (faces[:, 0] != faces[:, 1])
        & (faces[:, 1] != faces[:, 2])
        & (faces[:, 0] != faces[:, 2])
    )


def _edge_to_faces(mesh: trimesh.Trimesh) -> dict[tuple[int, int], list[int]]:
    """Build edge→face adjacency, skipping degenerate faces."""
    result: dict[tuple[int, int], list[int]] = defaultdict(list)
    faces = np.asarray(mesh.faces)  # snapshot — avoids trimesh caching overhead
    good = _non_degenerate(faces)
    # Vectorised edge extraction: 3 edges per face
    fi_idx = np.where(good)[0]
    gf = faces[fi_idx]
    for col_a, col_b in ((0, 1), (1, 2), (0, 2)):
        va = gf[:, col_a].astype(np.intp)
        vb = gf[:, col_b].astype(np.intp)
        lo = np.minimum(va, vb)
        hi = np.maximum(va, vb)
        for k in range(len(fi_idx)):
            result[(int(lo[k]), int(hi[k]))].append(int(fi_idx[k]))
    return result


def _face_edge_components(mesh: trimesh.Trimesh) -> tuple[np.ndarray, int]:
    """Connected components of faces using edge adjacency.

    Two faces are in the same component only if they share an edge
    (not just a vertex).

    Returns
    -------
    labels : np.ndarray
        (nFaces,) int array — component id for each face.
    main : int
        Component id of the largest component.
    """
    faces = mesh.faces
    n = len(faces)

    # Mark degenerate faces (duplicate vertices) so they are skipped
    degen = ~_non_degenerate(faces)

    edge_to_faces: dict[tuple[int, int], list[int]] = defaultdict(list)
    for fi in range(n):
        if degen[fi]:
            continue
        face = faces[fi]
        for i in range(3):
            e = (
                min(int(face[i]), int(face[(i + 1) % 3])),
                max(int(face[i]), int(face[(i + 1) % 3])),
            )
            edge_to_faces[e].append(fi)

    # -1 = unlabeled, -2 = degenerate (permanently skipped)
    labels = np.full(n, -1, dtype=np.intp)
    labels[degen] = -2
    comp_id = 0
    best_id, best_size = 0, 0

    for seed in range(n):
        if labels[seed] >= 0 or labels[seed] == -2:
            continue
        queue = [seed]
        labels[seed] = comp_id
        size = 0
        while queue:
            fi = queue.pop()
            size += 1
            face = mesh.faces[fi]
            for i in range(3):
                e = (
                    min(int(face[i]), int(face[(i + 1) % 3])),
                    max(int(face[i]), int(face[(i + 1) % 3])),
                )
                for nb in edge_to_faces[e]:
                    if labels[nb] < 0:
                        labels[nb] = comp_id
                        queue.append(nb)
        if size > best_size:
            best_id, best_size = comp_id, size
        comp_id += 1

    return labels, best_id


def find_holes(
    mesh: trimesh.Trimesh,
    *,
    verbose: bool = False,
) -> list[list[int]]:
    """Find boundary loops (holes) in the main surface of the mesh.

    A hole is a closed loop of boundary edges — edges used by exactly
    one face.  Only holes in the main edge-adjacency component are
    returned; floating fragments (faces connected by at most a single
    vertex) are ignored.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Input mesh.
    verbose : bool, default False
        Print summary.

    Returns
    -------
    list[list[int]]
        Each element is an ordered list of vertex indices forming one
        boundary loop.
    """
    face_comp, main_comp = _face_edge_components(mesh)

    # Build edge → face mapping and identify boundary edges
    edge_to_faces = _edge_to_faces(mesh)

    # Boundary edges whose face is in the main component
    boundary_edges = []
    for e, fis in edge_to_faces.items():
        if len(fis) == 1 and face_comp[fis[0]] == main_comp:
            boundary_edges.append(e)

    if verbose:
        print(f"[skeliner.pre] Holes: {len(boundary_edges)} boundary edges")

    if not boundary_edges:
        if verbose:
            print("[skeliner.pre] No holes found")
        return []

    # Trace closed loops from boundary edges
    adj: dict[int, set[int]] = defaultdict(set)
    for v1, v2 in boundary_edges:
        adj[v1].add(v2)
        adj[v2].add(v1)

    visited: set[int] = set()
    loops: list[list[int]] = []

    for start in sorted(adj):
        if start in visited:
            continue
        loop = [start]
        visited.add(start)
        current = start
        prev = -1
        while True:
            nbs = [n for n in adj[current] if n != prev]
            if not nbs:
                break
            closed = False
            for n in nbs:
                if n == start and len(loop) > 2:
                    loops.append(loop)
                    closed = True
                    break
            if closed:
                break
            nxt = next((n for n in nbs if n not in visited), None)
            if nxt is None:
                break
            visited.add(nxt)
            loop.append(nxt)
            prev = current
            current = nxt

    if verbose:
        print(f"[skeliner.pre] Found {len(loops)} holes")
        for i, loop in enumerate(loops):
            pts = mesh.vertices[loop]
            perimeter = sum(
                np.linalg.norm(pts[(j + 1) % len(pts)] - pts[j])
                for j in range(len(pts))
            )
            print(f"  Hole {i}: {len(loop)} vertices, perimeter={perimeter:.1f}")

    return loops


def _ear_clip_2d(pts_2d: np.ndarray) -> list[tuple[int, int, int]]:
    """Ear-clipping triangulation of a 2D polygon.

    Parameters
    ----------
    pts_2d : np.ndarray
        (N, 2) ordered polygon vertices.

    Returns
    -------
    list[tuple[int, int, int]]
        Triangle vertex indices into *pts_2d*.
    """
    n = len(pts_2d)
    if n < 3:
        return []
    if n == 3:
        return [(0, 1, 2)]

    # Ensure CCW winding
    indices = list(range(n))
    signed_area = 0.0
    for i in range(n):
        j = (i + 1) % n
        signed_area += pts_2d[i, 0] * pts_2d[j, 1]
        signed_area -= pts_2d[j, 0] * pts_2d[i, 1]
    if signed_area < 0:
        indices = indices[::-1]

    def _cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    def _pt_in_tri(p, a, b, c):
        d1, d2, d3 = _cross(p, a, b), _cross(p, b, c), _cross(p, c, a)
        return not ((d1 < 0 or d2 < 0 or d3 < 0) and (d1 > 0 or d2 > 0 or d3 > 0))

    def _is_ear(idxs, i):
        m = len(idxs)
        p_i, c_i, n_i = (i - 1) % m, i, (i + 1) % m
        a, b, c = pts_2d[idxs[p_i]], pts_2d[idxs[c_i]], pts_2d[idxs[n_i]]
        if _cross(a, b, c) <= 0:
            return False
        for j in range(m):
            if j in (p_i, c_i, n_i):
                continue
            if _pt_in_tri(pts_2d[idxs[j]], a, b, c):
                return False
        return True

    triangles: list[tuple[int, int, int]] = []
    for _ in range(n * n):
        m = len(indices)
        if m == 3:
            triangles.append((indices[0], indices[1], indices[2]))
            break
        if m < 3:
            break
        ear_found = False
        for i in range(m):
            if _is_ear(indices, i):
                p_i, n_i = (i - 1) % m, (i + 1) % m
                triangles.append((indices[p_i], indices[i], indices[n_i]))
                indices.pop(i)
                ear_found = True
                break
        if not ear_found:
            for i in range(1, m - 1):
                triangles.append((indices[0], indices[i], indices[i + 1]))
            break

    return triangles


def _fill_single_hole(
    mesh: trimesh.Trimesh,
    loop: list[int],
    target_edge: float,
    vert_offset: int,
    vtree: KDTree | None = None,
    vert_to_faces: list[list[int]] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Fill one boundary loop with curvature-aware tessellation.

    Returns (new_vertices (N,3), new_faces (M,3)).
    Face indices use original mesh indices for boundary vertices
    and ``vert_offset + i`` for newly created vertices.
    """
    from scipy.interpolate import RBFInterpolator

    verts = mesh.vertices
    boundary_pts = verts[loop]
    n_bnd = len(loop)

    # ── Local 2D frame via SVD ────────────────────────────────────────
    centroid_3d = boundary_pts.mean(axis=0)
    centered = boundary_pts - centroid_3d
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    u_ax, v_ax, n_ax = Vt[0], Vt[1], Vt[2]

    bnd_2d = centered @ np.column_stack([u_ax, v_ax])
    bnd_h = centered @ n_ax

    # ── RBF surface from boundary + nearby mesh vertices ──────────────
    if vtree is None:
        vtree = KDTree(verts)
    nearby: set[int] = set()
    for vi in loop:
        nearby.update(vtree.query_ball_point(verts[vi], 3.0 * target_edge))
    nearby -= set(loop)

    if nearby:
        nrb_centered = verts[sorted(nearby)] - centroid_3d
        nrb_2d = nrb_centered @ np.column_stack([u_ax, v_ax])
        nrb_h = nrb_centered @ n_ax
        # Filter: keep only nearby vertices that project OUTSIDE the
        # boundary polygon in 2D.  Vertices from the opposite side of
        # a tube project inside and cause RBF overshoot (spikes).
        n_poly = len(bnd_2d)
        inside = np.zeros(len(nrb_2d), dtype=bool)
        px, py = nrb_2d[:, 0], nrb_2d[:, 1]
        j = n_poly - 1
        for ii in range(n_poly):
            xi, yi = bnd_2d[ii]
            xj, yj = bnd_2d[j]
            cond = ((yi > py) != (yj > py)) & (
                px < (xj - xi) * (py - yi) / (yj - yi + 1e-30) + xi
            )
            inside ^= cond
            j = ii
        outside = ~inside
        nrb_2d = nrb_2d[outside]
        nrb_h = nrb_h[outside]
        if len(nrb_2d) > 0:
            rbf_pts = np.vstack([bnd_2d, nrb_2d])
            rbf_vals = np.concatenate([bnd_h, nrb_h])
        else:
            rbf_pts = bnd_2d
            rbf_vals = bnd_h
    else:
        rbf_pts = bnd_2d
        rbf_vals = bnd_h

    try:
        rbf = RBFInterpolator(rbf_pts, rbf_vals, kernel="thin_plate_spline")
    except np.linalg.LinAlgError:
        rbf = None

    # ── Ear-clip triangulation in 2D ──────────────────────────────────
    tri_list = _ear_clip_2d(bnd_2d)

    # ── Refine by longest-edge bisection ──────────────────────────────
    # Boundary edges (shared with original mesh) must not be split to
    # avoid T-junctions.
    bnd_edge_set: set[tuple[int, int]] = set()
    for idx in range(n_bnd):
        a, b = idx, (idx + 1) % n_bnd
        bnd_edge_set.add((min(a, b), max(a, b)))

    pts: list[np.ndarray] = list(bnd_2d)
    faces = [list(t) for t in tri_list]
    threshold = target_edge * 1.5
    edge_midpoints: dict[tuple[int, int], int] = {}

    for _ in range(200):
        new_faces: list[list[int]] = []
        changed = False
        for f in faces:
            i, j, k = f
            pi, pj, pk = pts[i], pts[j], pts[k]
            edges_sorted = sorted(
                [
                    (float(np.linalg.norm(pj - pi)), i, j),
                    (float(np.linalg.norm(pk - pj)), j, k),
                    (float(np.linalg.norm(pi - pk)), k, i),
                ],
                reverse=True,
            )
            split_done = False
            for elen, va, vb in edges_sorted:
                if elen <= threshold:
                    break
                ekey = (min(va, vb), max(va, vb))
                if ekey in bnd_edge_set:
                    continue
                if ekey in edge_midpoints:
                    mi = edge_midpoints[ekey]
                else:
                    mi = len(pts)
                    pts.append((pts[va] + pts[vb]) / 2.0)
                    edge_midpoints[ekey] = mi
                vc = [v for v in (i, j, k) if v != va and v != vb][0]
                new_faces.append([va, mi, vc])
                new_faces.append([mi, vb, vc])
                split_done = True
                changed = True
                break
            if not split_done:
                new_faces.append(f)
        faces = new_faces
        if not changed:
            break

    pts_arr = np.array(pts)
    faces_arr = np.array(faces, dtype=np.intp)
    n_new = len(pts_arr) - n_bnd

    # ── Map new 2D points → 3D via RBF height ────────────────────────
    if n_new > 0:
        new_2d = pts_arr[n_bnd:]
        if rbf is not None:
            new_h = np.clip(rbf(new_2d), bnd_h.min(), bnd_h.max())
        else:
            # Fallback: inverse-distance weighted interpolation from boundary
            dists = np.linalg.norm(
                new_2d[:, np.newaxis, :] - bnd_2d[np.newaxis, :, :], axis=2
            )
            weights = 1.0 / np.maximum(dists, 1e-10) ** 2
            new_h = (weights * bnd_h[np.newaxis, :]).sum(axis=1) / weights.sum(axis=1)
        new_3d = (
            centroid_3d
            + new_2d[:, 0:1] * u_ax
            + new_2d[:, 1:2] * v_ax
            + new_h[:, np.newaxis] * n_ax
        )
    else:
        new_3d = np.empty((0, 3))

    # ── Remap face indices ────────────────────────────────────────────
    loop_arr = np.array(loop, dtype=np.intp)
    remapped = np.empty_like(faces_arr)
    bnd_mask = faces_arr < n_bnd
    remapped[bnd_mask] = loop_arr[faces_arr[bnd_mask]]
    remapped[~bnd_mask] = vert_offset + (faces_arr[~bnd_mask] - n_bnd)

    # ── Consistent face orientation ───────────────────────────────────
    local_v = np.vstack([boundary_pts, new_3d]) if n_new > 0 else boundary_pts
    tri_normals = np.cross(
        local_v[faces_arr[:, 1]] - local_v[faces_arr[:, 0]],
        local_v[faces_arr[:, 2]] - local_v[faces_arr[:, 0]],
    )
    # Find adjacent faces via vert_to_faces index (fast) or fallback
    adj_fi: list[int] = []
    if vert_to_faces is not None:
        seen: set[int] = set()
        for vi in loop:
            for fi in vert_to_faces[vi]:
                if fi not in seen:
                    adj_fi.append(fi)
                    seen.add(fi)
                    if len(seen) > 50:  # enough for a good reference normal
                        break
            if len(seen) > 50:
                break
    else:
        loop_set = set(loop)
        _good = _non_degenerate(mesh.faces)
        adj_fi = [
            fi
            for fi, face in enumerate(mesh.faces)
            if _good[fi] and set(int(v) for v in face) & loop_set
        ]
    if adj_fi:
        ref_n = mesh.face_normals[adj_fi].mean(axis=0)
        if np.dot(tri_normals.mean(axis=0), ref_n) < 0:
            remapped = remapped[:, ::-1]

    return new_3d, remapped


def _fill_advancing_front(
    loop: list[int],
    vertices: np.ndarray,
    face_normals: np.ndarray,
    vert_to_faces: list[list[int]] | None = None,
) -> np.ndarray:
    """Fill a boundary loop using advancing front.

    Repeatedly picks the sharpest angle on the boundary and closes it
    with a triangle. No new vertices are created — only existing
    boundary vertices are used.

    Returns (M, 3) int array of new faces.
    """
    n = len(loop)
    if n < 3:
        return np.empty((0, 3), dtype=np.int64)
    if n == 3:
        tri = np.array([[loop[0], loop[1], loop[2]]], dtype=np.int64)
        # Orient consistently with neighbors
        if vert_to_faces is not None:
            ref_fi = []
            for vi in loop[:1]:
                ref_fi.extend(vert_to_faces[vi][:5])
            if ref_fi:
                ref_n = face_normals[ref_fi].mean(axis=0)
                v0, v1, v2 = vertices[loop[0]], vertices[loop[1]], vertices[loop[2]]
                tri_n = np.cross(v1 - v0, v2 - v0)
                if np.dot(tri_n, ref_n) < 0:
                    tri = tri[:, ::-1]
        return tri

    # Determine consistent winding from adjacent mesh faces
    flip = False
    if vert_to_faces is not None:
        ref_fi = []
        for vi in loop[:3]:
            ref_fi.extend(vert_to_faces[vi][:5])
        if ref_fi:
            ref_n = face_normals[ref_fi].mean(axis=0)
            v0, v1, v2 = vertices[loop[0]], vertices[loop[1]], vertices[loop[2]]
            tri_n = np.cross(v1 - v0, v2 - v0)
            if np.dot(tri_n, ref_n) < 0:
                flip = True

    # Work with a mutable list of boundary vertex indices
    bnd = list(loop)
    new_faces = []

    for _ in range(n * 2):  # safety limit
        m = len(bnd)
        if m < 3:
            break
        if m == 3:
            if flip:
                new_faces.append([bnd[0], bnd[2], bnd[1]])
            else:
                new_faces.append([bnd[0], bnd[1], bnd[2]])
            break

        # Find the ear with the smallest interior angle
        # (sharpest corner = best triangle to close)
        best_i = -1
        best_angle = float("inf")

        for i in range(m):
            p = vertices[bnd[(i - 1) % m]]
            c = vertices[bnd[i]]
            n_pt = vertices[bnd[(i + 1) % m]]

            e1 = p - c
            e2 = n_pt - c
            len1 = np.linalg.norm(e1)
            len2 = np.linalg.norm(e2)
            if len1 < 1e-10 or len2 < 1e-10:
                best_i = i
                best_angle = 0
                break
            cos_a = np.dot(e1, e2) / (len1 * len2)
            cos_a = np.clip(cos_a, -1, 1)
            angle = np.arccos(cos_a)

            if angle < best_angle:
                best_angle = angle
                best_i = i

        if best_i < 0:
            break

        # Create triangle at this ear
        prev_i = (best_i - 1) % m
        next_i = (best_i + 1) % m
        if flip:
            new_faces.append([bnd[prev_i], bnd[next_i], bnd[best_i]])
        else:
            new_faces.append([bnd[prev_i], bnd[best_i], bnd[next_i]])

        # Remove the vertex from the boundary
        bnd.pop(best_i)

    if not new_faces:
        return np.empty((0, 3), dtype=np.int64)
    return np.array(new_faces, dtype=np.int64)


def _fill_dome(
    loop: list[int],
    vertices: np.ndarray,
    face_normals: np.ndarray,
    vert_to_faces: list[list[int]] | None = None,
    dome_factor: float = 0.5,
    vert_offset: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Fill a boundary loop with a convex dome cap.

    Creates a center vertex pushed outward along the average boundary
    normal, with concentric rings for larger holes.

    Returns ``(new_vertices (K, 3), new_faces (M, 3))``.  Face indices
    use original mesh indices for boundary vertices and
    ``vert_offset + i`` for newly created vertices.
    """
    n = len(loop)
    if n < 3:
        return np.empty((0, 3)), np.empty((0, 3), dtype=np.int64)

    boundary_pts = vertices[loop]
    centroid = boundary_pts.mean(axis=0)

    # ── Outward normal from adjacent faces ────────────────────────────
    avg_normal = None
    if vert_to_faces is not None:
        adj_normals = []
        for vi in loop:
            for fi in vert_to_faces[vi]:
                adj_normals.append(face_normals[fi])
        if adj_normals:
            avg_normal = np.mean(adj_normals, axis=0)
            norm = np.linalg.norm(avg_normal)
            if norm > 1e-10:
                avg_normal = avg_normal / norm
            else:
                avg_normal = None

    if avg_normal is None:
        centered = boundary_pts - centroid
        _, _, Vt = np.linalg.svd(centered, full_matrices=False)
        avg_normal = Vt[2]
        if vert_to_faces is not None and vert_to_faces[loop[0]]:
            ref = face_normals[vert_to_faces[loop[0]][0]]
            if np.dot(avg_normal, ref) < 0:
                avg_normal = -avg_normal

    # ── Small holes: flat fill (no dome needed) ───────────────────────
    if n == 3:
        tri = np.array([[loop[0], loop[1], loop[2]]], dtype=np.int64)
        v0, v1, v2 = vertices[loop[0]], vertices[loop[1]], vertices[loop[2]]
        tri_n = np.cross(v1 - v0, v2 - v0)
        if np.dot(tri_n, avg_normal) < 0:
            tri = tri[:, ::-1]
        return np.empty((0, 3)), tri

    # ── Dome geometry ─────────────────────────────────────────────────
    radii = np.linalg.norm(boundary_pts - centroid, axis=1)
    avg_radius = float(radii.mean())
    dome_height = avg_radius * dome_factor

    # Intermediate rings for larger holes (better triangle quality)
    n_rings = max(0, (n - 4) // 15)

    new_verts_list: list[np.ndarray] = []
    ring_indices: list[list[int]] = []

    for r in range(1, n_rings + 1):
        t = r / (n_rings + 1)  # 0 → boundary, 1 → center
        frac = 1.0 - t  # radial fraction
        h = dome_height * (1.0 - frac * frac)  # parabolic profile
        ring_pts = centroid + (boundary_pts - centroid) * frac + avg_normal * h
        base = vert_offset + len(new_verts_list)
        ring_indices.append(list(range(base, base + n)))
        for pt in ring_pts:
            new_verts_list.append(pt)

    # Center vertex
    center = centroid + avg_normal * dome_height
    center_idx = vert_offset + len(new_verts_list)
    new_verts_list.append(center)

    # ── Triangulation ─────────────────────────────────────────────────
    faces: list[list[int]] = []
    prev = loop  # original vertex indices

    # Boundary → rings (quad strips → 2 triangles each)
    for ring in ring_indices:
        for i in range(n):
            j = (i + 1) % n
            faces.append([prev[i], prev[j], ring[j]])
            faces.append([prev[i], ring[j], ring[i]])
        prev = ring

    # Last ring (or boundary) → center fan
    for i in range(n):
        j = (i + 1) % n
        faces.append([prev[i], prev[j], center_idx])

    new_verts = np.array(new_verts_list)

    # ── Consistent orientation ────────────────────────────────────────
    def _pt(idx: int) -> np.ndarray:
        return vertices[idx] if idx < vert_offset else new_verts[idx - vert_offset]

    f0 = faces[0]
    tri_n = np.cross(_pt(f0[1]) - _pt(f0[0]), _pt(f0[2]) - _pt(f0[0]))
    if np.dot(tri_n, avg_normal) < 0:
        faces = [[f[1], f[0], f[2]] for f in faces]

    return new_verts, np.array(faces, dtype=np.int64)


def fill_holes(
    mesh: trimesh.Trimesh,
    *,
    holes: list[list[int]] | None = None,
    method: str = "advancing_front",
    max_perimeter_mult: float | None = None,
    dome_factor: float = 0.5,
    verbose: bool = False,
) -> trimesh.Trimesh:
    """Fill holes in the mesh.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Input mesh (typically after ``remove_organelles``).
    holes : list[list[int]] or None
        Pre-computed boundary loops from :func:`find_holes`.  If
        provided, detection is skipped and these loops are used directly.
    method : str, default "advancing_front"
        Filling method:
        - ``"advancing_front"`` — fast, closes boundary edges one by one
          by picking the sharpest angle. No new vertices. Works for all
          hole sizes.
        - ``"dome"`` — fills with a convex dome cap pushed outward along
          the boundary normal. Creates new vertices (center + optional
          concentric rings). Fast, O(n). Best for neurite tip holes.
        - ``"rbf"`` — projects to 2D, ear-clips, refines, lifts with RBF.
          Better surface quality for small planar holes but slow and fails
          for large curved holes.
    max_perimeter_mult : float or None
        Skip holes whose perimeter exceeds this multiple of the median
        edge length.  ``None`` = no limit (fill all holes).
        Default ``None`` for advancing_front/dome, ``50.0`` for rbf.
    dome_factor : float, default 0.5
        Height of dome as fraction of hole radius (only for ``"dome"``).
        0 = flat, 0.5 = gentle dome, 1.0 = hemisphere.
    verbose : bool, default False
        Print progress.

    Returns
    -------
    trimesh.Trimesh
        Mesh with holes filled.
    """
    if holes is not None:
        loops = holes
        if verbose:
            print(f"[skeliner.pre] Using provided holes ({len(loops)} loops)")
    else:
        loops = find_holes(mesh, verbose=verbose)
    if not loops:
        return mesh

    target_edge = float(np.median(mesh.edges_unique_length))

    # Default max_perimeter depends on method
    if max_perimeter_mult is None:
        if method == "rbf":
            max_perimeter_mult = 50.0
        # advancing_front: no limit
    max_perimeter = (
        max_perimeter_mult * target_edge
        if max_perimeter_mult is not None
        else float("inf")
    )

    if verbose:
        print(
            f"[skeliner.pre] Method: {method}, target edge: {target_edge:.1f}"
            + (
                f", max perimeter: {max_perimeter:.0f}"
                if max_perimeter < float("inf")
                else ", no perimeter limit"
            )
        )

    # Pre-build vert_to_faces for orientation (skip degenerate faces)
    _good = _non_degenerate(mesh.faces)
    vert_to_faces: list[list[int]] = [[] for _ in range(len(mesh.vertices))]
    for fi, face in enumerate(mesh.faces):
        if _good[fi]:
            for v in face:
                vert_to_faces[int(v)].append(fi)

    # Filter loops by perimeter
    valid_loops: list[list[int]] = []
    n_skipped = 0
    for loop in loops:
        pts = mesh.vertices[loop]
        perimeter = float(
            np.linalg.norm(np.diff(np.vstack([pts, pts[:1]]), axis=0), axis=1).sum()
        )
        if perimeter > max_perimeter:
            n_skipped += 1
        else:
            valid_loops.append(loop)

    if verbose:
        print(
            f"[skeliner.pre] Filling {len(valid_loops)} holes"
            + (f" (skipped {n_skipped} too large)" if n_skipped else "")
        )

    if method == "advancing_front":
        new_faces: list[np.ndarray] = []
        for loop in valid_loops:
            nf = _fill_advancing_front(
                loop, mesh.vertices, mesh.face_normals, vert_to_faces
            )
            if len(nf):
                new_faces.append(nf)

        if not new_faces:
            return mesh

        all_faces = np.vstack([mesh.faces] + new_faces)
        result = trimesh.Trimesh(vertices=mesh.vertices, faces=all_faces, process=False)

    elif method == "dome":
        new_verts: list[np.ndarray] = []
        new_face_list: list[np.ndarray] = []
        offset = len(mesh.vertices)

        for loop in valid_loops:
            nv, nf = _fill_dome(
                loop,
                mesh.vertices,
                mesh.face_normals,
                vert_to_faces,
                dome_factor=dome_factor,
                vert_offset=offset,
            )
            if len(nv) > 0:
                new_verts.append(nv)
            if len(nf) > 0:
                new_face_list.append(nf)
            offset += len(nv)

        if not new_face_list:
            return mesh

        v_parts = [mesh.vertices] + new_verts
        f_parts = [mesh.faces] + new_face_list
        result = trimesh.Trimesh(
            vertices=np.vstack(v_parts),
            faces=np.vstack(f_parts),
            process=False,
        )

    elif method == "rbf":
        vtree = KDTree(mesh.vertices)
        new_verts_rbf: list[np.ndarray] = []
        new_face_rbf: list[np.ndarray] = []
        offset = len(mesh.vertices)

        for loop in valid_loops:
            nv, nf = _fill_single_hole(
                mesh, loop, target_edge, offset, vtree, vert_to_faces
            )
            new_verts_rbf.append(nv)
            new_face_rbf.append(nf)
            offset += len(nv)

        v_parts = [mesh.vertices] + [v for v in new_verts_rbf if len(v) > 0]
        f_parts = [mesh.faces] + [f for f in new_face_rbf if len(f) > 0]
        result = trimesh.Trimesh(
            vertices=np.vstack(v_parts),
            faces=np.vstack(f_parts),
            process=False,
        )
    else:
        raise ValueError(
            f"Unknown method: {method!r}. Use 'advancing_front', 'dome', or 'rbf'."
        )

    if verbose:
        print(
            f"[skeliner.pre] Result: {len(result.vertices):,} verts, "
            f"{len(result.faces):,} faces"
            + (f" ({n_skipped} holes skipped)" if n_skipped else "")
        )
    return result


# ── Merge selected faces ────────────────────────────────────────────


def _trace_border_loops(
    mesh: trimesh.Trimesh,
    sel: set[int],
    edge_to_faces: dict[tuple[int, int], list[int]],
) -> list[list[int]]:
    """Trace closed loops along the border between *sel* and kept faces."""
    border_edges: list[tuple[int, int]] = []
    for e, fis in edge_to_faces.items():
        has_sel = any(fi in sel for fi in fis)
        has_kept = any(fi not in sel for fi in fis)
        if has_sel and has_kept:
            border_edges.append(e)

    if len(border_edges) < 3:
        return []

    adj: dict[int, set[int]] = defaultdict(set)
    for v1, v2 in border_edges:
        adj[v1].add(v2)
        adj[v2].add(v1)

    visited: set[int] = set()
    loops: list[list[int]] = []
    for start in sorted(adj):
        if start in visited:
            continue
        loop = [start]
        visited.add(start)
        current, prev = start, -1
        while True:
            nbs = [n for n in adj[current] if n != prev]
            if not nbs:
                break
            closed = False
            for n in nbs:
                if n == start and len(loop) > 2:
                    loops.append(loop)
                    closed = True
                    break
            if closed:
                break
            nxt = next((n for n in nbs if n not in visited), None)
            if nxt is None:
                break
            visited.add(nxt)
            loop.append(nxt)
            prev, current = current, nxt

    return loops


def _validate_loop_pair(
    la: list[int],
    lb: list[int],
    *,
    min_verts: int = 4,
    max_ratio: float = 5.0,
) -> str | None:
    """Return a reason string if the loop pair is unsuitable for stitching.

    Reject 3-vertex "loops" (degenerate triangle holes that no stitcher
    can bridge to a larger rim manifoldly), and reject pairs whose size
    ratio exceeds *max_ratio* (highly mismatched rims force the zipper
    or the tube fitter to fan multiple ring verts onto a single rim
    vert, producing fan vertices and non-manifold edges).

    Returns ``None`` if the pair passes.
    """
    na, nb = len(la), len(lb)
    if na < min_verts or nb < min_verts:
        return f"loop too small ({na}v + {nb}v, min {min_verts})"
    ratio = max(na, nb) / min(na, nb)
    if ratio > max_ratio:
        return (
            f"loop size mismatch ({na}v + {nb}v, ratio {ratio:.1f} > {max_ratio:.1f})"
        )
    return None


def _expand_tip_to_good_rim(
    mesh: trimesh.Trimesh,
    tip: list[int],
    edge_to_faces: dict[tuple[int, int], list[int]],
    face_adj: dict[int, set[int]],
    *,
    target_verts: int = 6,
    max_iters: int = 20,
    max_eat_factor: int = 30,
) -> tuple[set[int], list[int] | None]:
    """Iteratively peel BFS rings from *tip* until its border has at least
    *target_verts* vertices.

    Returns ``(expanded_tip, best_loop)``. ``best_loop`` is the largest
    border loop found at the final tip (``None`` if no loop ever traced).

    The initial tip from :func:`find_gaps` can produce a degenerate
    "rim" (e.g. 3 verts when the tip is a tongue attached at a single
    triangle). Each expansion step adds the next BFS ring of
    edge-neighbors and re-traces the border. Expansion is capped at
    ``len(tip) * max_eat_factor`` faces to avoid swallowing the entire
    component.
    """
    sel: set[int] = set(int(fi) for fi in tip)
    if not sel:
        return sel, None
    max_size = max(len(sel) * max_eat_factor, target_verts * 4)

    best_loop: list[int] | None = None
    for _ in range(max_iters):
        loops = _trace_border_loops(mesh, sel, edge_to_faces)
        loops = [lp for lp in loops if len(lp) >= 3]
        if loops:
            biggest = max(loops, key=len)
            best_loop = biggest
            if len(biggest) >= target_verts:
                return sel, biggest
        # Expand by one BFS ring
        next_sel = set(sel)
        for fi in sel:
            for nfi in face_adj.get(fi, ()):
                next_sel.add(nfi)
        if next_sel == sel or len(next_sel) > max_size:
            break
        sel = next_sel

    return sel, best_loop


def _expand_selection_per_group(
    mesh: trimesh.Trimesh,
    sel: list[int] | set[int],
    edge_to_faces: dict[tuple[int, int], list[int]],
    face_adj: dict[int, set[int]],
    *,
    target_verts: int = 6,
    max_iters: int = 20,
    max_eat_factor: int = 30,
) -> tuple[set[int], list[list[int]]]:
    """Split *sel* into edge-connected sub-groups, expand each
    independently with :func:`_expand_tip_to_good_rim`, then re-trace
    the border loops on the merged expanded selection.

    This is the user-driven merge analogue of the per-side expansion
    in :func:`remove_gaps`. Expanding each connected sub-selection on
    its own (rather than the whole selection at once) lets each side's
    rim grow toward similar sizes — `_zipper_stitch`'s KDTree weld
    only behaves manifoldly when the two rims are matched in vertex
    count, so symmetric per-side expansion is what unlocks a clean
    bridge.
    """
    sel_set: set[int] = set(int(fi) for fi in sel)
    if not sel_set:
        return sel_set, []

    # Split sel into edge-connected sub-groups via BFS over face_adj,
    # restricted to faces inside sel_set.
    groups: list[set[int]] = []
    visited: set[int] = set()
    for seed in sel_set:
        if seed in visited:
            continue
        group: set[int] = set()
        stack = [seed]
        while stack:
            f = stack.pop()
            if f in visited:
                continue
            visited.add(f)
            group.add(f)
            for nf in face_adj.get(f, ()):
                if nf in sel_set and nf not in visited:
                    stack.append(nf)
        groups.append(group)

    # Expand each group independently
    expanded: set[int] = set()
    for grp in groups:
        sub_sel, _ = _expand_tip_to_good_rim(
            mesh,
            list(grp),
            edge_to_faces,
            face_adj,
            target_verts=target_verts,
            max_iters=max_iters,
            max_eat_factor=max_eat_factor,
        )
        expanded |= sub_sel

    # Re-trace loops on the final merged selection
    loops = _trace_border_loops(mesh, expanded, edge_to_faces)
    loops = [lp for lp in loops if len(lp) >= 3]
    return expanded, loops


def _fit_loop_circle(pts: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    """Fit a circle to a 3D point loop.  Returns (center, radius, normal)."""
    center = pts.mean(axis=0)
    radii = np.linalg.norm(pts - center, axis=1)
    radius = float(np.mean(radii))
    # Normal from PCA: smallest eigenvector of covariance
    cov = np.cov((pts - center).T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    normal = eigvecs[:, 0]  # smallest eigenvalue = normal direction
    return center, radius, normal


def _hermite_spline(
    p0: np.ndarray,
    t0: np.ndarray,
    p1: np.ndarray,
    t1: np.ndarray,
    n: int,
) -> np.ndarray:
    """Evaluate cubic Hermite spline at *n* evenly spaced stations.

    Returns ``(n, 3)`` array including endpoints.
    """
    s = np.linspace(0, 1, n)[:, None]
    h00 = 2 * s**3 - 3 * s**2 + 1
    h10 = s**3 - 2 * s**2 + s
    h01 = -2 * s**3 + 3 * s**2
    h11 = s**3 - s**2
    return h00 * p0 + h10 * t0 + h01 * p1 + h11 * t1


def _zipper_stitch(
    mesh: trimesh.Trimesh,
    loop_a: list[int],
    loop_b: list[int],
    vert_to_faces: list[list[int]],
    new_verts: list[np.ndarray] | None = None,
) -> list[list[int]]:
    """Bridge two boundary loops with a tubular surface, returning new triangles.

    Generates circular cross-sections along a curved Hermite-spline
    centerline so the result looks like a realistic neurite segment.
    The radius interpolates smoothly between the two loop radii, and
    parallel transport keeps the frame twist-free.

    Parameters
    ----------
    new_verts : list or None
        Mutable list to which any newly created vertex coordinates are
        appended.  Vertex indices in the returned triangles reference
        ``mesh.vertices`` for existing verts and
        ``len(mesh.vertices) + i`` for ``new_verts[i]``.
    """

    if new_verts is None:
        new_verts_local: list[np.ndarray] = []
    else:
        new_verts_local = new_verts

    la = list(loop_a)
    lb = list(loop_b)

    # ── Fit loops and build curved centerline ───────────────────
    ca_fit, ra, na = _fit_loop_circle(mesh.vertices[la])
    cb_fit, rb, nb = _fit_loop_circle(mesh.vertices[lb])

    direction = cb_fit - ca_fit
    gap_dist = float(np.linalg.norm(direction))
    if np.dot(na, direction) < 0:
        na = -na
    if np.dot(nb, direction) < 0:
        nb = -nb

    median_edge = float(np.median(mesh.edges_unique_length))
    n_rings = max(1, int(round(gap_dist / median_edge)) - 1)

    # Hermite tangents: scale by gap distance for natural curvature
    tangent_a = na * gap_dist
    tangent_b = nb * gap_dist

    # n_rings intermediate + 2 endpoints = n_rings + 2 stations
    n_stations = n_rings + 2
    centers = _hermite_spline(ca_fit, tangent_a, cb_fit, tangent_b, n_stations)
    n_ring_pts = max(len(la), len(lb))
    n_existing = len(mesh.vertices)

    # Tangent at each station via finite differences
    tangents = np.zeros_like(centers)
    tangents[0] = centers[1] - centers[0]
    tangents[-1] = centers[-1] - centers[-2]
    for i in range(1, len(centers) - 1):
        tangents[i] = centers[i + 1] - centers[i - 1]
    tangents /= np.linalg.norm(tangents, axis=1, keepdims=True) + 1e-10

    # Parallel-transport a stable "up" vector along the centerline
    up = np.zeros_like(tangents)
    t0 = tangents[0]
    seed_up = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(seed_up, t0)) > 0.9:
        seed_up = np.array([0.0, 1.0, 0.0])
    up[0] = seed_up - np.dot(seed_up, t0) * t0
    up[0] /= np.linalg.norm(up[0]) + 1e-10
    for i in range(1, len(centers)):
        u = up[i - 1] - np.dot(up[i - 1], tangents[i]) * tangents[i]
        norm = np.linalg.norm(u)
        up[i] = u / norm if norm > 1e-10 else up[i - 1]

    # ── Resample both loops as polar offsets (angle, radius) ────────
    # Express each loop vertex as (angle, radius) in its station's
    # local frame, resample to n_ring_pts at evenly-spaced angles,
    # then blend between the two profiles.  Angle-based resampling
    # ensures the vertex correspondence is stable across the blend.
    def _to_polar_offsets(pts_3d, center, tangent, up_vec):
        """Project 3D loop to (angle, radius) in the local frame."""
        right_vec = np.cross(tangent, up_vec)
        right_vec /= np.linalg.norm(right_vec) + 1e-10
        rel = pts_3d - center
        u_coords = np.dot(rel, up_vec)
        r_coords = np.dot(rel, right_vec)
        angles = np.arctan2(r_coords, u_coords)
        radii = np.sqrt(u_coords**2 + r_coords**2)
        return angles, radii

    def _resample_polar(angles, radii, n):
        """Resample a polar profile to *n* evenly-spaced angles."""
        order = np.argsort(angles)
        a_sorted = angles[order]
        r_sorted = radii[order]
        # Close the loop
        a_closed = np.concatenate(
            [a_sorted - 2 * np.pi, a_sorted, a_sorted + 2 * np.pi]
        )
        r_closed = np.concatenate([r_sorted, r_sorted, r_sorted])
        target_angles = np.linspace(-np.pi, np.pi, n, endpoint=False)
        resampled_r = np.interp(target_angles, a_closed, r_closed)
        return target_angles, resampled_r

    ang_a, rad_a = _to_polar_offsets(mesh.vertices[la], centers[0], tangents[0], up[0])
    ang_b, rad_b = _to_polar_offsets(
        mesh.vertices[lb], centers[-1], tangents[-1], up[-1]
    )
    target_angles, profile_a = _resample_polar(ang_a, rad_a, n_ring_pts)
    _, profile_b = _resample_polar(ang_b, rad_b, n_ring_pts)

    # ── Generate ALL rings including at endpoint stations ─────────
    # We generate rings at every station (including 0 and N-1) so that
    # ALL ring-to-ring connections are between uniform vertex counts.
    # The boundary loops (la, lb) are stitched to their co-located
    # endpoint rings separately — this avoids the cross-opening fan
    # that occurs when stitching an irregular boundary loop directly
    # to a uniform ring at a different station.
    ring_ids: list[list[int]] = []

    for si in range(n_stations):
        t = si / (n_stations - 1)  # 0 at A, 1 at B
        c = centers[si]
        u_vec = up[si]
        right_vec = np.cross(tangents[si], u_vec)
        right_vec /= np.linalg.norm(right_vec) + 1e-10
        profile = profile_a * (1 - t) + profile_b * t
        ids = []
        for j in range(n_ring_pts):
            r = profile[j]
            angle = target_angles[j]
            pt = c + r * (np.cos(angle) * u_vec + np.sin(angle) * right_vec)
            ids.append(n_existing + len(new_verts_local))
            new_verts_local.append(pt)
        ring_ids.append(ids)

    # ── Stitch: boundary loops → endpoint rings → ring chain ─────
    # Sequence: la → ring[0] → ring[1] → ... → ring[-1] → lb
    triangles: list[list[int]] = []

    def _vpos(vid):
        if vid < n_existing:
            return mesh.vertices[vid]
        return new_verts_local[vid - n_existing]

    # Only stitch ring-to-ring (no boundary→ring bands that seal ends).
    all_bands: list[tuple[list[int], list[int]]] = []
    for ri in range(len(ring_ids) - 1):
        all_bands.append((ring_ids[ri], ring_ids[ri + 1]))

    for ra_ids, rb_ids in all_bands:
        na_r, nb_r = len(ra_ids), len(rb_ids)

        # Align start of rb to closest vertex in ra
        best_j = 0
        best_d = float("inf")
        p0 = _vpos(ra_ids[0])
        for j in range(nb_r):
            d = float(np.linalg.norm(_vpos(rb_ids[j]) - p0))
            if d < best_d:
                best_d = d
                best_j = j
        rb_ids = list(rb_ids[best_j:]) + list(rb_ids[:best_j])

        ia, ib, sa, sb = 0, 0, 0, 0
        while sa < na_r or sb < nb_r:
            ia_n = (ia + 1) % na_r
            ib_n = (ib + 1) % nb_r
            can_a, can_b = sa < na_r, sb < nb_r
            if can_a and can_b:
                da = float(np.linalg.norm(_vpos(ra_ids[ia_n]) - _vpos(rb_ids[ib])))
                db = float(np.linalg.norm(_vpos(ra_ids[ia]) - _vpos(rb_ids[ib_n])))
                adv_a = da <= db
            else:
                adv_a = can_a
            if adv_a:
                triangles.append([ra_ids[ia], ra_ids[ia_n], rb_ids[ib]])
                ia = ia_n
                sa += 1
            else:
                triangles.append([ra_ids[ia], rb_ids[ib_n], rb_ids[ib]])
                ib = ib_n
                sb += 1

    # ── Weld endpoint ring verts to boundary loop verts ──────────
    # Replace each ring[0] / ring[-1] vertex ID in the triangles with
    # the nearest boundary loop vertex.  This connects the tube to the
    # existing mesh without creating cap faces across the opening.
    def _build_weld_map(ring, loop):
        """Map each ring vert to its nearest loop vert."""
        ring_pts = np.array([_vpos(v) for v in ring])
        loop_pts = np.array([_vpos(v) for v in loop])
        tree = KDTree(loop_pts)
        _, idxs = tree.query(ring_pts)
        return {ring[i]: loop[int(idxs[i])] for i in range(len(ring))}

    weld = {}
    weld.update(_build_weld_map(ring_ids[0], la))
    weld.update(_build_weld_map(ring_ids[-1], lb))

    if weld:
        triangles = [[weld.get(v, v) for v in tri] for tri in triangles]

    # Orient consistently with surrounding mesh
    ref_fis: list[int] = []
    for vi in la[:5]:
        ref_fis.extend(vert_to_faces[vi][:5])
    if ref_fis:
        ref_n = mesh.face_normals[ref_fis].mean(axis=0)
        tri_normals = [
            np.cross(
                _vpos(t[1]) - _vpos(t[0]),
                _vpos(t[2]) - _vpos(t[0]),
            )
            for t in triangles
        ]
        if np.dot(np.mean(tri_normals, axis=0), ref_n) < 0:
            triangles = [[t[0], t[2], t[1]] for t in triangles]

    return triangles


def _stitch_and_rebuild(
    mesh: trimesh.Trimesh,
    sel: set[int],
    loop_pairs: list[tuple[list[int], list[int]]],
    *,
    verbose: bool = False,
) -> trimesh.Trimesh:
    """Remove *sel* faces, stitch each loop pair, rebuild mesh once."""
    # Vert-to-face for orientation (non-removed faces only)
    vert_to_faces: list[list[int]] = [[] for _ in range(len(mesh.vertices))]
    non_degen = _non_degenerate(mesh.faces)
    for fi, face in enumerate(mesh.faces):
        if fi not in sel and non_degen[fi]:
            for v in face:
                vert_to_faces[int(v)].append(fi)

    all_stitch: list[list[int]] = []
    new_verts: list[np.ndarray] = []
    for loop_a, loop_b in loop_pairs:
        tris = _zipper_stitch(mesh, loop_a, loop_b, vert_to_faces, new_verts)
        all_stitch.extend(tris)
        if verbose:
            print(
                f"[skeliner.pre]   Pair ({len(loop_a)}v + {len(loop_b)}v): "
                f"{len(tris)} stitch faces"
            )

    # Degenerate removed faces, append stitch faces at the end
    new_faces = mesh.faces.copy()
    for fi in sel:
        new_faces[fi] = 0

    # Combine existing + new vertices
    if new_verts:
        all_vertices = np.vstack([mesh.vertices, np.array(new_verts)])
    else:
        all_vertices = mesh.vertices

    if all_stitch:
        stitch_faces = np.array(all_stitch, dtype=np.int64)
        new_faces = np.vstack([new_faces, stitch_faces])

    result = trimesh.Trimesh(
        vertices=all_vertices,
        faces=new_faces,
        process=False,
    )

    if verbose:
        print(
            f"[skeliner.pre] Result: {len(result.faces):,} faces "
            f"({len(sel)} removed, {len(all_stitch)} stitched)"
        )

    return result


def merge_selected_faces(
    mesh: trimesh.Trimesh,
    face_indices: list[int],
    *,
    verbose: bool = False,
) -> trimesh.Trimesh:
    """Remove selected faces and stitch the resulting boundary loops.

    Workflow: lasso-select faces at the tips of two disconnected
    components, then merge.  The selected faces are removed, the
    border edges (between selected and remaining faces) are traced
    into loops, the two closest loops are paired, and a zipper
    stitch connects them with new triangles.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Input mesh.
    face_indices : list[int]
        Indices of faces to remove (the "bridge" region).
    verbose : bool

    Returns
    -------
    trimesh.Trimesh
        Mesh with selected faces replaced by a stitched strip.
    """
    sel_input = set(face_indices)
    if not sel_input:
        return mesh

    # Edge / face adjacency maps
    edge_to_faces = _edge_to_faces(mesh)
    face_adj = _face_adjacency(mesh, edge_to_faces)

    if verbose:
        print(f"[skeliner.pre] Merge: requested {len(sel_input)} faces")

    # Expand the user selection if needed so every border loop is a
    # real rim (not a 3-vert tongue base). Each edge-connected
    # sub-group of the selection is expanded independently — same fix
    # as remove_gaps's per-side expansion — so the resulting rims
    # converge to matching sizes (`_zipper_stitch` only welds
    # manifoldly when both rims have similar vertex counts).
    sel, loops = _expand_selection_per_group(
        mesh, sel_input, edge_to_faces, face_adj
    )

    if verbose:
        if len(sel) != len(sel_input):
            print(
                f"[skeliner.pre] Merge: expanded selection "
                f"{len(sel_input)} -> {len(sel)} faces"
            )
        for i, lp in enumerate(loops):
            pts = mesh.vertices[lp]
            perimeter = float(
                np.linalg.norm(np.diff(np.vstack([pts, pts[:1]]), axis=0), axis=1).sum()
            )
            print(
                f"[skeliner.pre]   Loop {i}: {len(lp)} verts, perimeter={perimeter:.1f}"
            )

    if len(loops) < 2:
        if verbose:
            print("[skeliner.pre] Fewer than 2 loops — cannot stitch")
        keep = np.ones(len(mesh.faces), dtype=bool)
        for fi in sel:
            keep[fi] = False
        return _rebuild_mesh(mesh, keep)

    # Pair loops by closest centroid
    centroids = [mesh.vertices[lp].mean(axis=0) for lp in loops]
    pairs: list[tuple[list[int], list[int]]] = []
    available = list(range(len(loops)))
    while len(available) >= 2:
        best_d = float("inf")
        best_i, best_j = 0, 1
        for ii in range(len(available)):
            for jj in range(ii + 1, len(available)):
                d = float(
                    np.linalg.norm(centroids[available[ii]] - centroids[available[jj]])
                )
                if d < best_d:
                    best_d = d
                    best_i, best_j = ii, jj
        la = loops[available[best_i]]
        lb = loops[available[best_j]]
        # Safety net: if expansion couldn't fix this pair, skip it
        # rather than introduce a fusion. Both loops still get their
        # tip faces removed (so they become unbridged holes the user
        # can inspect), unlike `remove_gaps` where we keep the tips.
        reason = _validate_loop_pair(la, lb)
        if reason is not None:
            if verbose:
                print(
                    f"[skeliner.pre]   Pair skipped after expansion ({reason})"
                )
        else:
            pairs.append((la, lb))
        available.pop(best_j)
        available.pop(best_i)

    if verbose:
        print(f"[skeliner.pre] Stitching {len(pairs)} loop pair(s) ...")

    return _stitch_and_rebuild(mesh, sel, pairs, verbose=verbose)


def remove_selected_faces(
    mesh: trimesh.Trimesh,
    face_indices: list[int],
    *,
    verbose: bool = False,
) -> trimesh.Trimesh:
    """Remove the selected faces from the mesh, leaving open holes.

    Companion to :func:`merge_selected_faces`.  Where
    ``merge_selected_faces`` bridges the rims of the removed region
    with new triangles, this function leaves the boundary loops as
    open holes — useful when the user just wants to delete a chunk
    and inspect the result, or feed it into other repair steps.

    Face and vertex indices are preserved (removed faces become
    degenerate ``[0, 0, 0]``) so face-based annotations remain valid.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Input mesh.
    face_indices : list[int]
        Indices of faces to remove.
    verbose : bool

    Returns
    -------
    trimesh.Trimesh
        Mesh with selected faces removed.
    """
    if not face_indices:
        return mesh
    keep = np.ones(len(mesh.faces), dtype=bool)
    keep[np.asarray(face_indices, dtype=np.int64)] = False
    if verbose:
        n_remove = int((~keep).sum())
        print(f"[skeliner.pre] remove_selected_faces: {n_remove} faces")
    return _rebuild_mesh(mesh, keep)


def _find_island_faces(
    faces: np.ndarray,
    active: np.ndarray,
    min_faces: int = 3,
) -> np.ndarray:
    """Return mask of island faces among *active* faces.

    Islands are edge-connected components with fewer than *min_faces*.
    """
    active = active & _non_degenerate(faces)
    active_idx = np.where(active)[0]
    af = faces[active_idx]
    n_active = len(af)
    if n_active == 0:
        return np.zeros(len(faces), dtype=bool)

    # Build edge-to-face adjacency for active faces only
    edge_to_faces: dict[tuple[int, int], list[int]] = defaultdict(list)
    for li, face in enumerate(af):
        for i in range(3):
            a, b = int(face[i]), int(face[(i + 1) % 3])
            edge_to_faces[(min(a, b), max(a, b))].append(li)

    # BFS connected components (local indices)
    labels = np.full(n_active, -1, dtype=np.intp)
    comp_id = 0
    best_id, best_size = 0, 0
    for seed in range(n_active):
        if labels[seed] >= 0:
            continue
        queue = [seed]
        labels[seed] = comp_id
        size = 0
        while queue:
            li = queue.pop()
            size += 1
            face = af[li]
            for i in range(3):
                a, b = int(face[i]), int(face[(i + 1) % 3])
                for nb in edge_to_faces[(min(a, b), max(a, b))]:
                    if labels[nb] < 0:
                        labels[nb] = comp_id
                        queue.append(nb)
        if size > best_size:
            best_id, best_size = comp_id, size
        comp_id += 1

    comp_ids, counts = np.unique(labels, return_counts=True)
    small = set(
        int(c) for c, n in zip(comp_ids, counts) if n < min_faces and c != best_id
    )
    if not small:
        return np.zeros(len(faces), dtype=bool)

    island_local = np.array([int(labels[li]) in small for li in range(n_active)])
    mask = np.zeros(len(faces), dtype=bool)
    mask[active_idx[island_local]] = True
    return mask


def _find_fin_faces(
    faces: np.ndarray,
    active: np.ndarray,
) -> np.ndarray:
    """Return mask of fin faces among *active* faces (iterates until stable)."""
    work = active.copy() & _non_degenerate(faces)
    max_v = int(faces.max()) + 1
    total_mask = np.zeros(len(faces), dtype=bool)

    for _ in range(100):
        active_idx = np.where(work)[0]
        af = faces[active_idx]
        n_active = len(af)
        if n_active == 0:
            break

        v0, v1, v2 = af[:, 0], af[:, 1], af[:, 2]
        e_a = np.stack([np.minimum(v0, v1), np.maximum(v0, v1)], axis=1)
        e_b = np.stack([np.minimum(v1, v2), np.maximum(v1, v2)], axis=1)
        e_c = np.stack([np.minimum(v2, v0), np.maximum(v2, v0)], axis=1)
        all_edges = np.concatenate([e_a, e_b, e_c], axis=0)

        edge_keys = all_edges[:, 0].astype(np.int64) * max_v + all_edges[:, 1]
        _, inverse, counts = np.unique(
            edge_keys, return_inverse=True, return_counts=True
        )
        edge_counts = counts[inverse]

        is_boundary = edge_counts.reshape(3, n_active) == 1
        fin_local = is_boundary.sum(axis=0) >= 2
        n_fins = int(fin_local.sum())
        if n_fins == 0:
            break

        fin_global_idx = active_idx[fin_local]
        total_mask[fin_global_idx] = True
        work[fin_global_idx] = False

    return total_mask


def find_fragments(
    mesh: trimesh.Trimesh,
    *,
    min_faces: int = 3,
    verbose: bool = False,
) -> np.ndarray:
    """Detect all fragment faces (islands and fins) without modifying the mesh.

    Alternates island and fin detection on a boolean mask until stable,
    exactly mirroring the logic of :func:`remove_fragments`.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Input mesh.
    min_faces : int, default 3
        Edge-connected components with fewer faces than this are islands.
    verbose : bool, default False
        Print summary.

    Returns
    -------
    np.ndarray
        Boolean mask ``(nFaces,)`` — fragment faces (islands + fins).
    """
    faces = mesh.faces
    active = np.ones(len(faces), dtype=bool)
    fragment = np.zeros(len(faces), dtype=bool)

    for _ in range(100):
        prev = int(active.sum())
        islands = _find_island_faces(faces, active, min_faces)
        active &= ~islands
        fragment |= islands

        fins = _find_fin_faces(faces, active)
        active &= ~fins
        fragment |= fins

        if int(active.sum()) == prev:
            break

    if verbose:
        n_frag = int(fragment.sum())
        print(
            f"[skeliner.pre] Fragments: {n_frag:,} faces "
            f"({int(islands.sum()):,} islands, {int(fins.sum()):,} fins "
            f"in last pass)"
        )
    return fragment


def _otsu_threshold(values: np.ndarray) -> tuple[float, float]:
    """Otsu's method on continuous 1-D data.

    Returns ``(threshold, separability)`` where *separability* is the
    ratio of between-class variance to total variance (0 = no split,
    1 = perfect bimodal separation).
    """
    x = np.sort(values)
    n = len(x)
    if n < 2:
        return float(x[0]), 0.0

    total_var = float(np.var(x))
    if total_var < 1e-12:
        return float(x[0]), 0.0

    best_thresh = float(x[0])
    best_between = 0.0

    # Scan every midpoint between consecutive unique values
    for i in range(1, n):
        if x[i] == x[i - 1]:
            continue
        left, right = x[:i], x[i:]
        w0, w1 = i / n, (n - i) / n
        between = w0 * w1 * (left.mean() - right.mean()) ** 2
        if between > best_between:
            best_between = between
            best_thresh = float(x[i - 1] + x[i]) / 2.0

    return best_thresh, best_between / total_var


def _assign_soma_verts(
    mesh: trimesh.Trimesh,
    soma: "Soma",
    main_fi: np.ndarray | None = None,
    adj: dict | None = None,
    verbose: bool = False,
) -> "Soma":
    """Assign mesh vertices to an existing soma ellipsoid.

    Performs containment, edge-dilation, pocket absorption, and
    disconnected-component absorption.  Refits the ellipsoid to the
    final vertex set.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        The mesh (may differ from the one used to detect the soma).
    soma : Soma
        Soma with fitted geometry (center/axes/R).
    main_fi : np.ndarray or None
        Face indices of the main component.  Computed if not provided.
    adj : dict or None
        Vertex adjacency on the main component.  Computed if not provided.
    verbose : bool, default False
        Print progress.
    """

    _log = "[skeliner.pre]   soma verts:"

    if main_fi is None:
        labels, main = _face_edge_components(mesh)
        main_fi = np.where(labels == main)[0]

    if adj is None:
        adj = defaultdict(list)
        for fi in main_fi:
            v = mesh.faces[fi]
            for i in range(3):
                a, b = int(v[i]), int(v[(i + 1) % 3])
                adj[a].append(b)
                adj[b].append(a)

    # Containment: main-component vertices inside the ellipsoid
    all_main_verts = np.unique(mesh.faces[main_fi])
    inside = soma.contains(mesh.vertices[all_main_verts])
    soma_set: set[int] = set(all_main_verts[inside].tolist())
    if verbose:
        print(f"{_log} {len(soma_set):,} inside ellipsoid")

    # Dilate along mesh edges (a few average edge lengths past the
    # ellipsoid surface, measured in body-coord units).
    avg_edge = float(mesh.edges_unique_length.mean())
    edge_in_body = avg_edge / float(soma.axes.min())
    dilation_limit = 1.0 + 3.0 * edge_in_body

    n_dilated = 0
    prev_size = 0
    while len(soma_set) != prev_size:
        prev_size = len(soma_set)
        frontier = set()
        for v in soma_set:
            for nv in adj.get(v, []):
                if nv not in soma_set:
                    frontier.add(nv)
        if not frontier:
            break
        frontier_arr = np.fromiter(frontier, dtype=np.intp)
        body = soma._body_coords(mesh.vertices[frontier_arr])
        body_dist = np.sqrt((body**2).sum(axis=1))
        accept = frontier_arr[body_dist < dilation_limit]
        if len(accept) == 0:
            break
        n_dilated += len(accept)
        soma_set.update(accept.tolist())
    if verbose:
        print(f"{_log} +{n_dilated:,} dilated → {len(soma_set):,}")

    # Absorb pockets: connected components of non-soma verts that are
    # topologically trapped (only reachable through the soma).
    # A neuron can have multiple neurite trees stemming from the soma;
    # all of them are trapped by definition.  We separate neurites
    # (large, preserve) from pockets (small surface artifacts, absorb)
    # using Otsu on trapped component sizes.
    all_main_set = set(all_main_verts.tolist())
    n_absorbed_total = 0
    n_skipped_neurite = 0
    for _iteration in range(10):
        outside = all_main_set - soma_set
        visited: set[int] = set()
        trapped_comps: list[list[int]] = []
        for start in outside:
            if start in visited:
                continue
            comp: list[int] = []
            pocket_queue = deque([start])
            while pocket_queue:
                v = pocket_queue.popleft()
                if v in visited:
                    continue
                visited.add(v)
                comp.append(v)
                for nv in adj[v]:
                    if nv in outside and nv not in visited:
                        pocket_queue.append(nv)
            comp_set = set(comp)
            trapped = all(
                nv in soma_set or nv in comp_set for v in comp for nv in adj.get(v, [])
            )
            if trapped and len(comp) < len(soma_set):
                trapped_comps.append(comp)

        if not trapped_comps:
            break

        # Otsu on component sizes to split neurites from pockets
        sizes = np.array([len(c) for c in trapped_comps], dtype=np.float64)
        if len(sizes) >= 2:
            size_thresh, _ = _otsu_threshold(sizes)
        else:
            size_thresh = 0.0  # single component — absorb it

        absorbed = 0
        for comp in trapped_comps:
            if len(comp) > size_thresh:
                n_skipped_neurite += len(comp)
            else:
                soma_set.update(comp)
                absorbed += len(comp)
        n_absorbed_total += absorbed
        if absorbed == 0:
            break
    if verbose and n_absorbed_total:
        print(f"{_log} +{n_absorbed_total:,} absorbed pockets → {len(soma_set):,}")
    if verbose and n_skipped_neurite:
        print(
            f"{_log} {n_skipped_neurite:,} verts in {sum(1 for c in trapped_comps if len(c) > size_thresh)} neurite branches preserved"
        )

    # Absorb disconnected-component vertices inside the ellipsoid.
    all_verts = np.arange(len(mesh.vertices))
    non_main_mask = np.ones(len(mesh.vertices), dtype=bool)
    non_main_mask[all_main_verts] = False
    non_main_verts = all_verts[non_main_mask]
    n_disconn = 0
    if non_main_verts.size:
        inside_non_main = soma.contains(mesh.vertices[non_main_verts])
        n_disconn = int(inside_non_main.sum())
        soma_set.update(non_main_verts[inside_non_main].tolist())
    if verbose and n_disconn:
        print(f"{_log} +{n_disconn:,} disconnected → {len(soma_set):,}")

    # Refit ellipsoid to the final vertex set.
    soma_verts_arr = np.fromiter(sorted(soma_set), dtype=np.intp)
    if len(soma_verts_arr) >= 4:
        try:
            soma = Soma.fit(mesh.vertices[soma_verts_arr], verts=soma_verts_arr)
        except ValueError:
            soma.verts = soma_verts_arr
    else:
        soma.verts = soma_verts_arr

    return soma


def find_soma_via_ring_cutoff(
    mesh: trimesh.Trimesh,
    *,
    organelles: Organelles | None = None,
    verbose: bool = False,
    mesh_stats: MeshStats | None = None,
) -> Soma | None:
    """Soma detection via BFS ring analysis + Otsu cutoff.

    Uses :func:`find_nucleus_center` to locate the soma centre, then
    grows a BFS flood on the main-component surface outward from that
    centre.  The soma boundary is where the largest connected ring
    component peaks and drops (Otsu on post-peak sizes).

    All internal thresholds are derived from the mesh data.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Input mesh (before organelle removal).
    organelles : Organelles or None
        Pre-computed organelles from :func:`find_organelles`.  If
        provided, organelle detection is skipped.
    verbose : bool, default False
        Print summary.
    mesh_stats : MeshStats or None
        From :func:`compute_mesh_stats`.  Reuses ``face_comp`` and
        ``main_ci`` to skip redundant component detection.

    Returns
    -------
    Soma or None
        Fitted ellipsoidal soma, or *None* if no nucleus is found or
        too few indicators exist.
    """
    if mesh_stats is not None and mesh_stats.face_comp is not None:
        labels, main = mesh_stats.face_comp, mesh_stats.main_ci
    else:
        labels, main = _face_edge_components(mesh)

    # ── 1. Find soma centre via nucleus detection ──────────────
    nuc = find_nucleus_center(mesh, verbose=verbose)
    if nuc is None:
        if verbose:
            print("[skeliner.pre] Soma: no nucleus found")
        return None
    center = nuc["center"]
    if verbose:
        print(
            f"[skeliner.pre] Soma: nucleus center="
            f"({center[0]:.0f}, {center[1]:.0f}, {center[2]:.0f})"
        )

    # ── 1b. Organelle mask (for excluding from final soma) ──────
    if organelles is None:
        organelles = find_organelles(mesh, verbose=verbose)

    # ── 2. Build main-component vertex adjacency ─────────────────────
    if verbose:
        print("[skeliner.pre] Soma: building vertex adjacency...")
    main_fi = np.where(labels == main)[0]
    adj: dict[int, list[int]] = defaultdict(list)
    for fi in main_fi:
        v = mesh.faces[fi]
        for i in range(3):
            a, b = int(v[i]), int(v[(i + 1) % 3])
            adj[a].append(b)
            adj[b].append(a)

    # ── 3. BFS from soma surface at nucleus mid-Z ──────────────
    #       Seed ring 0 with soma-cluster vertices at the nucleus
    #       center Z-level (from find_nucleus_center).  These are
    #       outer-surface vertices, tilt-invariant, and form a
    #       symmetric band around the soma cross-section.
    if verbose:
        print("[skeliner.pre] Soma: BFS ring analysis...")
    all_main_verts = np.fromiter(adj.keys(), dtype=np.intp)
    main_set = set(adj.keys())
    seed_vi = nuc.get("soma_seed_vi", np.array([], dtype=np.intp))
    # Keep only seeds that are on the main component
    seed_verts = np.array([v for v in seed_vi if v in main_set], dtype=np.intp)
    if len(seed_verts) == 0:
        # Fallback: single nearest vertex to nucleus center
        seed_verts = all_main_verts[
            np.argmin(np.linalg.norm(mesh.vertices[all_main_verts] - center, axis=1))
        ].reshape(1)

    ring_level: dict[int, int] = {}
    queue: deque[int] = deque()
    ring_verts: dict[int, list[int]] = defaultdict(list)
    for sv in seed_verts:
        vi = int(sv)
        ring_level[vi] = 0
        queue.append(vi)
        ring_verts[0].append(vi)
    if verbose:
        print(
            f"[skeliner.pre] Soma: seed ring 0: "
            f"{len(seed_verts)} verts (soma surface at mid-Z)"
        )

    while queue:
        v = queue.popleft()
        lv = ring_level[v]
        for nv in adj[v]:
            if nv not in ring_level:
                ring_level[nv] = lv + 1
                queue.append(nv)
                ring_verts[lv + 1].append(nv)

    # ── 4. Find soma boundary: largest connected ring component ────
    #       Each BFS ring is split into connected components on the
    #       mesh surface.  On the soma the largest component dominates
    #       and grows; once the frontier enters neurites it fragments
    #       into narrow strips and the largest component shrinks.
    #       The cutoff is where the largest component peaks (Otsu on
    #       the post-peak sizes determines the boundary).
    max_ring = max(ring_verts.keys())
    largest_comp_size = np.zeros(max_ring + 1)

    for lv in range(max_ring + 1):
        verts_in_ring = ring_verts.get(lv, [])
        if not verts_in_ring:
            continue
        # Find connected components within this ring using mesh adjacency
        ring_set = set(verts_in_ring)
        visited_ring: set[int] = set()
        max_comp = 0
        for start in verts_in_ring:
            if start in visited_ring:
                continue
            size = 0
            rq = deque([start])
            while rq:
                v = rq.popleft()
                if v in visited_ring:
                    continue
                visited_ring.add(v)
                size += 1
                for nv in adj[v]:
                    if nv in ring_set and nv not in visited_ring:
                        rq.append(nv)
            if size > max_comp:
                max_comp = size
        largest_comp_size[lv] = max_comp

    # Skip ring 0 (the injected seed set, not a natural BFS ring).
    # Limit peak search to rings within 3× the nucleus Z-span
    # (converted to ring count via average edge length) — peaks
    # beyond that are neurite artifacts, not the soma.
    avg_edge = float(mesh.edges_unique_length.mean())
    z_span = nuc["z_range"][1] - nuc["z_range"][0]
    max_soma_rings = max(int(z_span * 3 / avg_edge), 10)
    search_end = min(max_soma_rings + 1, max_ring + 1)
    peak_ring = 1 + int(np.argmax(largest_comp_size[1:search_end]))
    post_peak = largest_comp_size[peak_ring:]
    if len(post_peak) > 1:
        count_thresh, _ = _otsu_threshold(post_peak)
    else:
        count_thresh = 0.0

    cutoff = max_ring
    for lv in range(peak_ring, max_ring + 1):
        if largest_comp_size[lv] < count_thresh:
            cutoff = lv
            break

    # Under-detection guard: the soma must at least span the nucleus
    # Z-extent.  If it doesn't, extend the cutoff.
    min_cutoff = max(int(z_span / avg_edge), 1)
    if cutoff < min_cutoff:
        if verbose:
            print(
                f"[skeliner.pre] Soma: cutoff {cutoff} < "
                f"min_cutoff {min_cutoff} (z_span/avg_edge), extending"
            )
        cutoff = min_cutoff

    if verbose:
        print(
            f"[skeliner.pre] Soma: peak ring {peak_ring}, "
            f"cutoff ring {cutoff}/{max_ring} "
            f"(peak search limit {search_end})"
        )

    # ── 5. Fit ellipsoid from BFS ring vertices ────────────────────
    #       Include ring 0 (nucleus seed) in the soma vertex set.
    bfs_set: set[int] = set()
    for lv in range(cutoff + 1):
        bfs_set.update(ring_verts[lv])
    bfs_verts_arr = np.fromiter(bfs_set, dtype=np.intp)

    initial_soma = Soma.fit(mesh.vertices[bfs_verts_arr])

    soma = _assign_soma_verts(mesh, initial_soma, main_fi, adj, verbose=verbose)

    # ── 6. Prune neurite tubes from absorbed soma ─────────────────
    #       Pocket absorption may swallow neurite stubs.  Find
    #       connected components of soma vertices outside the initial
    #       (pre-absorption) ellipsoid.  A component is a neurite
    #       tube if it has a narrow attachment to the soma body
    #       (ratio < 0.2) and is elongated (PCA: λ1 >> λ2 ≈ λ3).
    #       After pruning tubes, drop any fragments that become
    #       disconnected from the main soma body when the tube
    #       boundary (soma verts adjacent to pruned verts) is
    #       temporarily removed — these are axon sections inside
    #       the ellipsoid connected only through the pruned tube.
    soma_set = set(soma.verts.tolist())
    soma_arr = np.fromiter(soma_set, dtype=np.intp)

    if len(soma_arr) >= 4:
        body_dist = np.sqrt(
            (initial_soma._body_coords(mesh.vertices[soma_arr]) ** 2).sum(axis=1)
        )
        inside_set: set[int] = set()
        outside_set: set[int] = set()
        for i in range(len(soma_arr)):
            vi = int(soma_arr[i])
            (inside_set if body_dist[i] <= 1.0 else outside_set).add(vi)

        neurite_prune: set[int] = set()
        if outside_set:
            # ── 6a. Prune neurite stubs ──────────────────────────
            #        A stub connects the soma body to a neurite tree.
            #        Identify stubs as outside components that border
            #        a large non-soma connected component (the neurite
            #        tree).  Fins protrude into extracellular space
            #        and do NOT border a large non-soma CC.
            #        "Large" is determined by Otsu on non-soma CC
            #        sizes — fully data-driven, no hard thresholds.

            # Find non-soma CCs on the main component
            all_main_set = set(np.unique(mesh.faces[main_fi]).tolist())
            non_soma = all_main_set - soma_set
            vis_ns: set[int] = set()
            neurite_tree: set[int] = set()
            ns_sizes: list[int] = []
            ns_ccs: list[list[int]] = []
            for ns_start in non_soma:
                if ns_start in vis_ns:
                    continue
                cc: list[int] = []
                nsq = deque([ns_start])
                while nsq:
                    v = nsq.popleft()
                    if v in vis_ns:
                        continue
                    vis_ns.add(v)
                    cc.append(v)
                    for nv in adj.get(v, []):
                        if nv in non_soma and nv not in vis_ns:
                            nsq.append(nv)
                ns_ccs.append(cc)
                ns_sizes.append(len(cc))

            if len(ns_sizes) >= 2:
                ns_thresh, _ = _otsu_threshold(np.array(ns_sizes, dtype=np.float64))
                for cc, sz in zip(ns_ccs, ns_sizes):
                    if sz > ns_thresh:
                        neurite_tree.update(cc)
            elif len(ns_sizes) == 1:
                neurite_tree.update(ns_ccs[0])

            # Find outside components; collect features
            n_tubes = 0
            outside_comps: list[tuple[list[int], bool, float]] = []
            if neurite_tree:
                visited_out: set[int] = set()
                for start in outside_set:
                    if start in visited_out:
                        continue
                    comp: list[int] = []
                    oq = deque([start])
                    while oq:
                        v = oq.popleft()
                        if v in visited_out:
                            continue
                        visited_out.add(v)
                        comp.append(v)
                        for nv in adj.get(v, []):
                            if nv in outside_set and nv not in visited_out:
                                oq.append(nv)

                    if len(comp) < 20:
                        continue

                    # Does this component border the neurite tree?
                    borders = any(
                        nv in neurite_tree for v in comp for nv in adj.get(v, [])
                    )
                    # How far does it extend in body coords?
                    comp_body = initial_soma._body_coords(mesh.vertices[np.array(comp)])
                    max_ext = float(np.sqrt((comp_body**2).sum(axis=1)).max())
                    outside_comps.append((comp, borders, max_ext))

                # Prune components extending far beyond the typical
                # near-surface bump.  Threshold = 2× the 25th
                # percentile of extents — data-relative, adapts to
                # ellipsoid fit quality.
                all_ext = np.array(
                    [e for _, _, e in outside_comps],
                    dtype=np.float64,
                )
                q1 = float(np.percentile(all_ext, 25))
                ext_thresh = 2.0 * q1

                for comp, borders, ext in outside_comps:
                    if ext > ext_thresh:
                        neurite_prune.update(comp)
                        n_tubes += 1

            if neurite_prune:
                soma_set -= neurite_prune
                if verbose:
                    print(
                        f"[skeliner.pre]   soma prune: "
                        f"{n_tubes} neurite stub(s), "
                        f"removed {len(neurite_prune):,} → "
                        f"{len(soma_set):,} verts"
                    )

        # ── 6b. Per-exit stub erosion ─────────────────────────────
        #        For each neurite exit (cluster of boundary verts),
        #        BFS inward into the soma independently.  Ring widths
        #        start narrow (tube); stop when the ring widens into
        #        the soma body (Otsu).  Erode each exit separately so
        #        narrow tubes get deep erosion, wide junctions don't.
        main_vert_set = set(adj.keys())
        non_main_soma = soma_set - main_vert_set
        all_main_set = set(np.unique(mesh.faces[main_fi]).tolist())
        non_soma = all_main_set - soma_set

        # Find soma boundary verts adjacent to non-soma.
        # Only erode when step 6a pruned neurite chains.
        exit_verts: set[int] = set()
        if neurite_prune and non_soma:
            for v in soma_set & main_vert_set:
                for nv in adj.get(v, []):
                    if nv in non_soma:
                        exit_verts.add(v)
                        break

        if exit_verts:
            soma_main = soma_set & main_vert_set

            # Group exit verts into per-exit clusters
            vis_ex: set[int] = set()
            exit_clusters: list[set[int]] = []
            for ex_start in exit_verts:
                if ex_start in vis_ex:
                    continue
                cl: list[int] = []
                exq = deque([ex_start])
                while exq:
                    v = exq.popleft()
                    if v in vis_ex:
                        continue
                    vis_ex.add(v)
                    cl.append(v)
                    for nv in adj.get(v, []):
                        if nv in exit_verts and nv not in vis_ex:
                            exq.append(nv)
                exit_clusters.append(set(cl))

            # Erode each exit independently
            all_stub: set[int] = set()
            n_exits_eroded = 0
            global_visited: set[int] = set()

            for cluster in exit_clusters:
                visited_er = set(cluster)
                stub_verts = set(cluster)
                current_ring = cluster
                ring_sizes: list[int] = [len(current_ring)]

                for _depth in range(200):
                    next_ring: set[int] = set()
                    for v in current_ring:
                        for nv in adj.get(v, []):
                            if (
                                nv in soma_main
                                and nv not in visited_er
                                and nv not in global_visited
                            ):
                                next_ring.add(nv)
                                visited_er.add(nv)
                    if not next_ring:
                        break

                    # Stop at real branch (2+ CCs each ≥ 25%)
                    # or when ring doubles from entry (soma body).
                    if len(next_ring) > 2 * ring_sizes[0]:
                        break
                    vis_r: set[int] = set()
                    cc_sizes_r: list[int] = []
                    for rs in next_ring:
                        if rs in vis_r:
                            continue
                        sz = 0
                        rq = deque([rs])
                        while rq:
                            u = rq.popleft()
                            if u in vis_r:
                                continue
                            vis_r.add(u)
                            sz += 1
                            for nu in adj.get(u, []):
                                if nu in next_ring and nu not in vis_r:
                                    rq.append(nu)
                        cc_sizes_r.append(sz)
                    quarter = len(next_ring) * 0.25
                    if sum(1 for s in cc_sizes_r if s >= quarter) >= 2:
                        break

                    stub_verts.update(next_ring)
                    current_ring = next_ring
                    ring_sizes.append(len(next_ring))

                if next_ring:  # actually eroded
                    all_stub.update(stub_verts)
                    global_visited.update(stub_verts)
                    n_exits_eroded += 1

            if all_stub and len(all_stub) < len(soma_main) * 0.5:
                soma_set -= all_stub
                soma_set |= non_main_soma
                if verbose:
                    print(
                        f"[skeliner.pre]   soma prune: "
                        f"eroded {len(all_stub):,} stub verts "
                        f"from {n_exits_eroded} exit(s) → "
                        f"{len(soma_set):,} verts"
                    )

        # ── 6c. Drop small disconnected soma fragments ─────────────
        #        Axon sections inside the ellipsoid may form small
        #        CCs disconnected from the main soma body.
        main_soma_final = soma_set & main_vert_set
        vis_final: set[int] = set()
        largest_final: list[int] = []
        for cc_start in main_soma_final:
            if cc_start in vis_final:
                continue
            cc: list[int] = []
            ccq = deque([cc_start])
            while ccq:
                v = ccq.popleft()
                if v in vis_final:
                    continue
                vis_final.add(v)
                cc.append(v)
                for nv in adj.get(v, []):
                    if nv in main_soma_final and nv not in vis_final:
                        ccq.append(nv)
            if len(cc) > len(largest_final):
                largest_final = cc

        n_frag = len(main_soma_final) - len(largest_final)
        if n_frag > 0:
            soma_set = set(largest_final) | non_main_soma
            if verbose:
                print(
                    f"[skeliner.pre]   soma prune: "
                    f"dropped {n_frag:,} disconnected → "
                    f"{len(soma_set):,} verts"
                )

        # Refit ellipsoid to final pruned vertex set
        if len(soma_set) != len(soma.verts):
            sv = np.fromiter(sorted(soma_set), dtype=np.intp)
            if len(sv) >= 4:
                try:
                    soma = Soma.fit(mesh.vertices[sv], verts=sv)
                except ValueError:
                    soma.verts = sv
            else:
                soma.verts = sv

    # ── 7. Exclude organelle vertices from final soma ───────────
    #       Only remove vertices that appear exclusively on organelle
    #       faces.  Mouth vertices (shared between organelle and
    #       non-organelle faces) must stay in the soma.
    if organelles.mask.any():
        org_verts = set(np.unique(mesh.faces[organelles.mask]).tolist())
        non_org_verts = set(np.unique(mesh.faces[~organelles.mask]).tolist())
        exclusive_org = org_verts - non_org_verts
        soma_set = set(soma.verts.tolist()) - exclusive_org
        n_removed = len(soma.verts) - len(soma_set)
        if n_removed > 0:
            sv = np.fromiter(sorted(soma_set), dtype=np.intp)
            if len(sv) >= 4:
                try:
                    soma = Soma.fit(mesh.vertices[sv], verts=sv)
                except ValueError:
                    soma.verts = sv
            else:
                soma.verts = sv
            if verbose:
                print(
                    f"[skeliner.pre] Soma: excluded "
                    f"{n_removed:,} organelle verts → "
                    f"{len(soma.verts):,}"
                )

    if verbose:
        print(
            f"[skeliner.pre] Soma: center=["
            f"{soma.center[0]:.0f}, {soma.center[1]:.0f}, "
            f"{soma.center[2]:.0f}], "
            f"axes=[{soma.axes[0]:.0f}, {soma.axes[1]:.0f}, "
            f"{soma.axes[2]:.0f}], "
            f"{len(soma.verts):,} surface verts "
            f"(cutoff ring {cutoff})"
        )

    soma.nucleus = {
        "center": nuc["center"],
        "peak_r": nuc["peak_r"],
        "z_range": nuc["z_range"],
        "slices": nuc["slices"],
    }
    return soma


def _build_edge_to_faces(faces, mask):
    """Edge→face map for faces where *mask* is True."""
    fi_idx = np.where(mask)[0]
    gf = faces[fi_idx]
    result: dict[tuple[int, int], list[int]] = defaultdict(list)
    for col_a, col_b in ((0, 1), (1, 2), (0, 2)):
        va = gf[:, col_a].astype(np.intp)
        vb = gf[:, col_b].astype(np.intp)
        lo = np.minimum(va, vb)
        hi = np.maximum(va, vb)
        for k in range(len(fi_idx)):
            result[(int(lo[k]), int(hi[k]))].append(int(fi_idx[k]))
    return result


def _face_components(faces, edge_to_faces, face_indices):
    """BFS connected components on a subset of faces."""
    fi_set = (
        set(face_indices.tolist())
        if hasattr(face_indices, "tolist")
        else set(face_indices)
    )
    visited: set[int] = set()
    components: list[np.ndarray] = []
    for fi in face_indices:
        fi = int(fi)
        if fi in visited:
            continue
        comp: list[int] = []
        q = deque([fi])
        visited.add(fi)
        while q:
            cur = q.popleft()
            comp.append(cur)
            face = faces[cur]
            for i in range(3):
                a, b = int(face[i]), int(face[(i + 1) % 3])
                e = (min(a, b), max(a, b))
                for nb in edge_to_faces[e]:
                    if nb in fi_set and nb not in visited:
                        visited.add(nb)
                        q.append(nb)
        components.append(np.array(comp, dtype=np.intp))
    components.sort(key=len, reverse=True)
    return components


def _build_org_output(
    organelles: Organelles,
    expanded_mask: np.ndarray,
) -> Organelles:
    """Build an Organelles dataclass from break_up_mesh results."""
    return Organelles(
        pocket=organelles.pocket,
        isolated=organelles.isolated,
        expanded=expanded_mask & ~(organelles.pocket | organelles.isolated),
    )


def break_up_mesh(
    mesh: trimesh.Trimesh,
    soma: Soma,
    organelles: Organelles,
    *,
    verbose: bool = False,
) -> MeshComponents:
    """Break the mesh using soma and organelles, classify the pieces.

    Removes soma + organelle faces, finds connected components of
    the remainder, and classifies each as:

    - **missed organelle** — not reachable from the main mesh body
      without crossing organelle faces (topologically trapped).
    - **missed soma** — reachable, but boundary is mostly soma faces.
    - **neurite** — reachable, large enough to be a real branch.
    - **discarded** — reachable, but too small (below auto threshold).

    The discard threshold is auto-inferred: components are sorted by
    size descending; once the cumulative face count reaches 95% of
    the total, the remaining components are discarded.

    Parameters
    ----------
    mesh : trimesh.Trimesh
    soma : Soma
        From ``find_soma_via_ring_cutoff`` (must have ``.verts``).
    organelles : Organelles
        From :func:`find_organelles`.
    verbose : bool

    Returns
    -------
    MeshComponents
    """
    faces = np.asarray(mesh.faces)
    verts = mesh.vertices
    nF = len(faces)

    # Mesh-mutating steps (e.g. remove_gaps) may have appended faces
    # since find_organelles ran. The new faces are stitch geometry, never
    # organelle membrane — pad with False so the masks align with `nF`.
    n_org = len(organelles.pocket)
    if n_org < nF:
        pad = np.zeros(nF - n_org, dtype=bool)
        organelles = Organelles(
            pocket=np.concatenate([organelles.pocket, pad]),
            isolated=np.concatenate([organelles.isolated, pad]),
            expanded=np.concatenate([organelles.expanded, pad]),
        )

    good = _non_degenerate(faces)
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    nonzero_area = np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1) > 0
    usable = good & nonzero_area

    # --- soma face mask: face is soma if >=2 of 3 verts are soma ---
    soma_set = set(soma.verts.tolist())
    soma_face = np.zeros(nF, dtype=bool)
    for fi in range(nF):
        s = 0
        for v in faces[fi]:
            if int(v) in soma_set:
                s += 1
        if s >= 2:
            soma_face[fi] = True

    # --- Phase 1: reachability without crossing organelles ---
    # For each mesh component, find its largest non-organelle body.
    # Faces not in their component's body are trapped by organelles.
    # This is per-component so disconnected neurite fragments are
    # handled correctly (each has its own reachable body).
    non_org = usable & ~organelles.mask
    ef_non_org = _build_edge_to_faces(faces, non_org)
    non_org_fi = np.where(non_org)[0]
    non_org_comps = _face_components(faces, ef_non_org, non_org_fi)

    # Label every non-org face with its non-org component id.
    # The largest non-org component within each mesh component is
    # that component's "body".
    non_org_label = np.full(nF, -1, dtype=np.intp)
    for ci, comp in enumerate(non_org_comps):
        non_org_label[comp] = ci

    # For each mesh component, find its largest non-org sub-component.
    # Use usable-face edge components as the mesh components.
    ef_usable = _build_edge_to_faces(faces, usable)
    usable_fi = np.where(usable)[0]
    mesh_comps = _face_components(faces, ef_usable, usable_fi)

    body_labels: set[int] = set()
    for mc in mesh_comps:
        # non-org labels present in this mesh component
        labels_in_mc = non_org_label[mc]
        labels_in_mc = labels_in_mc[labels_in_mc >= 0]
        if len(labels_in_mc) == 0:
            continue
        unique, counts = np.unique(labels_in_mc, return_counts=True)
        body_labels.add(int(unique[counts.argmax()]))

    reachable = np.zeros(nF, dtype=bool)
    for ci, comp in enumerate(non_org_comps):
        if ci in body_labels:
            reachable[comp] = True

    # --- Phase 2: break at soma + organelles ---
    remain = usable & ~soma_face & ~organelles.mask
    remain_fi = np.where(remain)[0]

    if len(remain_fi) == 0:
        if verbose:
            print("break_up_mesh: no remaining faces after exclusion")
        org_out = _build_org_output(organelles, organelles.mask)
        return MeshComponents(
            soma=soma,
            organelles=org_out,
            neurites=Neurites([]),
            discarded=Discarded([]),
        )

    ef_all = _build_edge_to_faces(faces, usable)
    components = _face_components(faces, ef_all, remain_fi)

    # --- classify ---
    neurite_candidates: list[np.ndarray] = []
    extra_soma_vi: list[int] = []
    organelles_expanded = organelles.mask.copy()

    for comp in components:
        # Reachability: is any face in its mesh component's body?
        in_main = reachable[comp].any()

        if not in_main:
            # trapped by organelles
            organelles_expanded[comp] = True
            if verbose:
                print(
                    f"break_up_mesh: absorbed {len(comp)} faces "
                    f"as missed organelle (trapped)"
                )
            continue

        # Boundary analysis for soma detection
        comp_set = set(comp.tolist())
        n_soma = 0
        n_total = 0
        for fi in comp:
            face = faces[fi]
            for i in range(3):
                a, b = int(face[i]), int(face[(i + 1) % 3])
                e = (min(a, b), max(a, b))
                for nb in ef_all[e]:
                    if nb not in comp_set:
                        n_total += 1
                        if soma_face[nb]:
                            n_soma += 1

        soma_frac = n_soma / n_total if n_total > 0 else 0.0
        if soma_frac > 0.5:
            comp_vi = np.unique(faces[comp])
            extra_soma_vi.extend(comp_vi.tolist())
            if verbose:
                print(
                    f"break_up_mesh: absorbed {len(comp)} faces "
                    f"({len(comp_vi)} verts) as missed soma"
                )
        else:
            neurite_candidates.append(comp)

    # --- refit soma if we absorbed extra verts ---
    if extra_soma_vi:
        all_soma_vi = np.union1d(soma.verts, np.array(extra_soma_vi, dtype=np.intp))
        prev_nucleus = soma.nucleus
        soma = Soma.fit(verts[all_soma_vi], verts=all_soma_vi)
        soma.nucleus = prev_nucleus

    # --- auto-threshold: keep components covering 95% of total faces ---
    neurite_candidates.sort(key=len, reverse=True)
    total_faces = sum(len(c) for c in neurite_candidates)
    cumsum = 0
    split_idx = len(neurite_candidates)
    for i, comp in enumerate(neurite_candidates):
        cumsum += len(comp)
        if cumsum >= total_faces * 0.95:
            split_idx = i + 1
            break

    neurites = neurite_candidates[:split_idx]
    discarded = neurite_candidates[split_idx:]

    if verbose:
        n_disc_faces = sum(len(c) for c in discarded)
        thresh = len(neurites[-1]) if neurites else 0
        print(
            f"break_up_mesh: {len(neurites)} neurites, "
            f"{len(discarded)} discarded ({n_disc_faces:,} faces, "
            f"threshold ~{thresh} faces), "
            f"soma {len(soma.verts):,} verts, "
            f"organelles {organelles_expanded.sum():,} faces"
        )

    org_out = _build_org_output(organelles, organelles_expanded)
    return MeshComponents(
        soma=soma,
        organelles=org_out,
        neurites=Neurites(neurites),
        discarded=Discarded(discarded),
    )


# Keep old name as alias
break_at_soma = break_up_mesh


def find_soma_via_neurite_exclusion(
    mesh: trimesh.Trimesh,
    *,
    organelles: Organelles | None = None,
    verbose: bool = False,
    mesh_stats: MeshStats | None = None,
) -> Soma | None:
    """Estimate soma by per-tip neurite exclusion.

    .. deprecated::
        Prefer :func:`find_soma_via_ring_cutoff`, which is the promoted
        soma detection method.

    Approach:
      0. Organelle clustering → soma center
      1. BFS on external surface (no organelle faces) from center
      2. Keep all rings up to ~1.5× peak ring, find tip clusters at
         the outer boundary, filter false tips by equator distance
      3. Per-tip neurite stub removal: for each tip, BFS inward with
         perpendicular ring seeding, detect tube→soma transition via
         sustained positive velocity + acceleration in the ring-size
         profile, exclude everything on the tip side.  Cumulative
         exclusion handles shared branches automatically.
      4. Fit ellipsoid to the cleaned set
    """
    warnings.warn(
        "find_soma_via_neurite_exclusion is deprecated, "
        "use find_soma_via_ring_cutoff instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    if mesh_stats is not None and mesh_stats.face_comp is not None:
        labels, main = mesh_stats.face_comp, mesh_stats.main_ci
    else:
        labels, main = _face_edge_components(mesh)

    # ── 0. Organelle clustering → soma center ─────────────────────
    if organelles is None:
        organelles = find_organelles(mesh, verbose=verbose)
    if organelles.mask.sum() == 0:
        if verbose:
            print("[skeliner.pre] Soma: no organelles found")
        return None

    org_fi = np.where(organelles.mask)[0]
    org_labels, _ = _face_edge_components(
        trimesh.Trimesh(
            vertices=mesh.vertices,
            faces=mesh.faces[org_fi],
            process=False,
        )
    )
    centroids = []
    for cid in np.unique(org_labels):
        local_fi = org_fi[org_labels == cid]
        verts = np.unique(mesh.faces[local_fi])
        centroids.append(mesh.vertices[verts].mean(axis=0))
    centroids = np.asarray(centroids)

    if verbose:
        print(
            f"[skeliner.pre] Soma: {organelles.mask.sum():,} organelle faces, "
            f"{len(centroids)} clusters"
        )
    if len(centroids) < 3:
        if verbose:
            print("[skeliner.pre] Soma: too few organelle clusters")
        return None

    tree = KDTree(centroids)
    dd, _ = tree.query(centroids, k=min(2, len(centroids)))
    nn_dists = dd[:, 1] if dd.ndim > 1 else dd
    nn_thresh, _ = _otsu_threshold(nn_dists)
    pairs = tree.query_pairs(r=nn_thresh)
    prox_adj: dict[int, set[int]] = defaultdict(set)
    for i, j in pairs:
        prox_adj[i].add(j)
        prox_adj[j].add(i)

    visited_prox: set[int] = set()
    best_comp: list[int] = []
    for start in range(len(centroids)):
        if start in visited_prox:
            continue
        comp: list[int] = []
        pq = deque([start])
        while pq:
            v = pq.popleft()
            if v in visited_prox:
                continue
            visited_prox.add(v)
            comp.append(v)
            for nv in prox_adj[v]:
                if nv not in visited_prox:
                    pq.append(nv)
        if len(comp) > len(best_comp):
            best_comp = comp

    core = centroids[best_comp]
    if len(core) < 3:
        if verbose:
            print("[skeliner.pre] Soma: no dense fragment cluster found")
        return None

    center = np.median(core, axis=0)
    if verbose:
        print(
            f"[skeliner.pre] Soma: dense cluster {len(core)}/{len(centroids)} fragments"
        )

    # ── 1. BFS on external surface from center ────────────────────
    main_fi = np.where(labels == main)[0]
    non_org_main = main_fi[~organelles.mask[main_fi]]

    adj_bfs: dict[int, list[int]] = defaultdict(list)
    for fi in non_org_main:
        v = mesh.faces[fi]
        for i in range(3):
            a, b = int(v[i]), int(v[(i + 1) % 3])
            adj_bfs[a].append(b)
            adj_bfs[b].append(a)

    bfs_verts = np.fromiter(adj_bfs.keys(), dtype=np.intp)
    seed = int(
        bfs_verts[np.argmin(np.linalg.norm(mesh.vertices[bfs_verts] - center, axis=1))]
    )

    ring_level: dict[int, int] = {seed: 0}
    queue: deque[int] = deque([seed])
    ring_verts: dict[int, list[int]] = defaultdict(list)
    ring_verts[0].append(seed)
    while queue:
        v = queue.popleft()
        lv = ring_level[v]
        for nv in adj_bfs[v]:
            if nv not in ring_level:
                ring_level[nv] = lv + 1
                queue.append(nv)
                ring_verts[lv + 1].append(nv)

    # Find peak ring (largest connected component width)
    max_ring = max(ring_verts.keys())
    largest_comp_size = np.zeros(max_ring + 1)
    for lv in range(max_ring + 1):
        vr = ring_verts.get(lv, [])
        if not vr:
            continue
        ring_set = set(vr)
        vis_r: set[int] = set()
        mx = 0
        for st in vr:
            if st in vis_r:
                continue
            sz = 0
            rq = deque([st])
            while rq:
                u = rq.popleft()
                if u in vis_r:
                    continue
                vis_r.add(u)
                sz += 1
                for nu in adj_bfs[u]:
                    if nu in ring_set and nu not in vis_r:
                        rq.append(nu)
            if sz > mx:
                mx = sz
        largest_comp_size[lv] = mx

    peak_ring = int(np.argmax(largest_comp_size))

    # Reject cells with no localised bulge (no soma).
    # spread_ratio: how concentrated the wide rings are.  For a soma
    # the wide rings cluster at one spot (ratio < 1/3).  For a
    # uniform-width cell the wide rings are everywhere (ratio ~ 1).
    nonzero_mask = largest_comp_size > 0
    nonzero = largest_comp_size[nonzero_mask]
    spread_ratio = 1.0
    if len(nonzero) >= 3:
        width_thresh, _ = _otsu_threshold(nonzero)
        above_idx = np.where(nonzero_mask & (largest_comp_size > width_thresh))[
            0
        ].astype(float)
        all_idx = np.where(nonzero_mask)[0].astype(float)
        if len(above_idx) >= 2 and np.std(all_idx) > 0:
            spread_ratio = float(np.std(above_idx) / np.std(all_idx))

    if spread_ratio > 1.0 / 3:
        if verbose:
            print(
                f"[skeliner.pre] Soma: no localised bulge "
                f"(spread_ratio={spread_ratio:.3f})"
            )
        return None

    # ── 2. Keep rings 0 to 1.5× peak, find tip clusters ─────────
    boundary = peak_ring + peak_ring // 2
    soma_set: set[int] = set()
    for lv in range(min(boundary + 1, max_ring + 1)):
        soma_set.update(ring_verts[lv])

    # Outer edge: soma_set verts adjacent to non-soma_set verts
    outer_edge: set[int] = set()
    for v in soma_set:
        for nv in adj_bfs[v]:
            if nv not in soma_set:
                outer_edge.add(v)
                break

    # Dilated clustering: expand outer_edge by 1 ring for connectivity,
    # then cluster — keeps only original outer_edge verts per cluster.
    dilated = set(outer_edge)
    for v in outer_edge:
        for nv in adj_bfs[v]:
            if nv in soma_set:
                dilated.add(nv)

    visited_tip: set[int] = set()
    tip_clusters: list[list[int]] = []
    for start in outer_edge:
        if start in visited_tip:
            continue
        cluster: list[int] = []
        tq = deque([start])
        while tq:
            v = tq.popleft()
            if v in visited_tip:
                continue
            visited_tip.add(v)
            if v in outer_edge:
                cluster.append(v)
            for nv in adj_bfs[v]:
                if nv in dilated and nv not in visited_tip:
                    tq.append(nv)
        if cluster:
            tip_clusters.append(cluster)
    tip_clusters.sort(key=len, reverse=True)

    # Filter false tips: remove clusters whose median Euclidean distance
    # from soma center is below the equator distance (median distance at
    # peak ring).  These are BFS-inflated verts inside the soma body.
    equator_dists = [
        float(np.linalg.norm(mesh.vertices[v] - center))
        for v in ring_verts.get(peak_ring, [])
    ]
    equator_dist = float(np.median(equator_dists)) if equator_dists else 0.0

    filtered_tips: list[list[int]] = []
    for cluster in tip_clusters:
        med_dist = float(
            np.median([np.linalg.norm(mesh.vertices[v] - center) for v in cluster])
        )
        if med_dist >= equator_dist:
            filtered_tips.append(cluster)

    if verbose:
        print(
            f"[skeliner.pre] Soma: peak ring {peak_ring}, "
            f"boundary ring {boundary}, "
            f"{len(soma_set):,} verts, "
            f"{len(tip_clusters)} tip clusters → "
            f"{len(filtered_tips)} after equator filter"
        )

    # ── 3. Per-tip neurite stub removal ─────────────────────────
    #
    # For each tip, BFS inward through the domain with perpendicular
    # ring seeding.  Detect where the ring-size profile transitions
    # from tube (flat) to soma (sustained climb) using the first
    # derivative (velocity) and second derivative (acceleration).
    # Exclude everything on the tip side of the cut.  Cumulative
    # exclusion handles shared branches automatically.

    _PERP_REF = 10  # rings into tube before computing perpendicular seed
    _SMOOTH_W = 5  # smoothing window for ring-size profile

    def _perp_seed(tip_verts, domain):
        """BFS from tip, find perpendicular cross-section at _PERP_REF."""
        vis: set[int] = set()
        rngs: list[list[int]] = []
        cur = [v for v in tip_verts if v in domain]
        for v in cur:
            vis.add(v)
        if cur:
            rngs.append(cur)
        while cur:
            nx = []
            for v in cur:
                for nv in adj_bfs[v]:
                    if nv in domain and nv not in vis:
                        vis.add(nv)
                        nx.append(nv)
            if not nx:
                break
            rngs.append(nx)
            cur = nx
        ref = _PERP_REF
        if len(rngs) < ref + 3:
            return tip_verts  # tube too short, use original
        ctrs = np.array(
            [mesh.vertices[[v for v in r]].mean(axis=0) for r in rngs[: ref + 5]]
        )
        lo = max(0, ref - 2)
        hi = min(len(ctrs) - 1, ref + 2)
        axis = ctrs[hi] - ctrs[lo]
        nrm = np.linalg.norm(axis)
        if nrm == 0:
            return tip_verts
        axis /= nrm
        ctr = ctrs[ref]
        nbr: set[int] = set()
        for b in range(max(0, ref - 3), min(len(rngs), ref + 4)):
            nbr.update(rngs[b])
        el = float(
            np.median(
                [
                    np.linalg.norm(mesh.vertices[v] - mesh.vertices[nv])
                    for v in list(nbr)[:100]
                    for nv in adj_bfs[v]
                    if nv in nbr
                ]
            )
        )
        half = el * 1.5
        return [
            v for v in nbr if abs(float(np.dot(mesh.vertices[v] - ctr, axis))) < half
        ]

    def _bfs_from(seeds, domain):
        """BFS from seed verts, return list of rings."""
        vis: set[int] = set()
        rngs: list[list[int]] = []
        cur = [v for v in seeds if v in domain]
        for v in cur:
            vis.add(v)
        if cur:
            rngs.append(cur)
        while cur:
            nx = []
            for v in cur:
                for nv in adj_bfs[v]:
                    if nv in domain and nv not in vis:
                        vis.add(nv)
                        nx.append(nv)
            if not nx:
                break
            rngs.append(nx)
            cur = nx
        return rngs

    def _find_cut(ring_sizes):
        """Detect tube→soma transition via d1 > 0 AND d2 > 0 with
        adaptive sustain derived from tube noise level."""
        skip = _PERP_REF
        w = _SMOOTH_W
        if len(ring_sizes) < skip + 15:
            return skip
        smooth = np.convolve(ring_sizes, np.ones(w) / w, mode="valid")
        d1 = np.diff(smooth)
        d2 = np.diff(d1)
        off1 = w // 2 + 1
        off2 = w // 2 + 2
        start = max(skip, off2)
        # Measure tube noise: max consecutive d1>0 AND d2>0 in tube
        tube_end = min(start + 20, len(ring_sizes))
        max_con = 0
        con = 0
        for b in range(start, tube_end):
            i1, i2 = b - off1, b - off2
            if 0 <= i1 < len(d1) and 0 <= i2 < len(d2) and d1[i1] > 0 and d2[i2] > 0:
                con += 1
                if con > max_con:
                    max_con = con
            else:
                con = 0
        sustain = max_con + 1
        # Forward scan
        for b in range(start, len(ring_sizes) - sustain):
            i1, i2 = b - off1, b - off2
            if i1 + sustain > len(d1) or i2 + sustain > len(d2):
                break
            if all(d1[i1 + k] > 0 and d2[i2 + k] > 0 for k in range(sustain)):
                return b
        return skip

    domain = set(soma_set)
    n_excluded = 0
    for ti, tip in enumerate(filtered_tips):
        pseed = _perp_seed(tip, domain)
        rngs = _bfs_from(pseed, domain)
        reached: set[int] = set()
        for r in rngs:
            reached.update(r)
        if seed not in reached:
            # Dead-end: tip is on an already-excluded branch
            domain -= reached
            n_excluded += len(reached)
            continue
        rsizes = [len(r) for r in rngs]
        cut = _find_cut(rsizes)
        to_rm: set[int] = set()
        for b in range(cut):
            to_rm.update(rngs[b])
        domain -= to_rm
        n_excluded += len(to_rm)

    soma_set = domain

    if verbose:
        print(
            f"[skeliner.pre] Soma: excluded {n_excluded:,} neurite verts "
            f"from {len(filtered_tips)} tips, {len(soma_set):,} soma verts remain"
        )

    # ── 4. Fit ellipsoid ──────────────────────────────────────────
    if len(soma_set) < 4:
        if verbose:
            print("[skeliner.pre] Soma: too few verts")
        return None

    soma_arr = np.fromiter(sorted(soma_set), dtype=np.intp)
    soma = Soma.fit(mesh.vertices[soma_arr], verts=soma_arr)

    if verbose:
        print(
            f"[skeliner.pre] Soma: center=["
            f"{soma.center[0]:.0f}, {soma.center[1]:.0f}, "
            f"{soma.center[2]:.0f}], "
            f"axes=[{soma.axes[0]:.0f}, {soma.axes[1]:.0f}, "
            f"{soma.axes[2]:.0f}], "
            f"{len(soma.verts):,} surface verts"
        )

    return soma


def find_soma_via_geodesic(
    mesh: trimesh.Trimesh,
    *,
    organelles: Organelles | None = None,
    verbose: bool = False,
    mesh_stats: MeshStats | None = None,
) -> Soma | None:
    """Experimental soma detection — geodesic proximity + mass boundary.

    .. deprecated::
        Prefer :func:`find_soma_via_ring_cutoff`, which is the promoted
        soma detection method.

    Work-in-progress.  Key differences from
    :func:`find_soma_via_neurite_exclusion`:

      0. Organelle clustering uses **geodesic** (surface) proximity via
         multi-source BFS instead of Euclidean NN.  This prevents linking
         organelles across membranes.
      1. BFS from center on non-organelle surface (same).
      2. Candidate set boundary determined by **neighborhood mass** of
         soma organelle clusters (Otsu on own_size + neighbor_sizes),
         converted to a BFS ring boundary.  No fixed multiplier.
      3. Per-tip neurite exclusion (same).
      4. Fit ellipsoid (same).
    """
    warnings.warn(
        "find_soma_via_geodesic is deprecated, use find_soma_via_ring_cutoff instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    if mesh_stats is not None and mesh_stats.face_comp is not None:
        labels, main = mesh_stats.face_comp, mesh_stats.main_ci
    else:
        labels, main = _face_edge_components(mesh)

    # ── 0. Organelle clustering via geodesic proximity ───────────
    if organelles is None:
        organelles = find_organelles(mesh, verbose=verbose)
    if organelles.mask.sum() == 0:
        if verbose:
            print("[skeliner.pre] Soma-alt: no organelles found")
        return None

    org_fi = np.where(organelles.mask)[0]
    org_labels, _ = _face_edge_components(
        trimesh.Trimesh(
            vertices=mesh.vertices,
            faces=mesh.faces[org_fi],
            process=False,
        )
    )
    clusters: list[dict] = []
    for cid in np.unique(org_labels):
        local_fi = org_fi[org_labels == cid]
        verts = set(int(v) for v in np.unique(mesh.faces[local_fi]))
        clusters.append(
            {
                "verts": verts,
                "centroid": mesh.vertices[list(verts)].mean(axis=0),
                "size": int(len(local_fi)),
            }
        )

    if verbose:
        print(
            f"[skeliner.pre] Soma-alt: {organelles.mask.sum():,} organelle faces, "
            f"{len(clusters)} clusters"
        )
    if len(clusters) < 3:
        if verbose:
            print("[skeliner.pre] Soma-alt: too few organelle clusters")
        return None

    # Build non-organelle surface adjacency
    main_fi = np.where(labels == main)[0]
    non_org_main = main_fi[~organelles.mask[main_fi]]

    adj_bfs: dict[int, list[int]] = defaultdict(list)
    faces_arr = np.asarray(mesh.faces)
    for fi in non_org_main:
        v = faces_arr[fi]
        for i in range(3):
            a, b = int(v[i]), int(v[(i + 1) % 3])
            adj_bfs[a].append(b)
            adj_bfs[b].append(a)

    # Border verts: non-organelle surface verts touching each cluster
    border_verts: dict[int, set[int]] = {}
    for ci, cl in enumerate(clusters):
        borders: set[int] = set()
        for v in cl["verts"]:
            if v in adj_bfs:
                borders.add(v)
            for nv in adj_bfs.get(v, []):
                if nv not in cl["verts"]:
                    borders.add(nv)
        border_verts[ci] = borders

    # Multi-source BFS: label every non-organelle vert by nearest cluster
    vert_label: dict[int, int] = {}
    vert_dist: dict[int, int] = {}
    queue: deque[int] = deque()
    for ci, borders in border_verts.items():
        for v in borders:
            if v not in vert_dist:
                vert_dist[v] = 0
                vert_label[v] = ci
                queue.append(v)
    while queue:
        v = queue.popleft()
        d = vert_dist[v]
        for nv in adj_bfs[v]:
            if nv not in vert_dist:
                vert_dist[nv] = d + 1
                vert_label[nv] = vert_label[v]
                queue.append(nv)

    # Voronoi neighbors: clusters whose territories are adjacent
    neighbors: dict[int, set[int]] = defaultdict(set)
    for v in vert_dist:
        lv = vert_label[v]
        for nv in adj_bfs[v]:
            if nv in vert_label:
                lnv = vert_label[nv]
                if lv != lnv:
                    neighbors[lv].add(lnv)
                    neighbors[lnv].add(lv)

    # Neighborhood mass: own size + sum of neighbor sizes
    sizes = np.array([cl["size"] for cl in clusters])
    nbr_mass = np.zeros(len(clusters))
    for ci in range(len(clusters)):
        nbr_mass[ci] = sizes[ci] + sum(int(sizes[nj]) for nj in neighbors[ci])

    has_nbr = np.array([len(neighbors[ci]) > 0 for ci in range(len(clusters))])
    if has_nbr.sum() < 3:
        if verbose:
            print("[skeliner.pre] Soma-alt: too few connected clusters")
        return None

    mass_thresh, _ = _otsu_threshold(nbr_mass[has_nbr])
    soma_clusters = has_nbr & (nbr_mass > mass_thresh)

    # Soma center = median of soma cluster centroids
    centroids = np.array([cl["centroid"] for cl in clusters])
    core = centroids[soma_clusters]
    if len(core) < 3:
        if verbose:
            print("[skeliner.pre] Soma-alt: no dense cluster found")
        return None

    center = np.median(core, axis=0)
    if verbose:
        print(
            f"[skeliner.pre] Soma-alt: {soma_clusters.sum()}/{len(clusters)} "
            f"soma clusters (mass > {mass_thresh:.0f})"
        )

    # ── 1. BFS from center ───────────────────────────────────────
    bfs_verts = np.fromiter(adj_bfs.keys(), dtype=np.intp)
    seed = int(
        bfs_verts[np.argmin(np.linalg.norm(mesh.vertices[bfs_verts] - center, axis=1))]
    )

    ring_level: dict[int, int] = {seed: 0}
    bfs_q: deque[int] = deque([seed])
    ring_verts: dict[int, list[int]] = defaultdict(list)
    ring_verts[0].append(seed)
    while bfs_q:
        v = bfs_q.popleft()
        lv = ring_level[v]
        for nv in adj_bfs[v]:
            if nv not in ring_level:
                ring_level[nv] = lv + 1
                bfs_q.append(nv)
                ring_verts[lv + 1].append(nv)

    max_ring = max(ring_verts.keys())

    # Peak ring (largest connected component width)
    largest_comp_size = np.zeros(max_ring + 1)
    for lv in range(max_ring + 1):
        vr = ring_verts.get(lv, [])
        if not vr:
            continue
        ring_set = set(vr)
        vis_r: set[int] = set()
        mx = 0
        for st in vr:
            if st in vis_r:
                continue
            sz = 0
            rq = deque([st])
            while rq:
                u = rq.popleft()
                if u in vis_r:
                    continue
                vis_r.add(u)
                sz += 1
                for nu in adj_bfs[u]:
                    if nu in ring_set and nu not in vis_r:
                        rq.append(nu)
            if sz > mx:
                mx = sz
        largest_comp_size[lv] = mx

    peak_ring = int(np.argmax(largest_comp_size))

    # ── 2. Boundary from soma cluster extent ─────────────────────
    # Map soma cluster Voronoi territory to BFS ring levels.
    # Boundary = max ring reached by any soma cluster territory.
    soma_territory = {v for v, ci in vert_label.items() if soma_clusters[ci]}
    soma_ring_vals = [ring_level[v] for v in soma_territory if v in ring_level]
    boundary = max(soma_ring_vals) if soma_ring_vals else peak_ring

    soma_set: set[int] = set()
    for lv in range(min(boundary + 1, max_ring + 1)):
        soma_set.update(ring_verts[lv])

    # Outer edge → tip clusters
    outer_edge: set[int] = set()
    for v in soma_set:
        for nv in adj_bfs[v]:
            if nv not in soma_set:
                outer_edge.add(v)
                break

    dilated = set(outer_edge)
    for v in outer_edge:
        for nv in adj_bfs[v]:
            if nv in soma_set:
                dilated.add(nv)

    visited_tip: set[int] = set()
    tip_clusters: list[list[int]] = []
    for start in outer_edge:
        if start in visited_tip:
            continue
        cluster: list[int] = []
        tq = deque([start])
        while tq:
            v = tq.popleft()
            if v in visited_tip:
                continue
            visited_tip.add(v)
            if v in outer_edge:
                cluster.append(v)
            for nv in adj_bfs[v]:
                if nv in dilated and nv not in visited_tip:
                    tq.append(nv)
        if cluster:
            tip_clusters.append(cluster)
    tip_clusters.sort(key=len, reverse=True)

    # Filter false tips by equator distance
    equator_dists = [
        float(np.linalg.norm(mesh.vertices[v] - center))
        for v in ring_verts.get(peak_ring, [])
    ]
    equator_dist = float(np.median(equator_dists)) if equator_dists else 0.0

    filtered_tips: list[list[int]] = []
    for cluster in tip_clusters:
        med_dist = float(
            np.median([np.linalg.norm(mesh.vertices[v] - center) for v in cluster])
        )
        if med_dist >= equator_dist:
            filtered_tips.append(cluster)

    if verbose:
        print(
            f"[skeliner.pre] Soma-alt: peak ring {peak_ring}, "
            f"boundary ring {boundary}, "
            f"{len(soma_set):,} verts, "
            f"{len(tip_clusters)} tip clusters → "
            f"{len(filtered_tips)} after equator filter"
        )

    # ── 3. Per-tip neurite stub removal ──────────────────────────
    _PERP_REF = 10
    _SMOOTH_W = 5

    def _perp_seed(tip_verts, domain):
        vis: set[int] = set()
        rngs: list[list[int]] = []
        cur = [v for v in tip_verts if v in domain]
        for v in cur:
            vis.add(v)
        if cur:
            rngs.append(cur)
        while cur:
            nx = []
            for v in cur:
                for nv in adj_bfs[v]:
                    if nv in domain and nv not in vis:
                        vis.add(nv)
                        nx.append(nv)
            if not nx:
                break
            rngs.append(nx)
            cur = nx
        ref = _PERP_REF
        if len(rngs) < ref + 3:
            return tip_verts
        ctrs = np.array(
            [mesh.vertices[[v for v in r]].mean(axis=0) for r in rngs[: ref + 5]]
        )
        lo = max(0, ref - 2)
        hi = min(len(ctrs) - 1, ref + 2)
        axis = ctrs[hi] - ctrs[lo]
        nrm = np.linalg.norm(axis)
        if nrm == 0:
            return tip_verts
        axis /= nrm
        ctr = ctrs[ref]
        nbr: set[int] = set()
        for b in range(max(0, ref - 3), min(len(rngs), ref + 4)):
            nbr.update(rngs[b])
        el = float(
            np.median(
                [
                    np.linalg.norm(mesh.vertices[v] - mesh.vertices[nv])
                    for v in list(nbr)[:100]
                    for nv in adj_bfs[v]
                    if nv in nbr
                ]
            )
        )
        half = el * 1.5
        return [
            v for v in nbr if abs(float(np.dot(mesh.vertices[v] - ctr, axis))) < half
        ]

    def _bfs_from(seeds, domain):
        vis: set[int] = set()
        rngs: list[list[int]] = []
        cur = [v for v in seeds if v in domain]
        for v in cur:
            vis.add(v)
        if cur:
            rngs.append(cur)
        while cur:
            nx = []
            for v in cur:
                for nv in adj_bfs[v]:
                    if nv in domain and nv not in vis:
                        vis.add(nv)
                        nx.append(nv)
            if not nx:
                break
            rngs.append(nx)
            cur = nx
        return rngs

    def _find_cut(ring_sizes):
        skip = _PERP_REF
        w = _SMOOTH_W
        if len(ring_sizes) < skip + 15:
            return skip
        smooth = np.convolve(ring_sizes, np.ones(w) / w, mode="valid")
        d1 = np.diff(smooth)
        d2 = np.diff(d1)
        off1 = w // 2 + 1
        off2 = w // 2 + 2
        start = max(skip, off2)
        tube_end = min(start + 20, len(ring_sizes))
        max_con = 0
        con = 0
        for b in range(start, tube_end):
            i1, i2 = b - off1, b - off2
            if 0 <= i1 < len(d1) and 0 <= i2 < len(d2) and d1[i1] > 0 and d2[i2] > 0:
                con += 1
                if con > max_con:
                    max_con = con
            else:
                con = 0
        sustain = max_con + 1
        for b in range(start, len(ring_sizes) - sustain):
            i1, i2 = b - off1, b - off2
            if i1 + sustain > len(d1) or i2 + sustain > len(d2):
                break
            if all(d1[i1 + k] > 0 and d2[i2 + k] > 0 for k in range(sustain)):
                return b
        return skip

    domain = set(soma_set)
    n_excluded = 0
    for ti, tip in enumerate(filtered_tips):
        pseed = _perp_seed(tip, domain)
        rngs = _bfs_from(pseed, domain)
        reached: set[int] = set()
        for r in rngs:
            reached.update(r)
        if seed not in reached:
            domain -= reached
            n_excluded += len(reached)
            continue
        rsizes = [len(r) for r in rngs]
        cut = _find_cut(rsizes)
        to_rm: set[int] = set()
        for b in range(cut):
            to_rm.update(rngs[b])
        domain -= to_rm
        n_excluded += len(to_rm)

    soma_set = domain

    if verbose:
        print(
            f"[skeliner.pre] Soma-alt: excluded {n_excluded:,} neurite verts "
            f"from {len(filtered_tips)} tips, {len(soma_set):,} soma verts remain"
        )

    # ── 4. Fit ellipsoid ─────────────────────────────────────────
    if len(soma_set) < 4:
        if verbose:
            print("[skeliner.pre] Soma-alt: too few verts")
        return None

    soma_arr = np.fromiter(sorted(soma_set), dtype=np.intp)
    soma = Soma.fit(mesh.vertices[soma_arr], verts=soma_arr)

    if verbose:
        print(
            f"[skeliner.pre] Soma-alt: center=["
            f"{soma.center[0]:.0f}, {soma.center[1]:.0f}, "
            f"{soma.center[2]:.0f}], "
            f"axes=[{soma.axes[0]:.0f}, {soma.axes[1]:.0f}, "
            f"{soma.axes[2]:.0f}], "
            f"{len(soma.verts):,} surface verts"
        )

    return soma


def find_disconnected(
    mesh: trimesh.Trimesh,
    *,
    verbose: bool = False,
    soma: Soma | None = None,
    organelles: Organelles | None = None,
    mesh_stats: MeshStats | None = None,
) -> list[list[int]]:
    """Detect disconnected mesh components from segmentation errors.

    Returns disconnected components — broken neurite segments that are
    separate from the main mesh.  Soma-region components, organelle
    components, and components enclosed by the main mesh are excluded.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Input mesh.
    verbose : bool, default False
        Print summary.
    soma : Soma or None
        Pre-computed soma from any ``find_soma_via_*`` function.
    organelles : Organelles or None
        Pre-computed organelles from :func:`find_organelles`.
        Components that are entirely organelle are skipped.
    mesh_stats : MeshStats or None
        From :func:`compute_mesh_stats`.  Reuses ``face_comp`` and
        ``main_ci`` to skip redundant component detection.

    Returns
    -------
    list[list[int]]
        Each element is a list of face indices for one disconnected
        component, sorted largest-first.
    """
    if mesh_stats is not None and mesh_stats.face_comp is not None:
        labels, main = mesh_stats.face_comp, mesh_stats.main_ci
    else:
        labels, main = _face_edge_components(mesh)
    n_faces = len(mesh.faces)
    if verbose:
        n_total_comps = int(labels.max()) + 1 if len(labels) else 0
        print(f"[skeliner.pre] Disconnected: {n_total_comps} total components")

    # Locate soma so we can exclude components inside it
    if soma is None:
        soma = find_soma_via_ring_cutoff(
            mesh,
            organelles=organelles,
            mesh_stats=mesh_stats,
            verbose=verbose,
        )

    # Build KD-tree of main-component face centroids + normals
    # for inside/outside classification
    if verbose:
        print("[skeliner.pre] Disconnected: building KDTree...")
    main_face_idx = np.where(labels == main)[0]
    main_centroids = mesh.triangles_center[main_face_idx]
    main_normals = mesh.face_normals[main_face_idx]
    main_tree = KDTree(main_centroids)
    # Local mesh scale — used to gate the enclosed test below.
    median_edge_len = float(np.median(mesh.edges_unique_length))

    # Collect non-main components (skip degenerate faces with label -2)
    comp_faces: dict[int, list[int]] = {}
    for fi in range(n_faces):
        cid = int(labels[fi])
        if cid == main or cid < 0:
            continue
        comp_faces.setdefault(cid, []).append(fi)

    if verbose:
        n_small = sum(1 for fis in comp_faces.values() if len(fis) < 7)
        print(
            f"[skeliner.pre] Disconnected: {len(comp_faces)} non-main "
            f"components, {n_small} dropped by < 7 filter"
        )

    components = []
    n_soma_excluded = 0
    n_organelle_excluded = 0
    n_enclosed_excluded = 0
    for cid, fis in comp_faces.items():
        # Need at least 7 faces: 3 for each tip + 1 body face to
        # bridge back to two other parts
        if len(fis) < 7:
            continue

        # Skip components that are entirely organelle
        if organelles is not None:
            fis_arr = np.asarray(fis)
            if organelles.mask[fis_arr].all():
                n_organelle_excluded += 1
                continue

        verts = np.unique(mesh.faces[fis])
        coords = mesh.vertices[verts]
        centroid = coords.mean(axis=0)

        # Exclude components whose centroid falls inside the soma ellipsoid
        if soma is not None:
            if soma.contains(centroid.reshape(1, -1))[0]:
                n_soma_excluded += 1
                continue

        # Exclude components enclosed by the main mesh.
        # For each component vertex, find the nearest main mesh face and
        # check which side of that face the vertex is on: if the vector
        # from the main face centroid to the vertex is opposite the main
        # face normal, the vertex is on the interior side.
        #
        # The nearest-face-normal vote is only reliable when the
        # component sits close to main: a true enclosed organelle is
        # right against the inner surface. For components far from main,
        # the nearest face may lie on a curved/concave region whose
        # normal points the wrong way, producing false "inside" votes.
        # Skip the test entirely when the component is clearly external.
        _, nn_idx = main_tree.query(coords)
        vecs = coords - main_centroids[nn_idx]
        dists = np.linalg.norm(vecs, axis=1)
        if np.median(dists) <= 5.0 * median_edge_len:
            dots = np.einsum("ij,ij->i", vecs, main_normals[nn_idx])
            # Component is enclosed if the majority of vertices are on
            # the inward side (dot < 0)
            if (dots < 0).sum() > len(dots) / 2:
                n_enclosed_excluded += 1
                if verbose:
                    print(
                        f"[skeliner.pre]   Excluded enclosed component "
                        f"({len(fis):,} faces, "
                        f"{(dots < 0).sum()}/{len(dots)} verts inside)"
                    )
                continue

        components.append((cid, fis))

    # Sort largest-first
    components.sort(key=lambda x: -len(x[1]))

    if verbose:
        total = sum(len(fis) for _, fis in components)
        excluded = []
        if n_soma_excluded:
            excluded.append(f"{n_soma_excluded} in soma")
        if n_organelle_excluded:
            excluded.append(f"{n_organelle_excluded} organelle")
        if n_enclosed_excluded:
            excluded.append(f"{n_enclosed_excluded} enclosed")
        exc_msg = f", {', '.join(excluded)} excluded" if excluded else ""
        print(
            f"[skeliner.pre] Disconnected: {len(components)} components, "
            f"{total:,} faces{exc_msg}"
        )

    return [fis for _, fis in components]


def find_gaps(
    mesh: trimesh.Trimesh,
    *,
    tip_rings: int = 3,
    verbose: bool = False,
    soma: Soma | None = None,
    disconnected: list[list[int]] | None = None,
    organelles: Organelles | None = None,
    mesh_stats: MeshStats | None = None,
) -> list[tuple[list[int], list[int], float]]:
    """Detect gaps between disconnected components and the main mesh.

    Each gap is a pair of face groups — the tip faces on each side of
    the break — ready for bridging.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Input mesh.
    tip_rings : int, default 3
        Number of face-graph BFS rings from the closest face to
        collect as tip faces on each side of a gap.
    verbose : bool, default False
        Print summary.
    soma : Soma or None
        Pre-computed soma from any ``find_soma_via_*`` function.
    disconnected : list[list[int]] or None
        Pre-computed disconnected components from :func:`find_disconnected`.

    Returns
    -------
    list[tuple[list[int], list[int], float, int, int]]
        Each element is ``(faces_a, faces_b, gap_distance, disc_a, disc_b)``
        where *faces_a* and *faces_b* are face-index lists on each side of
        the gap, *disc_a* and *disc_b* are disconnected component indices
        (``-1`` for main), sorted by gap distance (smallest first).
    """
    if mesh_stats is not None and mesh_stats.face_comp is not None:
        labels, main = mesh_stats.face_comp, mesh_stats.main_ci
    else:
        labels, main = _face_edge_components(mesh)

    # Get disconnected components (reuse filtering logic)
    if disconnected is not None:
        disc = disconnected
    else:
        disc = find_disconnected(
            mesh,
            verbose=verbose,
            soma=soma,
            organelles=organelles,
            mesh_stats=mesh_stats,
        )

    if verbose:
        print(f"[skeliner.pre] Gaps: {len(disc)} disconnected components")

    if not disc:
        return []

    # Build KD-trees: main component + each disconnected component
    comp_data: dict[int, dict] = {}

    # Main component
    main_fi = np.where(labels == main)[0]
    main_verts = np.unique(mesh.faces[main_fi])
    comp_data[main] = {
        "fi": main_fi,
        "verts": main_verts,
        "coords": mesh.vertices[main_verts],
        "tree": KDTree(mesh.vertices[main_verts]),
    }

    # Disconnected components — identify by their label
    for fis in disc:
        cid = int(labels[fis[0]])
        verts = np.unique(mesh.faces[fis])
        comp_data[cid] = {
            "fi": np.asarray(fis),
            "verts": verts,
            "coords": mesh.vertices[verts],
            "tree": KDTree(mesh.vertices[verts]),
        }

    if verbose:
        print(f"[skeliner.pre] Gaps: built KDTrees for {len(disc) + 1} components")

    # For each disconnected component, find its nearest neighbour.
    # Deduplicate: if A→B and B→A both exist, keep only one.
    gaps = []
    seen_pairs: set[tuple[int, int]] = set()
    disc_cids = [int(labels[fis[0]]) for fis in disc]
    all_cids = [main] + disc_cids

    # Build face adjacency for BFS-based tip selection
    if verbose:
        print("[skeliner.pre] Gaps: building face adjacency...")

    non_degen = _non_degenerate(mesh.faces)
    edge_to_faces_gap: dict[tuple[int, int], list[int]] = defaultdict(list)
    for fi in range(len(mesh.faces)):
        if not non_degen[fi]:
            continue
        face = mesh.faces[fi]
        for i in range(3):
            e = (
                min(int(face[i]), int(face[(i + 1) % 3])),
                max(int(face[i]), int(face[(i + 1) % 3])),
            )
            edge_to_faces_gap[e].append(fi)

    face_adj: dict[int, list[int]] = defaultdict(list)
    for e, fis_e in edge_to_faces_gap.items():
        for i in range(len(fis_e)):
            for j in range(i + 1, len(fis_e)):
                face_adj[fis_e[i]].append(fis_e[j])
                face_adj[fis_e[j]].append(fis_e[i])

    def _tip_faces_bfs(comp_fi, tip_vert_idx, n_rings=tip_rings):
        """BFS on face graph from the face nearest to tip vertex."""
        comp_set = set(comp_fi.tolist())
        vcoord = mesh.vertices[tip_vert_idx]
        centroids = mesh.vertices[mesh.faces[comp_fi]].mean(axis=1)
        seed_local = int(np.argmin(np.linalg.norm(centroids - vcoord, axis=1)))
        seed = int(comp_fi[seed_local])
        visited: set[int] = {seed}
        frontier: set[int] = {seed}
        for _ in range(n_rings):
            next_frontier: set[int] = set()
            for fi in frontier:
                for nf in face_adj[fi]:
                    if nf not in visited and nf in comp_set:
                        visited.add(nf)
                        next_frontier.add(nf)
            if not next_frontier:
                break
            frontier = next_frontier
        return sorted(visited)

    def _add_gap(cid_a, cid_b, idx_a, idx_b, dist):
        pair_key = (min(cid_a, cid_b), max(cid_a, cid_b))
        if pair_key in seen_pairs:
            return
        seen_pairs.add(pair_key)
        if dist < 1.0:  # vertex-connected = fusion, not gap
            return
        va = int(comp_data[cid_a]["verts"][idx_a])
        vb = int(comp_data[cid_b]["verts"][idx_b])
        fa = _tip_faces_bfs(comp_data[cid_a]["fi"], va)
        fb = _tip_faces_bfs(comp_data[cid_b]["fi"], vb)
        if fa and fb:
            # Map component IDs to disc indices (-1 for main)
            disc_idx_a = disc_cids.index(cid_a) if cid_a in disc_cids else -1
            disc_idx_b = disc_cids.index(cid_b) if cid_b in disc_cids else -1
            gaps.append((fa, fb, dist, disc_idx_a, disc_idx_b))

    # Precompute pairwise distances between all components (main + disc).
    # Then build an MST rooted at main so every disc component has a
    # bridge path back to main — no missing chain links, no duplicates.
    all_cids = [main] + disc_cids
    # edge_info keyed by sorted pair
    edge_info: dict[tuple[int, int], tuple[float, int, int, int, int]] = {}

    for i, cid_a in enumerate(all_cids):
        for cid_b in all_cids[i + 1 :]:
            da = comp_data[cid_a]
            db = comp_data[cid_b]
            dists, idxs = db["tree"].query(da["coords"])
            min_i = int(np.argmin(dists))
            raw_dist = float(dists[min_i])
            # Vertex-connected pairs are fusions (will be broken later),
            # not real gaps — treat as infinite cost so the MST avoids them.
            cost = float("inf") if raw_dist < 1.0 else raw_dist
            key = (min(cid_a, cid_b), max(cid_a, cid_b))
            edge_info[key] = (
                cost,
                cid_a,
                min_i,  # idx into cid_a's coords
                cid_b,
                int(idxs[min_i]),  # idx into cid_b's coords
            )

    if verbose:
        print(f"[skeliner.pre] Gaps: computed {len(edge_info)} pairwise distances")

    # Prim's MST from main
    connected: set[int] = {main}
    remaining: set[int] = set(disc_cids)

    while remaining:
        best_edge = None
        best_cost = float("inf")
        for cid_r in remaining:
            for cid_c in connected:
                key = (min(cid_r, cid_c), max(cid_r, cid_c))
                cost = edge_info[key][0]
                if cost < best_cost:
                    best_cost = cost
                    best_edge = (cid_r, cid_c, key)
        if best_edge is None:
            break
        cid_r, cid_c, key = best_edge
        connected.add(cid_r)
        remaining.discard(cid_r)

        dist, ca, idx_a, cb, idx_b = edge_info[key]
        _add_gap(ca, cb, idx_a, idx_b, dist)

    if verbose:
        print(
            f"[skeliner.pre] Gaps: MST connected {len(connected)} components, {len(gaps)} gaps found"
        )

    # Sort by gap distance
    gaps.sort(key=lambda x: x[2])

    if verbose:
        print(f"[skeliner.pre] Gaps: {len(gaps)} gaps found")
        for i, (fa, fb, dist, da, db) in enumerate(gaps):
            label_a = "main" if da == -1 else f"disc {da}"
            label_b = "main" if db == -1 else f"disc {db}"
            print(
                f"  gap {i}: {label_a} ({len(fa)}f) ↔ {label_b} ({len(fb)}f), "
                f"dist={dist:.0f}"
            )

    return gaps


def remove_gaps(
    mesh: trimesh.Trimesh,
    *,
    min_faces: int = 100,
    verbose: bool = False,
    soma: Soma | None = None,
    gaps: list | None = None,
    organelles: Organelles | None = None,
    mesh_stats: MeshStats | None = None,
) -> trimesh.Trimesh:
    """Bridge all detected gaps in a single mesh rebuild.

    All gap tip faces are removed at once and boundary loops from each
    gap are paired and zipper-stitched, then the mesh is rebuilt once.
    Vertex indices are preserved.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Input mesh.
    min_faces : int, default 100
        Passed to :func:`find_gaps`.
    verbose : bool, default False
        Print progress.
    soma : Soma or None
        Pre-computed soma.
    gaps : list or None
        Pre-computed gaps from :func:`find_gaps`.

    Returns
    -------
    trimesh.Trimesh
        Mesh with gaps bridged.
    """
    if gaps is not None:
        gaps = gaps
    else:
        gaps = find_gaps(
            mesh,
            verbose=verbose,
            soma=soma,
            organelles=organelles,
            mesh_stats=mesh_stats,
        )

    if not gaps:
        if verbose:
            print("[skeliner.pre] No gaps to bridge")
        return mesh

    # Build edge / face adjacency once
    edge_to_faces = _edge_to_faces(mesh)
    face_adj = _face_adjacency(mesh, edge_to_faces)

    if verbose:
        print(f"[skeliner.pre] Bridging {len(gaps)} gaps")

    # For each gap, expand each side's tip until both rims are real
    # loops. The initial tip from find_gaps can produce a degenerate
    # "rim" (e.g. 3 verts when the tip is a tongue attached at a single
    # triangle); peeling more rings exposes a clean cross-section that
    # the stitcher can bridge manifoldly.
    faces_to_remove: set[int] = set()
    loop_pairs: list[tuple[list[int], list[int]]] = []
    n_skipped = 0
    for gap_i, (faces_a, faces_b, dist, *_comp_ids) in enumerate(gaps):
        sel_a, loop_a = _expand_tip_to_good_rim(mesh, faces_a, edge_to_faces, face_adj)
        sel_b, loop_b = _expand_tip_to_good_rim(mesh, faces_b, edge_to_faces, face_adj)

        if loop_a is None or loop_b is None:
            if verbose:
                print(
                    f"[skeliner.pre]   Gap {gap_i} (dist={dist:.0f}): "
                    f"could not trace rim loops on one side, skipping"
                )
            n_skipped += 1
            continue

        # Final safety net: if expansion still didn't reach a clean
        # pair (degenerate or wildly mismatched), skip rather than
        # introduce a fusion.
        reason = _validate_loop_pair(loop_a, loop_b)
        if reason is not None:
            if verbose:
                print(
                    f"[skeliner.pre]   Gap {gap_i} (dist={dist:.0f}): "
                    f"skipped after expansion ({reason})"
                )
            n_skipped += 1
            continue

        loop_pairs.append((loop_a, loop_b))
        faces_to_remove |= sel_a | sel_b

        if verbose:
            extra = ""
            if len(sel_a) != len(faces_a) or len(sel_b) != len(faces_b):
                extra = (
                    f" [expanded a:{len(faces_a)}->{len(sel_a)}f, "
                    f"b:{len(faces_b)}->{len(sel_b)}f]"
                )
            print(
                f"[skeliner.pre]   Gap {gap_i} (dist={dist:.0f}): "
                f"{len(loop_a)}v + {len(loop_b)}v{extra}"
            )

    if not loop_pairs:
        if verbose:
            print("[skeliner.pre] No valid loop pairs; nothing to bridge")
        return mesh

    if verbose:
        print(f"[skeliner.pre] Removing {len(faces_to_remove)} tip faces")

    result = _stitch_and_rebuild(mesh, faces_to_remove, loop_pairs, verbose=verbose)

    # Invalidate topology (components merged); pad outward_dots for appended faces
    if mesh_stats is not None:
        n_added = len(result.faces) - len(mesh.faces)
        if n_added > 0 and mesh_stats.outward_dots is not None:
            mesh_stats.outward_dots = np.concatenate(
                [
                    mesh_stats.outward_dots,
                    np.ones(n_added, dtype=mesh_stats.outward_dots.dtype),
                ]
            )
        mesh_stats.invalidate_topology()

    return result


def remove_islands(
    mesh: trimesh.Trimesh,
    *,
    min_faces: int = 3,
    verbose: bool = False,
) -> trimesh.Trimesh:
    """Remove small edge-disconnected components (islands).

    An island is a cluster of faces that shares no edge with the main
    surface — it may touch via a vertex, but has no edge connectivity.
    This performs a single pass: all islands present at call time are
    removed.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Input mesh.
    min_faces : int, default 3
        Edge-connected components with fewer faces than this are removed.
    verbose : bool, default False
        Print summary.

    Returns
    -------
    trimesh.Trimesh
        Mesh with islands removed.
    """
    active = np.ones(len(mesh.faces), dtype=bool)
    islands = _find_island_faces(mesh.faces, active, min_faces)
    n_removed = int(islands.sum())

    if n_removed == 0:
        if verbose:
            print("[skeliner.pre] No islands to remove")
        return mesh

    clean = _rebuild_mesh(mesh, ~islands)

    if verbose:
        print(f"[skeliner.pre] Removed {n_removed} island faces")
    return clean


def remove_fins(
    mesh: trimesh.Trimesh,
    *,
    verbose: bool = False,
) -> trimesh.Trimesh:
    """Remove fin faces — faces with 2+ boundary edges.

    A fin is a face hanging off the surface by a single edge (2 of its
    3 edges are boundary edges).  This iterates until no more fins
    remain, since removing a fin can expose new fins.
    """
    active = np.ones(len(mesh.faces), dtype=bool)
    fins = _find_fin_faces(mesh.faces, active)
    n_removed = int(fins.sum())

    if n_removed == 0:
        if verbose:
            print("[skeliner.pre] No fins to remove")
        return mesh

    result = _rebuild_mesh(mesh, ~fins)

    if verbose:
        print(f"[skeliner.pre] Removed {n_removed} fin faces")

    return result


def remove_fragments(
    mesh: trimesh.Trimesh,
    *,
    min_faces: int = 3,
    verbose: bool = False,
    fragments: np.ndarray | None = None,
) -> trimesh.Trimesh:
    """Remove all fragments (islands and fins) by alternating until convergence.

    Islands are small edge-disconnected components; fins are faces
    hanging by a single edge.  Removing one type can create the other,
    so this alternates island and fin detection on a mask until stable,
    then rebuilds the mesh once.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Input mesh.
    min_faces : int, default 3
        Passed to island detection.
    verbose : bool, default False
        Print progress.

    Returns
    -------
    trimesh.Trimesh
        Mesh with all fragments removed.
    """
    if fragments is not None:
        fragment = fragments
        if verbose:
            print(
                f"[skeliner.pre] Using provided fragment mask ({int(fragment.sum()):,} faces)"
            )
    else:
        fragment = find_fragments(mesh, min_faces=min_faces, verbose=verbose)

    n_removed = int(fragment.sum())
    if n_removed == 0:
        return mesh

    result = _rebuild_mesh(mesh, ~fragment)

    if verbose:
        print(f"[skeliner.pre] Removed {n_removed:,} fragment faces")

    return result


def ensure_watertight(
    mesh: trimesh.Trimesh,
    *,
    verbose: bool = False,
) -> trimesh.Trimesh:
    """Remove fragments, fill holes, and verify watertightness.

    Chains ``remove_fragments`` → ``fill_holes``, iterating until
    stable.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Input mesh (typically after ``remove_organelles``).
    verbose : bool, default False
        Print progress.

    Returns
    -------
    trimesh.Trimesh
        Watertight mesh (or best-effort if non-manifold edges remain).
    """
    result = remove_fragments(mesh, verbose=verbose)

    # Fill holes iteratively — filling can create new fragments
    for iteration in range(10):
        prev_faces = len(result.faces)
        result = fill_holes(result, verbose=verbose)
        result = remove_fragments(result, verbose=verbose)
        if len(result.faces) == prev_faces:
            break
        if verbose:
            print(
                f"[skeliner.pre] Iteration {iteration + 1}: {len(result.faces):,} faces"
            )

    if verbose:
        wt = result.is_watertight
        print(f"[skeliner.pre] Watertight: {wt}")

    return result


def _outward_dot(
    mesh: trimesh.Trimesh,
    radius: float,
    vert_comp: np.ndarray | None = None,
) -> np.ndarray:
    """Per-face outward score: dot(face_normal, direction_from_local_COM).

    For each face, finds all vertices within *radius* of its centroid
    **on the same component**, computes their center of mass, and dots
    the face normal against the direction from that COM to the face
    centroid.

    Surface faces point outward (positive), internal faces point
    inward (negative).

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Input mesh.
    radius : float
        Ball-query radius for local centre-of-mass.
    vert_comp : np.ndarray or None
        Per-vertex component ID ``(nVertices,)``.  When provided, only
        vertices from the same component are used for the local COM.
        This gives correct outward-dot values for disconnected
        components.

    Returns
    -------
    np.ndarray
        (nFaces,) float64 array of outward dot products.
    """
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    face_centers = np.asarray(mesh.triangles_center, dtype=np.float64)
    face_normals = np.asarray(mesh.face_normals, dtype=np.float64)
    n_faces = len(mesh.faces)
    outward_dots = np.zeros(n_faces, dtype=np.float64)

    def _compute_batch(vtree, v, face_idx):
        """Compute outward-dot via sparse_distance_matrix (fully vectorized)."""
        from scipy.sparse import csr_matrix

        fc = face_centers[face_idx]
        fn = face_normals[face_idx]

        # Build a KDTree of face centers for this batch
        fc_tree = KDTree(fc)

        # All (face, vert) pairs within radius — returned as sparse matrix
        sdm = fc_tree.sparse_distance_matrix(vtree, radius, output_type="coo_matrix")

        if sdm.nnz == 0:
            return

        # Count neighbors per face
        row = sdm.row
        counts = np.bincount(row, minlength=len(face_idx))

        # Build weight matrix: each entry = 1/count (for mean)
        valid_entries = counts[row] >= 4
        weights = np.zeros(len(row), dtype=np.float64)
        weights[valid_entries] = 1.0 / counts[row[valid_entries]]

        W = csr_matrix(
            (weights, (row, sdm.col)),
            shape=(len(face_idx), len(v)),
            dtype=np.float64,
        )

        # COM = W @ v  (weighted mean of neighbor vertices)
        local_com = W @ v  # (n_faces, 3)

        # Direction
        direction = fc - local_com
        norms = np.linalg.norm(direction, axis=1)
        valid = (counts >= 4) & (norms > 1e-10)

        # Normalize and dot
        direction[valid] /= norms[valid, None]
        dots = np.einsum("ij,ij->i", fn, direction)

        outward_dots[face_idx[valid]] = dots[valid]

    if vert_comp is None:
        vtree = KDTree(verts)
        _compute_batch(vtree, verts, np.arange(n_faces))
    else:
        face_comp = vert_comp[mesh.faces[:, 0]]
        for ci in np.unique(face_comp):
            ci = int(ci)
            comp_vert_mask = vert_comp == ci
            comp_verts_idx = np.where(comp_vert_mask)[0]
            if len(comp_verts_idx) < 4:
                continue
            comp_verts = verts[comp_verts_idx]
            vtree = KDTree(comp_verts)
            comp_face_idx = np.where(face_comp == ci)[0]
            _compute_batch(vtree, comp_verts, comp_face_idx)

    return outward_dots


def _filter_small_clusters(
    mesh: trimesh.Trimesh,
    face_mask: np.ndarray,
    min_cluster_size: int,
) -> np.ndarray:
    """Remove connected components of flagged faces below a size threshold.

    Parameters
    ----------
    mesh
        The mesh.
    face_mask
        Boolean mask (nFaces,) — True for faces to consider.
    min_cluster_size
        Clusters with fewer faces than this are dropped.

    Returns
    -------
    np.ndarray
        Filtered boolean mask.
    """
    flagged = set(int(fi) for fi in np.where(face_mask)[0])
    if not flagged:
        return face_mask

    # Build face adjacency restricted to flagged faces
    faces_arr = np.asarray(mesh.faces)  # snapshot — avoids trimesh caching
    edge_to_faces: dict[tuple[int, int], list[int]] = defaultdict(list)
    for fi in flagged:
        f = faces_arr[fi]
        for i in range(3):
            e = (
                min(int(f[i]), int(f[(i + 1) % 3])),
                max(int(f[i]), int(f[(i + 1) % 3])),
            )
            edge_to_faces[e].append(fi)

    int_list = sorted(flagged)
    int_remap = {fi: i for i, fi in enumerate(int_list)}
    edges = set()
    for fi in int_list:
        f = faces_arr[fi]
        for i in range(3):
            e = (
                min(int(f[i]), int(f[(i + 1) % 3])),
                max(int(f[i]), int(f[(i + 1) % 3])),
            )
            for nfi in edge_to_faces[e]:
                if nfi != fi:
                    a, b = int_remap[fi], int_remap[nfi]
                    edges.add((min(a, b), max(a, b)))

    g = ig.Graph(n=len(int_list), edges=list(edges), directed=False)
    clusters = g.connected_components()

    keep = set()
    for cl in clusters:
        if len(cl) >= min_cluster_size:
            keep.update(int_list[i] for i in cl)

    filtered = np.zeros(len(mesh.faces), dtype=bool)
    for fi in keep:
        filtered[fi] = True
    return filtered


def compute_mesh_stats(
    mesh: trimesh.Trimesh,
    radius: float | None = None,
    radius_multiplier: float = 5.0,
    verbose: bool = False,
):
    """Shared precomputation for organelle detection.

    Returns a :class:`~skeliner.dataclass.MeshStats`.
    """
    if radius is None:
        median_edge = float(np.median(mesh.edges_unique_length))
        radius = radius_multiplier * median_edge
        if verbose:
            print(
                f"[skeliner.pre] Auto radius: {radius:.1f} "
                f"({radius_multiplier}x median edge {median_edge:.1f})"
            )

    # Compute connected components using edge adjacency so that
    # faces sharing only a vertex (not an edge) are correctly
    # separated.  This matters after _rebuild_mesh where degenerate
    # faces ([0,0,0]) share vertex 0 and removed patches leave
    # shared boundary vertices.
    face_comp, main_ci = _face_edge_components(mesh)
    main_face_mask = face_comp == main_ci

    # Vertex component labels (needed by _outward_dot for per-component COM)
    vert_comp = np.full(len(mesh.vertices), -1, dtype=np.intp)
    for fi in range(len(mesh.faces)):
        ci = int(face_comp[fi])
        if ci < 0:
            continue
        for v in mesh.faces[fi]:
            vert_comp[int(v)] = ci

    if verbose:
        n_comps = int(face_comp.max()) + 1 if len(face_comp) else 0
        from collections import Counter

        comp_sizes = Counter(
            int(face_comp[fi]) for fi in range(len(face_comp)) if face_comp[fi] >= 0
        )
        n_structural = sum(1 for n in comp_sizes.values() if n >= 100)
        print(
            f"[skeliner.pre] Components: {n_comps} total, "
            f"{n_structural} structural (>= 100 faces)"
        )

    outward_dots = _outward_dot(mesh, radius, vert_comp=vert_comp)

    if verbose:
        raw_count = int((outward_dots < 0).sum())
        print(
            f"[skeliner.pre] Raw internal faces: {raw_count:,} "
            f"({100 * raw_count / len(mesh.faces):.1f}%)"
        )

    return MeshStats(
        outward_dots=outward_dots,
        face_comp=face_comp,
        main_ci=int(main_ci),
    )


def _rim_enclosed_area(
    boundary_edges: list[tuple[int, int]],
    vertices: np.ndarray,
) -> float:
    """Compute total enclosed planar area of closed loops in boundary edges.

    For each closed loop: project vertices onto a best-fit plane, then
    compute polygon area via the shoelace formula.
    """

    if not boundary_edges:
        return 0.0

    vert_to_be: dict[int, list[int]] = defaultdict(list)
    for i, e in enumerate(boundary_edges):
        vert_to_be[e[0]].append(i)
        vert_to_be[e[1]].append(i)

    visited: set[int] = set()
    total_area = 0.0

    for start in range(len(boundary_edges)):
        if start in visited:
            continue
        comp: list[int] = []
        queue = deque([start])
        while queue:
            ei = queue.popleft()
            if ei in visited:
                continue
            visited.add(ei)
            comp.append(ei)
            for v in boundary_edges[ei]:
                for nei in vert_to_be[v]:
                    if nei not in visited:
                        queue.append(nei)

        # Check closed ring
        vcount: dict[int, int] = defaultdict(int)
        for ei in comp:
            for v in boundary_edges[ei]:
                vcount[v] += 1
        if not all(c == 2 for c in vcount.values()):
            continue

        # Order vertices into a loop
        edge_list = [boundary_edges[ei] for ei in comp]
        adj_v: dict[int, list[int]] = defaultdict(list)
        for a, b in edge_list:
            adj_v[a].append(b)
            adj_v[b].append(a)

        ordered = [edge_list[0][0]]
        while len(ordered) < len(edge_list):
            curr = ordered[-1]
            prev = ordered[-2] if len(ordered) > 1 else -1
            for nb in adj_v[curr]:
                if nb != prev:
                    ordered.append(nb)
                    break
            else:
                break

        if len(ordered) < 3:
            continue

        # Project onto best-fit plane and compute area
        pts = vertices[ordered]
        centroid = pts.mean(axis=0)
        centered = pts - centroid
        _, _, vh = np.linalg.svd(centered)
        u_ax, v_ax = vh[0], vh[1]
        pts_2d = np.column_stack([centered @ u_ax, centered @ v_ax])
        x, y = pts_2d[:, 0], pts_2d[:, 1]
        area = 0.5 * abs(
            np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]) + x[-1] * y[0] - x[0] * y[-1]
        )
        total_area += area

    return total_area


def _count_edge_loops(edges: list[tuple[int, int]]) -> int:
    """Count connected components of an edge list."""

    if not edges:
        return 0
    vert_to_idx: dict[int, list[int]] = defaultdict(list)
    for i, e in enumerate(edges):
        vert_to_idx[e[0]].append(i)
        vert_to_idx[e[1]].append(i)

    visited: set[int] = set()
    n_loops = 0
    for start in range(len(edges)):
        if start in visited:
            continue
        queue = deque([start])
        while queue:
            ei = queue.popleft()
            if ei in visited:
                continue
            visited.add(ei)
            for v in edges[ei]:
                for nei in vert_to_idx[v]:
                    if nei not in visited:
                        queue.append(nei)
        n_loops += 1
    return n_loops


def find_pocket_mouths(
    mesh: trimesh.Trimesh,
    *,
    radius: float | None = None,
    radius_multiplier: float = 5.0,
    min_pocket_size: int = 5,
    min_fold_ratio: float = 3.0,
    verbose: bool = False,
    mesh_stats: MeshStats | None = None,
    _adj: dict[int, set[int]] | None = None,
    _edge_to_face: dict[tuple[int, int], list[int]] | None = None,
) -> list[list[tuple[int, int]]]:
    """Find mouth edges — boundaries of negative-dot face clusters.

    A valid pocket must satisfy:

    1. Multiple boundary loops (not a flat patch with a single outline).
    2. Fold ratio > *min_fold_ratio*: the pocket surface area must be
       much larger than the mouth's enclosed planar area, indicating the
       surface folds inward through a small opening.

    Returns
    -------
    list[list[tuple[int, int]]]
        One list of edges per pocket mouth.
    """

    if mesh_stats is not None and mesh_stats.outward_dots is not None:
        outward_dots = mesh_stats.outward_dots
        if mesh_stats.face_comp is not None:
            main_face_mask = mesh_stats.main_face_mask
        else:
            _fc, _mc = _face_edge_components(mesh)
            main_face_mask = _fc == _mc
    else:
        mesh_stats = compute_mesh_stats(
            mesh,
            radius,
            radius_multiplier,
            verbose,
        )
        outward_dots = mesh_stats.outward_dots
        main_face_mask = mesh_stats.main_face_mask
    edge_to_face = _edge_to_face if _edge_to_face is not None else _edge_to_faces(mesh)
    adj = _adj if _adj is not None else _face_adjacency(mesh, edge_to_face)

    # Connected components of negative-dot faces on main component
    neg_idx = set(np.where((outward_dots < 0) & main_face_mask)[0].tolist())
    visited: set[int] = set()
    clusters: list[list[int]] = []
    for fi in neg_idx:
        if fi in visited:
            continue
        cluster: list[int] = []
        queue = deque([fi])
        while queue:
            curr = queue.popleft()
            if curr in visited:
                continue
            visited.add(curr)
            cluster.append(curr)
            for nfi in adj.get(curr, set()):
                if nfi in neg_idx and nfi not in visited:
                    queue.append(nfi)
        clusters.append(cluster)

    # For each cluster, collect boundary edges and count boundary loops.
    # A real pocket has multiple boundary loops (multiple openings).
    # A flat concave patch has exactly one boundary loop — not a pocket.
    faces_arr = np.asarray(mesh.faces)  # snapshot — avoids trimesh caching
    area_arr = np.asarray(mesh.area_faces)
    verts_arr = np.asarray(mesh.vertices)
    mouths: list[list[tuple[int, int]]] = []
    for cluster in clusters:
        if len(cluster) < min_pocket_size:
            continue
        cset = set(cluster)
        boundary_edges: list[tuple[int, int]] = []
        seen_edges: set[tuple[int, int]] = set()
        for fi in cluster:
            f = faces_arr[fi]
            for i in range(3):
                e = (
                    min(int(f[i]), int(f[(i + 1) % 3])),
                    max(int(f[i]), int(f[(i + 1) % 3])),
                )
                if e in seen_edges:
                    continue
                seen_edges.add(e)
                faces_on_edge = edge_to_face[e]
                if len(faces_on_edge) == 2:
                    other = (
                        faces_on_edge[1]
                        if faces_on_edge[0] in cset
                        else faces_on_edge[0]
                    )
                    if other not in cset:
                        boundary_edges.append(e)
        if not boundary_edges:
            continue

        # Fold ratio: pocket surface area / mouth enclosed planar area.
        # A real pocket folds inward through a small opening (high ratio).
        # A flat patch has ratio near 1.
        pocket_area = float(area_arr[cluster].sum())
        opening_area = _rim_enclosed_area(boundary_edges, verts_arr)
        if opening_area <= 0:
            continue  # no measurable opening = not a pocket entrance
        fold_ratio = pocket_area / opening_area
        if fold_ratio < min_fold_ratio:
            continue

        mouths.append(boundary_edges)

    if verbose:
        print(
            f"[skeliner.pre] Rims: {len(mouths)} pockets "
            f"(fold >= {min_fold_ratio}), "
            f"{sum(len(mouth) for mouth in mouths):,} mouth edges"
        )

    return mouths


def _face_adjacency(
    mesh: trimesh.Trimesh,
    edge_to_face: dict[tuple[int, int], list[int]] | None = None,
) -> dict[int, set[int]]:
    """Build face adjacency map (edge-connected neighbors)."""
    if edge_to_face is None:
        edge_to_face = _edge_to_faces(mesh)

    adj: dict[int, set[int]] = defaultdict(set)
    for faces in edge_to_face.values():
        for i in range(len(faces)):
            for j in range(i + 1, len(faces)):
                adj[faces[i]].add(faces[j])
                adj[faces[j]].add(faces[i])
    return adj


def find_pocket_organelles(
    mesh: trimesh.Trimesh,
    *,
    radius: float | None = None,
    radius_multiplier: float = 5.0,
    grow_threshold: float = 0.1,
    bridge_threshold: float = 0.3,
    max_hole_size: int = 500,
    hole_enclosure_ratio: float = 0.5,
    min_pocket_size: int = 5,
    min_fold_ratio: float = 3.0,
    min_cluster_size: int = 5,
    verbose: bool = False,
    mesh_stats: MeshStats | None = None,
) -> np.ndarray:
    """Detect pocket organelles — membrane folds connected to the neuron surface.

    Uses mouths (boundaries of negative-dot clusters) as seeds:

    1. Call :func:`find_pocket_mouths` to get mouth edges for each pocket.
    2. Seed from the **negative-dot faces** of each mouth's pocket cluster.
    3. Flood-fill from seeds, stopping at mouth edges and faces with
       ``outward_dot > grow_threshold``.
    4. Bridging: flood-fill from pocket boundary using the relaxed
       ``bridge_threshold`` to cross narrow positive-dot barriers.
    5. Hole filling: small non-pocket clusters mostly enclosed by pocket
       faces are filled in.

    Only regions behind a mouth get detected — curved surfaces without a
    mouth are correctly excluded.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Input neuron mesh.
    radius : float or None
        Radius for outward_dot computation. Auto-computed if None.
    radius_multiplier : float
        Multiplier for auto radius.
    grow_threshold : float
        Flood-fill will not enter faces with ``outward_dot > grow_threshold``.
    bridge_threshold : float
        Relaxed threshold used in a second flood-fill from the pocket
        boundary to cross narrow positive-dot barriers within a pocket.
    max_hole_size : int
        Non-pocket clusters with at most this many faces that are
        mostly enclosed by pocket faces are filled in.
    hole_enclosure_ratio : float
        Minimum fraction of a hole cluster's boundary neighbors that
        must be pocket faces for the hole to be filled.
    min_pocket_size : int
        Minimum negative-face cluster size to produce a mouth / be detected.
    min_cluster_size : int
        Final pocket clusters smaller than this are discarded.
    verbose : bool

    Returns
    -------
    np.ndarray
        Boolean mask ``(nFaces,)`` — pocket organelle faces.
    """

    if mesh_stats is not None and mesh_stats.outward_dots is not None:
        outward_dots = mesh_stats.outward_dots
        if mesh_stats.face_comp is not None:
            main_face_mask = mesh_stats.main_face_mask
        else:
            _fc, _mc = _face_edge_components(mesh)
            main_face_mask = _fc == _mc
    else:
        mesh_stats = compute_mesh_stats(
            mesh,
            radius,
            radius_multiplier,
            verbose,
        )
        outward_dots = mesh_stats.outward_dots
        main_face_mask = mesh_stats.main_face_mask
    n_faces = len(mesh.faces)
    # Compute shared structures once
    edge_to_face = _edge_to_faces(mesh)
    adj = _face_adjacency(mesh, edge_to_face)

    # Use find_pocket_mouths to get mouth edges for each pocket
    mouths = find_pocket_mouths(
        mesh,
        radius=radius,
        radius_multiplier=radius_multiplier,
        min_pocket_size=min_pocket_size,
        min_fold_ratio=min_fold_ratio,
        verbose=verbose,
        mesh_stats=mesh_stats,
        _adj=adj,
        _edge_to_face=edge_to_face,
    )

    if not mouths:
        if verbose:
            print("[skeliner.pre] No pockets found")
        return np.zeros(n_faces, dtype=bool)

    # Collect all mouth edges and seed faces
    # Seeds = negative-dot faces adjacent to mouth edges (inward side)
    mouth_edge_set: set[tuple[int, int]] = set()
    for mouth in mouths:
        mouth_edge_set.update(mouth)

    # Seeds: negative-dot faces touching mouth edges
    seeds: set[int] = set()
    for e in mouth_edge_set:
        for fi in edge_to_face.get(e, []):
            if main_face_mask[fi] and outward_dots[fi] < 0:
                seeds.add(fi)

    # Mouth faces: positive-dot faces touching mouth edges (block flood-fill)
    mouth_faces: set[int] = set()
    for e in mouth_edge_set:
        for fi in edge_to_face.get(e, []):
            if fi not in seeds:
                mouth_faces.add(fi)

    if verbose:
        print(
            f"[skeliner.pre] Pockets: {len(mouths)}, "
            f"pocket mouth edges: {len(mouth_edge_set):,}, "
            f"seeds: {len(seeds):,}"
        )

    # Flood-fill from seeds, blocked by mouth faces and grow_threshold
    pocket = np.zeros(n_faces, dtype=bool)
    visited = np.zeros(n_faces, dtype=bool)
    queue = deque(seeds)

    while queue:
        fi = queue.popleft()
        if visited[fi]:
            continue
        visited[fi] = True
        if fi in mouth_faces:
            continue
        if not main_face_mask[fi]:
            continue
        if outward_dots[fi] > grow_threshold:
            continue
        pocket[fi] = True
        for nfi in adj.get(fi, set()):
            if not visited[nfi]:
                queue.append(nfi)

    initial_count = int(pocket.sum())

    # Phase 1 — Bridging: flood-fill from pocket boundary using the
    # relaxed bridge_threshold to cross positive-dot barriers.
    bridge_queue = deque()
    for fi in np.where(pocket)[0]:
        for nfi in adj.get(int(fi), set()):
            if not pocket[nfi]:
                bridge_queue.append(nfi)

    while bridge_queue:
        fi = bridge_queue.popleft()
        if pocket[fi]:
            continue
        if fi in mouth_faces or not main_face_mask[fi]:
            continue
        if outward_dots[fi] > bridge_threshold:
            continue
        pocket[fi] = True
        for nfi in adj.get(fi, set()):
            if not pocket[nfi]:
                bridge_queue.append(nfi)

    bridge_count = int(pocket.sum()) - initial_count

    # Phase 2 — Hole filling: small non-pocket clusters mostly enclosed
    # by pocket faces are filled in.  Sub-folds inside pockets can have
    # positive outward_dot, so we use enclosure ratio only.
    non_pocket_main = main_face_mask & ~pocket
    non_pocket_idx = set(np.where(non_pocket_main)[0].tolist())
    np_visited: set[int] = set()
    hole_count = 0

    for fi in non_pocket_idx:
        if fi in np_visited:
            continue
        cluster: list[int] = []
        n_pocket_boundary = 0
        n_total_boundary = 0
        bfs_queue = deque([fi])
        while bfs_queue:
            curr = bfs_queue.popleft()
            if curr in np_visited:
                continue
            np_visited.add(curr)
            cluster.append(curr)
            for nfi in adj.get(curr, set()):
                if nfi in non_pocket_idx and nfi not in np_visited:
                    bfs_queue.append(nfi)
                elif nfi not in non_pocket_idx:
                    n_total_boundary += 1
                    if pocket[nfi]:
                        n_pocket_boundary += 1
        if len(cluster) == 0:
            continue
        enclosure = n_pocket_boundary / n_total_boundary if n_total_boundary else 0
        # Small holes: fill if enclosure meets threshold.
        # Large holes: only fill if fully entrapped (enclosure ~1.0)
        #   and not the main mesh body (< 5% of faces).
        n_faces = len(pocket)
        is_small = len(cluster) <= max_hole_size
        is_entrapped = enclosure >= 0.99 and len(cluster) < n_faces * 0.05
        if (is_small and enclosure >= hole_enclosure_ratio) or is_entrapped:
            for c in cluster:
                pocket[c] = True
            hole_count += len(cluster)

    # Cluster filter to remove noise
    pocket = _filter_small_clusters(mesh, pocket, min_cluster_size)

    # Post-validate: reject pocket components that are just surface dips.
    # A real pocket has much more surface area than its adjacent non-pocket
    # faces (the "cap").  A shallow dip has pocket_area ≈ cap_area.
    pocket_idx = set(np.where(pocket)[0].tolist())
    pocket_visited: set[int] = set()
    n_rejected = 0
    for start in list(pocket_idx):
        if start in pocket_visited:
            continue
        comp: list[int] = []
        pq = deque([start])
        while pq:
            fi = pq.popleft()
            if fi in pocket_visited:
                continue
            pocket_visited.add(fi)
            comp.append(fi)
            for nfi in adj.get(fi, set()):
                if nfi in pocket_idx and nfi not in pocket_visited:
                    pq.append(nfi)

        comp_set = set(comp)
        cap_faces: set[int] = set()
        for fi in comp:
            for nfi in adj.get(fi, set()):
                if nfi not in comp_set:
                    cap_faces.add(nfi)

        pocket_area = float(mesh.area_faces[comp].sum())
        cap_area = float(mesh.area_faces[list(cap_faces)].sum()) if cap_faces else 0.0
        if cap_area <= 0 or pocket_area / cap_area < min_fold_ratio:
            for fi in comp:
                pocket[fi] = False
            n_rejected += 1

    if verbose:
        print(
            f"[skeliner.pre] Pocket organelles (>= {min_cluster_size}): "
            f"{pocket.sum():,} faces "
            f"(initial {initial_count:,}, "
            f"bridged +{bridge_count:,}, "
            f"holes +{hole_count:,}, "
            f"rejected {n_rejected})"
        )
    return pocket


def find_isolated_organelles(
    mesh: trimesh.Trimesh,
    *,
    radius: float | None = None,
    radius_multiplier: float = 5.0,
    verbose: bool = False,
    mesh_stats: MeshStats | None = None,
) -> np.ndarray:
    """Detect vertex-disconnected internal fragments.

    These are organelle membranes that form entirely separate connected
    components enclosed within the neuron body.

    Returns
    -------
    np.ndarray
        Boolean mask ``(nFaces,)`` — isolated organelle faces.
    """
    if mesh_stats is None or mesh_stats.outward_dots is None:
        mesh_stats = compute_mesh_stats(
            mesh,
            radius,
            radius_multiplier,
            verbose,
        )
    outward_dots = mesh_stats.outward_dots
    if mesh_stats.face_comp is not None:
        face_comp = mesh_stats.face_comp
        main_ci = mesh_stats.main_ci
    else:
        face_comp, main_ci = _face_edge_components(mesh)
    n_comps = face_comp.max() + 1
    isolated = np.zeros(len(mesh.faces), dtype=bool)
    n_internal_frags = 0
    n_internal_frag_faces = 0
    n_kept_frags = 0
    n_via_dots = 0
    n_via_geo = 0

    # Classify each non-main component using outward_dots.
    # Organelle membranes have predominantly negative outward_dots
    # (faces point inward); structural components have positive.
    # For tiny components where outward_dots are all zero (too few
    # neighbours for valid local COM), fall back to the geometric
    # inside/outside test against the main component.
    main_face_idx = np.where(face_comp == main_ci)[0]
    main_centroids = mesh.triangles_center[main_face_idx]
    main_normals = mesh.face_normals[main_face_idx]
    main_tree = None  # built lazily for fallback

    for ci in range(n_comps):
        if ci == main_ci:
            continue
        comp_face_idx = np.where(face_comp == ci)[0]
        if len(comp_face_idx) == 0:
            continue

        comp_dots = outward_dots[comp_face_idx]
        n_valid = np.count_nonzero(comp_dots)

        if n_valid > 0:
            # Use outward_dots: majority negative → isolated organelle
            is_internal = (comp_dots < 0).sum() > n_valid / 2
            n_via_dots += 1
        else:
            # Fallback: geometric test against main surface
            if main_tree is None:
                main_tree = KDTree(main_centroids)
            comp_verts = np.unique(mesh.faces[comp_face_idx])
            coords = mesh.vertices[comp_verts]
            _, nn_idx = main_tree.query(coords)
            vecs = coords - main_centroids[nn_idx]
            dots = np.einsum("ij,ij->i", vecs, main_normals[nn_idx])
            is_internal = (dots < 0).sum() > len(dots) / 2
            n_via_geo += 1

        if is_internal:
            isolated[comp_face_idx] = True
            n_internal_frags += 1
            n_internal_frag_faces += len(comp_face_idx)
        else:
            n_kept_frags += 1

    if verbose:
        print(
            f"[skeliner.pre] Isolated: {n_via_dots:,} via outward_dots, "
            f"{n_via_geo:,} via geometric fallback"
        )
        print(
            f"[skeliner.pre] Isolated fragments: {n_internal_frags:,} "
            f"({n_internal_frag_faces:,} faces), "
            f"{n_kept_frags:,} external (kept)"
        )
    return isolated


def find_organelles(
    mesh: trimesh.Trimesh,
    *,
    radius: float | None = None,
    radius_multiplier: float = 5.0,
    min_cluster_size: int = 5,
    verbose: bool = False,
    mesh_stats: MeshStats | None = None,
    return_mesh_stats: bool = False,
) -> Organelles | tuple[Organelles, MeshStats]:
    """Detect internal mesh fragments (organelle membranes) in a neuron mesh.

    Returns an :class:`~skeliner.dataclass.Organelles` with two
    non-overlapping masks:

    * **pocket** — membrane folds connected to the neuron surface,
      detected via gradient-based mouth finding + flood-fill.
    * **isolated** — entire vertex-disconnected components whose
      mean outward dot is negative (fully enclosed fragments).

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Input neuron mesh.
    radius : float or None, default None
        Radius (in mesh units, typically nm) for the local center-of-mass
        neighbourhood.  If None, automatically set to
        ``radius_multiplier * median_edge_length``.
    radius_multiplier : float, default 5.0
        Multiplier for automatic radius computation.  Only used when
        *radius* is None.
    min_cluster_size : int, default 5
        Connected components of internal faces smaller than this are
        kept (assumed to be noise).
    verbose : bool, default False
        Print summary statistics.
    mesh_stats : MeshStats or None, default None
        Pre-computed :func:`compute_mesh_stats` output to reuse.  If
        None, fresh stats are computed internally.
    return_mesh_stats : bool, default False
        If True, also return the :class:`MeshStats` used during
        detection (freshly computed when *mesh_stats* was not provided).

    Returns
    -------
    Organelles
        Always returned.
    MeshStats
        Only returned when ``return_mesh_stats=True``.
    """
    import time as _time

    _p = "[skeliner.pre]"
    t_total = _time.perf_counter()

    # ── 1. Precompute outward dots and components ─────────────────
    if mesh_stats is None or mesh_stats.outward_dots is None:
        mesh_stats = compute_mesh_stats(
            mesh,
            radius,
            radius_multiplier,
            verbose,
        )
    if mesh_stats.face_comp is not None:
        face_comp, main_ci = mesh_stats.face_comp, mesh_stats.main_ci
    else:
        face_comp, main_ci = _face_edge_components(mesh)

    # ── 2. Find isolated organelles (small internal components) ───
    isolated = find_isolated_organelles(
        mesh,
        radius=radius,
        radius_multiplier=radius_multiplier,
        verbose=verbose,
        mesh_stats=mesh_stats,
    )

    # ── 3. Identify structural components (non-isolated) ──────────
    #       Build a mask covering all non-isolated components so that
    #       find_pocket_organelles detects pockets on all of them in
    #       a single pass (the outward dots are already per-component).
    isolated_faces_set = set(np.where(isolated)[0].tolist())
    structural_comps = []
    for ci in np.unique(face_comp):
        comp_face_idx = np.where(face_comp == ci)[0]
        non_iso_count = sum(1 for fi in comp_face_idx if fi not in isolated_faces_set)
        if non_iso_count >= min_cluster_size:
            structural_comps.append(int(ci))

    if verbose and len(structural_comps) > 1:
        other = [ci for ci in structural_comps if ci != main_ci]
        sizes = sorted([int((face_comp == ci).sum()) for ci in other], reverse=True)
        print(
            f"{_p} Structural components: main + "
            f"{len(other)} disconnected "
            f"({', '.join(f'{s:,}f' for s in sizes[:5])}"
            + ("..." if len(sizes) > 5 else "")
            + ")"
        )

    # ── 4. Run pocket detection on all structural components ──────
    #       Replace main_face_mask with structural_face_mask in the
    #       precomputed tuple so pocket detection covers all components.
    structural_mask = np.zeros(len(mesh.faces), dtype=bool)
    for ci in structural_comps:
        structural_mask[face_comp == ci] = True

    # Build a MeshStats where main_face_mask covers all structural components
    # by mapping main_ci to a synthetic value that matches structural_mask.
    # We achieve this by setting face_comp to 0 for structural faces and
    # main_ci=0, so main_face_mask == structural_mask.
    structural_face_comp = np.where(structural_mask, 0, -1).astype(face_comp.dtype)
    precomputed_structural = MeshStats(
        outward_dots=mesh_stats.outward_dots,
        face_comp=structural_face_comp,
        main_ci=0,
    )

    pocket = find_pocket_organelles(
        mesh,
        radius=radius,
        radius_multiplier=radius_multiplier,
        min_cluster_size=min_cluster_size,
        verbose=verbose,
        mesh_stats=precomputed_structural,
    )

    if verbose and len(structural_comps) > 1:
        main_pocket = int(pocket[face_comp == main_ci].sum())
        other_pocket = int(pocket.sum()) - main_pocket
        print(
            f"{_p} Pocket organelles: {int(pocket.sum()):,} faces "
            f"(main: {main_pocket:,}, other: {other_pocket:,})"
        )

    dt_total = _time.perf_counter() - t_total
    if verbose:
        print(
            f"{_p} find_organelles done: pocket={int(pocket.sum()):,}, "
            f"isolated={int(isolated.sum()):,} ({dt_total:.1f}s)"
        )

    org = Organelles(
        pocket=pocket,
        isolated=isolated,
        expanded=np.zeros(len(mesh.faces), dtype=bool),
    )
    if return_mesh_stats:
        return org, mesh_stats
    return org


def _find_nonmanifold_fusions(
    mesh: trimesh.Trimesh,
    *,
    radius: float | None = None,
    radius_multiplier: float = 5.0,
    grow_rings: int = 20,
    min_branch_size: int = 5,
    verbose: bool = False,
    mesh_stats: MeshStats | None = None,
) -> list[list[int]]:
    """Detect non-manifold fusions (shared edges, duplicate faces, pinch vertices)."""
    from collections import Counter, deque

    areas = mesh.area_faces
    zero_faces = set(np.where(areas < 1e-6)[0].tolist())

    edge_to_face = _edge_to_faces(mesh)

    def _get_neighbors(fi: int) -> set[int]:
        f = mesh.faces[fi]
        nb: set[int] = set()
        for i in range(3):
            a, b = int(f[i]), int(f[(i + 1) % 3])
            for nfi in edge_to_face[(min(a, b), max(a, b))]:
                if nfi != fi:
                    nb.add(nfi)
        return nb

    nm_edges = sum(1 for faces in edge_to_face.values() if len(faces) > 2)
    if verbose:
        print(
            f"[skeliner.pre] Mesh: {len(mesh.faces):,} faces, "
            f"{len(zero_faces)} zero-area, {nm_edges} non-manifold edges"
        )

    if mesh_stats is not None and mesh_stats.outward_dots is not None:
        outward_dots = mesh_stats.outward_dots
    else:
        if verbose:
            print("[skeliner.pre] Computing outward dots ...")
        outward_dots = _outward_dot(
            mesh,
            radius
            if radius is not None
            else radius_multiplier * float(np.median(mesh.edges_unique_length)),
        )
        if verbose:
            print("[skeliner.pre] Outward dots computed")

    # Signal 1: negative-dot faces at non-manifold edges
    nm_neg_faces: set[int] = set()
    for e, faces in edge_to_face.items():
        if len(faces) > 2:
            for fi in faces:
                if outward_dots[fi] < 0:
                    nm_neg_faces.add(fi)

    # Signal 2: exact duplicate faces with >3 neighbors
    face_tuples = [tuple(sorted(int(v) for v in f)) for f in mesh.faces]
    dupe_set = {ft for ft, c in Counter(face_tuples).items() if c > 1}
    dupe_faces: set[int] = set()
    for fi, ft in enumerate(face_tuples):
        if ft in dupe_set and fi not in zero_faces:
            nb: set[int] = set()
            f = mesh.faces[fi]
            for i in range(3):
                a, b = int(f[i]), int(f[(i + 1) % 3])
                for nfi in edge_to_face[(min(a, b), max(a, b))]:
                    if nfi != fi:
                        nb.add(nfi)
            if len(nb) > 3:
                dupe_faces.add(fi)

    seed_faces = nm_neg_faces | dupe_faces

    # Signal 3: fan vertices
    _good = _non_degenerate(mesh.faces)
    vert_to_face: list[list[int]] = [[] for _ in range(len(mesh.vertices))]
    for fi, f in enumerate(mesh.faces):
        if not _good[fi] or fi in zero_faces:
            continue
        for v in f:
            vert_to_face[int(v)].append(fi)

    fan_vertex_clusters: list[list[int]] = []
    n_verts = len(mesh.vertices)
    if verbose:
        print(f"[skeliner.pre] Scanning {n_verts:,} vertices for fan fusions ...")
    for vid in range(n_verts):
        fan = vert_to_face[vid]
        if len(fan) < 2:
            continue
        fan_set = set(fan)
        fan_adj: dict[int, set[int]] = defaultdict(set)
        for fi in fan:
            f = mesh.faces[fi]
            for i in range(3):
                a, b = int(f[i]), int(f[(i + 1) % 3])
                e = (min(a, b), max(a, b))
                for nfi in edge_to_face[e]:
                    if nfi in fan_set and nfi != fi:
                        fan_adj[fi].add(nfi)
        vis: set[int] = set()
        q = deque([fan[0]])
        while q:
            curr = q.popleft()
            if curr in vis:
                continue
            vis.add(curr)
            for nfi in fan_adj.get(curr, set()):
                if nfi not in vis:
                    q.append(nfi)
        if len(vis) < len(fan_set):
            fan_vertex_clusters.append(sorted(fan))
            if verbose:
                _vis2: set[int] = set()
                _comp_sizes: list[int] = []
                for _fi in fan:
                    if _fi in _vis2:
                        continue
                    _sz = 0
                    _q = deque([_fi])
                    while _q:
                        _c = _q.popleft()
                        if _c in _vis2:
                            continue
                        _vis2.add(_c)
                        _sz += 1
                        for _nfi in fan_adj.get(_c, set()):
                            if _nfi not in _vis2:
                                _q.append(_nfi)
                    _comp_sizes.append(_sz)
                _comp_sizes.sort(reverse=True)
                print(
                    f"[skeliner.pre]   Fan vertex {vid}: "
                    f"{len(_comp_sizes)} components, "
                    f"sizes {_comp_sizes}"
                )

    if verbose:
        print(
            f"[skeliner.pre] Fusions: scanned {n_verts:,} vertices, "
            f"{len(fan_vertex_clusters)} fan fusions found"
        )

    if not seed_faces and not fan_vertex_clusters:
        return []

    # Cluster seeds
    seed_adj: dict[int, set[int]] = defaultdict(set)
    for e, faces in edge_to_face.items():
        for i in range(len(faces)):
            for j in range(i + 1, len(faces)):
                if faces[i] in seed_faces and faces[j] in seed_faces:
                    seed_adj[faces[i]].add(faces[j])
                    seed_adj[faces[j]].add(faces[i])

    visited: set[int] = set()
    seed_clusters: list[list[int]] = []
    for fi in seed_faces:
        if fi in visited:
            continue
        cluster: list[int] = []
        queue = deque([fi])
        while queue:
            curr = queue.popleft()
            if curr in visited:
                continue
            visited.add(curr)
            cluster.append(curr)
            for nfi in seed_adj.get(curr, set()):
                if nfi not in visited:
                    queue.append(nfi)
        seed_clusters.append(cluster)

    if verbose:
        print(
            f"[skeliner.pre] Fusion seeds: {len(seed_clusters)} clusters, "
            f"{len(seed_faces)} faces "
            f"(nm_neg={len(nm_neg_faces)}, dupes={len(dupe_faces)}, "
            f"fan_verts={len(fan_vertex_clusters)})"
        )

    # Grow region & split per cluster
    def _manifold_components(region: set[int]) -> list[set[int]]:
        m_adj: dict[int, set[int]] = defaultdict(set)
        for fi in region:
            f = mesh.faces[fi]
            for i in range(3):
                a, b = int(f[i]), int(f[(i + 1) % 3])
                e = (min(a, b), max(a, b))
                if len(edge_to_face[e]) == 2:
                    for nfi in edge_to_face[e]:
                        if nfi != fi and nfi in region:
                            m_adj[fi].add(nfi)
        vis: set[int] = set()
        comps: list[set[int]] = []
        for fi in region:
            if fi in vis:
                continue
            comp: set[int] = set()
            q = deque([fi])
            while q:
                curr = q.popleft()
                if curr in vis:
                    continue
                vis.add(curr)
                comp.add(curr)
                for nfi in m_adj[curr]:
                    if nfi not in vis:
                        q.append(nfi)
            comps.append(comp)
        comps.sort(key=len, reverse=True)
        return comps

    result_clusters: list[list[int]] = []
    n_seed_clusters = len(seed_clusters)

    for ci, seed_cluster in enumerate(seed_clusters):
        region = set(seed_cluster)
        if verbose:
            print(
                f"[skeliner.pre] Growing seed cluster "
                f"{ci + 1}/{n_seed_clusters} ({len(seed_cluster)}f) ..."
            )
        found = False
        for ring in range(1, grow_rings + 1):
            boundary: set[int] = set()
            for fi in region:
                boundary.update(_get_neighbors(fi))
            region = (region | boundary) - zero_faces
            if ring < 3:
                continue
            comps = _manifold_components(region)
            big = [c for c in comps if len(c) >= min_branch_size]
            if verbose:
                print(
                    f"[skeliner.pre]   ring {ring}: "
                    f"{len(region)}f, {len(comps)} comps "
                    f"({len(big)} >= {min_branch_size}f)"
                )
            if len(big) >= 2:
                branch0 = big[0]
                branch1 = big[1]
                fusion_boundary: set[int] = set()
                for fi in branch0:
                    for nfi in _get_neighbors(fi):
                        if nfi in branch1:
                            fusion_boundary.add(fi)
                            fusion_boundary.add(nfi)
                result_clusters.append(sorted(fusion_boundary))
                found = True
                if verbose:
                    print(
                        f"[skeliner.pre]   Seed cluster ({len(seed_cluster)}f): "
                        f"split at ring {ring}, region {len(region)}f, "
                        f"branches {len(branch0)}+{len(branch1)}f, "
                        f"boundary {len(fusion_boundary)}f"
                    )
                break
        if not found and verbose:
            print(
                f"[skeliner.pre]   Seed cluster ({len(seed_cluster)}f): "
                f"no split after {grow_rings} rings, "
                f"region {len(region)}f"
            )

    result_clusters.extend(fan_vertex_clusters)
    return result_clusters


def find_fusions(
    mesh: trimesh.Trimesh,
    *,
    radius: float | None = None,
    radius_multiplier: float = 5.0,
    grow_rings: int = 20,
    min_branch_size: int = 5,
    verbose: bool = False,
    mesh_stats: MeshStats | None = None,
) -> list[list[int]]:
    """Detect non-manifold fusion points where two branches are wrongly connected.

    Detects non-manifold edges, duplicate faces, and pinch vertices.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Input neuron mesh (ideally after organelle removal).
    radius : float or None
        Radius for outward_dot computation. Auto-computed if None.
    radius_multiplier : float
        Multiplier for auto radius.
    grow_rings : int
        Maximum rings to grow around each seed cluster.
    min_branch_size : int
        Minimum faces in a component to count as a branch.
    verbose : bool
    mesh_stats : tuple or None
        Precomputed ``(outward_dots, face_comp, main_ci, main_face_mask)``
        from :func:`compute_mesh_stats`.  Reuses ``outward_dots`` to skip
        redundant computation.

    Returns
    -------
    list[list[int]]
        Each inner list is one fusion cluster (face indices at the
        fusion boundary).

    Notes
    -----
    This detects **non-manifold fusions** only (shared edges, duplicate
    faces, pinch vertices).  **Loop fusions** — manifold cycles where
    two branches diverge and reconverge — are not detected here.  Loop
    fusion detection is planned for the skeletonization refactoring
    (detect cycles in the pre-MST skeleton graph).
    """
    clusters = _find_nonmanifold_fusions(
        mesh,
        radius=radius,
        radius_multiplier=radius_multiplier,
        grow_rings=grow_rings,
        min_branch_size=min_branch_size,
        verbose=verbose,
        mesh_stats=mesh_stats,
    )
    clusters.sort(key=len, reverse=True)

    if verbose:
        print(
            f"[skeliner.pre] Fusions: {len(clusters)} regions, "
            f"{sum(len(c) for c in clusters)} boundary faces"
        )

    return clusters


def _split_fan_vertices(
    mesh: trimesh.Trimesh,
    verbose: bool = False,
) -> trimesh.Trimesh:
    """Split vertices whose face fan is disconnected by face-edge adjacency.

    At a manifold fusion, two branches share vertices but not face-edges.
    Duplicating these vertices so each face-edge component gets its own
    copy disconnects the branches in the vertex graph.

    Returns a new mesh with split vertices (more vertices, same faces).
    """

    _good = _non_degenerate(mesh.faces)
    vert_to_face: list[list[int]] = [[] for _ in range(len(mesh.vertices))]
    for fi, f in enumerate(mesh.faces):
        if _good[fi]:
            for v in f:
                vert_to_face[int(v)].append(fi)

    edge_to_face = _edge_to_faces(mesh)

    verts = mesh.vertices.copy()
    faces = mesh.faces.copy()
    new_verts = list(verts)
    n_split = 0

    for vid in range(len(verts)):
        fan = vert_to_face[vid]
        if len(fan) < 2:
            continue

        fan_set = set(fan)
        fan_adj: dict[int, set[int]] = defaultdict(set)
        for fi in fan:
            f = faces[fi]
            for i in range(3):
                a, b = int(f[i]), int(f[(i + 1) % 3])
                e = (min(a, b), max(a, b))
                for nfi in edge_to_face[e]:
                    if nfi in fan_set and nfi != fi:
                        fan_adj[fi].add(nfi)

        # Find connected components of the face fan
        vis: set[int] = set()
        comps: list[set[int]] = []
        for fi in fan:
            if fi in vis:
                continue
            comp: set[int] = set()
            q = deque([fi])
            while q:
                curr = q.popleft()
                if curr in vis:
                    continue
                vis.add(curr)
                comp.add(curr)
                for nfi in fan_adj[curr]:
                    if nfi not in vis:
                        q.append(nfi)
            comps.append(comp)

        if len(comps) <= 1:
            continue

        # Keep first component with original vertex, duplicate for rest
        for comp_faces in comps[1:]:
            new_vid = len(new_verts)
            new_verts.append(verts[vid].copy())
            for fi in comp_faces:
                for j in range(3):
                    if int(faces[fi][j]) == vid:
                        faces[fi][j] = new_vid
            n_split += 1

    if n_split == 0:
        return mesh

    result = trimesh.Trimesh(
        vertices=np.array(new_verts),
        faces=faces,
        process=False,
    )

    if verbose:
        print(
            f"[skeliner.pre] Split {n_split} fan vertices "
            f"({len(verts):,} -> {len(new_verts):,} vertices)"
        )

    return result


def remove_fusions(
    mesh: trimesh.Trimesh,
    *,
    fusions: list[list[int]] | None = None,
    radius: float | None = None,
    radius_multiplier: float = 5.0,
    verbose: bool = False,
    mesh_stats: MeshStats | None = None,
) -> trimesh.Trimesh:
    """Remove fusion faces and split shared vertices between branches.

    Two-step process:

    1. **Remove non-manifold fusion faces** — detected by
       :func:`find_fusions` (non-manifold edges, duplicate faces).
    2. **Split shared vertices** — vertices whose face fan is
       disconnected by face-edge adjacency, indicating two branches
       share a vertex without sharing face-edges.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Input neuron mesh (ideally after organelle removal).
    fusion_clusters : list[list[int]] or None
        Pre-computed clusters from :func:`find_fusions`.  If provided,
        detection is skipped and these clusters are used directly.
    radius : float or None
        Radius for outward_dot computation. Auto-computed if None.
    radius_multiplier : float
        Multiplier for auto radius.
    verbose : bool

    Returns
    -------
    trimesh.Trimesh
        Mesh with fusions removed and shared vertices split.
    """
    # Step 1: remove non-manifold fusion faces
    if fusions is not None:
        if verbose:
            n = sum(len(c) for c in fusions)
            print(
                f"[skeliner.pre] Using provided fusion clusters "
                f"({len(fusions)} regions, {n} faces)"
            )
    else:
        fusions = find_fusions(
            mesh,
            radius=radius,
            radius_multiplier=radius_multiplier,
            verbose=verbose,
        )

    all_fusions: set[int] = set()
    for c in fusions:
        all_fusions.update(c)

    if all_fusions:
        keep = np.ones(len(mesh.faces), dtype=bool)
        for fi in all_fusions:
            keep[fi] = False
        mesh = _rebuild_mesh(mesh, keep)
        if verbose:
            print(
                f"[skeliner.pre] Removed {len(all_fusions)} fusion faces "
                f"({len(fusions)} regions)"
            )

    # Step 2: split shared vertices
    mesh = _split_fan_vertices(mesh, verbose=verbose)

    if verbose:
        print(
            f"[skeliner.pre] Fusions: {len(all_fusions)} faces removed, vertices split"
        )

    # Invalidate topology (components split at fan vertices)
    if mesh_stats is not None:
        mesh_stats.invalidate_topology()

    return mesh


def _rebuild_mesh(
    mesh: trimesh.Trimesh,
    keep_mask: np.ndarray,
) -> trimesh.Trimesh:
    """Remove faces from a mesh, preserving all indices.

    Removed faces are replaced with degenerate triangles ``[0, 0, 0]``
    so that both vertex and face indices remain stable.  Downstream
    data (``soma.verts``, face-based annotations) stays valid without
    any remapping.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Original mesh.
    keep_mask : np.ndarray
        Boolean mask ``(nFaces,)`` — True for faces to keep.

    Returns
    -------
    trimesh.Trimesh
        Mesh with removed faces degenerated.  Same vertex and face
        array sizes as the input.
    """
    new_faces = mesh.faces.copy()
    new_faces[~keep_mask] = 0
    return trimesh.Trimesh(
        vertices=mesh.vertices,
        faces=new_faces,
        process=False,
    )


def remove_organelles(
    mesh: trimesh.Trimesh,
    *,
    organelles: Organelles | None = None,
    radius: float | None = None,
    radius_multiplier: float = 5.0,
    min_cluster_size: int = 5,
    verbose: bool = False,
) -> trimesh.Trimesh:
    """Remove internal mesh fragments (organelle membranes) from a neuron mesh.

    Organelle membranes (mitochondria, ER, etc.) often appear as
    connected or semi-connected components sitting *inside* the
    neuron body.  They bias skeleton-node positions and radius estimates
    and should be removed before skeletonisation.

    Vertex indices are preserved (unreferenced vertices are left in
    the array) so that vertex-based data like ``soma.verts`` remains
    valid without remapping.

    Note: this does NOT remove the nucleus membrane inside the soma.
    Use :func:`remove_nucleus` after skeletonisation for that.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Input neuron mesh.
    organelles : Organelles or None
        Pre-computed organelles from :func:`find_organelles`.  If
        provided, detection is skipped.
    radius : float or None, default None
        Radius (in mesh units, typically nm) for the local center-of-mass
        neighbourhood.  If None, automatically set to
        ``radius_multiplier * median_edge_length``.
    radius_multiplier : float, default 5.0
        Multiplier for automatic radius computation.  Only used when
        *radius* is None.
    min_cluster_size : int, default 5
        Connected components of internal faces smaller than this are
        kept (assumed to be noise).
    verbose : bool, default False
        Print summary statistics.

    Returns
    -------
    trimesh.Trimesh
        Cleaned mesh with internal fragments removed.  Vertex indices
        are unchanged from the input mesh.

    Examples
    --------
    org = find_organelles(mesh, verbose=True)
    soma = find_soma_via_ring_cutoff(mesh, organelles=org)
    clean = remove_organelles(mesh, organelles=org)
    # soma.verts is still valid on clean — no remapping needed
    """
    if organelles is None:
        organelles = find_organelles(
            mesh,
            radius=radius,
            radius_multiplier=radius_multiplier,
            min_cluster_size=min_cluster_size,
            verbose=verbose,
        )
    elif verbose:
        print(
            f"[skeliner.pre] Using provided organelles mask "
            f"({int(organelles.mask.sum()):,} faces)"
        )

    if not organelles.mask.any():
        if verbose:
            print("[skeliner.pre] Nothing to remove")
        return mesh

    clean = _rebuild_mesh(mesh, ~organelles.mask)

    if verbose:
        print(
            f"[skeliner.pre] Result: {len(clean.faces):,} faces "
            f"(removed {organelles.mask.sum():,} faces)"
        )

    return clean


def remove_nucleus(
    mesh: trimesh.Trimesh,
    skeleton,
    *,
    soma_inside_frac: float = 0.9,
    min_nucleus_faces: int = 100,
    verbose: bool = False,
) -> trimesh.Trimesh:
    """Remove the nucleus membrane from inside the soma.

    Uses the soma ellipsoid from a skeleton to identify the nucleus —
    a large internal shell inside the soma that ``remove_organelles``
    cannot detect because it resembles a normal surface.

    Algorithm:

    1. Find faces inside the soma ellipsoid.
    2. Classify each face as outward-facing or inward-facing relative
       to the soma center.
    3. Cut the face-adjacency graph at outward/inward transitions.
    4. The largest inward-facing component is the nucleus — remove it
       along with any other inward components above *min_nucleus_faces*.

    Intended pipeline::

        clean = pre.remove_organelles(raw_mesh)
        skel = skeliner.skeletonize(clean)
        final = pre.remove_nucleus(clean, skel)

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Input mesh (typically after ``remove_organelles``).
    skeleton : Skeleton
        Skeleton with soma detection (from ``skeletonize``).
    soma_inside_frac : float, default 0.9
        Scale factor for the soma ellipsoid boundary test.
    min_nucleus_faces : int, default 100
        Only remove inward components with at least this many faces.
    verbose : bool, default False
        Print summary.

    Returns
    -------
    trimesh.Trimesh
        Mesh with nucleus membrane removed.
    """
    soma = skeleton.soma
    if soma is None:
        if verbose:
            print("[skeliner.pre] No soma detected — skipping nucleus removal")
        return mesh

    centroids = mesh.triangles_center
    normals = mesh.face_normals
    n_faces = len(mesh.faces)

    # Step 1: faces inside soma ellipsoid (skip degenerate faces)
    good = _non_degenerate(mesh.faces)
    inside = soma.contains(centroids, inside_frac=soma_inside_frac) & good
    inside_idx = np.where(inside)[0]

    if len(inside_idx) == 0:
        if verbose:
            print("[skeliner.pre] No faces inside soma ellipsoid")
        return mesh

    # Step 2: classify outward vs inward relative to soma center
    dir_from_soma = centroids[inside] - soma.center
    nrm = np.linalg.norm(dir_from_soma, axis=1, keepdims=True)
    dir_from_soma /= np.maximum(nrm, 1e-10)
    soma_dots = np.einsum("ij,ij->i", normals[inside], dir_from_soma)
    is_outward = soma_dots >= 0

    if verbose:
        print(
            f"[skeliner.pre] Faces inside soma: {len(inside_idx):,} "
            f"(outward: {is_outward.sum():,}, inward: {(~is_outward).sum():,})"
        )

    # Step 3: build face graph inside soma, cut at outward/inward transitions
    inside_set = set(int(fi) for fi in inside_idx)
    fi_remap = {int(fi): i for i, fi in enumerate(inside_idx)}

    edge_to_faces: dict[tuple[int, int], list[int]] = defaultdict(list)
    for fi in inside_idx:
        face = mesh.faces[fi]
        for i in range(3):
            e = (
                min(int(face[i]), int(face[(i + 1) % 3])),
                max(int(face[i]), int(face[(i + 1) % 3])),
            )
            edge_to_faces[e].append(int(fi))

    edges: set[tuple[int, int]] = set()
    for fi in inside_idx:
        fi_int = int(fi)
        local_i = fi_remap[fi_int]
        face = mesh.faces[fi]
        for i in range(3):
            e = (
                min(int(face[i]), int(face[(i + 1) % 3])),
                max(int(face[i]), int(face[(i + 1) % 3])),
            )
            for nfi in edge_to_faces[e]:
                if nfi != fi_int and nfi in inside_set:
                    local_j = fi_remap[nfi]
                    if is_outward[local_i] == is_outward[local_j]:
                        a, b = min(local_i, local_j), max(local_i, local_j)
                        edges.add((a, b))

    g = ig.Graph(n=len(inside_idx), edges=list(edges), directed=False)
    comps = g.connected_components()

    # Step 4: collect large inward components = nucleus
    nucleus_mask = np.zeros(n_faces, dtype=bool)
    n_nucleus_comps = 0
    n_small_inward = 0

    for cl in comps:
        if is_outward[cl[0]]:
            continue
        if len(cl) < min_nucleus_faces:
            n_small_inward += 1
            continue
        for i in cl:
            nucleus_mask[int(inside_idx[i])] = True
        n_nucleus_comps += 1

    n_nucleus = int(nucleus_mask.sum())
    if verbose:
        print(
            f"[skeliner.pre] Nucleus: {n_nucleus_comps} component(s), "
            f"{n_nucleus:,} faces"
        )
        if n_small_inward > 0:
            print(
                f"[skeliner.pre]   Skipped {n_small_inward} inward component(s) "
                f"below min_nucleus_faces={min_nucleus_faces}"
            )

    if n_nucleus == 0:
        if verbose:
            print("[skeliner.pre] No nucleus found")
        return mesh

    keep = ~nucleus_mask
    clean = _rebuild_mesh(mesh, keep)

    if verbose:
        print(
            f"[skeliner.pre] Result: {len(clean.faces):,} faces "
            f"(removed {n_nucleus:,} nucleus faces)"
        )

    return clean


# ── Chunk boundary / sandwich detection ────────────────────────────


def find_chunk_boundaries(
    mesh: trimesh.Trimesh,
    *,
    verbose: bool = False,
) -> dict[int, np.ndarray]:
    """Detect chunk boundary planes from vertex coordinate clustering.

    EM meshes are built by running marching cubes per chunk and merging
    at chunk boundaries.  At each boundary plane, many vertices share
    the exact same coordinate on the boundary axis.  This function finds
    those planes by detecting local vertex-count spikes on each axis.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Input mesh.
    verbose : bool, default False
        Print detected boundaries.

    Returns
    -------
    dict[int, np.ndarray]
        Mapping ``{axis: array_of_boundary_coordinates}`` where *axis*
        is 0 (X), 1 (Y), or 2 (Z).  Each array is sorted.
    """
    verts = mesh.vertices
    boundaries: dict[int, np.ndarray] = {}

    for axis in range(3):
        coords = np.round(verts[:, axis]).astype(np.int64)
        unique, counts = np.unique(coords, return_counts=True)

        if len(unique) < 3:
            continue

        # Chunk boundaries are sharp spikes: a single coordinate value
        # with many more vertices than its neighbors.  Detect as values
        # where count exceeds the local background by a large factor.
        median_count = float(np.median(counts))
        # Spike ratio: each value's count vs mean of its ±5 neighbors
        n = len(counts)
        local_bg = np.empty(n, dtype=float)
        for i in range(n):
            lo = max(0, i - 5)
            hi = min(n, i + 6)
            # Exclude self from local background
            neighbors = np.concatenate([counts[lo:i], counts[i + 1 : hi]])
            local_bg[i] = neighbors.mean() if len(neighbors) > 0 else 1.0
        spike_ratio = counts / np.maximum(local_bg, 1.0)

        # A boundary is any coordinate with a clear local spike
        is_boundary = spike_ratio > 5.0

        if not is_boundary.any():
            continue

        # Remove false boundaries that are too close to a real one.
        # Real chunk boundaries have consistent spacing; a spike much
        # closer than the median spacing is a hard-cutoff or other
        # non-chunk artifact.
        bvals = unique[is_boundary]
        if len(bvals) >= 3:
            spacings = np.diff(bvals)
            median_sp = float(np.median(spacings))
            keep = np.ones(len(bvals), dtype=bool)
            for i in range(len(bvals)):
                left = spacings[i - 1] if i > 0 else median_sp
                right = spacings[i] if i < len(spacings) else median_sp
                # If BOTH neighbors are much closer than median, it's
                # a cluster — keep the one with the highest spike.
                # If only one side is short, this boundary is the
                # non-chunk one if it has a lower spike than its
                # close neighbor.
                if min(left, right) < median_sp * 0.25:
                    # Find the close neighbor
                    if left < median_sp * 0.25 and i > 0:
                        # Close to left neighbor — keep the stronger
                        left_ratio = spike_ratio[np.where(unique == bvals[i - 1])[0][0]]
                        this_ratio = spike_ratio[np.where(unique == bvals[i])[0][0]]
                        if this_ratio < left_ratio:
                            keep[i] = False
                    if right < median_sp * 0.25 and i < len(spacings):
                        right_ratio = spike_ratio[
                            np.where(unique == bvals[i + 1])[0][0]
                        ]
                        this_ratio = spike_ratio[np.where(unique == bvals[i])[0][0]]
                        if this_ratio < right_ratio:
                            keep[i] = False
            bvals = bvals[keep]

        boundaries[axis] = bvals

        if verbose:
            name = "XYZ"[axis]
            print(f"[skeliner.pre] Chunk boundaries {name}: {len(bvals)} planes")
            for val in bvals:
                idx = np.where(unique == val)[0][0]
                print(
                    f"  {name}={val}: {counts[idx]} verts (spike {spike_ratio[idx]:.1f}x)"
                )

    return boundaries


def find_parallel_patches(
    mesh: trimesh.Trimesh,
    *,
    boundaries: dict[int, np.ndarray] | None = None,
    normal_thresh: float = 0.99,
    boundary_dist: float = 30.0,
    expand_tilted: bool = True,
    loose_normal_thresh: float = 0.9,
    min_sandwich_overlap: float = 0.3,
    verbose: bool = False,
) -> list[dict]:
    """Detect parallel-patch artifacts at chunk boundaries.

    At chunk merge failures, marching cubes produces pairs of perfectly
    axis-aligned face patches on opposite sides of the boundary plane.
    This function finds connected patches of such faces and flags those
    that have an opposing patch (same boundary, overlapping in the
    perpendicular plane).

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Input mesh.
    boundaries : dict or None
        Output of :func:`find_chunk_boundaries`.  Computed if *None*.
    normal_thresh : float
        Minimum |normal·axis| to count as axis-aligned (default 0.99).
    boundary_dist : float
        Max distance from a boundary plane in nm (default 30).
    expand_tilted : bool, default True
        Run :func:`_expand_parallel_patches` at the end to absorb
        tilted transition faces at the rim of each seed patch.
    loose_normal_thresh : float, default 0.9
        Looser threshold used during rim expansion (passed to
        :func:`_expand_parallel_patches`).
    min_sandwich_overlap : float, default 0.3
        After detection, drop patch pairs whose triangle-area overlap
        (in the perpendicular plane) is below this fraction of the
        smaller partner's area.  Filters out organelle-to-main
        parallels that are not marching-cubes sandwich artifacts.
        Set to ``0.0`` to disable.
    verbose : bool
        Print summary.

    Returns
    -------
    list[dict]
        Each entry: ``{"axis": int, "bval": int, "faces": list[int],
        "up_faces": list[int], "down_faces": list[int]}``.
    """
    from collections import defaultdict, deque

    from scipy.spatial import cKDTree

    if boundaries is None:
        boundaries = find_chunk_boundaries(mesh)

    verts = mesh.vertices
    faces = mesh.faces
    normals = mesh.face_normals
    centroids = verts[faces].mean(axis=1)

    # Perpendicular axes for overlap check
    perp = {0: [1, 2], 1: [0, 2], 2: [0, 1]}

    results = []

    for axis, bvals in boundaries.items():
        n_ax = normals[:, axis]
        aligned = np.abs(n_ax) > normal_thresh

        # For each face, find nearest boundary on this axis
        near_mask = np.zeros(len(faces), dtype=bool)
        face_bval = np.full(len(faces), -1, dtype=np.int64)
        for bval in bvals:
            m = np.abs(centroids[:, axis] - bval) < boundary_dist
            near_mask |= m
            face_bval[m] = int(round(bval))

        target_mask = aligned & near_mask
        target = np.where(target_mask)[0]
        if len(target) < 2:
            continue
        target_set = set(int(fi) for fi in target)

        # Face adjacency among target faces
        edge_to_faces: dict[tuple, list[int]] = defaultdict(list)
        for fi in target:
            f = faces[fi]
            for i in range(3):
                a, b = int(f[i]), int(f[(i + 1) % 3])
                e = (min(a, b), max(a, b))
                edge_to_faces[e].append(int(fi))

        # Connected components — same boundary only
        visited: set[int] = set()
        patches: list[tuple[list[int], int]] = []
        for seed in target:
            seed = int(seed)
            if seed in visited:
                continue
            seed_bv = face_bval[seed]
            comp: list[int] = []
            queue = deque([seed])
            visited.add(seed)
            while queue:
                fi = queue.popleft()
                comp.append(fi)
                f = faces[fi]
                for i in range(3):
                    a, b = int(f[i]), int(f[(i + 1) % 3])
                    e = (min(a, b), max(a, b))
                    for nb in edge_to_faces[e]:
                        if (
                            nb not in visited
                            and nb in target_set
                            and face_bval[nb] == seed_bv
                        ):
                            visited.add(nb)
                            queue.append(nb)
            patches.append((comp, int(seed_bv)))

        # Check each patch for opposing faces
        perp_axes = perp[axis]

        def _any_face_overlaps(faces_a, faces_b):
            """Check if any face from A overlaps any face from B in perp plane.

            Projects triangles onto the perpendicular plane and checks
            for vertex-in-triangle overlap.
            """
            # Get triangle vertices projected onto perp plane
            verts_a = verts[faces[faces_a]][:, :, perp_axes]  # (Na, 3, 2)
            verts_b = verts[faces[faces_b]][:, :, perp_axes]  # (Nb, 3, 2)

            # Quick bbox pre-filter
            a_min, a_max = verts_a.reshape(-1, 2).min(0), verts_a.reshape(-1, 2).max(0)
            b_min, b_max = verts_b.reshape(-1, 2).min(0), verts_b.reshape(-1, 2).max(0)
            if not (np.all(a_min <= b_max) and np.all(b_min <= a_max)):
                return False

            # Check if any vertex from A is strictly inside any triangle
            # of B (and vice versa).  Use cross-product sign test with
            # strict inequalities — points on edges/vertices don't count
            # as "inside" so that adjacent (but non-overlapping) triangles
            # sharing a vertex or edge are not flagged.
            def _point_in_tri(px, py, tri):
                """Test points strictly inside triangle tri (3,2)."""
                x0, y0 = tri[0]
                x1, y1 = tri[1]
                x2, y2 = tri[2]
                d00 = (x1 - x0) * (py - y0) - (y1 - y0) * (px - x0)
                d01 = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
                d02 = (x0 - x2) * (py - y2) - (y0 - y2) * (px - x2)
                return ((d00 > 0) & (d01 > 0) & (d02 > 0)) | (
                    (d00 < 0) & (d01 < 0) & (d02 < 0)
                )

            # Exclude shared vertices between A and B
            vidx_a = set(int(v) for v in faces[faces_a].ravel())
            vidx_b = set(int(v) for v in faces[faces_b].ravel())
            shared_vidx = vidx_a & vidx_b

            # Collect vertices from A (excluding shared), test against B
            pts_a = verts_a.reshape(-1, 2)
            mask_a = np.ones(len(pts_a), dtype=bool)
            flat_vidx_a = faces[faces_a].ravel()
            for k, vi in enumerate(flat_vidx_a):
                if int(vi) in shared_vidx:
                    mask_a[k] = False
            pts_a_filtered = pts_a[mask_a]
            if len(pts_a_filtered) > 0:
                for tri in verts_b:
                    if np.any(
                        _point_in_tri(pts_a_filtered[:, 0], pts_a_filtered[:, 1], tri)
                    ):
                        return True

            # Vice versa
            pts_b = verts_b.reshape(-1, 2)
            mask_b = np.ones(len(pts_b), dtype=bool)
            flat_vidx_b = faces[faces_b].ravel()
            for k, vi in enumerate(flat_vidx_b):
                if int(vi) in shared_vidx:
                    mask_b[k] = False
            pts_b_filtered = pts_b[mask_b]
            if len(pts_b_filtered) > 0:
                for tri in verts_a:
                    if np.any(
                        _point_in_tri(pts_b_filtered[:, 0], pts_b_filtered[:, 1], tri)
                    ):
                        return True

            return False

        for comp, bv in patches:
            fi_arr = np.array(comp)
            n_ax_comp = n_ax[fi_arr]
            pos_fi = fi_arr[n_ax_comp > normal_thresh]
            neg_fi = fi_arr[n_ax_comp < -normal_thresh]

            has_overlap = False

            # Self-opposing: up/down within same patch
            if len(pos_fi) > 0 and len(neg_fi) > 0:
                if _any_face_overlaps(pos_fi, neg_fi):
                    has_overlap = True

            # Nearby opposing: other patches at same boundary
            if not has_overlap:
                comp_set = set(comp)
                others = [
                    int(fi)
                    for fi in target
                    if face_bval[fi] == bv and fi not in comp_set
                ]
                if others:
                    other_nax = n_ax[others]
                    mean_dir = n_ax_comp.mean()
                    if mean_dir > 0:
                        opp = [
                            others[j]
                            for j in range(len(others))
                            if other_nax[j] < -normal_thresh
                        ]
                    else:
                        opp = [
                            others[j]
                            for j in range(len(others))
                            if other_nax[j] > normal_thresh
                        ]
                    if opp:
                        if _any_face_overlaps(fi_arr, np.array(opp)):
                            has_overlap = True

            if has_overlap:
                results.append(
                    {
                        "axis": axis,
                        "bval": bv,
                        "faces": [int(f) for f in fi_arr],
                        "up_faces": [int(f) for f in pos_fi],
                        "down_faces": [int(f) for f in neg_fi],
                    }
                )

    # ── Neighbor expansion across all patches ──────────────────────
    # Include non-target faces adjacent to 2+ faces from ANY flagged
    # patch (catches slightly-tilted faces between patches).
    all_patch_faces: set[int] = set()
    for r in results:
        all_patch_faces.update(r["faces"])

    # Build full edge→face for faces near any boundary
    all_near_set: set[int] = set()
    for axis_k in boundaries:
        near_k = np.abs(centroids[:, axis_k, np.newaxis] - boundaries[axis_k]) < 30.0
        all_near_set.update(int(i) for i in np.where(near_k.any(axis=1))[0])

    full_e2f: dict[tuple, list[int]] = defaultdict(list)
    for fi in all_near_set:
        f = faces[fi]
        for i in range(3):
            a, b = int(f[i]), int(f[(i + 1) % 3])
            full_e2f[(min(a, b), max(a, b))].append(fi)

    neighbor_count: dict[int, int] = defaultdict(int)
    for fi in all_patch_faces:
        f = faces[fi]
        for i in range(3):
            a, b = int(f[i]), int(f[(i + 1) % 3])
            e = (min(a, b), max(a, b))
            for nb in full_e2f[e]:
                if nb not in all_patch_faces:
                    neighbor_count[nb] += 1

    expanded = {nb for nb, cnt in neighbor_count.items() if cnt >= 2}
    all_patch_faces.update(expanded)

    # ── Merge into connected patches ──────────────────────────────
    # Re-run connected components on all flagged faces so that
    # neighboring patches (possibly linked through expanded faces)
    # form a single patch.
    merge_e2f: dict[tuple, list[int]] = defaultdict(list)
    for fi in all_patch_faces:
        f = faces[fi]
        for i in range(3):
            a, b = int(f[i]), int(f[(i + 1) % 3])
            merge_e2f[(min(a, b), max(a, b))].append(fi)

    visited_merge: set[int] = set()
    merged: list[list[int]] = []
    for seed in all_patch_faces:
        if seed in visited_merge:
            continue
        comp: list[int] = []
        queue = deque([seed])
        visited_merge.add(seed)
        while queue:
            fi = queue.popleft()
            comp.append(fi)
            f = faces[fi]
            for i in range(3):
                a, b = int(f[i]), int(f[(i + 1) % 3])
                e = (min(a, b), max(a, b))
                for nb in merge_e2f[e]:
                    if nb not in visited_merge and nb in all_patch_faces:
                        visited_merge.add(nb)
                        queue.append(nb)
        merged.append(comp)

    # Rebuild results from merged components.  Determine axis from
    # the dominant normal direction among perfectly-aligned faces.
    results = []
    for comp in merged:
        fi_arr = np.array(comp)
        # Find dominant axis: the one with most |n| > normal_thresh faces
        best_axis = 0
        best_count = 0
        for ax in range(3):
            cnt = int(np.sum(np.abs(normals[fi_arr, ax]) > normal_thresh))
            if cnt > best_count:
                best_count = cnt
                best_axis = ax
        # Determine boundary value from centroids
        bv = int(round(float(np.median(centroids[fi_arr, best_axis]))))
        # Find nearest actual boundary
        bvals_ax = boundaries.get(best_axis, np.array([]))
        if len(bvals_ax) > 0:
            idx = np.argmin(np.abs(bvals_ax - bv))
            bv = int(round(bvals_ax[idx]))

        n_ax_comp = normals[fi_arr, best_axis]
        pos_fi = fi_arr[n_ax_comp > 0.5]
        neg_fi = fi_arr[n_ax_comp < -0.5]
        results.append(
            {
                "axis": best_axis,
                "bval": bv,
                "faces": [int(f) for f in fi_arr],
                "up_faces": [int(f) for f in pos_fi],
                "down_faces": [int(f) for f in neg_fi],
            }
        )

    if expand_tilted:
        results = _expand_parallel_patches(
            mesh,
            results,
            loose_normal_thresh=loose_normal_thresh,
            boundary_dist=boundary_dist,
        )

    if min_sandwich_overlap > 0:
        results = _filter_weak_sandwich_pairs(
            mesh,
            results,
            min_overlap=min_sandwich_overlap,
        )

    results.sort(key=lambda r: (r["axis"], r["bval"], -len(r["faces"])))

    if verbose:
        n_faces = sum(len(r["faces"]) for r in results)
        print(
            f"[skeliner.pre] Parallel patches: {len(results)} patches, {n_faces} faces"
        )
        for r in results:
            name = "XYZ"[r["axis"]]
            print(
                f"  {name}={r['bval']}: {len(r['faces'])}f "
                f"(+{len(r['up_faces'])} / -{len(r['down_faces'])})"
            )

    return results


def _trim_patches_to_overlap(
    mesh: trimesh.Trimesh,
    patches: list[dict],
) -> list[dict]:
    """Trim each patch to the subset of faces whose projection onto
    the plane perpendicular to the patch axis intersects an opposing
    partner at the same ``(axis, bval)``.

    Only the portion of a parallel patch that actually sits above an
    opposing sandwich partner is a marching-cubes double-layer.  The
    rest of the same connected flat region (a large axis-aligned sheet
    that happens to be at a chunk boundary) is load-bearing topology
    and must not be removed — otherwise the mesh gets disconnected.

    Patches with no opposing partner in their group are dropped
    (empty ``faces``).  Mixed patches with both up and down faces are
    self-opposing: each sign is trimmed against the other sign's
    union.
    """
    from collections import defaultdict

    if not patches:
        return patches

    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    verts = mesh.vertices
    faces = mesh.faces
    normals = mesh.face_normals
    perp = {0: [1, 2], 1: [0, 2], 2: [0, 1]}

    groups: dict[tuple, list[int]] = defaultdict(list)
    for pi, p in enumerate(patches):
        groups[(int(p["axis"]), int(p["bval"]))].append(pi)

    new_patches: list[dict] = []
    for (axis, bval), group_pis in groups.items():
        perp_axes = perp[axis]

        # Union polygon per sign, pooled from all patches in the group.
        sign_union: dict[int, object] = {}
        for sign_val in (1, -1):
            polys = []
            for pj in group_pis:
                p = patches[pj]
                side_faces = p["up_faces"] if sign_val > 0 else p["down_faces"]
                if not side_faces:
                    continue
                for tri in verts[faces[side_faces]][:, :, perp_axes]:
                    try:
                        poly = Polygon(tri)
                        if poly.is_valid and poly.area > 0:
                            polys.append(poly)
                    except Exception:
                        continue
            if polys:
                sign_union[sign_val] = unary_union(polys)

        for pi in group_pis:
            p = patches[pi]
            kept_faces: list[int] = []
            kept_up: list[int] = []
            kept_down: list[int] = []

            for fi in p["faces"]:
                n_ax = normals[fi, axis]
                is_up = n_ax > 0
                opp_sign = -1 if is_up else 1
                opp = sign_union.get(opp_sign)
                if opp is None:
                    continue
                tri = verts[faces[fi]][:, perp_axes]
                try:
                    face_poly = Polygon(tri)
                    if not face_poly.is_valid:
                        continue
                    if face_poly.intersects(opp):
                        fi_int = int(fi)
                        kept_faces.append(fi_int)
                        if is_up:
                            kept_up.append(fi_int)
                        else:
                            kept_down.append(fi_int)
                except Exception:
                    continue

            if kept_faces:
                new_patches.append(
                    {
                        "axis": axis,
                        "bval": bval,
                        "faces": kept_faces,
                        "up_faces": kept_up,
                        "down_faces": kept_down,
                    }
                )

    return new_patches


def _sandwich_overlap_ratio(
    mesh: trimesh.Trimesh,
    faces_a: list[int],
    faces_b: list[int],
    axis: int,
) -> float:
    """Triangle-area overlap of two patches projected onto the plane
    perpendicular to *axis*, normalised by the smaller patch's area.

    Returns 0.0 if either side is empty or the polygons are invalid.
    """
    if not faces_a or not faces_b:
        return 0.0
    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    perp = [i for i in range(3) if i != axis]
    verts = mesh.vertices
    tris_a = verts[mesh.faces[faces_a]][:, :, perp]
    tris_b = verts[mesh.faces[faces_b]][:, :, perp]

    polys_a = [Polygon(t) for t in tris_a if Polygon(t).is_valid]
    polys_b = [Polygon(t) for t in tris_b if Polygon(t).is_valid]
    if not polys_a or not polys_b:
        return 0.0

    union_a = unary_union(polys_a)
    union_b = unary_union(polys_b)
    if union_a.is_empty or union_b.is_empty:
        return 0.0

    area_a = float(union_a.area)
    area_b = float(union_b.area)
    if area_a <= 0 or area_b <= 0:
        return 0.0

    inter_area = float(union_a.intersection(union_b).area)
    return inter_area / min(area_a, area_b)


def _filter_weak_sandwich_pairs(
    mesh: trimesh.Trimesh,
    patches: list[dict],
    *,
    min_overlap: float = 0.3,
) -> list[dict]:
    """Drop patches whose best opposing partner at the same (axis,bval)
    has triangle-area overlap below *min_overlap* in the perpendicular
    plane.  Keeps only patches that are genuine marching-cubes sandwich
    sides — filters out organelle-to-main parallels where the two
    sheets barely intersect.
    """
    from collections import defaultdict

    if not patches or min_overlap <= 0:
        return patches

    groups: dict[tuple, list[int]] = defaultdict(list)
    for pi, p in enumerate(patches):
        groups[(int(p["axis"]), int(p["bval"]))].append(pi)

    keep = [False] * len(patches)

    for (axis, _bval), group_pis in groups.items():
        # For each patch, find the best opposing overlap ratio.
        for pi in group_pis:
            p = patches[pi]
            up_i = p["up_faces"]
            dn_i = p["down_faces"]
            # Self-opposing (mixed patch): its own up vs down
            best = 0.0
            if up_i and dn_i:
                best = max(best, _sandwich_overlap_ratio(mesh, up_i, dn_i, axis))
            # Pure plus or minus: look at other patches at same boundary
            for pj in group_pis:
                if pj == pi:
                    continue
                q = patches[pj]
                if up_i and q["down_faces"]:
                    best = max(
                        best,
                        _sandwich_overlap_ratio(mesh, up_i, q["down_faces"], axis),
                    )
                if dn_i and q["up_faces"]:
                    best = max(
                        best,
                        _sandwich_overlap_ratio(mesh, dn_i, q["up_faces"], axis),
                    )
            if best >= min_overlap:
                keep[pi] = True

    return [p for pi, p in enumerate(patches) if keep[pi]]


def _expand_parallel_patches(
    mesh: trimesh.Trimesh,
    patches: list[dict],
    *,
    loose_normal_thresh: float = 0.9,
    boundary_dist: float = 30.0,
) -> list[dict]:
    """Absorb tilted transition faces adjacent to strict-parallel seed
    patches.

    At the rim of a parallel sandwich patch, marching cubes often emits
    a few slightly-tilted triangles that bridge the flat sheet to the
    surrounding geometry.  These faces fail the strict ``normal_thresh``
    test but should still be treated as part of the patch, otherwise
    they split one logical sandwich into multiple disconnected seed
    patches (and leave the rim contour broken).

    For each seed patch this function BFS-expands through edge-adjacent
    faces whose axis-aligned normal component satisfies

    * same sign as the patch (``n_axis * sign > 0``),
    * ``|n_axis| > loose_normal_thresh`` (default 0.9 — about 26° tilt),
    * centroid within ``2 * boundary_dist`` of the patch's ``bval``,
    * projection (in the plane perpendicular to the axis) overlapping
      the bbox of an *opposing* patch at the same axis/bval.

    Patches whose expansions share any absorbed face get merged into
    one in the returned list.
    """
    from collections import defaultdict, deque

    if not patches:
        return patches

    verts = mesh.vertices
    faces = mesh.faces
    normals = mesh.face_normals
    centroids = verts[faces].mean(axis=1)
    perp = {0: [1, 2], 1: [0, 2], 2: [0, 1]}
    expand_dist = 2.0 * boundary_dist

    # Mesh-wide edge adjacency
    full_ef: dict[tuple, list[int]] = defaultdict(list)
    for fi in range(len(faces)):
        f = faces[fi]
        if f[0] == f[1] or f[1] == f[2] or f[0] == f[2]:
            continue
        for i in range(3):
            a, b = int(f[i]), int(f[(i + 1) % 3])
            full_ef[(min(a, b), max(a, b))].append(int(fi))

    def patch_sign(p):
        if p["up_faces"]:
            return 1
        if p["down_faces"]:
            return -1
        return 0

    # Group patches by (axis, bval) and compute opposing XY bbox per sign.
    groups: dict[tuple, list[int]] = defaultdict(list)
    for pi, p in enumerate(patches):
        groups[(int(p["axis"]), int(p["bval"]))].append(pi)

    expansions: list[set[int]] = [set(p["faces"]) for p in patches]

    for (axis, bval), group_pis in groups.items():
        perp_axes = perp[axis]

        opp_bbox: dict[int, tuple] = {}
        for sign_val in (1, -1):
            opp_vs: list[np.ndarray] = []
            for pj in group_pis:
                if patch_sign(patches[pj]) == -sign_val:
                    opp_vs.append(verts[faces[patches[pj]["faces"]]].reshape(-1, 3))
            if not opp_vs:
                continue
            stacked = np.vstack(opp_vs)
            opp_bbox[sign_val] = (
                stacked[:, perp_axes].min(0),
                stacked[:, perp_axes].max(0),
            )

        for pi in group_pis:
            sign_p = patch_sign(patches[pi])
            if sign_p == 0 or sign_p not in opp_bbox:
                continue
            xy_min, xy_max = opp_bbox[sign_p]
            q = deque(patches[pi]["faces"])
            while q:
                fi = q.popleft()
                f = faces[fi]
                for i in range(3):
                    a, b = int(f[i]), int(f[(i + 1) % 3])
                    e = (min(a, b), max(a, b))
                    for nb in full_ef[e]:
                        if nb in expansions[pi]:
                            continue
                        n_ax = normals[nb, axis]
                        if n_ax * sign_p <= 0:
                            continue
                        if abs(n_ax) < loose_normal_thresh:
                            continue
                        if abs(centroids[nb, axis] - bval) > expand_dist:
                            continue
                        v = verts[faces[nb]][:, perp_axes]
                        if not (
                            (v.min(0) < xy_max).all() and (v.max(0) > xy_min).all()
                        ):
                            continue
                        expansions[pi].add(int(nb))
                        q.append(int(nb))

    # Merge patches whose expansions share absorbed faces (same axis,
    # bval, and sign).
    parent = list(range(len(patches)))

    def uf_find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def uf_union(a, b):
        ra, rb = uf_find(a), uf_find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(len(patches)):
        for j in range(i + 1, len(patches)):
            if patches[i]["axis"] != patches[j]["axis"]:
                continue
            if patches[i]["bval"] != patches[j]["bval"]:
                continue
            if patch_sign(patches[i]) != patch_sign(patches[j]):
                continue
            if expansions[i] & expansions[j]:
                uf_union(i, j)

    merged_groups: dict[int, list[int]] = defaultdict(list)
    for i in range(len(patches)):
        merged_groups[uf_find(i)].append(i)

    new_patches: list[dict] = []
    for root, members in merged_groups.items():
        axis = int(patches[root]["axis"])
        bval = int(patches[root]["bval"])
        merged_faces: set[int] = set()
        merged_up: set[int] = set()
        merged_down: set[int] = set()
        orig: set[int] = set()
        for mi in members:
            merged_faces |= expansions[mi]
            merged_up.update(patches[mi]["up_faces"])
            merged_down.update(patches[mi]["down_faces"])
            orig.update(patches[mi]["faces"])
        # Classify absorbed faces by their signed normal component
        for fi in merged_faces - orig:
            if normals[fi, axis] > 0:
                merged_up.add(fi)
            else:
                merged_down.add(fi)
        new_patches.append(
            {
                "axis": axis,
                "bval": bval,
                "faces": sorted(int(f) for f in merged_faces),
                "up_faces": sorted(int(f) for f in merged_up),
                "down_faces": sorted(int(f) for f in merged_down),
            }
        )

    return new_patches


def _find_fold_faces(mesh, patch):
    """Find faces on fold edges within a patch.

    A fold edge is where an up face shares an edge with a down face —
    the surface turns 180°.  Returns the set of face indices directly
    on fold edges.
    """
    from collections import defaultdict

    up_set = set(patch["up_faces"])
    down_set = set(patch["down_faces"])
    faces = mesh.faces

    edge_faces: dict[tuple, list[int]] = defaultdict(list)
    for fi in patch["faces"]:
        f = faces[fi]
        for i in range(3):
            e = (
                min(int(f[i]), int(f[(i + 1) % 3])),
                max(int(f[i]), int(f[(i + 1) % 3])),
            )
            edge_faces[e].append(fi)

    fold: set[int] = set()
    for e, fis in edge_faces.items():
        has_up = any(fi in up_set for fi in fis)
        has_down = any(fi in down_set for fi in fis)
        if has_up and has_down:
            fold.update(fis)

    return fold


def _trace_edge_loops(
    edges: list[tuple[int, int]],
) -> list[tuple[list[int], bool]]:
    """Trace a set of undirected edges into ordered vertex sequences.

    Returns a list of ``(vertex_sequence, is_closed)`` pairs.  For
    closed loops the last vertex is implicitly connected back to the
    first (i.e. the wrap-around edge exists in the input).  For open
    chains the first and last vertices are the dangling endpoints.

    Open chains are emitted first, starting at any degree-1 vertex.
    Remaining unused edges form closed loops.  Non-manifold vertices
    (degree ≥ 3) are split — each incident edge is walked at most
    once, so a figure-8 becomes two separate loops touching at the
    shared vertex.
    """
    from collections import defaultdict

    if not edges:
        return []

    ve: dict[int, list[tuple[int, int]]] = defaultdict(list)
    eset: set[tuple[int, int]] = set()
    for e in edges:
        key = (min(e[0], e[1]), max(e[0], e[1]))
        if key in eset:
            continue
        eset.add(key)
        ve[key[0]].append(key)
        ve[key[1]].append(key)

    used: set[tuple[int, int]] = set()
    out: list[tuple[list[int], bool]] = []

    def walk(start_edge: tuple[int, int]) -> tuple[list[int], bool]:
        chain = [start_edge[0], start_edge[1]]
        used.add(start_edge)
        curr = start_edge[1]
        closed = False
        while True:
            nxt_e = None
            for cand in ve[curr]:
                if cand not in used:
                    nxt_e = cand
                    break
            if nxt_e is None:
                break
            used.add(nxt_e)
            nxt = nxt_e[1] if nxt_e[0] == curr else nxt_e[0]
            if nxt == chain[0]:
                closed = True
                break
            chain.append(nxt)
            curr = nxt
        return chain, closed

    # Open chains: start at degree-1 endpoints
    endpoints = [v for v, es in ve.items() if len(es) == 1]
    for v in endpoints:
        for e in ve[v]:
            if e in used:
                continue
            oriented = e if e[0] == v else (e[1], e[0])
            chain, closed = walk(oriented)
            out.append((chain, closed))

    # Closed loops from any remaining unused edge
    for e in eset:
        if e in used:
            continue
        chain, closed = walk(e)
        out.append((chain, closed))

    return out


def _zip_vertex_loops(
    loop_a: list[int],
    loop_b: list[int],
    verts: np.ndarray,
    *,
    closed_a: bool = True,
    closed_b: bool = True,
) -> list[tuple[int, int, int]]:
    """Zip two ordered vertex loops into a strip of triangles.

    Both loops are walked in lock-step with a "shortest diagonal"
    decision rule: at each step, advance whichever side produces the
    shorter bridge edge to the other side's current vertex.  Each
    source edge of loop_a and loop_b is consumed in exactly one new
    triangle, so the result is manifold by construction (no fans, no
    duplicate edges).

    Before walking, this tries every starting offset and both winding
    orders on ``loop_b`` and picks the alignment that minimises the
    sum of bridge-edge lengths.  This is O((m + n) * n) which is fine
    for small rim loops.
    """
    m = len(loop_a)
    n = len(loop_b)
    if m < 2 or n < 2:
        return []

    def walk(shifted_b: list[int]) -> tuple[float, list[tuple[int, int, int]]]:
        tris: list[tuple[int, int, int]] = []
        cost = 0.0
        i, j = 0, 0
        # Closed loops walk m + n steps total; open chains walk m-1 + n-1.
        steps_a = m if closed_a else m - 1
        steps_b = n if closed_b else n - 1
        while i < steps_a or j < steps_b:
            a_curr = loop_a[i % m]
            a_next = loop_a[(i + 1) % m]
            t_curr = shifted_b[j % n]
            t_next = shifted_b[(j + 1) % n]
            if i >= steps_a:
                tris.append((a_curr, t_next, t_curr))
                cost += float(np.linalg.norm(verts[t_next] - verts[a_curr]))
                j += 1
            elif j >= steps_b:
                tris.append((a_curr, a_next, t_curr))
                cost += float(np.linalg.norm(verts[a_next] - verts[t_curr]))
                i += 1
            else:
                d_adv_a = float(np.linalg.norm(verts[a_next] - verts[t_curr]))
                d_adv_b = float(np.linalg.norm(verts[a_curr] - verts[t_next]))
                if d_adv_a <= d_adv_b:
                    tris.append((a_curr, a_next, t_curr))
                    cost += d_adv_a
                    i += 1
                else:
                    tris.append((a_curr, t_next, t_curr))
                    cost += d_adv_b
                    j += 1
        return cost, tris

    best: tuple[float, list[tuple[int, int, int]]] | None = None
    # Try both winding directions and all starting offsets on loop_b
    # (for closed loops).  For open chains, only offset 0 is valid.
    reverse_options = [False, True]
    shift_options = range(n) if closed_b else [0]
    for reverse in reverse_options:
        base = list(loop_b[::-1] if reverse else loop_b)
        for shift in shift_options:
            shifted = base[shift:] + base[:shift]
            cost, tris = walk(shifted)
            if best is None or cost < best[0]:
                best = (cost, tris)

    return best[1] if best is not None else []


def _stitch_mode_b(mesh, orig_faces, mode_b_faces, patches, verbose=False):
    """Bridge gaps left by Mode B parallel-patch removal using
    per-sandwich rim tracing and contour zipping.

    For each ``(axis, bval)`` sandwich group in *patches*, collect the
    rim edges on each sign side (edges where an up-removed or
    down-removed face meets a surviving face), trace them into ordered
    vertex loops, pair up-side loops with down-side loops by spatial
    proximity, and zip each pair into a manifold triangle strip.

    Bridges are emitted with :func:`_zip_vertex_loops`, which consumes
    each rim edge exactly once in exactly one new triangle — so the
    result has no fans, no duplicate faces, and no non-manifold edges
    by construction.
    """
    from collections import defaultdict

    verts = mesh.vertices
    faces = mesh.faces
    degen = np.all(faces == 0, axis=1)

    # Edge → surviving faces map (excludes degenerates)
    surv_ef: dict[tuple[int, int], set[int]] = defaultdict(set)
    for fi in range(len(faces)):
        if degen[fi]:
            continue
        f = faces[fi]
        a, b, c = int(f[0]), int(f[1]), int(f[2])
        if a == b or b == c or a == c:
            continue
        surv_ef[(min(a, b), max(a, b))].add(fi)
        surv_ef[(min(b, c), max(b, c))].add(fi)
        surv_ef[(min(a, c), max(a, c))].add(fi)

    mode_b_set = set(int(fi) for fi in mode_b_faces)

    def rim_edges_for(removed_side: set[int]) -> list[tuple[int, int]]:
        """Return edges of *orig_faces* in removed_side where the
        other side of the edge is a surviving face (not a degen or
        another removed face)."""
        out: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        for fi in removed_side:
            f = orig_faces[fi]
            for i in range(3):
                a, b = int(f[i]), int(f[(i + 1) % 3])
                if a == b:
                    continue
                e = (min(a, b), max(a, b))
                if e in seen:
                    continue
                surv = surv_ef.get(e, set())
                if len(surv) == 1:
                    seen.add(e)
                    out.append(e)
        return out

    # Group Mode-B patches by (axis, bval)
    groups: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for p in patches:
        pf = set(p["faces"])
        if not (pf & mode_b_set):
            continue
        groups[(int(p["axis"]), int(p["bval"]))].append(p)

    if not groups:
        if verbose:
            print("[skeliner.pre] Stitch: no Mode-B sandwich groups")
        return mesh

    all_new_tris: list[tuple[int, int, int]] = []
    n_pairs = 0
    n_skipped_loops = 0

    for (_axis, _bval), group in groups.items():
        up_removed: set[int] = set()
        dn_removed: set[int] = set()
        for p in group:
            up_removed.update(int(f) for f in p["up_faces"] if int(f) in mode_b_set)
            dn_removed.update(int(f) for f in p["down_faces"] if int(f) in mode_b_set)

        up_rim = rim_edges_for(up_removed)
        dn_rim = rim_edges_for(dn_removed)
        if not up_rim or not dn_rim:
            continue

        up_traces = _trace_edge_loops(up_rim)
        dn_traces = _trace_edge_loops(dn_rim)
        # A real contour needs at least 3 vertices (closed) or 2
        # vertices for a single-edge chain.  Drop anything smaller.
        up_traces = [
            (lp, c)
            for lp, c in up_traces
            if (c and len(lp) >= 3) or (not c and len(lp) >= 2)
        ]
        dn_traces = [
            (lp, c)
            for lp, c in dn_traces
            if (c and len(lp) >= 3) or (not c and len(lp) >= 2)
        ]
        if not up_traces or not dn_traces:
            continue

        # Pair loops greedily by centroid distance — largest loops first.
        # Perpendicular plane of the sandwich axis for bbox overlap check.
        perp_axes = [i for i in range(3) if i != _axis]
        up_info = []
        for lp, c in up_traces:
            pts = verts[lp]
            up_info.append(
                {
                    "loop": lp,
                    "closed": c,
                    "pts": pts,
                    "center": pts.mean(axis=0),
                    "bmin": pts[:, perp_axes].min(0),
                    "bmax": pts[:, perp_axes].max(0),
                }
            )
        dn_info = []
        for lp, c in dn_traces:
            pts = verts[lp]
            dn_info.append(
                {
                    "loop": lp,
                    "closed": c,
                    "pts": pts,
                    "center": pts.mean(axis=0),
                    "bmin": pts[:, perp_axes].min(0),
                    "bmax": pts[:, perp_axes].max(0),
                }
            )

        up_order = sorted(range(len(up_info)), key=lambda i: -len(up_info[i]["loop"]))
        dn_used: set[int] = set()

        for ui in up_order:
            u = up_info[ui]
            best_dn = -1
            best_d = float("inf")
            for di in range(len(dn_info)):
                if di in dn_used:
                    continue
                d_info = dn_info[di]
                # Require bbox overlap in the perpendicular plane —
                # without this, a mixed-patch rim whose up and down
                # sides cover different XY regions would get zipped
                # into a twisted strip crossing empty space.
                imin = np.maximum(u["bmin"], d_info["bmin"])
                imax = np.minimum(u["bmax"], d_info["bmax"])
                if not (imax > imin).all():
                    continue
                d = float(np.linalg.norm(u["center"] - d_info["center"]))
                if d < best_d:
                    best_d = d
                    best_dn = di
            if best_dn < 0:
                n_skipped_loops += 1
                continue
            dn_used.add(best_dn)
            d_info = dn_info[best_dn]
            tris = _zip_vertex_loops(
                u["loop"],
                d_info["loop"],
                verts,
                closed_a=u["closed"],
                closed_b=d_info["closed"],
            )
            if tris:
                all_new_tris.extend(tris)
                n_pairs += 1
            else:
                n_skipped_loops += 1

        n_skipped_loops += len(dn_info) - len(dn_used)

    if not all_new_tris:
        if verbose:
            print(
                f"[skeliner.pre] Stitch: no contours zipped "
                f"({n_skipped_loops} loops unpaired)"
            )
        return mesh

    new_faces = np.array(all_new_tris, dtype=np.int64)
    combined = np.vstack([faces, new_faces])

    if verbose:
        print(
            f"[skeliner.pre] Stitch: {n_pairs} rim pairs zipped, "
            f"{len(new_faces)} bridge faces"
            + (f" ({n_skipped_loops} loops unpaired)" if n_skipped_loops else "")
        )

    return trimesh.Trimesh(vertices=verts, faces=combined, process=False)


def remove_parallel_patches(
    mesh: trimesh.Trimesh,
    *,
    patches: list[dict] | None = None,
    boundaries: dict[int, np.ndarray] | None = None,
    verbose: bool = False,
    mesh_stats: MeshStats | None = None,
) -> trimesh.Trimesh:
    """Remove parallel-patch artifacts at chunk boundaries.

    1. Classify each patch as Mode A (has fold edges) or Mode B (no
       fold edges) using :func:`_find_fold_faces`.
    2. Remove all detected patch faces.
    3. Find new disconnected components created by the removal.
    4. For each new component, check which removed patches it borders:
       - borders only Mode A → orphan from fold removal → remove
       - borders any Mode B → real neurite piece → keep (for stitching)

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Input mesh.
    patches : list[dict] or None
        Output of :func:`find_parallel_patches`.  Computed if *None*.
    boundaries : dict or None
        Passed through to :func:`find_parallel_patches` if *patches* is None.
    verbose : bool
        Print summary.

    Returns
    -------
    trimesh.Trimesh
        Mesh with patch artifacts and Mode A orphans removed.
    """
    from collections import defaultdict

    if patches is None:
        patches = find_parallel_patches(mesh, boundaries=boundaries, verbose=verbose)

    if not patches:
        if verbose:
            print("[skeliner.pre] No parallel patches to remove")
        return mesh

    # ── Step 0: trim patches to overlap with opposing partner ─────
    # Only the portion of a parallel patch that sits above an opposing
    # sandwich partner is a marching-cubes double-layer.  Trimming
    # avoids disconnecting the mesh when a large flat axis-aligned
    # sheet is load-bearing topology with only a small MC sandwich
    # sitting in one corner (see site_02 case).
    n_faces_before_trim = sum(len(p["faces"]) for p in patches)
    patches = _trim_patches_to_overlap(mesh, patches)
    n_faces_after_trim = sum(len(p["faces"]) for p in patches)
    if verbose and n_faces_after_trim != n_faces_before_trim:
        print(
            f"[skeliner.pre] Trimmed patches to overlap: "
            f"{n_faces_before_trim} → {n_faces_after_trim} faces"
        )
    if not patches:
        if verbose:
            print("[skeliner.pre] No patch-overlap regions to remove")
        return mesh

    faces = mesh.faces

    # ── Step 1: classify patches ──────────────────────────────────
    mode_a_faces: set[int] = set()  # fold patches
    mode_b_faces: set[int] = set()  # parallel patches
    all_removed: set[int] = set()
    n_a = 0
    n_b = 0

    for patch in patches:
        fold = _find_fold_faces(mesh, patch)
        patch_set = set(patch["faces"])
        all_removed.update(patch_set)

        if fold:
            mode_a_faces.update(patch_set)
            n_a += 1
        else:
            mode_b_faces.update(patch_set)
            n_b += 1

    if verbose:
        print(
            f"[skeliner.pre] Parallel patches: "
            f"{n_a} Mode A (fold), {n_b} Mode B (parallel)"
        )
        print(f"[skeliner.pre] Removing {len(all_removed)} patch faces")

    # ── Step 2: remove all patch faces ────────────────────────────
    keep_mask = np.ones(len(faces), dtype=bool)
    keep_mask[list(all_removed)] = False
    result = _rebuild_mesh(mesh, keep_mask)

    # ── Step 3: find NEWLY disconnected components ─────────────────
    # Only consider faces that were in the main component before
    # removal but ended up in a non-main component after.
    labels_before, main_before = _face_edge_components(mesh)
    was_main = set(int(i) for i in np.where(labels_before == main_before)[0])

    labels_after, main_after = _face_edge_components(result)
    degen = np.all(result.faces == 0, axis=1)

    # Faces that were main, survived removal, but are now non-main
    newly_disconnected: set[int] = set()
    for fi in was_main:
        if not degen[fi] and labels_after[fi] != main_after:
            newly_disconnected.add(fi)

    # ── Step 4: classify new orphan components ────────────────────
    # Only applies to *newly* disconnected pieces (those that were in
    # main_before but are non-main_after).  Mode B pre-existing
    # disconnects are handled by the stitch step unconditionally.
    if newly_disconnected:
        # Build edge→type map for removed faces
        removed_edge_type: dict[tuple, str] = {}
        for fi in all_removed:
            f = faces[fi]
            typ = "a" if fi in mode_a_faces else "b"
            for i in range(3):
                e = (
                    min(int(f[i]), int(f[(i + 1) % 3])),
                    max(int(f[i]), int(f[(i + 1) % 3])),
                )
                # If any bordering removed face is Mode B, mark as "b"
                if removed_edge_type.get(e) != "b":
                    removed_edge_type[e] = typ

        # Group newly disconnected faces by component
        new_comp_ids = set(int(labels_after[fi]) for fi in newly_disconnected)
        orphan_remove: set[int] = set()

        for comp_id in new_comp_ids:
            comp_face_idxs = [
                fi
                for fi in np.where(labels_after == comp_id)[0]
                if fi in newly_disconnected or not degen[fi]
            ]
            borders_mode_b = False

            for fi in comp_face_idxs:
                if borders_mode_b:
                    break
                f = faces[fi]
                for i in range(3):
                    e = (
                        min(int(f[i]), int(f[(i + 1) % 3])),
                        max(int(f[i]), int(f[(i + 1) % 3])),
                    )
                    if removed_edge_type.get(e) == "b":
                        borders_mode_b = True
                        break

            if not borders_mode_b:
                orphan_remove.update(int(fi) for fi in comp_face_idxs)

        if orphan_remove:
            if verbose:
                n_comps = len({int(labels_after[fi]) for fi in orphan_remove})
                print(
                    f"[skeliner.pre] Removing {len(orphan_remove)} orphan faces "
                    f"({n_comps} components from Mode A removal)"
                )
            keep_mask2 = np.ones(len(faces), dtype=bool)
            keep_mask2[list(all_removed)] = False
            keep_mask2[list(orphan_remove)] = False
            result = _rebuild_mesh(mesh, keep_mask2)
        elif verbose:
            print(
                f"[skeliner.pre] Keeping {len(new_comp_ids)} new disconnected "
                f"components (border Mode B, need stitching)"
            )
    elif verbose:
        print("[skeliner.pre] No newly orphaned components from removal")

    # ── Step 5: stitch Mode B gaps ───────────────────────────────
    n_before = len(result.faces)
    if mode_b_faces:
        result = _stitch_mode_b(
            result,
            faces,
            mode_b_faces,
            patches,
            verbose=verbose,
        )

    # Invalidate topology; pad outward_dots for appended stitch faces
    if mesh_stats is not None:
        n_added = len(result.faces) - n_before
        if n_added > 0 and mesh_stats.outward_dots is not None:
            mesh_stats.outward_dots = np.concatenate(
                [
                    mesh_stats.outward_dots,
                    np.ones(n_added, dtype=mesh_stats.outward_dots.dtype),
                ]
            )
        mesh_stats.invalidate_topology()

    return result


def compact_mesh(
    mesh: trimesh.Trimesh,
    components: MeshComponents,
    *,
    return_maps: bool = False,
    verbose: bool = False,
) -> (
    tuple[trimesh.Trimesh, MeshComponents]
    | tuple[trimesh.Trimesh, MeshComponents, np.ndarray, np.ndarray]
):
    """Remove degenerate faces, drop unreferenced vertices, reindex.

    Call once at the end of preprocessing, after all face removals.
    Remaps all face/vertex-indexed data inside *components*.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Mesh after face removals.
    components : MeshComponents
        All face-indexed data is remapped to match the compacted mesh.
    return_maps : bool, default False
        If True, also return ``vert_map`` and ``face_map`` arrays for
        remapping external data (e.g. viewer annotations).
    verbose : bool, default False
        Print summary.

    Returns
    -------
    clean : trimesh.Trimesh
        Compacted mesh with no unreferenced vertices.
    components : MeshComponents
        Remapped components.
    vert_map : np.ndarray *(only when return_maps=True)*
        ``(nOldVerts,)`` int64 — ``vert_map[old]`` → new index, or -1.
    face_map : np.ndarray *(only when return_maps=True)*
        ``(nOldFaces,)`` int64 — ``face_map[old]`` → new index, or -1.
    """
    # Strip degenerate faces (from _rebuild_mesh) and compact vertices
    good = ~np.all(mesh.faces == mesh.faces[:, :1], axis=1)
    live_faces = mesh.faces[good]

    referenced = np.sort(np.unique(live_faces.ravel()))
    n_old = len(mesh.vertices)
    vert_map = np.full(n_old, -1, dtype=np.int64)
    vert_map[referenced] = np.arange(len(referenced), dtype=np.int64)

    clean = trimesh.Trimesh(
        vertices=mesh.vertices[referenced],
        faces=vert_map[live_faces],
        process=False,
    )

    if verbose:
        n_removed_v = n_old - len(clean.vertices)
        n_removed_f = len(mesh.faces) - len(clean.faces)
        print(
            f"[skeliner.pre] Compact: {n_old:,} → {len(clean.vertices):,} verts "
            f"({n_removed_v:,} removed), "
            f"{len(mesh.faces):,} → {len(clean.faces):,} faces "
            f"({n_removed_f:,} removed)"
        )

    # Remap soma
    remapped_soma = None
    if components.soma is not None:
        remapped_soma = components.soma.remap(vert_map)
        if verbose:
            n_before = (
                len(components.soma.verts) if components.soma.verts is not None else 0
            )
            n_after = len(remapped_soma.verts) if remapped_soma.verts is not None else 0
            print(
                f"[skeliner.pre] Soma remap: {n_before:,} → {n_after:,} verts "
                f"({n_before - n_after:,} dropped)"
            )

    # Build face index map: old_fi → new_fi (or -1 if removed)
    face_map = np.full(len(mesh.faces), -1, dtype=np.int64)
    face_map[good] = np.arange(int(good.sum()), dtype=np.int64)

    def _remap_face_list(arrays):
        out = []
        for arr in arrays:
            mapped = face_map[arr]
            mapped = mapped[mapped >= 0]
            if len(mapped) > 0:
                out.append(mapped)
        return out

    organelles = components.organelles
    remapped_org = Organelles(
        pocket=organelles.pocket[good],
        isolated=organelles.isolated[good],
        expanded=organelles.expanded[good],
    )

    remapped = MeshComponents(
        soma=remapped_soma,
        organelles=remapped_org,
        neurites=Neurites(_remap_face_list(components.neurites)),
        discarded=Discarded(_remap_face_list(components.discarded)),
    )

    if return_maps:
        return clean, remapped, vert_map, face_map
    return clean, remapped


# -----------------------------------------------------------------------------
# High-level preprocessing pipeline
# -----------------------------------------------------------------------------
from dataclasses import dataclass, field


@dataclass
class PreprocessResult:
    """Result of :func:`preprocess` — cleaned mesh + break_up_mesh output."""

    mesh: trimesh.Trimesh
    components: MeshComponents
    mesh_stats: MeshStats | None = None


def preprocess(
    mesh: trimesh.Trimesh,
    *,
    compact: bool = False,
    verbose: bool = False,
) -> PreprocessResult:
    """Run the full preprocessing pipeline in one call.

    Pipeline order:

      1. find_parallel_patches → remove_parallel_patches
      2. find_organelles
      3. find_soma_via_ring_cutoff
      4. find_disconnected → find_gaps → remove_gaps
      5. find_fusions → remove_fusions
      6. break_up_mesh
      7. compact_mesh (optional)

    Each ``find_*`` step is skipped if it returns nothing.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Input mesh.
    compact : bool, default False
        If True, run :func:`compact_mesh` as a final step to drop
        unreferenced vertices and degenerate faces.  This invalidates
        original mesh indices, so external data tied to them (e.g.
        viewer annotations) will need to be remapped.  Skeletonization
        works correctly on uncompacted meshes.
    verbose : bool, default False
        Print progress.

    Returns
    -------
    PreprocessResult
        Cleaned mesh, mesh components, and the :class:`MeshStats`
        threaded through the pipeline.
    """
    # 1. Parallel patches
    patches = find_parallel_patches(mesh, verbose=verbose)
    if patches:
        mesh = remove_parallel_patches(mesh, patches=patches, verbose=verbose)

    # 2. Organelles (also yields mesh_stats for downstream stages)
    org, mesh_stats = find_organelles(mesh, verbose=verbose, return_mesh_stats=True)

    # 3. Soma
    soma = find_soma_via_ring_cutoff(
        mesh, organelles=org, mesh_stats=mesh_stats, verbose=verbose
    )

    # 4. Disconnected → gaps
    disconnected = find_disconnected(
        mesh,
        soma=soma,
        organelles=org,
        mesh_stats=mesh_stats,
        verbose=verbose,
    )

    gaps = find_gaps(
        mesh,
        soma=soma,
        disconnected=disconnected,
        mesh_stats=mesh_stats,
        verbose=verbose,
    )
    if gaps:
        mesh = remove_gaps(mesh, gaps=gaps, mesh_stats=mesh_stats, verbose=verbose)

    # 5. Fusions
    fusions = find_fusions(mesh, mesh_stats=mesh_stats, verbose=verbose)
    if fusions:
        mesh = remove_fusions(
            mesh, fusions=fusions, mesh_stats=mesh_stats, verbose=verbose
        )

    # 6. Break up mesh
    if soma is not None:
        components = break_up_mesh(mesh, soma, org, verbose=verbose)
    else:
        components = MeshComponents(
            soma=None,
            organelles=org,
            neurites=Neurites([]),
            discarded=Discarded([]),
        )

    # 7. Compact (optional — remaps everything in MeshComponents)
    if compact:
        mesh, components = compact_mesh(mesh, components=components, verbose=verbose)
        # Compaction invalidates mesh_stats; drop the stale reference
        # so callers don't accidentally reuse it against the new mesh.
        mesh_stats = None

    return PreprocessResult(
        mesh=mesh,
        components=components,
        mesh_stats=mesh_stats,
    )


# =====================================================================
#  Z-slice helpers (shared by find_nucleus_center & find_soma_via_z_contour)
# =====================================================================


def _z_slice_cluster(
    verts_xy: np.ndarray,
    grid_res: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int] | None:
    """Rasterize 2D points, find the largest connected cluster.

    Returns ``(occ, soma_region, ix, iy, xy_min)`` or None if too few
    points.  *occ* is the raw occupancy grid, *soma_region* is the
    dilated largest-component mask, *ix/iy* are per-vertex grid indices,
    *xy_min* is the grid origin.
    """
    from scipy.ndimage import binary_dilation, label

    pts = verts_xy
    if len(pts) < 10:
        return None

    xy_min = pts.min(axis=0) - grid_res * 3
    nx = int((pts[:, 0].max() - xy_min[0] + grid_res * 6) / grid_res) + 1
    ny = int((pts[:, 1].max() - xy_min[1] + grid_res * 6) / grid_res) + 1

    occ = np.zeros((nx, ny), dtype=bool)
    ix = ((pts[:, 0] - xy_min[0]) / grid_res).astype(int).clip(0, nx - 1)
    iy = ((pts[:, 1] - xy_min[1]) / grid_res).astype(int).clip(0, ny - 1)
    occ[ix, iy] = True

    struct3 = np.ones((3, 3), dtype=bool)
    dilated = binary_dilation(occ, struct3)
    labeled, n_labels = label(dilated)
    if n_labels == 0:
        return None

    sizes = np.bincount(labeled.ravel())[1:]
    lid = int(np.argmax(sizes)) + 1
    soma_region = labeled == lid

    return occ, soma_region, ix, iy, xy_min


def _z_slice_void(
    occ: np.ndarray,
    soma_region: np.ndarray,
    xy_min: np.ndarray,
    grid_res: float,
) -> tuple[float, float, float, np.ndarray | None] | None:
    """Detect the nucleus void within a soma cluster.

    Returns ``(cx, cy, void_r, void_xy)`` or None if no void.
    *void_xy* is ``(M, 2)`` world-coordinate points of the void region.
    """
    from scipy.ndimage import binary_fill_holes, distance_transform_edt, label

    occ_s = occ & soma_region

    # 4-ray enclosure
    enc = np.ones_like(occ_s)
    for ax in range(2):
        enc &= np.maximum.accumulate(occ_s, axis=ax)
        enc &= np.flip(
            np.maximum.accumulate(np.flip(occ_s, axis=ax), axis=ax),
            axis=ax,
        )
    # Confine to filled soma cluster (can't leak through pocket mouth)
    soma_filled = binary_fill_holes(soma_region)
    enc &= soma_filled

    void = enc & ~occ_s
    if not void.any():
        return None

    dt = distance_transform_edt(~occ_s)
    dtv = dt * void
    pk = np.unravel_index(dtv.argmax(), dtv.shape)
    cx = float(xy_min[0] + pk[0] * grid_res + grid_res / 2)
    cy = float(xy_min[1] + pk[1] * grid_res + grid_res / 2)
    void_r = float(dtv.max()) * grid_res

    # Keep only the connected component containing the peak,
    # with dt > 2 to exclude small vertex-spacing gaps.
    deep_void = void & (dt > 2)
    if not deep_void.any():
        return cx, cy, void_r, None

    void_labeled, _ = label(deep_void)
    peak_label = void_labeled[pk[0], pk[1]]
    if peak_label == 0:
        deep_ij = np.argwhere(deep_void)
        dists = np.abs(deep_ij[:, 0] - pk[0]) + np.abs(deep_ij[:, 1] - pk[1])
        peak_label = void_labeled[
            deep_ij[np.argmin(dists)][0], deep_ij[np.argmin(dists)][1]
        ]
    nucleus_void = void_labeled == peak_label

    void_ij = np.argwhere(nucleus_void)
    void_xy = np.column_stack(
        [
            xy_min[0] + void_ij[:, 0] * grid_res + grid_res / 2,
            xy_min[1] + void_ij[:, 1] * grid_res + grid_res / 2,
        ]
    )
    return cx, cy, void_r, void_xy


#  find_nucleus_center — nucleus detection via Z-slice void analysis
# =====================================================================


def find_nucleus_center(
    mesh: trimesh.Trimesh,
    *,
    grid_res: float = 200.0,
    z_tol: float = 150.0,
    max_shift: float = 3000.0,
    min_void_r: float = 500.0,
    verbose: bool = False,
) -> np.ndarray | None:
    """Locate the nucleus center from the mesh geometry alone.

    The nucleus membrane folds inward from the soma surface, creating
    a large void visible in Z-slice cross-sections.  At each Z-level
    the algorithm rasterizes vertex positions into a 2D occupancy grid,
    isolates the largest connected cluster (the soma cross-section),
    and measures the maximum distance-to-nearest-vertex within a
    4-ray-enclosed region.  The nucleus shows up as a sustained void
    across many consecutive Z-levels at a stable XY location.

    **Algorithm**

    1. For each Z-level, rasterize mesh vertex XY positions into a
       2D grid at *grid_res* resolution.
    2. Dilate by 1 px and take connected components; keep only the
       largest (the soma cross-section, which is bigger than any
       neurite cross-section).
    3. On the *original* (undilated) grid within that component, run a
       4-ray enclosure test (occupied cell must exist in +x, −x, +y,
       −y).  Enclosed empty cells are void candidates.
    4. Distance-transform the void candidates; the peak gives the void
       centre and radius for that Z-level.
    5. Chain consecutive Z-levels whose void centres are within
       *max_shift* of each other.  The longest chain = nucleus.
    6. Return the mean void centre across the chain.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Input neuron mesh (any cell type).
    grid_res : float
        2D rasterization cell size in nm.  200 nm works well: fine
        enough to resolve the nucleus void (~2 000–3 000 nm radius),
        coarse enough that vertex gaps inside the soma surface close.
    z_tol : float
        Half-width of the Z slab for collecting vertices at each level.
    max_shift : float
        Maximum XY displacement (nm) between consecutive Z-levels for
        a void to be considered part of the same chain.
    min_void_r : float
        Minimum void radius (nm) to be considered a candidate.
    verbose : bool
        Print progress.

    Returns
    -------
    dict or None
        ``None`` if no sustained void is found.  Otherwise a dict with:

        - ``center`` — ``(3,)`` nucleus centre in world coordinates.
        - ``z_range`` — ``(z_lo, z_hi)`` Z extent of the detected void.
        - ``peak_r`` — peak void radius (nm) across the chain.
        - ``slices`` — ``(N, 4)`` array where each row is
          ``(z, cx, cy, void_r)`` for the N Z-levels in the best chain.
          *cx, cy* are the void centre at each level and *void_r* is
          the void radius at that level.
    """
    _p = "[find_nucleus_center]"

    verts = np.asarray(mesh.vertices, dtype=np.float64)
    if verbose:
        z_unique = np.unique(verts[:, 2])
        n_levels = int((z_unique.max() - z_unique.min()) / 210)
        print(f"{_p} {len(verts):,} vertices, ~{n_levels} Z-levels")

    raw, best, center = _z_scan(verts, grid_res, z_tol, min_void_r, max_shift)

    if best is None:
        if verbose:
            print(f"{_p} no sustained void found")
        return None

    slices_numeric = np.array(
        [(raw[i][0], raw[i][1], raw[i][2], raw[i][3]) for i in best]
    )
    contours = {raw[i][0]: raw[i][4] for i in best}

    peak_r = float(slices_numeric[:, 3].max())
    z_lo = float(slices_numeric[0, 0])
    z_hi = float(slices_numeric[-1, 0])

    if verbose:
        print(
            f"{_p} nucleus: center=({center[0]:.0f}, {center[1]:.0f}, "
            f"{center[2]:.0f}), Z=[{z_lo:.0f},{z_hi:.0f}], "
            f"peak_r={peak_r:.0f} nm, {len(best)} Z-levels"
        )

    # Soma-cluster vertex indices at the mid-chain Z-level.
    # These are outer-surface vertices (not void/pocket), suitable
    # as BFS seeds for soma detection.
    mid_i = best[len(best) // 2]
    mid_vi = raw[mid_i][5]  # vert_indices at this Z
    mid_sm = raw[mid_i][6]  # soma_mask (bool over vert_indices)
    soma_seed_vi = mid_vi[mid_sm] if mid_sm is not None else np.array([], dtype=np.intp)

    return {
        "center": center,
        "z_range": (z_lo, z_hi),
        "peak_r": peak_r,
        "slices": slices_numeric,
        "contours": contours,
        "soma_seed_vi": soma_seed_vi,
    }


#  find_soma_via_z_contour — soma detection using Z-slice contours
# =====================================================================


def _z_scan(
    verts: np.ndarray,
    grid_res: float = 200.0,
    z_tol: float = 150.0,
    min_void_r: float = 500.0,
    max_shift: float = 3000.0,
    retry_broken: bool = True,
) -> tuple[list[tuple], list[int] | None, np.ndarray | None]:
    """Shared Z-scan: cluster + void detection + spatial coherence.

    Two-pass approach:

    1. Per-Z independent clustering + void detection → find best chain.
    2. Hull-guided void detection for Z-levels where the soma ring has
       holes (mesh artifacts / missing faces).  The gap breaks the
       4-ray enclosure test, so the void goes undetected.  Pass 2
       borrows the convex hull from the nearest good neighbor to close
       the gap and recover the void.

    Returns ``(raw, best_run, nucleus_center)`` where *raw* is the
    per-Z-level data ``[(z, cx, cy, void_r, void_xy, vert_indices,
    soma_mask), ...]``, *best_run* is the indices into *raw* of the
    nucleus chain (or None), and *nucleus_center* is ``(3,)`` or None.
    """
    from scipy.ndimage import binary_dilation, label
    from scipy.spatial import ConvexHull

    z_unique = np.unique(verts[:, 2])
    z_step = 210.0
    z_levels = np.arange(z_unique.min() + z_step, z_unique.max() - z_step, z_step)

    all_vi = np.arange(len(verts))
    raw: list[tuple] = []
    # Each entry: (z, cx, cy, void_r, void_xy, vert_indices, soma_mask)
    #   vert_indices: indices into mesh.vertices for verts near this Z
    #   soma_mask: bool array over vert_indices, True = in soma cluster

    # --- Pass 1: independent per-Z detection ---
    # Also store the grid data for pass 2 re-processing.
    grid_data: list[tuple | None] = []  # (occ, ix, iy, xy_min, labeled)

    for z in z_levels:
        near_z = np.abs(verts[:, 2] - z) < z_tol
        vi = all_vi[near_z]
        pts = verts[vi, :2]

        cluster = _z_slice_cluster(pts, grid_res)
        if cluster is None:
            raw.append((z, np.nan, np.nan, 0.0, None, vi, None))
            grid_data.append(None)
            continue

        occ, soma_region, ix, iy, xy_min = cluster
        soma_mask = soma_region[ix, iy]

        # Store labeled grid for pass 2
        struct3 = np.ones((3, 3), dtype=bool)
        dilated = binary_dilation(occ, struct3)
        labeled, _ = label(dilated)
        grid_data.append((occ, ix, iy, xy_min, labeled))

        void_result = _z_slice_void(occ, soma_region, xy_min, grid_res)
        if void_result is None:
            raw.append((z, np.nan, np.nan, 0.0, None, vi, soma_mask))
        else:
            cx, cy, void_r, void_xy = void_result
            raw.append((z, cx, cy, void_r, void_xy, vi, soma_mask))

    # Spatial-coherence chains
    runs: list[list[int]] = []
    cur: list[int] = []
    for i, entry in enumerate(raw):
        z, cx, cy, r = entry[0], entry[1], entry[2], entry[3]
        if r < min_void_r or np.isnan(cx):
            if cur:
                runs.append(cur)
                cur = []
            continue
        if not cur:
            cur = [i]
        else:
            pcx, pcy = raw[cur[-1]][1], raw[cur[-1]][2]
            if np.sqrt((cx - pcx) ** 2 + (cy - pcy) ** 2) < max_shift:
                cur.append(i)
            else:
                runs.append(cur)
                cur = [i]
    if cur:
        runs.append(cur)

    runs = [r for r in runs if len(r) >= 2]

    if not runs:
        return raw, None, None

    best = max(runs, key=len)

    if not retry_broken:
        slices = np.array([(raw[i][0], raw[i][1], raw[i][2], raw[i][3]) for i in best])
        center = np.array(
            [np.nanmean(slices[:, 1]), np.nanmean(slices[:, 2]), np.mean(slices[:, 0])]
        )
        return raw, best, center

    # --- Pass 2: hull-guided void detection for broken Z-levels ---
    # When the soma ring has a gap (surface hole), the 4-ray enclosure
    # and binary_fill_holes both fail.  Use the convex hull from the
    # nearest good neighbor to define "inside" at broken levels.
    best_set = set(best)
    chain_lo, chain_hi = best[0], best[-1]

    # Scan levels adjacent to the chain in both directions
    retry_indices = []
    for i in range(chain_lo - 1, -1, -1):
        if grid_data[i] is None:
            break
        retry_indices.append(i)
    for i in range(chain_hi + 1, len(raw)):
        if grid_data[i] is None:
            break
        retry_indices.append(i)

    for i in retry_indices:
        if i in best_set:
            continue
        r_existing = raw[i][3]
        if r_existing >= min_void_r and not np.isnan(raw[i][1]):
            continue  # already good

        gd = grid_data[i]
        if gd is None:
            continue
        occ, ix, iy, xy_min, labeled = gd

        # Find nearest good neighbor in the chain
        ref_i = chain_lo if i < chain_lo else chain_hi

        # Build convex hull from the reference soma vertices
        ref_vi = raw[ref_i][5]
        ref_sm = raw[ref_i][6]
        if ref_sm is None:
            continue
        ref_pts = verts[ref_vi[ref_sm], :2]
        if len(ref_pts) < 4:
            continue
        try:
            ref_hull = ConvexHull(ref_pts)
        except Exception:
            continue

        # Build hull mask on this level's grid: for each grid cell,
        # test if its centre falls inside the reference convex hull.
        nx, ny = occ.shape
        gi, gj = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
        gx = xy_min[0] + gi * grid_res + grid_res / 2
        gy = xy_min[1] + gj * grid_res + grid_res / 2
        grid_pts = np.column_stack([gx.ravel(), gy.ravel()])
        inside_hull = np.ones(len(grid_pts), dtype=bool)
        for eq in ref_hull.equations:
            inside_hull &= grid_pts @ eq[:-1] + eq[-1] <= 0
        hull_mask = inside_hull.reshape(nx, ny)

        # Merge all clusters whose centroid is inside the hull
        n_labels = labeled.max()
        merged_region = np.zeros_like(occ)
        for lid in range(1, n_labels + 1):
            cij = np.argwhere(labeled == lid)
            centroid = np.array(
                [
                    xy_min[0] + cij[:, 0].mean() * grid_res + grid_res / 2,
                    xy_min[1] + cij[:, 1].mean() * grid_res + grid_res / 2,
                ]
            )
            if all(
                eq[:-1] @ centroid + eq[-1] <= grid_res for eq in ref_hull.equations
            ):
                merged_region |= labeled == lid
        if not merged_region.any():
            continue
        soma_mask_new = merged_region[ix, iy]

        # Hull-guided void detection: use hull_mask instead of
        # binary_fill_holes to define the interior.
        from scipy.ndimage import distance_transform_edt as _edt
        from scipy.ndimage import label as _label

        occ_h = occ & hull_mask
        enc = np.ones_like(occ_h)
        for ax in range(2):
            enc &= np.maximum.accumulate(occ_h, axis=ax)
            enc &= np.flip(
                np.maximum.accumulate(np.flip(occ_h, axis=ax), axis=ax), axis=ax
            )
        enc &= hull_mask
        void = enc & ~occ_h
        if not void.any():
            continue

        dt = _edt(~occ_h)
        dtv = dt * void
        pk = np.unravel_index(dtv.argmax(), dtv.shape)
        void_r = float(dtv.max()) * grid_res
        cx = float(xy_min[0] + pk[0] * grid_res + grid_res / 2)
        cy = float(xy_min[1] + pk[1] * grid_res + grid_res / 2)

        deep_void = void & (dt > 2)
        void_xy = None
        if deep_void.any():
            vl, _ = _label(deep_void)
            pl = vl[pk[0], pk[1]]
            if pl == 0:
                dij = np.argwhere(deep_void)
                dd = np.abs(dij[:, 0] - pk[0]) + np.abs(dij[:, 1] - pk[1])
                pl = vl[dij[dd.argmin()][0], dij[dd.argmin()][1]]
            nv = vl == pl
            vij = np.argwhere(nv)
            void_xy = np.column_stack(
                [
                    xy_min[0] + vij[:, 0] * grid_res + grid_res / 2,
                    xy_min[1] + vij[:, 1] * grid_res + grid_res / 2,
                ]
            )

        z = raw[i][0]
        vi = raw[i][5]
        raw[i] = (z, cx, cy, void_r, void_xy, vi, soma_mask_new)

    # Re-run spatial coherence with potentially new void detections
    runs2: list[list[int]] = []
    cur2: list[int] = []
    for i, entry in enumerate(raw):
        z, cx, cy, r = entry[0], entry[1], entry[2], entry[3]
        if r < min_void_r or np.isnan(cx):
            if cur2:
                runs2.append(cur2)
                cur2 = []
            continue
        if not cur2:
            cur2 = [i]
        else:
            pcx, pcy = raw[cur2[-1]][1], raw[cur2[-1]][2]
            if np.sqrt((cx - pcx) ** 2 + (cy - pcy) ** 2) < max_shift:
                cur2.append(i)
            else:
                runs2.append(cur2)
                cur2 = [i]
    if cur2:
        runs2.append(cur2)

    runs2 = [r for r in runs2 if len(r) >= 2]
    if runs2:
        best = max(runs2, key=len)

    slices = np.array([(raw[i][0], raw[i][1], raw[i][2], raw[i][3]) for i in best])
    center = np.array(
        [
            np.nanmean(slices[:, 1]),
            np.nanmean(slices[:, 2]),
            np.mean(slices[:, 0]),
        ]
    )

    return raw, best, center


def _soma_hulls(
    raw: list[tuple],
    best: list[int],
    center: np.ndarray,
    verts: np.ndarray,
    grid_res: float = 200.0,
) -> tuple[dict, bool, bool]:
    """Build per-Z soma hulls with neurite protrusions stripped.

    For each Z-level with a soma cluster, rasterizes the soma vertices,
    fills the ring interior to form a solid disk, then applies
    morphological opening to remove thin structures (neurite
    protrusions).  The convex hull is built from the surviving
    (soma-only) grid cells.

    Returns ``(hulls, extend_lo, extend_hi)`` where *hulls* is a dict
    mapping raw-index → Shapely Polygon, and *extend_lo* / *extend_hi*
    indicate whether the soma extends to the mesh boundary on each side
    (True = nothing beyond the soma, extend to mesh edge; False = soma
    is bounded by neurites, don't extend).
    """
    from scipy.ndimage import (
        binary_dilation,
        binary_erosion,
        binary_fill_holes,
        distance_transform_edt,
    )
    from scipy.spatial import ConvexHull
    from shapely.geometry import Polygon as _Poly

    soma_levels = [
        i for i in range(len(raw)) if raw[i][6] is not None and raw[i][6].sum() >= 4
    ]

    hulls: dict[int, _Poly] = {}
    struct3 = np.ones((3, 3), dtype=bool)

    for i in soma_levels:
        sm = raw[i][6]
        vi = raw[i][5]
        soma_pts = verts[vi[sm], :2]
        if len(soma_pts) < 4:
            continue

        # Rasterize soma vertices
        xy_min = soma_pts.min(axis=0) - grid_res * 3
        nx = int((soma_pts[:, 0].max() - xy_min[0] + grid_res * 6) / grid_res) + 1
        ny = int((soma_pts[:, 1].max() - xy_min[1] + grid_res * 6) / grid_res) + 1
        occ = np.zeros((nx, ny), dtype=bool)
        ix = ((soma_pts[:, 0] - xy_min[0]) / grid_res).astype(int).clip(0, nx - 1)
        iy = ((soma_pts[:, 1] - xy_min[1]) / grid_res).astype(int).clip(0, ny - 1)
        occ[ix, iy] = True

        # Close ring gaps → fill interior → open to strip neurites.
        closed = binary_dilation(occ, struct3, iterations=2)
        closed = binary_erosion(closed, struct3, iterations=2)
        filled = binary_fill_holes(closed)
        dt = distance_transform_edt(filled)
        peak_dist = dt.max()

        if peak_dist >= 6:
            r = max(int(peak_dist * 0.3), 3)
            se = np.zeros((2 * r + 1, 2 * r + 1), dtype=bool)
            yy, xx = np.ogrid[-r : r + 1, -r : r + 1]
            se[xx**2 + yy**2 <= r**2] = True
            opened = binary_erosion(filled, se)
            opened = binary_dilation(opened, se)
            if opened.any():
                # Build hull from opened grid cell centres
                gi, gj = np.where(opened)
                hull_pts = np.column_stack(
                    [
                        xy_min[0] + gi * grid_res + grid_res / 2,
                        xy_min[1] + gj * grid_res + grid_res / 2,
                    ]
                )
                try:
                    h = ConvexHull(hull_pts)
                    p = _Poly(hull_pts[h.vertices])
                    if p.is_valid:
                        hulls[i] = p
                        continue
                except Exception:
                    pass

        # Fallback: raw convex hull (opening too small or failed)
        try:
            h = ConvexHull(soma_pts)
            p = _Poly(soma_pts[h.vertices])
            if p.is_valid:
                hulls[i] = p
        except Exception:
            pass

    # Walk outward from the nucleus chain in both Z-directions.
    # Stop when the hull area drops well below the peak (soma is
    # tapering into neurites).  If the area is still large at the
    # mesh boundary, include everything — the soma is cut off by
    # the segmentation volume, not by a natural taper.
    chain_areas = [hulls[i].area for i in best if i in hulls]
    peak_area = max(chain_areas) if chain_areas else 0
    max_r = np.sqrt(peak_area / np.pi) if peak_area > 0 else 5000
    max_dist = max_r * 1.5
    # Collect all levels with hulls, sorted by index
    hull_levels = sorted(hulls.keys())

    # Find the chain's position in hull_levels
    chain_set = set(best)
    chain_positions = [pos for pos, i in enumerate(hull_levels) if i in chain_set]
    if not chain_positions:
        return {}, False, False
    mid_lo = min(chain_positions)
    mid_hi = max(chain_positions)

    def _walk(cutoff):
        """Walk outward from the chain, return (accepted, ext_lo, ext_hi)."""
        acc = set()
        for pos in range(mid_lo, mid_hi + 1):
            acc.add(hull_levels[pos])

        e_lo = True
        for pos in range(mid_lo - 1, -1, -1):
            i = hull_levels[pos]
            p = hulls[i]
            if p.area < cutoff:
                e_lo = False
                break
            cx, cy = p.centroid.x, p.centroid.y
            dist = np.sqrt((cx - center[0]) ** 2 + (cy - center[1]) ** 2)
            if dist > max_dist:
                e_lo = False
                break
            acc.add(i)

        e_hi = True
        for pos in range(mid_hi + 1, len(hull_levels)):
            i = hull_levels[pos]
            p = hulls[i]
            if p.area < cutoff:
                e_hi = False
                break
            cx, cy = p.centroid.x, p.centroid.y
            dist = np.sqrt((cx - center[0]) ** 2 + (cy - center[1]) ** 2)
            if dist > max_dist:
                e_hi = False
                break
            acc.add(i)

        return acc, e_lo, e_hi

    # Two area cutoffs derived from the data:
    # - Low cutoff (5% of peak): for sides where the soma extends to
    #   the mesh boundary — keeps levels that are still clearly soma.
    # - High cutoff (50% of min chain area): for sides bounded by
    #   neurites — stops before neurite-sized levels.
    cutoff_lo = peak_area * 0.05
    min_chain_area = min(chain_areas) if chain_areas else peak_area
    cutoff_hi = min_chain_area * 0.5

    # First walk with low cutoff to determine which sides are bounded.
    acc_probe, extend_lo, extend_hi = _walk(cutoff_lo)

    # Override: if the soma walk reached close to the mesh boundary,
    # the soma is at the edge of the segmentation volume — extend
    # regardless of why the walk stopped.
    mesh_z_lo = float(verts[:, 2].min())
    mesh_z_hi = float(verts[:, 2].max())
    z_step = 210.0
    if not extend_lo and acc_probe:
        lo_z = min(raw[i][0] for i in acc_probe)
        if lo_z - mesh_z_lo < z_step * 3:
            extend_lo = True
    if not extend_hi and acc_probe:
        hi_z = max(raw[i][0] for i in acc_probe)
        if mesh_z_hi - hi_z < z_step * 3:
            extend_hi = True

    # Second walk: use tight cutoff for bounded sides, low for unbounded.
    # If both sides have the same cutoff, one walk suffices.
    if extend_lo == extend_hi:
        cutoff = cutoff_lo if extend_lo else cutoff_hi
        accepted, _, _ = _walk(cutoff)
    else:
        # Different cutoffs per side — walk with tight cutoff, then
        # re-extend the unbounded side with the low cutoff.
        accepted, _, _ = _walk(cutoff_hi)
        # Re-walk the unbounded side with the low cutoff
        if extend_lo:
            for pos in range(mid_lo - 1, -1, -1):
                i = hull_levels[pos]
                p = hulls[i]
                if p.area < cutoff_lo:
                    break
                cx, cy = p.centroid.x, p.centroid.y
                dist = np.sqrt((cx - center[0]) ** 2 + (cy - center[1]) ** 2)
                if dist > max_dist:
                    break
                accepted.add(i)
        if extend_hi:
            for pos in range(mid_hi + 1, len(hull_levels)):
                i = hull_levels[pos]
                p = hulls[i]
                if p.area < cutoff_lo:
                    break
                cx, cy = p.centroid.x, p.centroid.y
                dist = np.sqrt((cx - center[0]) ** 2 + (cy - center[1]) ** 2)
                if dist > max_dist:
                    break
                accepted.add(i)

    return {i: hulls[i] for i in accepted}, extend_lo, extend_hi


def find_soma_via_z_contour(
    mesh: trimesh.Trimesh,
    *,
    organelles: Organelles | None = None,
    grid_res: float = 200.0,
    z_tol: float = 150.0,
    verbose: bool = False,
) -> "Soma | None":
    """Detect the soma by first locating the nucleus void.

    .. deprecated::
        Prefer :func:`find_soma_via_ring_cutoff`, which is the promoted
        soma detection method.

    Uses :func:`_z_scan` (fast) to find the nucleus, then builds
    per-Z soma hulls with neurite protrusions stripped via
    morphological opening.  Classifies faces whose centroids fall
    inside the per-Z hulls as soma.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Input neuron mesh.
    organelles : Organelles or None
        Pre-computed organelles from :func:`find_organelles`.  If
        provided, organelle faces are excluded from the soma vertex set.
    grid_res : float
        2D rasterization cell size in nm.
    z_tol : float
        Half-width of the Z slab.
    verbose : bool
        Print progress.

    Returns
    -------
    Soma or None
    """
    warnings.warn(
        "find_soma_via_z_contour is deprecated, use find_soma_via_ring_cutoff instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    from shapely.geometry import Point

    _p = "[find_soma_via_z_contour]"

    verts = np.asarray(mesh.vertices, dtype=np.float64)
    if verbose:
        print(f"{_p} {len(verts):,} vertices")

    raw, best, nc = _z_scan(verts, grid_res, z_tol)

    if nc is None:
        if verbose:
            print(f"{_p} no nucleus found")
        return None

    if verbose:
        n_chain = len(best)
        peak_r = max(raw[i][3] for i in best)
        print(
            f"{_p} nucleus: ({nc[0]:.0f}, {nc[1]:.0f}, {nc[2]:.0f}), "
            f"r={peak_r:.0f}nm, {n_chain} Z-levels"
        )

    soma_hulls, extend_lo, extend_hi = _soma_hulls(raw, best, nc, verts, grid_res)

    from shapely import prepare

    hull_polys: list[tuple[float, object]] = []
    for i in sorted(soma_hulls.keys()):
        poly = soma_hulls[i]
        if poly is None or poly.is_empty:
            continue
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_valid:
            prepare(poly)
            hull_polys.append((raw[i][0], poly))

    if not hull_polys:
        if verbose:
            print(f"{_p} no soma contours found")
        return None

    # Small expansion to account for grid-resolution tolerance.
    # The hull is built from grid cell centres (grid_res=200nm), so
    # surface faces up to ~1 cell away may be just outside.
    expanded_polys: list[tuple[float, object]] = []
    for z_val, poly in hull_polys:
        ep = poly.buffer(grid_res)
        if not ep.is_valid:
            ep = ep.buffer(0)
        prepare(ep)
        expanded_polys.append((z_val, ep))

    hull_z = np.array([h[0] for h in expanded_polys])
    z_lo, z_hi = hull_z.min(), hull_z.max()
    n_z_hit = len(expanded_polys)

    # Extend the classification range to the mesh boundary on sides
    # where the soma walk didn't encounter a natural taper (nothing
    # beyond the soma — it's cut off by the segmentation volume).
    # Faces beyond the last hull are tested against the nearest hull.
    if extend_lo:
        z_lo = float(verts[:, 2].min())
    if extend_hi:
        z_hi = float(verts[:, 2].max())

    # Classify faces: centroid inside the nearest Z-level's contour → soma.
    faces = np.asarray(mesh.faces)
    centroids = verts[faces].mean(axis=1)  # (n_faces, 3)

    z_in_range = (centroids[:, 2] >= z_lo) & (centroids[:, 2] <= z_hi)
    cand_fi = np.where(z_in_range)[0]

    soma_face = np.zeros(len(faces), dtype=bool)
    cand_z = centroids[cand_fi, 2]
    nearest_idx = np.searchsorted(hull_z, cand_z).clip(0, len(hull_z) - 1)

    # Check nearest and the one before (searchsorted gives insertion point)
    for offset in [0, -1]:
        idx = (nearest_idx + offset).clip(0, len(hull_z) - 1)
        for hi in range(n_z_hit):
            mask = idx == hi
            if not mask.any():
                continue
            fi_batch = cand_fi[mask]
            pts_xy = centroids[fi_batch, :2]
            inside = np.array(
                [expanded_polys[hi][1].contains(Point(p)) for p in pts_xy]
            )
            soma_face[fi_batch] |= inside

    # Exclude organelle faces from soma
    if organelles is not None:
        soma_face &= ~organelles.mask

    # Keep only the largest connected component of soma faces
    from collections import deque

    adj = _face_adjacency(mesh)
    soma_idx = set(np.where(soma_face)[0].tolist())
    visited: set[int] = set()
    largest_comp: list[int] = []
    for fi in soma_idx:
        if fi in visited:
            continue
        comp: list[int] = []
        q = deque([fi])
        while q:
            c = q.popleft()
            if c in visited:
                continue
            visited.add(c)
            comp.append(c)
            for nfi in adj.get(c, set()):
                if nfi in soma_idx and nfi not in visited:
                    q.append(nfi)
        if len(comp) > len(largest_comp):
            largest_comp = comp
    soma_face[:] = False
    for fi in largest_comp:
        soma_face[fi] = True

    n_soma_faces = int(soma_face.sum())
    if n_soma_faces < 4:
        if verbose:
            print(f"{_p} too few soma faces ({n_soma_faces})")
        return None

    # Soma vertices = all vertices of soma faces
    soma_vert_set = set(faces[soma_face].ravel().tolist())
    soma_arr = np.fromiter(sorted(soma_vert_set), dtype=np.intp)
    soma = Soma.fit(mesh.vertices[soma_arr], verts=soma_arr)

    # Build nucleus dict from _z_scan results
    nuc_slices = np.array([(raw[i][0], raw[i][1], raw[i][2], raw[i][3]) for i in best])
    soma.nucleus = {
        "center": nc,
        "peak_r": float(nuc_slices[:, 3].max()),
        "z_range": (float(nuc_slices[0, 0]), float(nuc_slices[-1, 0])),
        "slices": nuc_slices,
    }

    if verbose:
        print(
            f"{_p} soma: {int(soma_face.sum()):,} faces, "
            f"{len(soma.verts):,} verts from {n_z_hit} Z-levels, "
            f"center=[{soma.center[0]:.0f}, {soma.center[1]:.0f}, "
            f"{soma.center[2]:.0f}], "
            f"axes=[{soma.axes[0]:.0f}, {soma.axes[1]:.0f}, "
            f"{soma.axes[2]:.0f}]"
        )

    return soma


# Backward-compat wrapper — will be removed in a future release.


def _deprecated_alias(old_name: str, new_func):
    import functools
    import warnings

    @functools.wraps(new_func)
    def wrapper(*args, **kwargs):
        warnings.warn(
            f"{old_name}() is deprecated, use {new_func.__name__}() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return new_func(*args, **kwargs)

    return wrapper


find_soma = _deprecated_alias("find_soma", find_soma_via_neurite_exclusion)
