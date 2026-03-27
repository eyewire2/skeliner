"""skeliner.pre – mesh preprocessing utilities."""

from collections import defaultdict

import igraph as ig
import numpy as np
import trimesh
from scipy.spatial import KDTree

from skeliner.dataclass import Soma

__all__ = [
    "compact_mesh",
    "compute_mesh_stats",
    "fill_holes",
    "find_disconnected",
    "find_gaps",
    "find_holes",
    "find_nucleus_center",
    "find_offsets",
    "find_soma",
    "find_soma_from_nucleus",
    "find_soma_void",
    "preprocess",
    "PreprocessResult",
    "remove_fins",
    "remove_fragments",
    "remove_fusions",
    "remove_islands",
    "remove_offsets",
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


def _zipper_stitch(
    mesh: trimesh.Trimesh,
    loop_a: list[int],
    loop_b: list[int],
    vert_to_faces: list[list[int]],
    new_verts: list[np.ndarray] | None = None,
) -> list[list[int]]:
    """Zipper-stitch two boundary loops, returning new triangles.

    When the gap between loops is much larger than the mesh's median
    edge length, intermediate vertex rings are interpolated so that
    the resulting faces match the typical mesh resolution.

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

    pts_a = mesh.vertices[loop_a]
    pts_b = mesh.vertices[loop_b]
    ca = pts_a.mean(axis=0)
    cb = pts_b.mean(axis=0)

    # Align starting points
    tree_b = KDTree(pts_b)
    dists_q, idxs_q = tree_b.query(pts_a)
    start_a = int(np.argmin(dists_q))
    start_b = int(idxs_q[start_a])

    # Orient loops to run in opposite directions
    axis = cb - ca
    axis_len = np.linalg.norm(axis)
    axis = axis / axis_len if axis_len > 1e-10 else np.array([0.0, 0.0, 1.0])

    def _winding(pts_loop, ax):
        c = pts_loop.mean(axis=0)
        total = 0.0
        for k in range(len(pts_loop)):
            e1 = pts_loop[k] - c
            e2 = pts_loop[(k + 1) % len(pts_loop)] - c
            total += float(np.dot(np.cross(e1, e2), ax))
        return total

    la = list(loop_a)
    lb = list(loop_b)
    if _winding(pts_a, axis) * _winding(pts_b, axis) > 0:
        lb = lb[::-1]
        start_b = len(lb) - 1 - start_b

    la = la[start_a:] + la[:start_a]
    lb = lb[start_b:] + lb[:start_b]

    # ── Determine if intermediate rings are needed ────────────────
    gap_dist = float(np.linalg.norm(ca - cb))
    median_edge = float(np.median(mesh.edges_unique_length))
    n_rings = max(0, int(round(gap_dist / median_edge)) - 1)

    if n_rings > 0:
        # Build matched correspondences: resample both loops to the
        # same vertex count, then interpolate intermediate rings.
        n_pts = max(len(la), len(lb))
        pts_la = mesh.vertices[la]
        pts_lb = mesh.vertices[lb]

        # Resample each loop to n_pts evenly spaced points
        def _resample(pts, n):
            """Resample closed loop to *n* evenly spaced points."""
            closed = np.vstack([pts, pts[:1]])
            seg_lens = np.linalg.norm(np.diff(closed, axis=0), axis=1)
            cum = np.concatenate([[0], np.cumsum(seg_lens)])
            total = cum[-1]
            targets = np.linspace(0, total, n, endpoint=False)
            resampled = np.empty((n, 3))
            for i, t in enumerate(targets):
                idx = np.searchsorted(cum, t, side="right") - 1
                idx = min(idx, len(pts) - 1)
                frac = (t - cum[idx]) / max(seg_lens[idx], 1e-10)
                nxt = (idx + 1) % len(pts)
                resampled[i] = pts[idx] * (1 - frac) + pts[nxt] * frac
            return resampled

        ring_a = _resample(pts_la, n_pts)
        ring_b = _resample(pts_lb, n_pts)

        # Create intermediate rings as new vertices
        n_existing = len(mesh.vertices)
        rings: list[list[int]] = []  # each ring is a list of vert indices
        rings.append(la)  # ring 0 = loop_a (existing verts)
        for r in range(1, n_rings + 1):
            t = r / (n_rings + 1)
            ring_pts = ring_a * (1 - t) + ring_b * t
            ring_ids = []
            for pt in ring_pts:
                ring_ids.append(n_existing + len(new_verts_local))
                new_verts_local.append(pt)
            rings.append(ring_ids)
        rings.append(lb)  # last ring = loop_b (existing verts)

        # Stitch consecutive rings
        triangles: list[list[int]] = []

        def _vpos(vid):
            if vid < n_existing:
                return mesh.vertices[vid]
            return new_verts_local[vid - n_existing]

        for ri in range(len(rings) - 1):
            ra_ids = rings[ri]
            rb_ids = rings[ri + 1]
            na_r, nb_r = len(ra_ids), len(rb_ids)
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
    else:
        # Direct zipper (gap is small enough)
        na, nb = len(la), len(lb)
        triangles: list[list[int]] = []
        ia, ib, steps_a, steps_b = 0, 0, 0, 0

        while steps_a < na or steps_b < nb:
            ia_next = (ia + 1) % na
            ib_next = (ib + 1) % nb
            can_a, can_b = steps_a < na, steps_b < nb

            if can_a and can_b:
                da = float(
                    np.linalg.norm(mesh.vertices[la[ia_next]] - mesh.vertices[lb[ib]])
                )
                db = float(
                    np.linalg.norm(mesh.vertices[la[ia]] - mesh.vertices[lb[ib_next]])
                )
                advance_a = da <= db
            else:
                advance_a = can_a

            if advance_a:
                triangles.append([la[ia], la[ia_next], lb[ib]])
                ia = ia_next
                steps_a += 1
            else:
                triangles.append([la[ia], lb[ib_next], lb[ib]])
                ib = ib_next
                steps_b += 1

    # Orient consistently with surrounding mesh
    ref_fis: list[int] = []
    for vi in la[:5]:
        ref_fis.extend(vert_to_faces[vi][:5])
    if ref_fis:
        ref_n = mesh.face_normals[ref_fis].mean(axis=0)

        def _vpos(vid):
            if vid < len(mesh.vertices):
                return mesh.vertices[vid]
            return new_verts_local[vid - len(mesh.vertices)]

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
    sel = set(face_indices)
    if not sel:
        return mesh

    # Edge-to-face map
    edge_to_faces = _edge_to_faces(mesh)

    if verbose:
        print(f"[skeliner.pre] Merge: removing {len(sel)} faces")

    loops = _trace_border_loops(mesh, sel, edge_to_faces)

    if verbose:
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
        pairs.append((loops[available[best_i]], loops[available[best_j]]))
        available.pop(best_j)
        available.pop(best_i)

    if verbose:
        print(f"[skeliner.pre] Stitching {len(pairs)} loop pair(s) ...")

    return _stitch_and_rebuild(mesh, sel, pairs, verbose=verbose)


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
    from collections import deque

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


def find_soma_deprecated(
    mesh: trimesh.Trimesh,
    *,
    organelles: np.ndarray | None = None,
    verbose: bool = False,
    mesh_stats: tuple | None = None,
) -> Soma | None:
    """**Deprecated** — use :func:`find_soma` instead.

    Original soma detection via BFS + Otsu ring cutoff.  Replaced by
    the per-tip neurite exclusion approach which handles branching
    neurites and non-monotonic tube widths.

    Organelles (mitochondria, ER, nucleus membrane) are densely packed
    inside the soma.  Their centroids cluster tightly in 3-D, giving a
    robust soma centre.  A BFS flood on the main-component surface
    grows outward from that centre; the soma boundary is where the
    largest connected ring component peaks and drops (Otsu).

    All internal thresholds are derived from the mesh data.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Input mesh (before organelle removal).
    organelles : np.ndarray or None
        Pre-computed boolean face mask from :func:`find_organelles`
        (``pocket | isolated``).  If provided, organelle detection is
        skipped and this mask is used directly.
    verbose : bool, default False
        Print summary.
    mesh_stats : tuple or None
        ``(outward_dots, face_comp, main_ci, main_mask)`` from
        :func:`compute_mesh_stats`.  Reuses ``face_comp`` and
        ``main_ci`` to skip redundant component detection.

    Returns
    -------
    Soma or None
        Fitted ellipsoidal soma, or *None* if too few indicators exist.
    """
    from collections import deque

    if mesh_stats is not None:
        _, labels, main, _ = mesh_stats
    else:
        labels, main = _face_edge_components(mesh)

    # ── 1. Locate organelle clusters and compute centroids ──────
    if organelles is None:
        pocket, isolated = find_organelles(mesh, verbose=verbose)
        organelles = pocket | isolated
    if organelles.sum() == 0:
        if verbose:
            print("[skeliner.pre] Soma: no organelles found")
        return None

    org_fi = np.where(organelles)[0]
    org_labels, _ = _face_edge_components(
        trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces[org_fi], process=False)
    )
    centroids = []
    for cid in np.unique(org_labels):
        local_fi = org_fi[org_labels == cid]
        verts = np.unique(mesh.faces[local_fi])
        centroids.append(mesh.vertices[verts].mean(axis=0))

    centroids = np.asarray(centroids)
    if verbose:
        print(
            f"[skeliner.pre] Soma: {organelles.sum():,} organelle faces, "
            f"{len(centroids)} clusters"
        )
    if len(centroids) < 3:
        if verbose:
            print("[skeliner.pre] Soma: too few organelle clusters")
        return None

    # ── 1b. Find the dense cluster of fragments (soma) ──────────
    #        Build a proximity graph: Otsu on nearest-neighbour
    #        distances gives a data-driven radius.  The largest
    #        connected component is the soma fragment cluster.

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
            f"[skeliner.pre] Soma: dense cluster {len(core)}/{len(centroids)} "
            f"fragments, nn_thresh={nn_thresh:.0f}"
        )

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

    # BFS adjacency: external surface only (exclude organelle faces).
    # Organelle pockets inside the soma create topological shortcuts
    # that fragment ring components and cause premature cutoff.
    adj_bfs: dict[int, list[int]] = defaultdict(list)
    non_org_main_fi = main_fi[~organelles[main_fi]]
    for fi in non_org_main_fi:
        v = mesh.faces[fi]
        for i in range(3):
            a, b = int(v[i]), int(v[(i + 1) % 3])
            adj_bfs[a].append(b)
            adj_bfs[b].append(a)

    # ── 3. BFS from nearest vertex to centre (no hard cap) ───────
    if verbose:
        print("[skeliner.pre] Soma: BFS ring analysis...")
    bfs_verts = np.fromiter(adj_bfs.keys(), dtype=np.intp)
    seed = int(
        bfs_verts[
            np.argmin(np.linalg.norm(mesh.vertices[bfs_verts] - center, axis=1))
        ]
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
                for nv in adj_bfs[v]:
                    if nv in ring_set and nv not in visited_ring:
                        rq.append(nv)
            if size > max_comp:
                max_comp = size
        largest_comp_size[lv] = max_comp

    peak_ring = int(np.argmax(largest_comp_size))
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

    # ── 4b. Validate: is the wide region a localized bulge (soma)?
    #        Otsu splits ring component sizes into wide vs narrow.
    #        spread_ratio = std(wide ring positions) / std(all ring positions).
    #        For a soma this is small (wide rings concentrate at one spot).
    #        For a uniform-width cell it approaches 1.0.
    #        Tested on 28 cells (BCs + SACs): soma 0.03–0.21, no-soma 0.46–1.04.
    nonzero_mask = largest_comp_size > 0
    nonzero = largest_comp_size[nonzero_mask]
    spread_ratio = 1.0  # default: assume no soma
    if len(nonzero) >= 3:
        width_thresh, _ = _otsu_threshold(nonzero)
        above_indices = np.where(nonzero_mask & (largest_comp_size > width_thresh))[
            0
        ].astype(float)
        all_indices = np.where(nonzero_mask)[0].astype(float)

        if len(above_indices) >= 2 and np.std(all_indices) > 0:
            spread_ratio = float(np.std(above_indices) / np.std(all_indices))

    # Reject if spread_ratio is too high (wide rings not localized).
    # Use Otsu on [spread_ratio, 1.0 - spread_ratio] to decide:
    # if spread_ratio is closer to 0 than to 1, it's concentrated.
    if verbose:
        print(
            f"[skeliner.pre] Soma: peak ring {peak_ring}, "
            f"cutoff ring {cutoff}/{max_ring}, "
            f"spread_ratio={spread_ratio:.3f}"
        )
    if spread_ratio > 1.0 / 3:
        if verbose:
            print(
                f"[skeliner.pre] Soma: no localized bulge "
                f"(spread_ratio={spread_ratio:.3f})"
            )
        return None

    # ── 5. Fit ellipsoid from BFS ring vertices ────────────────────
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
            (initial_soma._body_coords(mesh.vertices[soma_arr]) ** 2)
            .sum(axis=1)
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
                ns_thresh, _ = _otsu_threshold(
                    np.array(ns_sizes, dtype=np.float64)
                )
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
                        nv in neurite_tree
                        for v in comp
                        for nv in adj.get(v, [])
                    )
                    # How far does it extend in body coords?
                    comp_body = initial_soma._body_coords(
                        mesh.vertices[np.array(comp)]
                    )
                    max_ext = float(
                        np.sqrt((comp_body**2).sum(axis=1)).max()
                    )
                    outside_comps.append(
                        (comp, borders, max_ext)
                    )

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
                            if (nv in soma_main
                                    and nv not in visited_er
                                    and nv not in global_visited):
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
                                if (nu in next_ring
                                        and nu not in vis_r):
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

    if verbose:
        print(
            f"[skeliner.pre] Soma: center=["
            f"{soma.center[0]:.0f}, {soma.center[1]:.0f}, "
            f"{soma.center[2]:.0f}], "
            f"axes=[{soma.axes[0]:.0f}, {soma.axes[1]:.0f}, "
            f"{soma.axes[2]:.0f}], "
            f"{len(soma.verts):,} surface verts "
            f"({len(core)}/{len(centroids)} indicator fragments, "
            f"cutoff ring {cutoff})"
        )

    return soma


def find_soma(
    mesh: trimesh.Trimesh,
    *,
    organelles: np.ndarray | None = None,
    verbose: bool = False,
    mesh_stats: tuple | None = None,
) -> Soma | None:
    """Estimate soma by per-tip neurite exclusion.

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
    from collections import deque

    if mesh_stats is not None:
        _, labels, main, _ = mesh_stats
    else:
        labels, main = _face_edge_components(mesh)

    # ── 0. Organelle clustering → soma center ─────────────────────
    if organelles is None:
        pocket, isolated = find_organelles(mesh, verbose=verbose)
        organelles = pocket | isolated
    if organelles.sum() == 0:
        if verbose:
            print("[skeliner.pre] Soma: no organelles found")
        return None

    org_fi = np.where(organelles)[0]
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
            f"[skeliner.pre] Soma: {organelles.sum():,} organelle faces, "
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
            f"[skeliner.pre] Soma: dense cluster "
            f"{len(core)}/{len(centroids)} fragments"
        )

    # ── 1. BFS on external surface from center ────────────────────
    main_fi = np.where(labels == main)[0]
    non_org_main = main_fi[~organelles[main_fi]]

    adj_bfs: dict[int, list[int]] = defaultdict(list)
    for fi in non_org_main:
        v = mesh.faces[fi]
        for i in range(3):
            a, b = int(v[i]), int(v[(i + 1) % 3])
            adj_bfs[a].append(b)
            adj_bfs[b].append(a)

    bfs_verts = np.fromiter(adj_bfs.keys(), dtype=np.intp)
    seed = int(
        bfs_verts[
            np.argmin(
                np.linalg.norm(mesh.vertices[bfs_verts] - center, axis=1)
            )
        ]
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
        above_idx = np.where(
            nonzero_mask & (largest_comp_size > width_thresh)
        )[0].astype(float)
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
    _SMOOTH_W = 5   # smoothing window for ring-size profile

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
            if (
                0 <= i1 < len(d1)
                and 0 <= i2 < len(d2)
                and d1[i1] > 0
                and d2[i2] > 0
            ):
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
            if all(
                d1[i1 + k] > 0 and d2[i2 + k] > 0 for k in range(sustain)
            ):
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


def find_soma_alt(
    mesh: trimesh.Trimesh,
    *,
    organelles: np.ndarray | None = None,
    verbose: bool = False,
    mesh_stats: tuple | None = None,
) -> Soma | None:
    """Experimental soma detection — geodesic proximity + mass boundary.

    Work-in-progress replacement for :func:`find_soma`.  Key differences:

      0. Organelle clustering uses **geodesic** (surface) proximity via
         multi-source BFS instead of Euclidean NN.  This prevents linking
         organelles across membranes.
      1. BFS from center on non-organelle surface (same as find_soma).
      2. Candidate set boundary determined by **neighborhood mass** of
         soma organelle clusters (Otsu on own_size + neighbor_sizes),
         converted to a BFS ring boundary.  No fixed multiplier.
      3. Per-tip neurite exclusion (same as find_soma).
      4. Fit ellipsoid (same as find_soma).
    """
    from collections import deque

    if mesh_stats is not None:
        _, labels, main, _ = mesh_stats
    else:
        labels, main = _face_edge_components(mesh)

    # ── 0. Organelle clustering via geodesic proximity ───────────
    if organelles is None:
        pocket, isolated = find_organelles(mesh, verbose=verbose)
        organelles = pocket | isolated
    if organelles.sum() == 0:
        if verbose:
            print("[skeliner.pre] Soma-alt: no organelles found")
        return None

    org_fi = np.where(organelles)[0]
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
        clusters.append({
            "verts": verts,
            "centroid": mesh.vertices[list(verts)].mean(axis=0),
            "size": int(len(local_fi)),
        })

    if verbose:
        print(
            f"[skeliner.pre] Soma-alt: {organelles.sum():,} organelle faces, "
            f"{len(clusters)} clusters"
        )
    if len(clusters) < 3:
        if verbose:
            print("[skeliner.pre] Soma-alt: too few organelle clusters")
        return None

    # Build non-organelle surface adjacency
    main_fi = np.where(labels == main)[0]
    non_org_main = main_fi[~organelles[main_fi]]

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
        bfs_verts[
            np.argmin(np.linalg.norm(mesh.vertices[bfs_verts] - center, axis=1))
        ]
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
            np.median(
                [np.linalg.norm(mesh.vertices[v] - center) for v in cluster]
            )
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
            v
            for v in nbr
            if abs(float(np.dot(mesh.vertices[v] - ctr, axis))) < half
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
            if (
                0 <= i1 < len(d1)
                and 0 <= i2 < len(d2)
                and d1[i1] > 0
                and d2[i2] > 0
            ):
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
            if all(
                d1[i1 + k] > 0 and d2[i2 + k] > 0 for k in range(sustain)
            ):
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
    organelles: np.ndarray | None = None,
    mesh_stats: tuple | None = None,
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
        Pre-computed soma from :func:`find_soma`.
    organelles : np.ndarray or None
        Boolean mask ``(nFaces,)`` of organelle faces.  Components that
        are entirely organelle are skipped.
    mesh_stats : tuple or None
        ``(outward_dots, face_comp, main_ci, main_mask)`` from
        :func:`compute_mesh_stats`.  Reuses ``face_comp`` and
        ``main_ci`` to skip redundant component detection.

    Returns
    -------
    list[list[int]]
        Each element is a list of face indices for one disconnected
        component, sorted largest-first.
    """
    if mesh_stats is not None:
        _, labels, main, _ = mesh_stats
    else:
        labels, main = _face_edge_components(mesh)
    n_faces = len(mesh.faces)
    if verbose:
        n_total_comps = int(labels.max()) + 1 if len(labels) else 0
        print(f"[skeliner.pre] Disconnected: {n_total_comps} total components")

    # Locate soma so we can exclude components inside it
    if soma is None:
        soma = find_soma(
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
            if organelles[fis_arr].all():
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
        _, nn_idx = main_tree.query(coords)
        vecs = coords - main_centroids[nn_idx]
        dots = np.einsum("ij,ij->i", vecs, main_normals[nn_idx])
        # Component is enclosed if the majority of vertices are on the
        # inward side (dot < 0)
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
    organelles: np.ndarray | None = None,
    mesh_stats: tuple | None = None,
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
        Pre-computed soma from :func:`find_soma`.
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
    if mesh_stats is not None:
        _, labels, main, _ = mesh_stats
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
    from collections import deque as _deque

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

    # Filter outlier gaps using Otsu on log distances.
    # Only filter when there are enough gaps to form distinct groups
    # and the max distance is at least 5x the median (clear outlier).
    if len(gaps) >= 3:
        dists_arr = np.array([g[2] for g in gaps], dtype=np.float64)
        if dists_arr[-1] > 5.0 * np.median(dists_arr):
            log_dists = np.log(np.maximum(dists_arr, 1.0))
            thresh, _ = _otsu_threshold(log_dists)
            n_before = len(gaps)
            gaps = [g for g in gaps if np.log(max(g[2], 1.0)) <= thresh]
            n_dropped = n_before - len(gaps)
            if verbose and n_dropped:
                print(
                    f"[skeliner.pre] Gaps: dropped {n_dropped} outlier gap(s) "
                    f"(> {np.exp(thresh):.0f}nm)"
                )

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
    organelles: np.ndarray | None = None,
    mesh_stats: tuple | None = None,
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

    # Build edge map once
    edge_to_faces = _edge_to_faces(mesh)

    if verbose:
        print(f"[skeliner.pre] Bridging {len(gaps)} gaps")

    # For each gap, trace its border loops. Only collect faces to
    # remove for gaps that successfully produce a loop pair.
    faces_to_remove: set[int] = set()
    loop_pairs: list[tuple[list[int], list[int]]] = []
    for gap_i, (faces_a, faces_b, dist, *_comp_ids) in enumerate(gaps):
        gap_sel = set(faces_a) | set(faces_b)
        loops = _trace_border_loops(mesh, gap_sel, edge_to_faces)

        if len(loops) < 2:
            if verbose:
                print(
                    f"[skeliner.pre]   Gap {gap_i} (dist={dist:.0f}): "
                    f"only {len(loops)} loop(s), skipping"
                )
            continue

        # Pair the two closest loops for this gap
        centroids = [mesh.vertices[lp].mean(axis=0) for lp in loops]
        best_d = float("inf")
        best_pair = (0, 1)
        for ii in range(len(loops)):
            for jj in range(ii + 1, len(loops)):
                d = float(np.linalg.norm(centroids[ii] - centroids[jj]))
                if d < best_d:
                    best_d = d
                    best_pair = (ii, jj)

        la, lb = loops[best_pair[0]], loops[best_pair[1]]
        loop_pairs.append((la, lb))
        faces_to_remove |= gap_sel

        if verbose:
            print(
                f"[skeliner.pre]   Gap {gap_i} (dist={dist:.0f}): "
                f"{len(la)}v + {len(lb)}v"
            )

    if verbose:
        print(f"[skeliner.pre] Removing {len(faces_to_remove)} tip faces")

    return _stitch_and_rebuild(mesh, faces_to_remove, loop_pairs, verbose=verbose)


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
) -> tuple[np.ndarray, np.ndarray, int, np.ndarray]:
    """Shared precomputation for organelle detection.

    Returns (outward_dots, face_comp, main_ci, main_face_mask).
    """
    if radius is None:
        median_edge = float(np.median(mesh.edges_unique_length))
        radius = radius_multiplier * median_edge
        if verbose:
            print(
                f"[skeliner.pre] Auto radius: {radius:.1f} "
                f"({radius_multiplier}x median edge {median_edge:.1f})"
            )

    # Compute connected components first so _outward_dot can use
    # per-component KDTrees for correct local COM.
    edge_list = set()
    for face in mesh.faces:
        for i in range(3):
            a, b = int(face[i]), int(face[(i + 1) % 3])
            edge_list.add((min(a, b), max(a, b)))

    g = ig.Graph(n=len(mesh.vertices), edges=list(edge_list), directed=False)
    comps = g.connected_components()
    main_ci = max(range(len(comps)), key=lambda i: len(comps[i]))

    vert_comp = np.full(len(mesh.vertices), -1, dtype=np.intp)
    for ci, cl in enumerate(comps):
        for v in cl:
            vert_comp[v] = ci
    face_comp = vert_comp[mesh.faces[:, 0]]
    main_face_mask = face_comp == main_ci

    if verbose:
        n_comps = len(comps)
        n_structural = sum(1 for c in comps if len(c) >= 100)
        print(
            f"[skeliner.pre] Components: {n_comps} total, "
            f"{n_structural} structural (>= 100 verts)"
        )

    outward_dots = _outward_dot(mesh, radius, vert_comp=vert_comp)

    if verbose:
        raw_count = int((outward_dots < 0).sum())
        print(
            f"[skeliner.pre] Raw internal faces: {raw_count:,} "
            f"({100 * raw_count / len(mesh.faces):.1f}%)"
        )

    return outward_dots, face_comp, main_ci, main_face_mask


def _rim_enclosed_area(
    boundary_edges: list[tuple[int, int]],
    vertices: np.ndarray,
) -> float:
    """Compute total enclosed planar area of closed loops in boundary edges.

    For each closed loop: project vertices onto a best-fit plane, then
    compute polygon area via the shoelace formula.
    """
    from collections import defaultdict, deque

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
    from collections import defaultdict, deque

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


def find_rims(
    mesh: trimesh.Trimesh,
    *,
    radius: float | None = None,
    radius_multiplier: float = 5.0,
    min_pocket_size: int = 5,
    min_fold_ratio: float = 3.0,
    verbose: bool = False,
    mesh_stats: tuple | None = None,
    _adj: dict[int, set[int]] | None = None,
    _edge_to_face: dict[tuple[int, int], list[int]] | None = None,
) -> list[list[tuple[int, int]]]:
    """Find rim edges — boundaries of negative-dot face clusters.

    A valid pocket must satisfy:

    1. Multiple boundary loops (not a flat patch with a single outline).
    2. Fold ratio > *min_fold_ratio*: the pocket surface area must be
       much larger than the rim's enclosed planar area, indicating the
       surface folds inward through a small opening.

    Returns
    -------
    list[list[tuple[int, int]]]
        One list of edges per pocket rim.
    """
    from collections import defaultdict, deque

    if mesh_stats is not None:
        outward_dots, _, _, main_face_mask = mesh_stats
    else:
        outward_dots, _, _, main_face_mask = compute_mesh_stats(
            mesh,
            radius,
            radius_multiplier,
            verbose,
        )
    n_faces = len(mesh.faces)
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
    rims: list[list[tuple[int, int]]] = []
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

        # Fold ratio: pocket surface area / rim enclosed planar area.
        # A real pocket folds inward through a small opening (high ratio).
        # A flat patch has ratio near 1.
        pocket_area = float(area_arr[cluster].sum())
        opening_area = _rim_enclosed_area(boundary_edges, verts_arr)
        if opening_area <= 0:
            continue  # no measurable opening = not a pocket entrance
        fold_ratio = pocket_area / opening_area
        if fold_ratio < min_fold_ratio:
            continue

        rims.append(boundary_edges)

    if verbose:
        print(
            f"[skeliner.pre] Rims: {len(rims)} pockets "
            f"(fold >= {min_fold_ratio}), "
            f"{sum(len(r) for r in rims):,} rim edges"
        )

    return rims


def find_pocket_mouths(
    mesh: trimesh.Trimesh,
    *,
    radius: float | None = None,
    radius_multiplier: float = 5.0,
    min_pocket_size: int = 5,
    min_fold_ratio: float = 3.0,
    verbose: bool = False,
    mesh_stats: tuple | None = None,
    _adj: dict[int, set[int]] | None = None,
    _edge_to_face: dict[tuple[int, int], list[int]] | None = None,
) -> list[list[tuple[int, int]]]:
    """Find closed mouth edge-loops of pockets.

    Same boundary-edge detection as *find_rims*, but gaps (degree-1
    endpoints caused by mesh boundary edges) are closed by shortest
    vertex-paths on the mesh surface.

    Returns
    -------
    list[list[tuple[int, int]]]
        One list of edges per pocket mouth (original rim edges +
        closure edges).  Same format as *find_rims*.
    """
    from collections import deque

    if mesh_stats is not None:
        outward_dots, _, _, main_face_mask = mesh_stats
    else:
        outward_dots, _, _, main_face_mask = compute_mesh_stats(
            mesh, radius, radius_multiplier, verbose,
        )
    n_faces = len(mesh.faces)
    edge_to_face = (
        _edge_to_face if _edge_to_face is not None else _edge_to_faces(mesh)
    )
    adj = _adj if _adj is not None else _face_adjacency(mesh, edge_to_face)

    faces_arr = np.asarray(mesh.faces)
    area_arr = np.asarray(mesh.area_faces)
    verts_arr = np.asarray(mesh.vertices)

    # -- neg-dot clusters (same as find_rims) --
    neg_mask = (outward_dots < 0) & main_face_mask
    neg_idx = set(np.where(neg_mask)[0].tolist())
    cluster_vis: set[int] = set()
    clusters: list[list[int]] = []
    for fi in neg_idx:
        if fi in cluster_vis:
            continue
        cluster: list[int] = []
        queue: deque[int] = deque([fi])
        while queue:
            curr = queue.popleft()
            if curr in cluster_vis:
                continue
            cluster_vis.add(curr)
            cluster.append(curr)
            for nfi in adj.get(curr, set()):
                if nfi in neg_idx and nfi not in cluster_vis:
                    queue.append(nfi)
        clusters.append(cluster)

    # -- vertex adjacency for shortest-path closure --
    vert_adj: dict[int, set[int]] = defaultdict(set)
    for e in edge_to_face:
        vert_adj[e[0]].add(e[1])
        vert_adj[e[1]].add(e[0])

    # -- per-cluster: collect boundary edges, validate, close gaps --
    mouths: list[list[tuple[int, int]]] = []

    for cluster in clusters:
        if len(cluster) < min_pocket_size:
            continue
        cset = set(cluster)

        # Boundary edges (identical to find_rims)
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
                foe = edge_to_face[e]
                if len(foe) == 2:
                    other = foe[1] if foe[0] in cset else foe[0]
                    if other not in cset:
                        boundary_edges.append(e)
        if not boundary_edges:
            continue

        # Fold-ratio validation (identical to find_rims)
        pocket_area = float(area_arr[cluster].sum())
        opening_area = _rim_enclosed_area(boundary_edges, verts_arr)
        if opening_area <= 0:
            continue
        if pocket_area / opening_area < min_fold_ratio:
            continue

        # -- close gaps: find degree-1 vertices and connect them --
        degree: dict[int, int] = defaultdict(int)
        for a, b in boundary_edges:
            degree[a] += 1
            degree[b] += 1

        endpoints = [v for v, d in degree.items() if d == 1]

        if not endpoints:
            # Already closed
            mouths.append(boundary_edges)
            continue

        # Greedily pair nearest endpoints and connect via shortest
        # vertex-path on the mesh.
        closure_edges: list[tuple[int, int]] = []
        remaining = list(endpoints)

        while len(remaining) >= 2:
            # Find closest pair
            best_dist = float("inf")
            best_i, best_j = 0, 1
            for i in range(len(remaining)):
                for j in range(i + 1, len(remaining)):
                    d = float(
                        np.linalg.norm(
                            verts_arr[remaining[i]]
                            - verts_arr[remaining[j]]
                        )
                    )
                    if d < best_dist:
                        best_dist = d
                        best_i, best_j = i, j

            v_start = remaining[best_i]
            v_end = remaining[best_j]
            # Remove in reverse index order to keep indices valid
            for idx in sorted([best_i, best_j], reverse=True):
                remaining.pop(idx)

            # BFS shortest vertex-path
            parent: dict[int, int] = {v_start: -1}
            bfs: deque[int] = deque([v_start])
            found = False
            while bfs:
                v = bfs.popleft()
                if v == v_end:
                    found = True
                    break
                for nv in vert_adj.get(v, set()):
                    if nv not in parent:
                        parent[nv] = v
                        bfs.append(nv)
            if found:
                # Trace path and collect edges
                v = v_end
                while parent[v] != -1:
                    pv = parent[v]
                    closure_edges.append((min(v, pv), max(v, pv)))
                    v = pv

        closed = boundary_edges + closure_edges
        # Drop mouths that still have degree-1 vertices (unsealed gaps)
        if any(d == 1 for d in _mouth_degree(closed).values()):
            continue
        mouths.append(closed)

    if verbose:
        n_edges = sum(len(m) for m in mouths)
        n_sealed = sum(
            1
            for m in mouths
            if all(d != 1 for d in _mouth_degree(m).values())
        )
        print(
            f"[skeliner.pre] Pocket mouths: {len(mouths)} pockets, "
            f"{n_edges:,} edges ({n_sealed}/{len(mouths)} sealed)"
        )

    return mouths


def _mouth_degree(edges: list[tuple[int, int]]) -> dict[int, int]:
    """Vertex degree map for an edge list."""
    deg: dict[int, int] = defaultdict(int)
    for a, b in edges:
        deg[a] += 1
        deg[b] += 1
    return deg


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


def _outward_dot_gradient(
    dots: np.ndarray,
    adj: dict[int, set[int]],
    n_faces: int,
) -> np.ndarray:
    """Per-face gradient: max absolute outward_dot difference to neighbors."""
    gradient = np.zeros(n_faces, dtype=np.float32)
    for fi in range(n_faces):
        nbs = adj.get(fi)
        if nbs:
            nbs_list = list(nbs)
            gradient[fi] = np.abs(dots[nbs_list] - dots[fi]).max()
    return gradient


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
    mesh_stats: tuple | None = None,
) -> np.ndarray:
    """Detect pocket organelles — membrane folds connected to the neuron surface.

    Uses rims (boundaries of negative-dot clusters) as seeds:

    1. Call :func:`find_rims` to get rim edges for each pocket.
    2. Seed from the **negative-dot faces** of each rim's pocket cluster.
    3. Flood-fill from seeds, stopping at rim edges and faces with
       ``outward_dot > grow_threshold``.
    4. Bridging: flood-fill from pocket boundary using the relaxed
       ``bridge_threshold`` to cross narrow positive-dot barriers.
    5. Hole filling: small non-pocket clusters mostly enclosed by pocket
       faces are filled in.

    Only regions behind a rim get detected — curved surfaces without a
    rim are correctly excluded.

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
        Minimum negative-face cluster size to produce a rim / be detected.
    min_cluster_size : int
        Final pocket clusters smaller than this are discarded.
    verbose : bool

    Returns
    -------
    np.ndarray
        Boolean mask ``(nFaces,)`` — pocket organelle faces.
    """
    from collections import defaultdict, deque

    if mesh_stats is not None:
        outward_dots, _, _, main_face_mask = mesh_stats
    else:
        outward_dots, _, _, main_face_mask = compute_mesh_stats(
            mesh,
            radius,
            radius_multiplier,
            verbose,
        )
    n_faces = len(mesh.faces)
    # Compute shared structures once
    edge_to_face = _edge_to_faces(mesh)
    adj = _face_adjacency(mesh, edge_to_face)

    # Use find_rims to get rim edges for each pocket
    rims = find_rims(
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

    if not rims:
        if verbose:
            print("[skeliner.pre] No pockets found")
        return np.zeros(n_faces, dtype=bool)

    # Collect all rim edges and seed faces
    # Seeds = negative-dot faces adjacent to rim edges (inward side)
    rim_edge_set: set[tuple[int, int]] = set()
    for rim in rims:
        rim_edge_set.update(rim)

    # Seeds: negative-dot faces touching rim edges
    seeds: set[int] = set()
    for e in rim_edge_set:
        for fi in edge_to_face.get(e, []):
            if main_face_mask[fi] and outward_dots[fi] < 0:
                seeds.add(fi)

    # Rim faces: positive-dot faces touching rim edges (block flood-fill)
    rim_faces: set[int] = set()
    for e in rim_edge_set:
        for fi in edge_to_face.get(e, []):
            if fi not in seeds:
                rim_faces.add(fi)

    if verbose:
        print(
            f"[skeliner.pre] Pockets: {len(rims)}, "
            f"rim edges: {len(rim_edge_set):,}, "
            f"seeds: {len(seeds):,}"
        )

    # Flood-fill from seeds, blocked by rim faces and grow_threshold
    pocket = np.zeros(n_faces, dtype=bool)
    visited = np.zeros(n_faces, dtype=bool)
    queue = deque(seeds)

    while queue:
        fi = queue.popleft()
        if visited[fi]:
            continue
        visited[fi] = True
        if fi in rim_faces:
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
        if fi in rim_faces or not main_face_mask[fi]:
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
        too_large = False
        bfs_queue = deque([fi])
        while bfs_queue:
            curr = bfs_queue.popleft()
            if curr in np_visited:
                continue
            np_visited.add(curr)
            cluster.append(curr)
            if len(cluster) > max_hole_size:
                too_large = True
                while bfs_queue:
                    c2 = bfs_queue.popleft()
                    if c2 not in np_visited:
                        np_visited.add(c2)
                        cluster.append(c2)
                        for n2 in adj.get(c2, set()):
                            if n2 in non_pocket_idx and n2 not in np_visited:
                                bfs_queue.append(n2)
                break
            for nfi in adj.get(curr, set()):
                if nfi in non_pocket_idx and nfi not in np_visited:
                    bfs_queue.append(nfi)
                elif nfi not in non_pocket_idx:
                    n_total_boundary += 1
                    if pocket[nfi]:
                        n_pocket_boundary += 1
        if too_large or len(cluster) == 0:
            continue
        enclosure = n_pocket_boundary / n_total_boundary if n_total_boundary else 0
        if enclosure >= hole_enclosure_ratio:
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


def find_pocket_organelles_alt(
    mesh: trimesh.Trimesh,
    *,
    radius: float | None = None,
    radius_multiplier: float = 5.0,
    min_pocket_size: int = 5,
    min_fold_ratio: float = 3.0,
    min_cluster_size: int = 5,
    verbose: bool = False,
    mesh_stats: tuple | None = None,
) -> np.ndarray:
    """Detect pocket organelles using closed mouth boundaries.

    Uses :func:`find_pocket_mouths` to get sealed edge-loops at each
    pocket opening, then floods from the exterior blocked by those
    edges.  Everything not reached is pocket.

    Returns
    -------
    np.ndarray
        Boolean mask ``(nFaces,)`` — pocket organelle faces.
    """
    from collections import deque

    if mesh_stats is not None:
        outward_dots, _, _, main_face_mask = mesh_stats
    else:
        outward_dots, _, _, main_face_mask = compute_mesh_stats(
            mesh, radius, radius_multiplier, verbose,
        )
    n_faces = len(mesh.faces)
    edge_to_face = _edge_to_faces(mesh)
    adj = _face_adjacency(mesh, edge_to_face)
    faces_arr = np.asarray(mesh.faces)

    # -- face → edge set lookup --
    face_edges: dict[int, set[tuple[int, int]]] = {}
    for fi in range(n_faces):
        f = faces_arr[fi]
        fe: set[tuple[int, int]] = set()
        for i in range(3):
            fe.add(
                (
                    min(int(f[i]), int(f[(i + 1) % 3])),
                    max(int(f[i]), int(f[(i + 1) % 3])),
                )
            )
        face_edges[fi] = fe

    # -- barrier: ALL neg/pos boundary edges + closure edges --
    # All neg/pos boundaries seal every neg-dot cluster (including
    # small/flat ones that don't qualify as pockets).  The closure
    # edges from find_pocket_mouths seal gaps at mesh boundary edges.
    neg_mask = (outward_dots < 0) & main_face_mask
    mouth_edge_set: set[tuple[int, int]] = set()
    for fi in np.where(neg_mask)[0]:
        for e in face_edges[int(fi)]:
            for nfi in edge_to_face.get(e, []):
                if nfi != fi and outward_dots[nfi] >= 0:
                    mouth_edge_set.add(e)

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

    # Add closure edges (the shortest-path bridges that seal gaps)
    for m in mouths:
        mouth_edge_set.update(m)

    if verbose:
        print(
            f"[skeliner.pre] Mouth barrier: {len(mouth_edge_set):,} edges "
            f"(all neg/pos + {len(mouths)} pocket closures)"
        )

    # -- flood from exterior, blocked by mouth edges --
    start_face = int(np.argmax(outward_dots * main_face_mask))
    outside = np.zeros(n_faces, dtype=bool)
    vis = np.zeros(n_faces, dtype=bool)
    queue: deque[int] = deque([start_face])

    while queue:
        fi = queue.popleft()
        if vis[fi]:
            continue
        vis[fi] = True
        if not main_face_mask[fi]:
            continue
        outside[fi] = True
        for nfi in adj.get(fi, set()):
            if vis[nfi]:
                continue
            shared = face_edges[fi] & face_edges[nfi]
            if any(e in mouth_edge_set for e in shared):
                continue
            queue.append(nfi)

    # -- pocket = main faces not reached from outside --
    pocket = main_face_mask & ~outside

    # -- filter small clusters --
    pocket = _filter_small_clusters(mesh, pocket, min_cluster_size)

    if verbose:
        print(
            f"[skeliner.pre] Pocket organelles (alt): "
            f"{int(pocket.sum()):,} faces"
        )

    return pocket


def find_isolated_organelles(
    mesh: trimesh.Trimesh,
    *,
    radius: float | None = None,
    radius_multiplier: float = 5.0,
    verbose: bool = False,
    mesh_stats: tuple | None = None,
) -> np.ndarray:
    """Detect vertex-disconnected internal fragments.

    These are organelle membranes that form entirely separate connected
    components enclosed within the neuron body.

    Returns
    -------
    np.ndarray
        Boolean mask ``(nFaces,)`` — isolated organelle faces.
    """
    if mesh_stats is not None:
        outward_dots, face_comp, main_ci, _ = mesh_stats
    else:
        outward_dots, face_comp, main_ci, _ = compute_mesh_stats(
            mesh,
            radius,
            radius_multiplier,
            verbose,
        )
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
    mesh_stats: tuple | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Detect internal mesh fragments (organelle membranes) in a neuron mesh.

    Returns two non-overlapping masks:

    * **pocket** — membrane folds connected to the neuron surface,
      detected via gradient-based rim finding + flood-fill.
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

    Returns
    -------
    pocket : np.ndarray
        Boolean mask ``(nFaces,)`` — pocket organelle faces (main component).
    isolated : np.ndarray
        Boolean mask ``(nFaces,)`` — isolated internal fragment faces.
    """
    import time as _time

    _p = "[skeliner.pre]"
    t_total = _time.perf_counter()

    # ── 1. Precompute outward dots and components ─────────────────
    if mesh_stats is not None:
        precomputed = mesh_stats
    else:
        precomputed = compute_mesh_stats(
            mesh,
            radius,
            radius_multiplier,
            verbose,
        )
    _, face_comp, main_ci, _ = precomputed

    # ── 2. Find isolated organelles (small internal components) ───
    isolated = find_isolated_organelles(
        mesh,
        radius=radius,
        radius_multiplier=radius_multiplier,
        verbose=verbose,
        mesh_stats=precomputed,
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

    precomputed_structural = (
        precomputed[0],  # outward_dots
        precomputed[1],  # face_comp
        precomputed[2],  # main_ci
        structural_mask,  # structural_face_mask (replaces main_face_mask)
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

    return pocket, isolated


def _find_nonmanifold_fusions(
    mesh: trimesh.Trimesh,
    *,
    radius: float | None = None,
    radius_multiplier: float = 5.0,
    grow_rings: int = 20,
    min_branch_size: int = 5,
    verbose: bool = False,
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
    from collections import deque

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
    fusion_clusters: list[list[int]] | None = None,
    radius: float | None = None,
    radius_multiplier: float = 5.0,
    verbose: bool = False,
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
    if fusion_clusters is not None:
        clusters = fusion_clusters
        if verbose:
            n = sum(len(c) for c in clusters)
            print(
                f"[skeliner.pre] Using provided fusion clusters "
                f"({len(clusters)} regions, {n} faces)"
            )
    else:
        clusters = find_fusions(
            mesh,
            radius=radius,
            radius_multiplier=radius_multiplier,
            verbose=verbose,
        )

    all_fusion: set[int] = set()
    for c in clusters:
        all_fusion.update(c)

    if all_fusion:
        keep = np.ones(len(mesh.faces), dtype=bool)
        for fi in all_fusion:
            keep[fi] = False
        mesh = _rebuild_mesh(mesh, keep)
        if verbose:
            print(
                f"[skeliner.pre] Removed {len(all_fusion)} fusion faces "
                f"({len(clusters)} regions)"
            )

    # Step 2: split shared vertices
    mesh = _split_fan_vertices(mesh, verbose=verbose)

    if verbose:
        print(
            f"[skeliner.pre] Fusions: {len(all_fusion)} faces removed, vertices split"
        )

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
    organelles: np.ndarray | None = None,
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
    organelles : np.ndarray or None
        Pre-computed boolean mask from :func:`find_organelles` (or
        ``find_pocket_organelles | find_isolated_organelles``).  If
        provided, detection is skipped and this mask is used directly.
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
    pocket, isolated = find_organelles(mesh, verbose=True)
    soma = find_soma(mesh, organelles=pocket | isolated)
    clean = remove_organelles(mesh, organelles=pocket | isolated)
    # soma.verts is still valid on clean — no remapping needed
    """
    if organelles is not None and len(organelles) == len(mesh.faces):
        organelle = np.asarray(organelles, dtype=bool)
        if verbose:
            print(
                f"[skeliner.pre] Using provided organelle mask "
                f"({int(organelle.sum()):,} faces)"
            )
    else:
        pocket, isolated = find_organelles(
            mesh,
            radius=radius,
            radius_multiplier=radius_multiplier,
            min_cluster_size=min_cluster_size,
            verbose=verbose,
        )
        organelle = pocket | isolated

    if not organelle.any():
        if verbose:
            print("[skeliner.pre] Nothing to remove")
        return mesh

    clean = _rebuild_mesh(mesh, ~organelle)

    if verbose:
        print(
            f"[skeliner.pre] Result: {len(clean.faces):,} faces "
            f"(removed {organelle.sum():,} faces)"
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


# ── Offset detection and correction ─────────────────────────────────


def _detect_tile_size(vertices: np.ndarray) -> int:
    """Auto-detect XY tile size from vertex coordinate clustering.

    EM volume tiles create vertex clusters at tile boundaries.
    Tests powers of 2 and picks the size with the strongest
    boundary clustering (highest ratio of observed to expected
    boundary vertices).
    """
    # At tile boundaries, many vertices share the exact same
    # X or Y coordinate.  Find these peak coordinates and
    # measure the most common spacing between them.
    # Find coordinates with high vertex counts (Otsu), then
    # measure spacing between the TOP peaks per axis.  Use the
    # largest consistent spacing as the tile size.
    best_spacing = 0
    for axis in [0, 1]:
        coords = np.round(vertices[:, axis]).astype(np.int64)
        unique, counts = np.unique(coords, return_counts=True)
        thresh, _ = _otsu_threshold(counts.astype(float))
        # Take the top peaks above threshold, sorted by count
        above = counts > thresh
        if above.sum() < 3:
            continue
        peak_coords = unique[above]
        peak_counts = counts[above]
        # Use only the top peaks (sorted by count descending)
        order = np.argsort(-peak_counts)
        top_peaks = np.sort(peak_coords[order[:max(3, len(order) // 5)]])
        if len(top_peaks) >= 2:
            diffs = np.diff(top_peaks)
            diffs = diffs[diffs > 100]
            if len(diffs) > 0:
                spacing = float(np.median(diffs))
                if spacing > best_spacing:
                    best_spacing = spacing

    if best_spacing > 100:
        return int(2 ** round(np.log2(best_spacing)))

    extent = vertices.max(axis=0) - vertices.min(axis=0)
    return int(max(extent[:2]) / 4)


def _detect_z_resolution(vertices: np.ndarray) -> float:
    """Auto-detect Z section spacing from vertex coordinates."""
    z_unique = np.unique(np.round(vertices[:, 2], 1))
    if len(z_unique) < 2:
        return 1.0
    diffs = np.diff(z_unique)
    return float(np.median(diffs[diffs > 0.5]))


def _cluster_xy(points: np.ndarray, radius: float) -> list[np.ndarray]:
    """Cluster 2-D points by proximity (connected components)."""
    if len(points) < 2:
        return [points] if len(points) else []
    tree = KDTree(points)
    visited = np.zeros(len(points), dtype=bool)
    clusters: list[np.ndarray] = []
    for seed in range(len(points)):
        if visited[seed]:
            continue
        queue = [seed]
        visited[seed] = True
        members = [seed]
        while queue:
            node = queue.pop()
            for nb in tree.query_ball_point(points[node], r=radius):
                if not visited[nb]:
                    visited[nb] = True
                    queue.append(nb)
                    members.append(nb)
        clusters.append(points[members])
    return clusters


def _match_contour_offset(
    target: np.ndarray,
    source: np.ndarray,
    voxel: float,
    search_radius: int = 10,
) -> tuple[np.ndarray, float]:
    """Find the XY translation that best aligns *source* onto *target*.

    Searches a grid of ``±search_radius`` voxels around multiple
    starting points (centroid difference and zero) to avoid missing
    the true offset when centroids are skewed by outlier vertices.
    Returns ``(offset_xy, mean_nn_distance)``.
    """
    init = target.mean(axis=0) - source.mean(axis=0)
    tree = KDTree(target)
    best_offset = init
    best_dist = float("inf")

    # Search around both the centroid difference AND zero offset
    # to handle cases where one contour has extra vertices
    seeds = [init, np.zeros(2)]
    for seed in seeds:
        for dx in np.arange(-search_radius, search_radius + 1) * voxel:
            for dy in np.arange(-search_radius, search_radius + 1) * voxel:
                offset = seed + np.array([dx, dy])
                dists, _ = tree.query(source + offset)
                mean_d = float(np.median(dists))
                if mean_d < best_dist:
                    best_dist = mean_d
                    best_offset = offset
    return best_offset, best_dist


def find_offsets(
    mesh: trimesh.Trimesh,
    *,
    verbose: bool = False,
) -> list[dict]:
    """Detect Z-plane alignment offsets from EM volume registration errors.

    Scans the mesh for flat Z-facing cap faces that close off the tube
    at broken Z-boundaries.  Pairs FLOOR (cap pointing up) and CEIL
    (cap pointing down) planes into gaps, clusters cap contours by XY
    proximity, and measures the XY offset via contour matching.

    Should be run **before** any other preprocessing step.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Input mesh.
    verbose : bool, default False
        Print progress.

    Returns
    -------
    list[dict]
        Each entry describes one gap with keys:

        - ``z_floor`` : float — Z of the FLOOR cap plane
        - ``z_ceil`` : float — Z of the CEIL cap plane
        - ``offset`` : ndarray shape ``(2,)`` — XY translation from
          shifted side to correct side
        - ``match_error`` : float — mean nearest-neighbour distance
          after alignment (nm)
        - ``shifted_verts`` : ndarray — vertex indices on the shifted
          side (the smaller piece)
        - ``cap_faces`` : ndarray — face indices of the flat caps at
          both FLOOR and CEIL planes
    """
    normals = mesh.face_normals
    nz = normals[:, 2]
    centroids = mesh.triangles_center
    n_faces = len(mesh.faces)

    # ── 1. Detect Z resolution and flat faces ──
    z_res = _detect_z_resolution(mesh.vertices)
    flat_mask = np.abs(nz) > 0.99
    face_z = np.round(centroids[:, 2] / z_res) * z_res
    vert_z = np.round(mesh.vertices[:, 2] / z_res) * z_res
    if verbose:
        print(
            f"[skeliner.pre] Offsets: Z resolution = {z_res:.1f} nm, "
            f"{flat_mask.sum():,} flat faces"
        )

    # ── 2. Find Z gaps per connected component ──
    # A gap exists when a component has faces at z=A and z=B
    # with nothing in between (B - A > 1.5 * z_res).
    # Also check between nearby components at similar XY.
    comp_labels, _ = _face_edge_components(mesh)
    n_comps = int(comp_labels.max()) + 1

    # Per-component: find Z-range and gaps
    gaps: list[tuple[float, float]] = []
    seen_gaps: set[tuple[float, float]] = set()

    for c in range(n_comps):
        c_mask = comp_labels == c
        if not c_mask.any():
            continue
        c_z = np.unique(face_z[c_mask])
        if len(c_z) < 2:
            continue
        c_z = np.sort(c_z)
        diffs = np.diff(c_z)
        for i, d in enumerate(diffs):
            if d > z_res * 1.5:
                gap = (float(c_z[i]), float(c_z[i + 1]))
                if gap not in seen_gaps:
                    seen_gaps.add(gap)
                    gaps.append(gap)

    # Also find gaps BETWEEN nearby significant components
    comp_z_ranges: list[tuple[int, float, float, np.ndarray]] = []
    for c in range(n_comps):
        c_mask = comp_labels == c
        if not c_mask.any() or c_mask.sum() < 50:
            continue
        c_fi = np.where(c_mask)[0]
        c_z_min = float(face_z[c_fi].min())
        c_z_max = float(face_z[c_fi].max())
        c_xy = centroids[c_fi, :2].mean(axis=0)
        comp_z_ranges.append((c, c_z_min, c_z_max, c_xy))

    # For each component, find NEAREST neighbor above and below
    for i, (ci, zi_min, zi_max, xyi) in enumerate(comp_z_ranges):
        best_above = None
        best_above_gap = float("inf")
        best_below = None
        best_below_gap = float("inf")
        for j, (cj, zj_min, zj_max, xyj) in enumerate(comp_z_ranges):
            if i == j:
                continue
            if float(np.linalg.norm(xyi - xyj)) > 20000:
                continue
            # cj is above ci
            if zj_min > zi_max and zj_min - zi_max < best_above_gap:
                best_above_gap = zj_min - zi_max
                best_above = (zi_max, zj_min)
            # cj is below ci
            if zi_min > zj_max and zi_min - zj_max < best_below_gap:
                best_below_gap = zi_min - zj_max
                best_below = (zj_max, zi_min)
        for gap in [best_above, best_below]:
            if gap and gap[1] - gap[0] > z_res * 1.5:
                if gap not in seen_gaps:
                    seen_gaps.add(gap)
                    gaps.append(gap)

    gaps.sort()
    if verbose:
        print(f"[skeliner.pre] Offsets: {len(gaps)} Z-gaps found")

    # ── 3. For each gap, check for flat caps at boundaries ──
    comp_labels, _ = _face_edge_components(mesh)
    cluster_radius = float(np.median(mesh.edges_unique_length)) * 20

    results: list[dict] = []
    for z_floor, z_ceil in gaps:
        # Flat faces at the gap boundaries
        cap_fi_floor = np.where(
            (face_z == z_floor) & flat_mask
        )[0]
        cap_fi_ceil = np.where(
            (face_z == z_ceil) & flat_mask
        )[0]

        if len(cap_fi_floor) < 3 or len(cap_fi_ceil) < 3:
            continue

        floor_comp_id = int(comp_labels[cap_fi_floor[0]])
        ceil_comp_id = int(comp_labels[cap_fi_ceil[0]])

        # The flat caps must cover significant area (closing a tube,
        # not a tiny organelle surface).  Use Otsu on all flat face
        # areas to separate real caps from incidental flat faces.
        floor_cap_area = float(mesh.area_faces[cap_fi_floor].sum())
        ceil_cap_area = float(mesh.area_faces[cap_fi_ceil].sum())
        median_face_area = float(np.median(mesh.area_faces))
        # Cap must be larger than a few typical faces
        if floor_cap_area < median_face_area * 5:
            continue
        if ceil_cap_area < median_face_area * 5:
            continue

        # Skip if the DOMINANT component at floor and ceil is the
        # same (minor components like main mesh may appear at both)
        floor_comp_counts: dict[int, int] = defaultdict(int)
        for fi in cap_fi_floor:
            floor_comp_counts[int(comp_labels[fi])] += 1
        ceil_comp_counts: dict[int, int] = defaultdict(int)
        for fi in cap_fi_ceil:
            ceil_comp_counts[int(comp_labels[fi])] += 1
        floor_dominant = max(floor_comp_counts, key=floor_comp_counts.get)
        ceil_dominant = max(ceil_comp_counts, key=ceil_comp_counts.get)
        if floor_dominant == ceil_dominant:
            continue

        # Verify clean gap: the floor cap's component must have
        # no faces between z_floor and z_ceil.  Other components
        # (like the main mesh) may have faces there — that's OK.
        floor_comp = int(comp_labels[cap_fi_floor[0]])
        in_floor_comp = comp_labels == floor_comp
        in_gap_z = (face_z > z_floor + z_res * 0.5) & (
            face_z < z_ceil - z_res * 0.5
        )
        faces_in_gap = (in_floor_comp & in_gap_z).sum()
        if faces_in_gap > 0:
            continue

        # Get cap vertices at each Z
        floor_vi = np.unique(mesh.faces[cap_fi_floor].ravel())
        floor_vi = floor_vi[vert_z[floor_vi] == z_floor]
        ceil_vi = np.unique(mesh.faces[cap_fi_ceil].ravel())
        ceil_vi = ceil_vi[vert_z[ceil_vi] == z_ceil]

        if len(floor_vi) < 3 or len(ceil_vi) < 3:
            continue

        floor_xy = mesh.vertices[floor_vi, :2]
        ceil_xy = mesh.vertices[ceil_vi, :2]

        # Cluster by XY proximity
        floor_clusters = _cluster_xy(floor_xy, cluster_radius)
        ceil_clusters = _cluster_xy(ceil_xy, cluster_radius)

        # Match ceil → floor
        used_floor: set[int] = set()
        for cc in ceil_clusters:
            if len(cc) < 3:
                continue
            cc_center = cc.mean(axis=0)
            best_fi = -1
            best_d = float("inf")
            for fi_idx, fc in enumerate(floor_clusters):
                if fi_idx in used_floor or len(fc) < 3:
                    continue
                d = float(np.linalg.norm(fc.mean(axis=0) - cc_center))
                if d < best_d:
                    best_d = d
                    best_fi = fi_idx
            if best_fi < 0:
                continue
            used_floor.add(best_fi)
            fc = floor_clusters[best_fi]

            # Filter floor cluster to vertices near ceiling
            ceil_extent = np.linalg.norm(
                cc - cc_center, axis=1
            ).max()
            fc_dists = np.linalg.norm(fc - cc_center, axis=1)
            near = fc[fc_dists < ceil_extent * 3]
            if len(near) >= 3:
                fc = near
            fc_center = fc.mean(axis=0)

            # Measure offset via shape matching
            offset, err = _match_contour_offset(cc, fc, z_res)

            # Determine shifted side
            floor_cap_local = [
                fi for fi in cap_fi_floor
                if np.linalg.norm(centroids[fi, :2] - fc_center)
                < cluster_radius
            ]
            ceil_cap_local = [
                fi for fi in cap_fi_ceil
                if np.linalg.norm(centroids[fi, :2] - cc_center)
                < cluster_radius
            ]
            f_comps = set(
                int(comp_labels[fi]) for fi in floor_cap_local
                if comp_labels[fi] >= 0
            )
            c_comps = set(
                int(comp_labels[fi]) for fi in ceil_cap_local
                if comp_labels[fi] >= 0
            )
            # Same component at branch level → skip
            if f_comps & c_comps:
                continue

            f_size = sum(
                int((comp_labels == c).sum()) for c in f_comps
            )
            c_size = sum(
                int((comp_labels == c).sum()) for c in c_comps
            )
            if f_size <= c_size:
                shifted_side = "below"
                s_comps = f_comps
            else:
                shifted_side = "above"
                s_comps = c_comps

            s_mask = np.zeros(n_faces, dtype=bool)
            for c in s_comps:
                s_mask |= comp_labels == c
            shifted_vi = np.unique(mesh.faces[s_mask].ravel())

            cap_faces = np.array(
                floor_cap_local + ceil_cap_local, dtype=np.intp
            )
            floor_pos = np.array(
                [fc_center[0], fc_center[1], z_floor]
            )
            corrected_pos = np.array(
                [fc_center[0] + offset[0],
                 fc_center[1] + offset[1], z_ceil]
            )

            results.append(
                {
                    "z_floor": z_floor,
                    "z_ceil": z_ceil,
                    "offset": offset,
                    "match_error": err,
                    "shifted_verts": shifted_vi,
                    "shifted_side": shifted_side,
                    "cap_faces": cap_faces,
                    "floor_center": floor_pos,
                    "ceil_center": corrected_pos,
                }
            )
            if verbose:
                print(
                    f"[skeliner.pre]   Gap z={z_floor:.0f}"
                    f"→{z_ceil:.0f}: "
                    f"offset=({offset[0]:.0f}, {offset[1]:.0f}), "
                    f"error={err:.0f}, shifted={shifted_side} "
                    f"({len(shifted_vi):,} verts)"
                )

    if verbose:
        print(f"[skeliner.pre] Offsets: {len(results)} offsets detected")

    return results


def remove_offsets(
    mesh: trimesh.Trimesh,
    *,
    offsets: list[dict] | None = None,
    verbose: bool = False,
) -> trimesh.Trimesh:
    """Correct vertex positions for detected Z-plane alignment offsets.

    Translates vertices on the shifted side of each gap by the inverse
    offset vector, then removes the flat cap faces at the gap boundaries.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Input mesh.
    offsets : list[dict] or None
        Pre-computed offsets from :func:`find_offsets`.  If None,
        ``find_offsets`` is called automatically.
    verbose : bool, default False
        Print progress.

    Returns
    -------
    trimesh.Trimesh
        Mesh with offset vertices realigned and cap faces removed.
    """
    if offsets is None:
        offsets = find_offsets(mesh, verbose=verbose)

    if not offsets:
        if verbose:
            print("[skeliner.pre] No offsets to correct")
        return mesh

    z_res = _detect_z_resolution(mesh.vertices)
    new_verts = mesh.vertices.copy()

    # Process bottom to top. Each gap moves everything below
    # its ceiling. Centroid is restored at the end.
    sorted_offsets = sorted(offsets, key=lambda r: r["z_ceil"])

    comp_labels, main_comp = _face_edge_components(mesh)
    tile_size = _detect_tile_size(mesh.vertices)
    # Validate: each tile must fully contain the cap contour
    # extent of every offset.  If not, double until it does.
    for rec in offsets:
        cap_vi = np.unique(mesh.faces[rec["cap_faces"]].ravel())
        if len(cap_vi) < 2:
            continue
        cap_extent = max(
            mesh.vertices[cap_vi, 0].max() - mesh.vertices[cap_vi, 0].min(),
            mesh.vertices[cap_vi, 1].max() - mesh.vertices[cap_vi, 1].min(),
        )
        while tile_size < cap_extent * 2:
            tile_size *= 2
    if verbose:
        print(f"[skeliner.pre] Offsets: tile size = {tile_size}")
    vert_tx = (mesh.vertices[:, 0] // tile_size).astype(int)
    vert_ty = (mesh.vertices[:, 1] // tile_size).astype(int)

    for rec in sorted_offsets:
        offset_xy = rec["offset"]
        z_floor = rec["z_floor"]
        z_ceil = rec["z_ceil"]

        # Start with both floor and ceiling tiles
        floor_xy = rec["floor_center"][:2]
        ceil_xy = rec["ceil_center"][:2]
        gap_tx = int(floor_xy[0] // tile_size)
        gap_ty = int(floor_xy[1] // tile_size)
        ceil_tx = int(ceil_xy[0] // tile_size)
        ceil_ty = int(ceil_xy[1] // tile_size)
        corrected_tiles = {(gap_tx, gap_ty), (ceil_tx, ceil_ty)}

        in_tile = (vert_tx == gap_tx) & (vert_ty == gap_ty)
        below_mask = in_tile & (new_verts[:, 2] < z_ceil - z_res * 0.5)

        # Expand: for faces spanning tile boundary, add just the
        # boundary vertices (not the whole adjacent tile)
        face_any = below_mask[mesh.faces].any(axis=1)
        face_all = below_mask[mesh.faces].all(axis=1)
        mixed = face_any & ~face_all
        if mixed.any():
            for fi in np.where(mixed)[0]:
                for vi in mesh.faces[fi]:
                    if not below_mask[vi] and new_verts[vi, 2] < z_ceil - z_res * 0.5:
                        below_mask[vi] = True

        n_moved = int(below_mask.sum())
        if n_moved == 0:
            continue

        # Also include isolated non-main components below z_ceil
        # that are near the corrected region
        moved_center = mesh.vertices[below_mask, :2].mean(axis=0)
        moved_extent = max(
            mesh.vertices[below_mask, 0].max()
            - mesh.vertices[below_mask, 0].min(),
            mesh.vertices[below_mask, 1].max()
            - mesh.vertices[below_mask, 1].min(),
        )
        for c in range(int(comp_labels.max()) + 1):
            if c == main_comp:
                continue
            c_mask = comp_labels == c
            if not c_mask.any():
                continue
            c_verts = np.unique(mesh.faces[c_mask].ravel())
            c_z = mesh.vertices[c_verts, 2]
            if c_z.max() > z_ceil or c_z.min() > z_ceil:
                continue
            c_center = mesh.vertices[c_verts, :2].mean(axis=0)
            dist = np.linalg.norm(c_center - moved_center)
            if dist < moved_extent:
                below_mask[c_verts] = True

        dz = z_ceil - z_floor
        new_verts[below_mask, 0] += offset_xy[0]
        new_verts[below_mask, 1] += offset_xy[1]
        new_verts[below_mask, 2] += dz

        if verbose:
            print(
                f"[skeliner.pre]   Corrected z={z_floor:.0f}"
                f"→{z_ceil:.0f}: "
                f"moved {n_moved:,} verts by "
                f"({offset_xy[0]:.0f}, {offset_xy[1]:.0f}, "
                f"{dz:.0f})"
            )

    # Remove flat cap faces at all gap Z-planes
    nz_abs = np.abs(mesh.face_normals[:, 2])
    face_z = np.round(mesh.triangles_center[:, 2] / z_res) * z_res
    keep = np.ones(len(mesh.faces), dtype=bool)
    for rec in offsets:
        for z in [rec["z_floor"], rec["z_ceil"]]:
            keep[(face_z == z) & (nz_abs > 0.95)] = False

    # Remove distorted faces (vertices moved by different amounts)
    total_diff = new_verts - mesh.vertices
    fd = total_diff[mesh.faces]
    same_01 = np.all(np.abs(fd[:, 0] - fd[:, 1]) < 0.1, axis=1)
    same_12 = np.all(np.abs(fd[:, 1] - fd[:, 2]) < 0.1, axis=1)
    keep[~(same_01 & same_12)] = False

    n_removed = int((~keep).sum())
    new_faces = mesh.faces.copy()
    new_faces[~keep] = 0

    # Restore global position
    orig_centroid = mesh.vertices.mean(axis=0)
    new_centroid = new_verts.mean(axis=0)
    new_verts += orig_centroid - new_centroid

    result = trimesh.Trimesh(
        vertices=new_verts,
        faces=new_faces,
        process=False,
    )

    if verbose:
        drift = np.linalg.norm(new_centroid - orig_centroid)
        print(
            f"[skeliner.pre] Offsets: corrected {len(offsets)} offsets, "
            f"removed {n_removed:,} faces, "
            f"centroid drift {drift:.0f} nm (restored)"
        )

    return result


def compact_mesh(
    mesh: trimesh.Trimesh,
    *,
    soma: Soma | None = None,
    verbose: bool = False,
) -> tuple[trimesh.Trimesh, np.ndarray, Soma | None]:
    """Remove unreferenced vertices and compact indices.

    Call once at the end of preprocessing, after all face removals.
    Returns a ``vert_map`` for translating any remaining vertex-index
    data, and a remapped soma if provided.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Mesh after face removals.
    soma : Soma or None
        If provided, ``soma.verts`` is remapped to the new indices.
    verbose : bool, default False
        Print summary.

    Returns
    -------
    clean : trimesh.Trimesh
        Compacted mesh with no unreferenced vertices.
    vert_map : np.ndarray
        ``(nOldVerts,)`` int64 — ``vert_map[old]`` is the new index,
        or ``-1`` if the vertex was removed.
    soma : Soma or None
        Remapped soma, or None if not provided.

    Examples
    --------
    pocket, isolated = find_organelles(mesh)
    soma = find_soma(mesh, organelles=pocket | isolated)
    mesh = remove_organelles(mesh, organelles=pocket | isolated)
    # ... more removals ...
    mesh, vert_map, soma = compact_mesh(mesh, soma=soma)
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

    remapped_soma = None
    if soma is not None:
        remapped_soma = soma.remap(vert_map)
        if verbose:
            n_before = len(soma.verts) if soma.verts is not None else 0
            n_after = len(remapped_soma.verts) if remapped_soma.verts is not None else 0
            print(
                f"[skeliner.pre] Soma remap: {n_before:,} → {n_after:,} verts "
                f"({n_before - n_after:,} dropped)"
            )

    return clean, vert_map, remapped_soma


# -----------------------------------------------------------------------------
# High-level preprocessing pipeline
# -----------------------------------------------------------------------------
from dataclasses import dataclass, field


@dataclass
class PreprocessResult:
    """Result of :func:`preprocess` — all detection artifacts + cleaned mesh."""

    mesh: trimesh.Trimesh
    soma: Soma | None
    organelles: np.ndarray  # bool mask on original mesh
    fragments: np.ndarray  # bool mask on original mesh
    disconnected: list[list[int]]
    gaps: list
    mesh_stats: tuple
    vert_map: np.ndarray


def preprocess(
    mesh: trimesh.Trimesh,
    *,
    verbose: bool = False,
) -> PreprocessResult:
    """Run the full preprocessing pipeline in one call.

    Runs detection and removal in the correct order, sharing
    precomputed data throughout.  Returns a :class:`PreprocessResult`
    with the cleaned mesh and all intermediate artifacts.

    Pipeline order:
      1. compute_mesh_stats
      2. find_organelles
      3. find_soma
      4. find_disconnected
      5. find_gaps
      6. find_fragments
      7. remove_fragments
      8. remove_organelles
      9. remove_gaps
      10. compact_mesh

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Input mesh.
    verbose : bool, default False
        Print progress.

    Returns
    -------
    PreprocessResult
        Cleaned mesh and all detection artifacts.
    """
    # ── Detection (all on original mesh, sharing stats) ───────────
    stats = compute_mesh_stats(mesh, verbose=verbose)

    pocket, isolated = find_organelles(mesh, mesh_stats=stats, verbose=verbose)
    organelles_mask = pocket | isolated

    soma = find_soma(
        mesh, organelles=organelles_mask, mesh_stats=stats, verbose=verbose
    )

    disconnected = find_disconnected(
        mesh,
        soma=soma,
        organelles=organelles_mask,
        mesh_stats=stats,
        verbose=verbose,
    )

    gaps = find_gaps(
        mesh,
        soma=soma,
        disconnected=disconnected,
        mesh_stats=stats,
        verbose=verbose,
    )

    fragments_mask = find_fragments(mesh, verbose=verbose)

    # ── Removal (order: fragments → organelles → gaps → compact) ──
    mesh = remove_fragments(mesh, fragments=fragments_mask, verbose=verbose)
    mesh = remove_organelles(mesh, organelles=organelles_mask, verbose=verbose)
    mesh = remove_gaps(mesh, gaps=gaps, verbose=verbose)
    mesh, vert_map, soma = compact_mesh(mesh, soma=soma, verbose=verbose)

    return PreprocessResult(
        mesh=mesh,
        soma=soma,
        organelles=organelles_mask,
        fragments=fragments_mask,
        disconnected=disconnected,
        gaps=gaps,
        mesh_stats=stats,
        vert_map=vert_map,
    )


# =====================================================================
#  Z-slice helpers (shared by find_nucleus_center & find_soma_from_nucleus)
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
        peak_label = void_labeled[deep_ij[np.argmin(dists)][0],
                                  deep_ij[np.argmin(dists)][1]]
    nucleus_void = void_labeled == peak_label

    void_ij = np.argwhere(nucleus_void)
    void_xy = np.column_stack([
        xy_min[0] + void_ij[:, 0] * grid_res + grid_res / 2,
        xy_min[1] + void_ij[:, 1] * grid_res + grid_res / 2,
    ])
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

    raw, best, center = _z_scan(verts, grid_res, z_tol,
                                min_void_r, max_shift)

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

    return {
        "center": center,
        "z_range": (z_lo, z_hi),
        "peak_r": peak_r,
        "slices": slices_numeric,
        "contours": contours,
    }


#  find_soma_from_nucleus — soma detection using nucleus center
# =====================================================================


def _z_scan(
    verts: np.ndarray,
    grid_res: float = 200.0,
    z_tol: float = 150.0,
    min_void_r: float = 500.0,
    max_shift: float = 3000.0,
) -> tuple[list[tuple], list[int] | None, np.ndarray | None]:
    """Shared Z-scan: cluster + void detection + spatial coherence.

    Returns ``(raw, best_run, nucleus_center)`` where *raw* is the
    per-Z-level data ``[(z, cx, cy, void_r, void_xy, vert_indices,
    soma_mask), ...]``, *best_run* is the indices into *raw* of the
    nucleus chain (or None), and *nucleus_center* is ``(3,)`` or None.
    """
    z_unique = np.unique(verts[:, 2])
    z_step = 210.0
    z_levels = np.arange(z_unique.min() + z_step, z_unique.max() - z_step,
                         z_step)

    all_vi = np.arange(len(verts))
    raw: list[tuple] = []
    # Each entry: (z, cx, cy, void_r, void_xy, vert_indices, soma_mask)
    #   vert_indices: indices into mesh.vertices for verts near this Z
    #   soma_mask: bool array over vert_indices, True = in soma cluster

    for z in z_levels:
        near_z = np.abs(verts[:, 2] - z) < z_tol
        vi = all_vi[near_z]
        pts = verts[vi, :2]

        cluster = _z_slice_cluster(pts, grid_res)
        if cluster is None:
            raw.append((z, np.nan, np.nan, 0.0, None, vi, None))
            continue

        occ, soma_region, ix, iy, xy_min = cluster
        soma_mask = soma_region[ix, iy]

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
    slices = np.array([(raw[i][0], raw[i][1], raw[i][2], raw[i][3])
                       for i in best])
    center = np.array([
        np.nanmean(slices[:, 1]),
        np.nanmean(slices[:, 2]),
        np.mean(slices[:, 0]),
    ])

    return raw, best, center


def find_soma_from_nucleus(
    mesh: trimesh.Trimesh,
    *,
    grid_res: float = 200.0,
    z_tol: float = 150.0,
    verbose: bool = False,
) -> "Soma | None":
    """Detect the soma by first locating the nucleus void.

    Single-pass Z-scan: finds the nucleus void via spatial coherence,
    then collects soma-cluster vertices at all Z-levels where the
    cluster contains the nucleus XY.  Returns the same ``Soma`` type
    as :func:`find_soma`.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Input neuron mesh.
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
    from scipy.spatial import ConvexHull
    from shapely.geometry import Point, Polygon

    _p = "[find_soma_from_nucleus]"

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
        print(f"{_p} nucleus: ({nc[0]:.0f}, {nc[1]:.0f}, {nc[2]:.0f}), "
              f"r={peak_r:.0f}nm, {n_chain} Z-levels")

    # Build per-Z convex hull polygons, restricted to Z-levels within
    # the nucleus chain range (the soma exists where the nucleus does).
    nc_point = Point(nc[0], nc[1])
    hull_polys: list[tuple[float, Polygon]] = []  # (z, polygon)
    # Extend the nucleus Z-range adaptively in each direction:
    # keep going while the soma cluster contains the nucleus XY
    # and its area stays above 25% of the peak area in the chain.
    chain_areas = []
    for i in best:
        z, cx, cy, r, void_xy, vi, sm = raw[i]
        if sm is not None:
            chain_areas.append(sm.sum())
    peak_area = max(chain_areas) if chain_areas else 0
    area_thresh = peak_area * 0.25

    nuc_z_lo = raw[best[0]][0]
    nuc_z_hi = raw[best[-1]][0]

    # Extend downward
    for i in range(best[0] - 1, -1, -1):
        z, cx, cy, r, void_xy, vi, sm = raw[i]
        if sm is None:
            break
        soma_pts = verts[vi[sm], :2]
        if len(soma_pts) < 4 or sm.sum() < area_thresh:
            break
        try:
            h = ConvexHull(soma_pts)
            p = Polygon(soma_pts[h.vertices])
            if not p.contains(nc_point):
                break
        except Exception:
            break
        nuc_z_lo = z

    # Extend upward
    for i in range(best[-1] + 1, len(raw)):
        z, cx, cy, r, void_xy, vi, sm = raw[i]
        if sm is None:
            break
        soma_pts = verts[vi[sm], :2]
        if len(soma_pts) < 4 or sm.sum() < area_thresh:
            break
        try:
            h = ConvexHull(soma_pts)
            p = Polygon(soma_pts[h.vertices])
            if not p.contains(nc_point):
                break
        except Exception:
            break
        nuc_z_hi = z

    from scipy.ndimage import binary_erosion, binary_dilation

    # Disk structuring element for morphological opening.
    _r = 2
    yy, xx = np.ogrid[-_r:_r + 1, -_r:_r + 1]
    disk = (xx ** 2 + yy ** 2) <= _r ** 2
    struct3 = np.ones((3, 3), dtype=bool)

    for entry in raw:
        z, cx, cy, r, void_xy, vi, soma_mask = entry
        if z < nuc_z_lo or z > nuc_z_hi:
            continue
        if soma_mask is None:
            continue

        # Reconstruct the soma cluster grid
        pts = verts[vi, :2]
        cluster = _z_slice_cluster(pts, grid_res)
        if cluster is None:
            continue
        occ, soma_region, ix, iy, xy_min = cluster
        occ_s = occ & soma_region

        # Close (fill vertex gaps) then open (remove thin spikes)
        closed = binary_dilation(occ_s, struct3)
        closed = binary_erosion(closed, struct3)
        opened = binary_dilation(binary_erosion(closed, disk), disk)

        # Convex hull of the opened (spike-free) region
        opened_ij = np.argwhere(opened)
        if len(opened_ij) < 4:
            continue
        opened_xy = np.column_stack([
            xy_min[0] + opened_ij[:, 0] * grid_res + grid_res / 2,
            xy_min[1] + opened_ij[:, 1] * grid_res + grid_res / 2,
        ])

        try:
            hull = ConvexHull(opened_xy)
            hull_poly = Polygon(opened_xy[hull.vertices])
            if not hull_poly.is_valid:
                hull_poly = hull_poly.buffer(0)
            if hull_poly.is_valid and hull_poly.contains(nc_point):
                from shapely import prepare
                prepare(hull_poly)
                hull_polys.append((z, hull_poly))
        except Exception:
            continue

    if not hull_polys:
        if verbose:
            print(f"{_p} no soma contours found")
        return None

    hull_z = np.array([h[0] for h in hull_polys])
    z_lo, z_hi = hull_z.min(), hull_z.max()

    # Classify every mesh vertex: Z within range → check XY against
    # nearest Z-level's convex hull polygon.
    z_in_range = (verts[:, 2] >= z_lo) & (verts[:, 2] <= z_hi)
    candidates = np.where(z_in_range)[0]

    soma_mask_all = np.zeros(len(candidates), dtype=bool)
    # For each candidate vertex, find the nearest hull Z
    cand_z = verts[candidates, 2]
    nearest_idx = np.searchsorted(hull_z, cand_z).clip(0, len(hull_z) - 1)
    # Also check the index before (searchsorted gives insertion point)
    for offset in [0, -1]:
        idx = (nearest_idx + offset).clip(0, len(hull_z) - 1)
        # Group by hull index for batch point-in-polygon
        for hi in range(len(hull_polys)):
            mask = idx == hi
            if not mask.any():
                continue
            ci = candidates[mask]
            pts_xy = verts[ci, :2]
            inside = np.array([hull_polys[hi][1].contains(Point(p))
                               for p in pts_xy])
            soma_mask_all[mask] |= inside

    soma_vert_set = set(candidates[soma_mask_all].tolist())
    n_z_hit = len(hull_polys)

    if len(soma_vert_set) < 4:
        if verbose:
            print(f"{_p} too few soma vertices ({len(soma_vert_set)})")
        return None

    soma_arr = np.fromiter(sorted(soma_vert_set), dtype=np.intp)
    soma = Soma.fit(mesh.vertices[soma_arr], verts=soma_arr)

    if verbose:
        print(
            f"{_p} soma: center=["
            f"{soma.center[0]:.0f}, {soma.center[1]:.0f}, "
            f"{soma.center[2]:.0f}], "
            f"axes=[{soma.axes[0]:.0f}, {soma.axes[1]:.0f}, "
            f"{soma.axes[2]:.0f}], "
            f"{len(soma.verts):,} verts from {n_z_hit} Z-levels"
        )

    return soma


#  find_soma_void — soma detection via organelle-void cross-sections
# =====================================================================


def find_soma_void(
    mesh: trimesh.Trimesh,
    pocket: np.ndarray,
    isolated: np.ndarray,
    *,
    mesh_stats: tuple | None = None,
    verbose: bool = False,
) -> np.ndarray:
    """Detect soma faces using organelle distribution and mesh cross-sections.

    The approach exploits the fact that organelles coat the nucleus membrane
    but are absent from the nucleus interior, creating a 3D void.

    **Algorithm**

    1. Find the nucleus void centre from the organelle field
       (density peak → local enclosure-based void detection).
    2. Build a bounding box centred on the void by expanding outward
       along each axis until organelle density drops off.
    3. Cross-section a submesh within the bounding box at every
       vertex Z-level.
    4. At each Z-level, select the contour that contains the void
       centre's XY projection — that is the soma contour.  Neurite
       contours (which don't contain the void centre) are excluded.
    5. Classify vertices inside the soma contour at their Z-level.
    6. A face is soma if any of its vertices is classified as soma.
       Organelle faces (pocket / isolated) are excluded.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Input neuron mesh.
    pocket : np.ndarray
        Boolean mask ``(nFaces,)`` of pocket organelle faces.
    isolated : np.ndarray
        Boolean mask ``(nFaces,)`` of isolated organelle faces.
    mesh_stats : tuple or None
        Precomputed ``(outward_dots, face_comp, main_ci, main_face_mask)``.
    verbose : bool
        Print progress information.

    Returns
    -------
    np.ndarray
        Boolean mask ``(nFaces,)`` — soma faces (includes nucleus membrane,
        excludes organelle faces).
    """
    import time as _time

    from shapely.geometry import Point, Polygon
    from shapely.ops import unary_union
    from shapely import prepare

    _p = "[find_soma_void]"
    t0 = _time.perf_counter()

    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces_arr = np.asarray(mesh.faces)
    n_faces = len(faces_arr)
    all_c = verts[faces_arr].mean(axis=1)

    # ── 0. Mesh stats ──────────────────────────────────────────────
    if mesh_stats is not None:
        _, face_comp, main_ci, _ = mesh_stats
    else:
        stats = compute_mesh_stats(mesh, verbose=verbose)
        _, face_comp, main_ci, _ = stats

    org_mask = pocket | isolated
    org_c = all_c[np.where(org_mask)[0]]

    if len(org_c) < 50:
        if verbose:
            print(f"{_p} too few organelle faces ({len(org_c)}), skipping")
        return np.zeros(n_faces, dtype=bool)

    # ── 1. Find the nucleus void centre ────────────────────────────
    void_center = _find_void_center(org_c, verbose=verbose)
    if void_center is None or np.any(np.isnan(void_center)):
        if verbose:
            print(f"{_p} no void found in organelle field")
        return np.zeros(n_faces, dtype=bool)

    if verbose:
        print(f"{_p} void centre: ({void_center[0]:.0f}, "
              f"{void_center[1]:.0f}, {void_center[2]:.0f})")

    # ── 2. Bounding box centred on the void ────────────────────────
    soma_box_min, soma_box_max = _find_void_bounding_box(
        org_c, void_center, verbose=verbose,
    )

    if verbose:
        print(f"{_p} soma box: "
              f"X=[{soma_box_min[0]:.0f},{soma_box_max[0]:.0f}] "
              f"Y=[{soma_box_min[1]:.0f},{soma_box_max[1]:.0f}] "
              f"Z=[{soma_box_min[2]:.0f},{soma_box_max[2]:.0f}] "
              f"({_time.perf_counter() - t0:.1f}s)")

    # ── 3. Cross-section a submesh within the bounding box ─────────
    face_in_box = (
        (all_c[:, 0] >= soma_box_min[0])
        & (all_c[:, 0] <= soma_box_max[0])
        & (all_c[:, 1] >= soma_box_min[1])
        & (all_c[:, 1] <= soma_box_max[1])
        & (all_c[:, 2] >= soma_box_min[2])
        & (all_c[:, 2] <= soma_box_max[2])
    )
    box_face_idx = np.where(face_in_box)[0]
    if len(box_face_idx) == 0:
        if verbose:
            print(f"{_p} no faces inside bounding box")
        return np.zeros(n_faces, dtype=bool)

    submesh = mesh.submesh([box_face_idx], append=True)

    if verbose:
        print(f"{_p} submesh: {len(submesh.faces):,} faces "
              f"(from {n_faces:,})")

    z_unique = np.unique(submesh.vertices[:, 2])
    z_near = z_unique[
        (z_unique >= soma_box_min[2]) & (z_unique <= soma_box_max[2])
    ]

    # Void centre XY as a shapely Point for contour selection
    vc_xy = Point(void_center[0], void_center[1])

    z_contours: dict[float, list[tuple[float, np.ndarray]]] = {}
    for z in z_near:
        contours = _section_contours(submesh, float(z))
        if contours:
            z_contours[z] = contours

    if not z_contours:
        if verbose:
            print(f"{_p} no cross-section contours found")
        return np.zeros(n_faces, dtype=bool)

    if verbose:
        print(f"{_p} collected {len(z_contours)} Z-levels "
              f"({_time.perf_counter() - t0:.1f}s)")

    # ── 4. Determine soma Z-range and build contour polygons ────────
    #       The void centre defines the Z-range: the soma exists at
    #       Z-levels where a contour contains the void centre's XY.
    #       Within that range, union ALL contours at each Z-level
    #       (not just the one containing the void centre) to capture
    #       the full soma surface including folds and pores.

    # First pass: find which Z-levels contain the void centre
    vc_z_levels: list[float] = []
    for z, contours in z_contours.items():
        for _area, pts in contours:
            if len(pts) < 3:
                continue
            try:
                p = Polygon(pts[:, :2])
                if not p.is_valid:
                    p = p.buffer(0)
                if p.is_valid and not p.is_empty and p.contains(vc_xy):
                    vc_z_levels.append(z)
                    break
            except Exception:
                pass

    if not vc_z_levels:
        if verbose:
            print(f"{_p} no contour contains the void centre")
        return np.zeros(n_faces, dtype=bool)

    soma_z_min = min(vc_z_levels)
    soma_z_max = max(vc_z_levels)

    # From the void-centre contours, derive the soma XY extent.
    # This constrains which contours to include — neurites extending
    # in XY beyond the soma are excluded.
    soma_x_min, soma_x_max = np.inf, -np.inf
    soma_y_min, soma_y_max = np.inf, -np.inf
    for z in vc_z_levels:
        contours = z_contours.get(z, [])
        for _area, pts in contours:
            if len(pts) < 3:
                continue
            try:
                p = Polygon(pts[:, :2])
                if not p.is_valid:
                    p = p.buffer(0)
                if p.is_valid and not p.is_empty and p.contains(vc_xy):
                    bx = p.bounds  # (minx, miny, maxx, maxy)
                    soma_x_min = min(soma_x_min, bx[0])
                    soma_y_min = min(soma_y_min, bx[1])
                    soma_x_max = max(soma_x_max, bx[2])
                    soma_y_max = max(soma_y_max, bx[3])
                    break
            except Exception:
                pass

    # Add padding for folds/pores just outside the void-centre contour
    xy_pad = 1000.0
    soma_x_min -= xy_pad
    soma_x_max += xy_pad
    soma_y_min -= xy_pad
    soma_y_max += xy_pad

    if verbose:
        print(f"{_p} soma XY extent: "
              f"X=[{soma_x_min:.0f},{soma_x_max:.0f}] "
              f"Y=[{soma_y_min:.0f},{soma_y_max:.0f}]")

    # Second pass: within the soma Z-range, union contours that
    # overlap with the soma XY extent
    soma_xy_box = Polygon([
        (soma_x_min, soma_y_min), (soma_x_max, soma_y_min),
        (soma_x_max, soma_y_max), (soma_x_min, soma_y_max),
    ])

    soma_polys: dict[float, object] = {}
    for z, contours in z_contours.items():
        if z < soma_z_min or z > soma_z_max:
            continue
        polys = []
        for _area, pts in contours:
            if len(pts) < 3:
                continue
            try:
                p = Polygon(pts[:, :2])
                if not p.is_valid:
                    p = p.buffer(0)
                if p.is_valid and not p.is_empty:
                    if p.intersects(soma_xy_box):
                        polys.append(p)
            except Exception:
                pass
        if polys:
            merged = unary_union(polys).buffer(5.0)
            prepare(merged)
            soma_polys[z] = merged

    if verbose:
        print(f"{_p} soma Z-range: [{soma_z_min:.0f}, {soma_z_max:.0f}], "
              f"{len(soma_polys)} Z-levels "
              f"({_time.perf_counter() - t0:.1f}s)")

    # ── 5. Classify vertices ───────────────────────────────────────
    vert_in_soma = np.zeros(len(verts), dtype=bool)
    vert_z = verts[:, 2]

    in_box = (
        (verts[:, 0] >= soma_box_min[0])
        & (verts[:, 0] <= soma_box_max[0])
        & (verts[:, 1] >= soma_box_min[1])
        & (verts[:, 1] <= soma_box_max[1])
    )

    for z, poly in soma_polys.items():
        at_z = np.where((vert_z == z) & in_box)[0]
        if len(at_z) == 0:
            continue
        inside = np.array(
            [poly.contains(Point(p)) for p in verts[at_z, :2]]
        )
        vert_in_soma[at_z] = inside

    if verbose:
        print(f"{_p} {vert_in_soma.sum():,} vertices in soma "
              f"({_time.perf_counter() - t0:.1f}s)")

    # ── 6. Face classification ─────────────────────────────────────
    face_soma = vert_in_soma[faces_arr].any(axis=1)
    clean_main = (face_comp == main_ci) & ~pocket & ~isolated
    soma_mask = face_soma & clean_main

    if verbose:
        print(
            f"{_p} soma: {int(soma_mask.sum()):,} faces "
            f"({_time.perf_counter() - t0:.1f}s)"
        )

    return soma_mask


def has_soma(
    org_c: np.ndarray,
    *,
    verbose: bool = False,
) -> tuple[bool, np.ndarray | None, np.ndarray | None]:
    """Detect whether the cell has a soma based on organelle distribution.

    The soma is characterised by organelles wrapping around a nucleus
    void — an empty interior enclosed by organelles.  On a coarse
    occupancy grid (500-unit resolution), inter-organelle gaps vanish
    and only the nucleus void remains as enclosed empty space.

    Parameters
    ----------
    org_c : np.ndarray
        ``(N, 3)`` organelle face centroids.

    Returns
    -------
    detected : bool
        True if a soma void is detected.
    void_approx : np.ndarray or None
        ``(3,)`` approximate void location (centroid of enclosed
        empty voxels on the coarse grid), or None.
    soma_mask : np.ndarray or None
        ``(N,)`` boolean mask over *org_c* selecting organelles in
        the soma neighbourhood, or None.
    """
    from scipy.ndimage import distance_transform_edt, gaussian_filter

    _p = "[has_soma]"

    if len(org_c) < 50:
        if verbose:
            print(f"{_p} too few organelles ({len(org_c)})")
        return False, None, None

    # ── Coarse occupancy grid (500-unit resolution) ───────────────
    coarse_res = 500
    mins = org_c.min(axis=0)
    bins = ((org_c - mins) / coarse_res).astype(int)
    gs = bins.max(axis=0) + 1
    grid = np.zeros(gs, dtype=np.float32)
    np.add.at(grid, (bins[:, 0], bins[:, 1], bins[:, 2]), 1)

    coarse_occ = grid > 0

    # ── 6-ray enclosure test ──────────────────────────────────────
    # At 500-unit resolution inter-organelle gaps are < 1 voxel and
    # vanish.  Only the nucleus void (5000+ units) creates enclosed
    # empty space.
    enclosed = np.ones(gs, dtype=bool)
    for axis in range(3):
        enclosed &= np.maximum.accumulate(coarse_occ, axis=axis)
        enclosed &= np.flip(
            np.maximum.accumulate(np.flip(coarse_occ, axis=axis), axis=axis),
            axis=axis,
        )
    ee = enclosed & ~coarse_occ
    n_enclosed = int(ee.sum())

    if n_enclosed == 0:
        if verbose:
            print(f"{_p} no enclosed empty voxels — no soma detected")
        return False, None, None

    # ── Approximate void location ─────────────────────────────────
    coords = np.argwhere(ee)
    coarse_dt = distance_transform_edt(~coarse_occ)
    dt_vals = coarse_dt[coords[:, 0], coords[:, 1], coords[:, 2]]

    # DT-weighted centroid of enclosed voxels
    void_vox = np.average(coords.astype(float), axis=0, weights=dt_vals)
    void_approx = mins + void_vox * coarse_res + coarse_res / 2

    # ── Soma-neighbourhood organelles ─────────────────────────────
    crop_radius = 10_000
    soma_mask = np.linalg.norm(org_c - void_approx, axis=1) < crop_radius

    if verbose:
        dt_max = float(dt_vals.max())
        print(
            f"{_p} {n_enclosed} enclosed voxels, "
            f"DT_max={dt_max:.1f} ({dt_max * coarse_res:.0f} units), "
            f"void≈({void_approx[0]:.0f}, {void_approx[1]:.0f}, "
            f"{void_approx[2]:.0f}), {soma_mask.sum():,} soma organelles"
        )

    return True, void_approx, soma_mask


def _find_void_seed(
    org_c: np.ndarray,
    *,
    verbose: bool = False,
) -> np.ndarray | None:
    """Find a rough point inside the nucleus void.

    Not the true centre — biased toward the dense side — but
    reliably inside the void.  Used as the ray-cast seed by
    ``_find_void_organelles``.

    Parameters
    ----------
    org_c : np.ndarray
        ``(N, 3)`` organelle face centroids.

    Returns
    -------
    np.ndarray or None
        ``(3,)`` approximate void location, or None if not found.
    """
    from scipy.ndimage import distance_transform_edt, gaussian_filter

    _p = "[_find_void_seed]"

    # ── Stage 1: density peak → soma neighbourhood ─────────────────
    coarse_res = 500
    mins = org_c.min(axis=0)
    bins = ((org_c - mins) / coarse_res).astype(int)
    grid_shape = bins.max(axis=0) + 1
    grid = np.zeros(grid_shape, dtype=np.float32)
    np.add.at(grid, (bins[:, 0], bins[:, 1], bins[:, 2]), 1)
    grid_smooth = gaussian_filter(grid, sigma=3)

    peak_idx = np.unravel_index(grid_smooth.argmax(), grid_smooth.shape)
    peak = mins + np.array(peak_idx) * coarse_res + coarse_res / 2

    # Crop to local region around density peak
    crop_radius = 10_000
    local = org_c[np.linalg.norm(org_c - peak, axis=1) < crop_radius]

    if len(local) < 50:
        if verbose:
            print(f"{_p} too few local organelles ({len(local)})")
        return None

    if verbose:
        print(
            f"{_p} density peak: ({peak[0]:.0f}, {peak[1]:.0f}, "
            f"{peak[2]:.0f}), {len(local):,} local organelles"
        )

    # ── Stage 2: enclosure-based void detection ────────────────────
    vox_res = 300
    vm = local.min(axis=0) - vox_res * 3
    vb = ((local - vm) / vox_res).astype(int)
    vs = vb.max(axis=0) + 4
    occ = np.zeros(vs, dtype=bool)
    occ[vb[:, 0], vb[:, 1], vb[:, 2]] = True

    # Distance to nearest organelle at each empty voxel
    dt = distance_transform_edt(~occ)

    # Enclosure: for each axis, an occupied voxel must exist both
    # before and after the current voxel along that axis.
    enclosed = np.ones(vs, dtype=bool)
    for axis in range(3):
        enclosed &= np.maximum.accumulate(occ, axis=axis)
        enclosed &= np.flip(
            np.maximum.accumulate(np.flip(occ, axis=axis), axis=axis),
            axis=axis,
        )

    # Enclosed empty voxels — the nucleus interior
    mask = enclosed & ~occ
    coords = np.argwhere(mask)

    if len(coords) == 0:
        if verbose:
            print(f"{_p} no enclosed empty voxels found, "
                  f"falling back to density peak")
        return peak.copy()

    # Weighted centroid (deeper = more weight) centres the result
    weights = dt[mask]
    center = vm + np.average(coords, axis=0, weights=weights) * vox_res \
        + vox_res / 2

    if verbose:
        dt_max = float(weights.max())
        print(
            f"{_p} void: {len(coords)} enclosed voxels, "
            f"DT_max={dt_max:.1f} vox ({dt_max * vox_res:.0f} units), "
            f"centre=({center[0]:.0f}, {center[1]:.0f}, {center[2]:.0f})"
        )

    return center


def _find_void_organelles(
    org_c: np.ndarray,
    *,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Identify organelle centroids surrounding the nucleus void.

    Balloon / ray-cast approach:

    1. Get a rough seed inside the void from ``_find_void_seed``
       (enclosure + DT-weighted centroid — imperfect but reliably
       inside the void).
    2. Build an occupancy grid and cast rays uniformly from the
       seed.  The first organelle voxel each ray hits is a
       void-surface organelle.  Rays that escape through the
       pocket mouth are ignored.

    Parameters
    ----------
    org_c : np.ndarray
        ``(N, 3)`` organelle face centroids.

    Returns
    -------
    mask : np.ndarray
        ``(N,)`` boolean — True for void-surrounding organelles.
    box_min : np.ndarray
        ``(3,)`` lower corner of selected-organelle bounding box.
    box_max : np.ndarray
        ``(3,)`` upper corner of selected-organelle bounding box.
    """
    from scipy.ndimage import distance_transform_edt, gaussian_filter

    _p = "[_find_void_organelles]"
    out = np.zeros(len(org_c), dtype=bool)
    nan3 = np.full(3, np.nan)

    # ── Rough seed from the old void-center logic ─────────────────
    seed_world = _find_void_seed(org_c, verbose=verbose)
    if seed_world is None:
        if verbose:
            print(f"{_p} no seed — _find_void_seed returned None")
        return out, nan3, nan3

    # ── Crop organelles around seed ───────────────────────────────
    crop_radius = 10_000
    local_mask = np.linalg.norm(org_c - seed_world, axis=1) < crop_radius
    local = org_c[local_mask]

    if len(local) < 50:
        if verbose:
            print(f"{_p} too few local organelles ({len(local)})")
        return out, nan3, nan3

    # ── Occupancy grid ────────────────────────────────────────────
    vox_res = 300
    vm = local.min(axis=0) - vox_res * 3
    vb = ((local - vm) / vox_res).astype(int)
    vs = vb.max(axis=0) + 4
    occ = np.zeros(vs, dtype=bool)
    occ[vb[:, 0], vb[:, 1], vb[:, 2]] = True

    # Seed in voxel coordinates
    seed = (seed_world - vm) / vox_res

    # ── Ray-cast from seed in all directions ──────────────────────
    # Fibonacci sphere sampling for uniform coverage.
    n_rays = 2000
    golden = (1 + np.sqrt(5)) / 2
    idx = np.arange(n_rays, dtype=float)
    theta = 2 * np.pi * idx / golden
    phi = np.arccos(1 - 2 * (idx + 0.5) / n_rays)
    dirs = np.column_stack([
        np.sin(phi) * np.cos(theta),
        np.sin(phi) * np.sin(theta),
        np.cos(phi),
    ])  # (n_rays, 3)

    # March all rays in parallel — step size 0.5 voxels
    max_r = float(max(vs))
    n_steps = int(np.ceil(max_r / 0.5))
    t = np.arange(1, n_steps + 1, dtype=float) * 0.5  # (T,)

    # positions: (n_rays, T, 3)
    pos = seed[None, None, :] + dirs[:, None, :] * t[None, :, None]
    ix = np.round(pos).astype(np.int32)

    # Bounds check
    valid = (
        (ix[:, :, 0] >= 0) & (ix[:, :, 0] < vs[0]) &
        (ix[:, :, 1] >= 0) & (ix[:, :, 1] < vs[1]) &
        (ix[:, :, 2] >= 0) & (ix[:, :, 2] < vs[2])
    )

    # Safe indices for look-up (out-of-bounds → 0, masked later)
    sx = np.where(valid, ix[:, :, 0], 0)
    sy = np.where(valid, ix[:, :, 1], 0)
    sz = np.where(valid, ix[:, :, 2], 0)

    is_occ = occ[sx, sy, sz] & valid  # (n_rays, T)

    # First hit along each ray
    has_hit = is_occ.any(axis=1)
    first_t = np.argmax(is_occ, axis=1)  # index of first True

    hit_ix = ix[np.arange(n_rays), first_t]  # (n_rays, 3)
    hit_ix = hit_ix[has_hit]

    # Unique hit voxels → set for fast lookup
    hit_set = set(map(tuple, hit_ix.tolist()))

    if not hit_set:
        if verbose:
            print(f"{_p} no organelle hit by any ray")
        return out, nan3, nan3

    # Map hit voxels back to org_c indices
    local_indices = np.where(local_mask)[0]
    for i in range(len(vb)):
        if tuple(vb[i]) in hit_set:
            out[local_indices[i]] = True

    if out.any():
        sel = org_c[out]
        box_min, box_max = sel.min(axis=0), sel.max(axis=0)
    else:
        box_min, box_max = nan3.copy(), nan3.copy()

    if verbose:
        n_hit = has_hit.sum()
        n_escaped = n_rays - n_hit
        print(
            f"{_p} {out.sum():,} / {len(org_c):,} void-surrounding "
            f"organelles ({len(hit_set)} hit voxels, "
            f"{n_hit} rays hit, {n_escaped} escaped)"
        )

    return out, box_min, box_max


def _find_void_center(
    org_c: np.ndarray,
    *,
    verbose: bool = False,
) -> np.ndarray | None:
    """Find the centre of the nucleus void.

    Midpoint of the bounding box of void-surrounding organelles
    (from ``_find_void_organelles``).  Equidistant from the extremes
    on each axis — immune to organelle density bias.

    Parameters
    ----------
    org_c : np.ndarray
        ``(N, 3)`` organelle face centroids.

    Returns
    -------
    np.ndarray or None
        ``(3,)`` void centre in world coordinates, or None if not found.
    """
    _p = "[_find_void_center]"
    void_org, box_min, box_max = _find_void_organelles(org_c, verbose=verbose)

    if not void_org.any() or np.any(np.isnan(box_min)):
        if verbose:
            print(f"{_p} no void organelles found")
        return None

    center = (box_min + box_max) / 2

    if verbose:
        print(
            f"{_p} centre=({center[0]:.0f}, {center[1]:.0f}, "
            f"{center[2]:.0f})"
        )

    return center


def _find_void_bounding_box(
    org_c: np.ndarray,
    void_center: np.ndarray,
    *,
    bin_size: float = 500,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Find the soma bounding box by expanding from the void centre.

    For each axis, bin organelle centroids into slabs and expand outward
    from the void centre while the count stays above a fraction of the
    peak count.  The box is naturally centred on the nucleus.

    Parameters
    ----------
    org_c : np.ndarray
        ``(N, 3)`` organelle face centroids.
    void_center : np.ndarray
        ``(3,)`` void centre in world coordinates.
    bin_size : float
        Width of each slab along each axis.

    Returns
    -------
    box_min, box_max : np.ndarray, np.ndarray
        ``(3,)`` arrays — lower and upper corners of the bounding box.
    """
    _p = "[_find_void_bounding_box]"
    half_bin = bin_size / 2
    padding = 1000.0

    # Crop to soma neighbourhood — exclude distant neurite organelles
    crop_radius = 10_000
    local = org_c[np.linalg.norm(org_c - void_center, axis=1) < crop_radius]

    box_min = np.full(3, -np.inf)
    box_max = np.full(3, np.inf)

    for axis in range(3):
        ax_vals = local[:, axis]

        if len(ax_vals) == 0:
            continue

        lo = ax_vals.min() - bin_size
        hi = ax_vals.max() + bin_size
        edges = np.arange(lo, hi + bin_size, bin_size)
        counts, edges = np.histogram(ax_vals, bins=edges)
        centres = (edges[:-1] + edges[1:]) / 2

        if len(counts) == 0 or counts.max() == 0:
            continue

        # Find the slab closest to the void centre
        vc_idx = int(np.argmin(np.abs(centres - void_center[axis])))

        # Expand outward from void centre in both directions.
        # Collect the full profile, then find the steepest relative
        # drop — the soma/neurite boundary.  No fixed threshold.
        for direction in [-1, +1]:
            rng = (range(vc_idx - 1, -1, -1) if direction == -1
                   else range(vc_idx + 1, len(counts)))
            indices = list(rng)
            if not indices:
                continue

            # Smooth the profile with a 3-bin running mean to
            # reduce noise before looking for the steepest drop.
            vals = counts[indices].astype(float)
            if len(vals) >= 3:
                smoothed = np.convolve(vals, np.ones(3) / 3, mode="same")
            else:
                smoothed = vals

            # Find the steepest relative drop: (prev - cur) / prev.
            # The soma boundary is the first large drop from a
            # still-high level.
            best_drop_ratio = 0.0
            best_drop_pos = len(indices) - 1
            for j in range(1, len(smoothed)):
                prev = smoothed[j - 1]
                if prev <= 0:
                    continue
                drop_ratio = (prev - smoothed[j]) / prev
                if drop_ratio > best_drop_ratio:
                    best_drop_ratio = drop_ratio
                    best_drop_pos = j

            # The box extends to the bin just before the steepest drop
            stop_idx = indices[max(best_drop_pos - 1, 0)]
            boundary = centres[stop_idx]
            if direction == -1:
                box_min[axis] = boundary - half_bin - padding
            else:
                box_max[axis] = boundary + half_bin + padding

    if verbose:
        print(
            f"{_p} box: "
            f"X=[{box_min[0]:.0f},{box_max[0]:.0f}] "
            f"Y=[{box_min[1]:.0f},{box_max[1]:.0f}] "
            f"Z=[{box_min[2]:.0f},{box_max[2]:.0f}]"
        )

    return box_min, box_max


def _section_contours(
    mesh: trimesh.Trimesh,
    z: float,
) -> list[tuple[float, np.ndarray]]:
    """Return ``[(area, vertices_Nx3), ...]`` for each contour at *z*."""
    section = mesh.section(
        plane_origin=[0, 0, z], plane_normal=[0, 0, 1]
    )
    if section is None:
        return []
    sv = section.vertices
    contours: list[tuple[float, np.ndarray]] = []
    for entity in section.entities:
        pts = sv[entity.points]
        if len(pts) < 3:
            continue
        x, y = pts[:, 0], pts[:, 1]
        area = 0.5 * abs(
            np.sum(x[:-1] * y[1:] - x[1:] * y[:-1])
            + x[-1] * y[0]
            - x[0] * y[-1]
        )
        contours.append((area, pts))
    return contours
