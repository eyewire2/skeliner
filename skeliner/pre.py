"""skeliner.pre – mesh preprocessing utilities."""

from collections import defaultdict

import igraph as ig
import numpy as np
import trimesh
from scipy.spatial import KDTree

from skeliner.dataclass import Soma

__all__ = [
    "ensure_watertight",
    "fill_holes",
    "find_disconnected",
    "find_gaps",
    "find_holes",
    "find_soma",
    "remove_fins",
    "remove_fragments",
    "remove_fusions",
    "remove_islands",
    "remove_nucleus",
    "remove_organelles",
]


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
    edge_to_faces: dict[tuple[int, int], list[int]] = defaultdict(list)
    for fi, face in enumerate(mesh.faces):
        for i in range(3):
            e = (
                min(int(face[i]), int(face[(i + 1) % 3])),
                max(int(face[i]), int(face[(i + 1) % 3])),
            )
            edge_to_faces[e].append(fi)

    n = len(mesh.faces)
    labels = np.full(n, -1, dtype=np.intp)
    comp_id = 0
    best_id, best_size = 0, 0

    for seed in range(n):
        if labels[seed] >= 0:
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
    edge_to_faces: dict[tuple[int, int], list[int]] = defaultdict(list)
    for fi, face in enumerate(mesh.faces):
        for i in range(3):
            e = (
                min(int(face[i]), int(face[(i + 1) % 3])),
                max(int(face[i]), int(face[(i + 1) % 3])),
            )
            edge_to_faces[e].append(fi)

    # Boundary edges whose face is in the main component
    boundary_edges = []
    for e, fis in edge_to_faces.items():
        if len(fis) == 1 and face_comp[fis[0]] == main_comp:
            boundary_edges.append(e)

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
        adj_fi = [
            fi
            for fi, face in enumerate(mesh.faces)
            if set(int(v) for v in face) & loop_set
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

    # Pre-build vert_to_faces for orientation
    vert_to_faces: list[list[int]] = [[] for _ in range(len(mesh.vertices))]
    for fi, face in enumerate(mesh.faces):
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
    for fi, face in enumerate(mesh.faces):
        if fi not in sel:
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

    keep = np.ones(len(mesh.faces), dtype=bool)
    for fi in sel:
        keep[fi] = False
    kept_faces = mesh.faces[keep]

    # Combine existing + new vertices
    if new_verts:
        all_vertices = np.vstack([mesh.vertices, np.array(new_verts)])
    else:
        all_vertices = mesh.vertices.copy()

    if all_stitch:
        stitch_faces = np.array(all_stitch, dtype=np.int64)
        all_faces = np.vstack([kept_faces, stitch_faces])
    else:
        all_faces = kept_faces

    result = trimesh.Trimesh(
        vertices=all_vertices,
        faces=all_faces,
        process=False,
    )
    result.remove_unreferenced_vertices()

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
    edge_to_faces: dict[tuple[int, int], list[int]] = defaultdict(list)
    for fi, face in enumerate(mesh.faces):
        for i in range(3):
            e = (
                min(int(face[i]), int(face[(i + 1) % 3])),
                max(int(face[i]), int(face[(i + 1) % 3])),
            )
            edge_to_faces[e].append(fi)

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
        result = trimesh.Trimesh(
            vertices=mesh.vertices,
            faces=mesh.faces[keep],
            process=False,
        )
        result.remove_unreferenced_vertices()
        return result

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
    work = active.copy()
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


def find_soma(
    mesh: trimesh.Trimesh,
    *,
    organelle_mask: np.ndarray | None = None,
    verbose: bool = False,
) -> Soma | None:
    """Estimate soma from the spatial clustering of organelles.

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
    organelle_mask : np.ndarray or None
        Pre-computed boolean face mask from :func:`find_organelles`
        (``pocket | isolated``).  If provided, organelle detection is
        skipped and this mask is used directly.
    verbose : bool, default False
        Print summary.

    Returns
    -------
    Soma or None
        Fitted ellipsoidal soma, or *None* if too few indicators exist.
    """
    from collections import deque

    labels, main = _face_edge_components(mesh)

    # ── 1. Locate organelle clusters and compute centroids ──────
    if organelle_mask is None:
        pocket, isolated = find_organelles(mesh, verbose=verbose)
        organelle_mask = pocket | isolated
    if organelle_mask.sum() == 0:
        if verbose:
            print("[skeliner.pre] Soma: no organelles found")
        return None

    org_fi = np.where(organelle_mask)[0]
    org_labels, _ = _face_edge_components(
        trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces[org_fi], process=False)
    )
    centroids = []
    for cid in np.unique(org_labels):
        local_fi = org_fi[org_labels == cid]
        verts = np.unique(mesh.faces[local_fi])
        centroids.append(mesh.vertices[verts].mean(axis=0))

    centroids = np.asarray(centroids)
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

    # ── 2. Build main-component vertex adjacency ─────────────────────
    main_fi = np.where(labels == main)[0]
    adj: dict[int, list[int]] = defaultdict(list)
    for fi in main_fi:
        v = mesh.faces[fi]
        for i in range(3):
            a, b = int(v[i]), int(v[(i + 1) % 3])
            adj[a].append(b)
            adj[b].append(a)

    # ── 3. BFS from nearest vertex to centre (no hard cap) ───────
    main_verts = np.fromiter(adj.keys(), dtype=np.intp)
    seed = int(
        main_verts[
            np.argmin(np.linalg.norm(mesh.vertices[main_verts] - center, axis=1))
        ]
    )

    ring_level: dict[int, int] = {seed: 0}
    queue: deque[int] = deque([seed])
    ring_verts: dict[int, list[int]] = defaultdict(list)
    ring_verts[0].append(seed)

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

    soma = Soma.fit(mesh.vertices[bfs_verts_arr])

    # ── 6. Assign verts: main-component vertices inside the ellipsoid
    all_main_verts = np.unique(mesh.faces[main_fi])
    inside = soma.contains(mesh.vertices[all_main_verts])
    soma_set: set[int] = set(all_main_verts[inside].tolist())

    # ── 7. Dilate the soma boundary, then absorb pockets ──────────
    #       The dilation distance and pocket absorption distance are
    #       derived from mesh resolution: a few average edge lengths
    #       expressed in body-coord units.
    avg_edge = float(mesh.edges_unique_length.mean())
    edge_in_body = avg_edge / float(soma.axes.min())
    dilation_limit = 1.0 + 3.0 * edge_in_body

    # 7a. Dilate: grow soma boundary along mesh edges, accepting
    #     neighbours within a few edge-lengths of the ellipsoid surface.
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
        soma_set.update(accept.tolist())

    # 7b. Absorb pockets: connected components of non-soma verts
    #     that are topologically trapped — their only path to the
    #     rest of the mesh goes through the soma.
    all_main_set = set(all_main_verts.tolist())
    for _iteration in range(10):
        outside = all_main_set - soma_set
        visited: set[int] = set()
        absorbed = 0
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
            # Trapped = every neighbour of the component is either
            # inside the component itself or inside the soma.
            # Only absorb if the component is small relative to the
            # current soma (a pocket, not a neurite branch).
            comp_set = set(comp)
            trapped = all(
                nv in soma_set or nv in comp_set for v in comp for nv in adj.get(v, [])
            )
            if trapped and len(comp) < len(soma_set):
                soma_set.update(comp)
                absorbed += len(comp)
        if absorbed == 0:
            break

    # ── 8. Absorb disconnected-component vertices inside the ellipsoid
    #       Small fragments (failed organelle removals) that sit inside
    #       the soma region should be part of the soma, not left as
    #       stray vertices that create noise in downstream binning.
    all_verts = np.arange(len(mesh.vertices))
    non_main_mask = np.ones(len(mesh.vertices), dtype=bool)
    non_main_mask[all_main_verts] = False
    non_main_verts = all_verts[non_main_mask]
    if non_main_verts.size:
        inside_non_main = soma.contains(mesh.vertices[non_main_verts])
        soma_set.update(non_main_verts[inside_non_main].tolist())

    # ── 9. Refit ellipsoid to the final soma vertices ───────────
    soma_verts_arr = np.fromiter(sorted(soma_set), dtype=np.intp)
    if len(soma_verts_arr) >= 4:
        try:
            soma = Soma.fit(mesh.vertices[soma_verts_arr], verts=soma_verts_arr)
        except ValueError:
            soma.verts = soma_verts_arr
    else:
        soma.verts = soma_verts_arr

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


def find_disconnected(
    mesh: trimesh.Trimesh,
    *,
    verbose: bool = False,
    _precomputed_soma: Soma | None = None,
) -> list[list[int]]:
    """Detect disconnected mesh components from segmentation errors.

    Returns disconnected components — broken neurite segments that are
    separate from the main mesh.  Soma-region components and components
    enclosed by the main mesh (residual organelles) are excluded.
    Should be run after remove_organelles.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Input mesh.
    verbose : bool, default False
        Print summary.
    _precomputed_soma : Soma or None
        Pre-computed soma from :func:`find_soma`.

    Returns
    -------
    list[list[int]]
        Each element is a list of face indices for one disconnected
        component, sorted largest-first.
    """
    labels, main = _face_edge_components(mesh)
    n_faces = len(mesh.faces)

    # Locate soma so we can exclude components inside it
    soma = (
        _precomputed_soma
        if _precomputed_soma is not None
        else find_soma(mesh, verbose=verbose)
    )

    # Build KD-tree of main-component face centroids + normals
    # for inside/outside classification
    main_face_idx = np.where(labels == main)[0]
    main_centroids = mesh.triangles_center[main_face_idx]
    main_normals = mesh.face_normals[main_face_idx]
    main_tree = KDTree(main_centroids)

    # Collect non-main components
    comp_faces: dict[int, list[int]] = {}
    for fi in range(n_faces):
        cid = int(labels[fi])
        if cid == main:
            continue
        comp_faces.setdefault(cid, []).append(fi)

    components = []
    n_soma_excluded = 0
    n_enclosed_excluded = 0
    for cid, fis in comp_faces.items():
        # Need at least 7 faces: 3 for each tip + 1 body face to
        # bridge back to two other parts
        if len(fis) < 7:
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
    _precomputed_soma: Soma | None = None,
    _precomputed_disconnected: list[list[int]] | None = None,
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
    _precomputed_soma : Soma or None
        Pre-computed soma from :func:`find_soma`.
    _precomputed_disconnected : list[list[int]] or None
        Pre-computed disconnected components from :func:`find_disconnected`.

    Returns
    -------
    list[tuple[list[int], list[int], float]]
        Each element is ``(faces_a, faces_b, gap_distance)`` where
        *faces_a* and *faces_b* are face-index lists on each side of
        the gap, sorted by gap distance (smallest first).
    """
    labels, main = _face_edge_components(mesh)

    # Get disconnected components (reuse filtering logic)
    if _precomputed_disconnected is not None:
        disc = _precomputed_disconnected
    else:
        disc = find_disconnected(
            mesh,
            verbose=verbose,
            _precomputed_soma=_precomputed_soma,
        )

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

    # For each disconnected component, find its nearest neighbour.
    # Deduplicate: if A→B and B→A both exist, keep only one.
    gaps = []
    seen_pairs: set[tuple[int, int]] = set()
    disc_cids = [int(labels[fis[0]]) for fis in disc]
    all_cids = [main] + disc_cids

    # Build face adjacency for BFS-based tip selection
    from collections import deque as _deque

    edge_to_faces_gap: dict[tuple[int, int], list[int]] = defaultdict(list)
    for fi, face in enumerate(mesh.faces):
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
            gaps.append((fa, fb, dist))

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

    # Sort by gap distance
    gaps.sort(key=lambda x: x[2])

    if verbose:
        for i, (fa, fb, dist) in enumerate(gaps):
            ca = mesh.vertices[mesh.faces[fa]].mean(axis=(0, 1))
            print(
                f"  gap {i}: {len(fa)}f ↔ {len(fb)}f, "
                f"dist={dist:.0f}, "
                f"near [{ca[0]:.0f}, {ca[1]:.0f}, {ca[2]:.0f}]"
            )

    return gaps


def remove_gaps(
    mesh: trimesh.Trimesh,
    *,
    min_faces: int = 100,
    verbose: bool = False,
    _precomputed_soma: Soma | None = None,
    _precomputed_gaps: list | None = None,
) -> trimesh.Trimesh:
    """Bridge all detected gaps in a single mesh rebuild.

    All gap tip faces are removed at once and boundary loops from each
    gap are paired and zipper-stitched, then the mesh is rebuilt once.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Input mesh.
    min_faces : int, default 100
        Passed to :func:`find_gaps`.
    verbose : bool, default False
        Print progress.
    _precomputed_soma : Soma or None
        Pre-computed soma.
    _precomputed_gaps : list or None
        Pre-computed gaps from :func:`find_gaps`.

    Returns
    -------
    trimesh.Trimesh
        Mesh with gaps bridged.
    """
    if _precomputed_gaps is not None:
        gaps = _precomputed_gaps
    else:
        gaps = find_gaps(
            mesh,
            min_faces=min_faces,
            verbose=verbose,
            _precomputed_soma=_precomputed_soma,
        )

    if not gaps:
        if verbose:
            print("[skeliner.pre] No gaps to bridge")
        return mesh

    # Build edge map once
    edge_to_faces: dict[tuple[int, int], list[int]] = defaultdict(list)
    for fi, face in enumerate(mesh.faces):
        for i in range(3):
            e = (
                min(int(face[i]), int(face[(i + 1) % 3])),
                max(int(face[i]), int(face[(i + 1) % 3])),
            )
            edge_to_faces[e].append(fi)

    if verbose:
        print(f"[skeliner.pre] Bridging {len(gaps)} gaps")

    # For each gap, trace its border loops. Only collect faces to
    # remove for gaps that successfully produce a loop pair.
    faces_to_remove: set[int] = set()
    loop_pairs: list[tuple[list[int], list[int]]] = []
    for gap_i, (faces_a, faces_b, dist) in enumerate(gaps):
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

    clean = mesh.submesh([np.where(~islands)[0]], append=True)
    clean.remove_unreferenced_vertices()

    if verbose:
        print(
            f"[skeliner.pre] Removed islands: {n_removed} faces, "
            f"{len(mesh.vertices) - len(clean.vertices)} verts"
        )
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

    result = mesh.submesh([np.where(~fins)[0]], append=True)
    result.remove_unreferenced_vertices()

    if verbose:
        print(f"[skeliner.pre] Removed {n_removed} fin faces")

    return result


def remove_fragments(
    mesh: trimesh.Trimesh,
    *,
    min_faces: int = 3,
    verbose: bool = False,
    _precomputed: np.ndarray | None = None,
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
    if _precomputed is not None:
        fragment = _precomputed
    else:
        fragment = find_fragments(mesh, min_faces=min_faces, verbose=verbose)

    n_removed = int(fragment.sum())
    if n_removed == 0:
        return mesh

    result = mesh.submesh([np.where(~fragment)[0]], append=True)
    result.remove_unreferenced_vertices()
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
    edge_to_faces: dict[tuple[int, int], list[int]] = defaultdict(list)
    for fi in flagged:
        f = mesh.faces[fi]
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
        f = mesh.faces[fi]
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


def _organelle_precompute(
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
    _precomputed: tuple | None = None,
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

    if _precomputed is not None:
        outward_dots, _, _, main_face_mask = _precomputed
    else:
        outward_dots, _, _, main_face_mask = _organelle_precompute(
            mesh,
            radius,
            radius_multiplier,
            verbose,
        )
    n_faces = len(mesh.faces)
    adj = _face_adjacency(mesh)

    # Build edge-to-face map
    edge_to_face: dict[tuple[int, int], list[int]] = defaultdict(list)
    for fi, f in enumerate(mesh.faces):
        for i in range(3):
            a, b = int(f[i]), int(f[(i + 1) % 3])
            edge_to_face[(min(a, b), max(a, b))].append(fi)

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
    rims: list[list[tuple[int, int]]] = []
    for cluster in clusters:
        if len(cluster) < min_pocket_size:
            continue
        cset = set(cluster)
        boundary_edges: list[tuple[int, int]] = []
        seen_edges: set[tuple[int, int]] = set()
        for fi in cluster:
            f = mesh.faces[fi]
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
        pocket_area = float(mesh.area_faces[cluster].sum())
        opening_area = _rim_enclosed_area(boundary_edges, mesh.vertices)
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


def _face_adjacency(mesh: trimesh.Trimesh) -> dict[int, set[int]]:
    """Build face adjacency map (edge-connected neighbors)."""
    from collections import defaultdict

    edge_to_face: dict[tuple[int, int], list[int]] = defaultdict(list)
    for fi, f in enumerate(mesh.faces):
        for i in range(3):
            a, b = int(f[i]), int(f[(i + 1) % 3])
            edge_to_face[(min(a, b), max(a, b))].append(fi)

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
    _precomputed: tuple | None = None,
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

    if _precomputed is not None:
        outward_dots, _, _, main_face_mask = _precomputed
    else:
        outward_dots, _, _, main_face_mask = _organelle_precompute(
            mesh,
            radius,
            radius_multiplier,
            verbose,
        )
    n_faces = len(mesh.faces)
    adj = _face_adjacency(mesh)

    # Use find_rims to get rim edges for each pocket
    rims = find_rims(
        mesh,
        radius=radius,
        radius_multiplier=radius_multiplier,
        min_pocket_size=min_pocket_size,
        min_fold_ratio=min_fold_ratio,
        verbose=verbose,
        _precomputed=_precomputed,
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

    # Build edge-to-face map
    edge_to_face: dict[tuple[int, int], list[int]] = defaultdict(list)
    for fi, f in enumerate(mesh.faces):
        for i in range(3):
            a, b = int(f[i]), int(f[(i + 1) % 3])
            edge_to_face[(min(a, b), max(a, b))].append(fi)

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


def find_isolated_organelles(
    mesh: trimesh.Trimesh,
    *,
    radius: float | None = None,
    radius_multiplier: float = 5.0,
    verbose: bool = False,
    _precomputed: tuple | None = None,
) -> np.ndarray:
    """Detect vertex-disconnected internal fragments.

    These are organelle membranes that form entirely separate connected
    components enclosed within the neuron body.

    Returns
    -------
    np.ndarray
        Boolean mask ``(nFaces,)`` — isolated organelle faces.
    """
    if _precomputed is not None:
        outward_dots, face_comp, main_ci, _ = _precomputed
    else:
        outward_dots, face_comp, main_ci, _ = _organelle_precompute(
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

    # Two-pass approach:
    # Pass 1: check every non-main component against the main component.
    #         Components classified as external are "structural" (disconnected
    #         neurites, offset layers).
    # Pass 2: re-check components that were external in pass 1 against ALL
    #         structural surfaces (main + pass-1 externals).  This catches
    #         organelles inside disconnected structural components.
    main_face_idx = np.where(face_comp == main_ci)[0]
    main_centroids = mesh.triangles_center[main_face_idx]
    main_normals = mesh.face_normals[main_face_idx]
    main_tree = KDTree(main_centroids)

    external_cis: set[int] = set()

    # Pass 1: classify against main component
    for ci in range(n_comps):
        if ci == main_ci:
            continue
        comp_face_idx = np.where(face_comp == ci)[0]
        if len(comp_face_idx) == 0:
            continue
        comp_verts = np.unique(mesh.faces[comp_face_idx])
        coords = mesh.vertices[comp_verts]
        _, nn_idx = main_tree.query(coords)
        vecs = coords - main_centroids[nn_idx]
        dots = np.einsum("ij,ij->i", vecs, main_normals[nn_idx])
        is_internal = (dots < 0).sum() > len(dots) / 2
        if is_internal:
            isolated[comp_face_idx] = True
            n_internal_frags += 1
            n_internal_frag_faces += len(comp_face_idx)
        else:
            external_cis.add(ci)
            n_kept_frags += 1

    # Pass 2: build ONE combined surface from main + all externals.
    # For each external component, query against it.  If the nearest
    # face belongs to the same component, use the second nearest.
    if external_cis:
        ext_face_list = []
        ext_face_comp_ids = []
        for eci in external_cis:
            eci_faces = np.where(face_comp == eci)[0]
            ext_face_list.append(eci_faces)
            ext_face_comp_ids.append(np.full(len(eci_faces), eci, dtype=np.intp))

        all_struct_faces = np.concatenate([main_face_idx] + ext_face_list)
        all_struct_comp = np.concatenate(
            [np.full(len(main_face_idx), main_ci, dtype=np.intp)] + ext_face_comp_ids
        )
        all_struct_centroids = mesh.triangles_center[all_struct_faces]
        all_struct_normals = mesh.face_normals[all_struct_faces]
        struct_tree = KDTree(all_struct_centroids)

        newly_isolated = set()
        for ci in list(external_cis):
            comp_face_idx = np.where(face_comp == ci)[0]
            comp_verts = np.unique(mesh.faces[comp_face_idx])
            coords = mesh.vertices[comp_verts]
            # Query k=2 so we can skip self-matches
            _, nn_idx = struct_tree.query(coords, k=2)
            # For each vertex, prefer the nearest face NOT from ci
            k0_comp = all_struct_comp[nn_idx[:, 0]]
            use_k1 = k0_comp == ci
            chosen = np.where(use_k1, nn_idx[:, 1], nn_idx[:, 0])
            vecs = coords - all_struct_centroids[chosen]
            dots = np.einsum("ij,ij->i", vecs, all_struct_normals[chosen])
            is_internal = (dots < 0).sum() > len(dots) / 2
            if is_internal:
                isolated[comp_face_idx] = True
                newly_isolated.add(ci)
                n_internal_frags += 1
                n_internal_frag_faces += len(comp_face_idx)
                n_kept_frags -= 1

    if verbose:
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
    _precomputed: tuple | None = None,
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
    if _precomputed is not None:
        precomputed = _precomputed
    else:
        precomputed = _organelle_precompute(
            mesh, radius, radius_multiplier, verbose,
        )
    _, face_comp, main_ci, _ = precomputed

    # ── 2. Find isolated organelles (small internal components) ───
    isolated = find_isolated_organelles(
        mesh, radius=radius, radius_multiplier=radius_multiplier,
        verbose=verbose, _precomputed=precomputed,
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
            + ("..." if len(sizes) > 5 else "") + ")"
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
        structural_mask, # structural_face_mask (replaces main_face_mask)
    )

    pocket = find_pocket_organelles(
        mesh, radius=radius, radius_multiplier=radius_multiplier,
        min_cluster_size=min_cluster_size,
        verbose=verbose, _precomputed=precomputed_structural,
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


def find_fusions(
    mesh: trimesh.Trimesh,
    *,
    radius: float | None = None,
    radius_multiplier: float = 5.0,
    grow_rings: int = 20,
    min_branch_size: int = 5,
    verbose: bool = False,
) -> list[list[int]]:
    """Detect fusion points where two branches are wrongly connected.

    A fusion is any place where two separated components are wrongly
    connected via shared edges or shared vertices.

    Algorithm:

    1. Find non-manifold seed faces: negative-outward-dot faces at edges
       shared by >2 faces, plus duplicate faces with >3 neighbors.
    2. Cluster seeds by adjacency.
    3. For each seed cluster, grow a local region outward ring by ring.
    4. Split the region using **manifold-only** edge connectivity — at
       a fusion, non-manifold edges connect the two branches, so removing
       them from the adjacency separates the branches.
    5. Report the boundary faces between the two largest components as
       the fusion zone.
    6. Detect **vertex-only fusions**: vertices whose face fan splits
       into multiple edge-disconnected components (pinch vertices).
       Report the fan faces as additional fusion clusters.

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
        Each inner list is one fusion cluster (boundary face indices
        between the two branches).
    """
    from collections import Counter, deque

    areas = mesh.area_faces
    zero_faces = set(np.where(areas < 1e-6)[0].tolist())

    # Build edge-to-face map, excluding zero-area faces
    edge_to_face: dict[tuple[int, int], list[int]] = defaultdict(list)
    for fi, f in enumerate(mesh.faces):
        if fi in zero_faces:
            continue
        for i in range(3):
            a, b = int(f[i]), int(f[(i + 1) % 3])
            edge_to_face[(min(a, b), max(a, b))].append(fi)

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

    # Outward dot for inward-face filtering
    if verbose:
        print("[skeliner.pre] Computing outward dots ...")
    outward_dots = _outward_dot(
        mesh,
        radius
        if radius is not None
        else radius_multiplier * float(np.median(mesh.edges_unique_length)),
    )

    # ── Seed detection ───────────────────────────────────────────────
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

    # Signal 3: fan vertices — vertices whose face fan splits into
    # multiple edge-connected components (vertex-only fusion)
    vert_to_face: list[list[int]] = [[] for _ in range(len(mesh.vertices))]
    for fi, f in enumerate(mesh.faces):
        if fi in zero_faces:
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

        # BFS from first face; if not all fan faces reached,
        # this vertex is a pinch point between disconnected branches.
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
                # count components for reporting
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

    if not seed_faces and not fan_vertex_clusters:
        if verbose:
            print("[skeliner.pre] No fusions found")
        return []

    # ── Cluster seeds ────────────────────────────────────────────────
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

    # ── Grow region & split per cluster ──────────────────────────────
    def _manifold_components(
        region: set[int],
    ) -> list[set[int]]:
        """Split region into components using manifold edges only."""
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

        # Grow outward until manifold-split gives 2 big components
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
                # Boundary faces between the two largest components
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
    result_clusters.sort(key=len, reverse=True)

    if verbose:
        print(
            f"[skeliner.pre] Fusions: {len(result_clusters)} regions, "
            f"{sum(len(c) for c in result_clusters)} boundary faces"
        )

    return result_clusters


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

    vert_to_face: list[list[int]] = [[] for _ in range(len(mesh.vertices))]
    for fi, f in enumerate(mesh.faces):
        for v in f:
            vert_to_face[int(v)].append(fi)

    edge_to_face: dict[tuple[int, int], list[int]] = defaultdict(list)
    for fi, f in enumerate(mesh.faces):
        for i in range(3):
            a, b = int(f[i]), int(f[(i + 1) % 3])
            edge_to_face[(min(a, b), max(a, b))].append(fi)

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
        mesh = mesh.submesh([np.where(keep)[0]], append=True)
        mesh.remove_unreferenced_vertices()
        if verbose:
            print(
                f"[skeliner.pre] Removed {len(all_fusion)} fusion faces "
                f"({len(clusters)} regions)"
            )

    # Step 2: split shared vertices
    mesh = _split_fan_vertices(mesh, verbose=verbose)

    return mesh


def remove_organelles(
    mesh: trimesh.Trimesh,
    *,
    organelle_mask: np.ndarray | None = None,
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

    Note: this does NOT remove the nucleus membrane inside the soma.
    Use :func:`remove_nucleus` after skeletonisation for that.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Input neuron mesh.
    organelle_mask : np.ndarray or None
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
        Cleaned mesh with internal fragments removed.

    Examples
    --------
    # Detect once
    pocket, isolated = find_organelles(mesh, verbose=True)

    # Reuse for removal — no recomputation
    clean = remove_organelles(mesh, organelle_mask=pocket | isolated)

    # Or the old way still works (auto-detects if no mask given):

    clean = remove_organelles(mesh)  # detects internally
    """
    if organelle_mask is not None and len(organelle_mask) == len(mesh.faces):
        organelle = np.asarray(organelle_mask, dtype=bool)
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

    keep = ~organelle
    clean = mesh.submesh([np.where(keep)[0]], append=True)
    clean.remove_unreferenced_vertices()

    if verbose:
        print(
            f"[skeliner.pre] Result: {len(clean.vertices):,} verts, "
            f"{len(clean.faces):,} faces "
            f"(removed {organelle.sum():,} faces, "
            f"{len(mesh.vertices) - len(clean.vertices):,} verts)"
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

    # Step 1: faces inside soma ellipsoid
    inside = soma.contains(centroids, inside_frac=soma_inside_frac)
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

    for cl in comps:
        if is_outward[cl[0]]:
            continue
        if len(cl) < min_nucleus_faces:
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

    if n_nucleus == 0:
        if verbose:
            print("[skeliner.pre] No nucleus found")
        return mesh

    keep = ~nucleus_mask
    clean = mesh.submesh([np.where(keep)[0]], append=True)
    clean.remove_unreferenced_vertices()

    if verbose:
        print(
            f"[skeliner.pre] Result: {len(clean.vertices):,} verts, "
            f"{len(clean.faces):,} faces "
            f"(removed {n_nucleus:,} nucleus faces)"
        )

    return clean
