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
    from collections import deque

    sel = set(face_indices)
    if not sel:
        return mesh

    # ── 1. Edge-to-face map ──────────────────────────────────────────
    edge_to_faces: dict[tuple[int, int], list[int]] = defaultdict(list)
    for fi, face in enumerate(mesh.faces):
        for i in range(3):
            e = (
                min(int(face[i]), int(face[(i + 1) % 3])),
                max(int(face[i]), int(face[(i + 1) % 3])),
            )
            edge_to_faces[e].append(fi)

    # ── 2. Border edges: one side selected, other side not ───────────
    border_edges: list[tuple[int, int]] = []
    for e, fis in edge_to_faces.items():
        has_sel = any(fi in sel for fi in fis)
        has_kept = any(fi not in sel for fi in fis)
        if has_sel and has_kept:
            border_edges.append(e)

    if verbose:
        print(
            f"[skeliner.pre] Merge: removing {len(sel)} faces, "
            f"{len(border_edges)} border edges"
        )

    if len(border_edges) < 3:
        if verbose:
            print("[skeliner.pre] Not enough border edges to stitch")
        # Just remove faces
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

    # ── 3. Trace border edges into closed loops ──────────────────────
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

    # ── 4. Pair loops by closest centroid ────────────────────────────
    centroids = [mesh.vertices[lp].mean(axis=0) for lp in loops]
    pairs: list[tuple[int, int]] = []
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
        pairs.append((available[best_i], available[best_j]))
        available.pop(best_j)
        available.pop(best_i)

    if verbose:
        print(f"[skeliner.pre] Stitching {len(pairs)} loop pair(s) ...")

    # ── 5. Vert-to-face for orientation (non-selected faces only) ────
    vert_to_faces: list[list[int]] = [[] for _ in range(len(mesh.vertices))]
    for fi, face in enumerate(mesh.faces):
        if fi not in sel:
            for v in face:
                vert_to_faces[int(v)].append(fi)

    # ── 6. Zipper-stitch each pair ──────────────────────────────────
    all_stitch: list[list[int]] = []

    for li_a, li_b in pairs:
        loop_a = list(loops[li_a])
        loop_b = list(loops[li_b])
        pts_a = mesh.vertices[loop_a]
        pts_b = mesh.vertices[loop_b]

        # 6a. Closest vertex pair as starting alignment
        from scipy.spatial import cKDTree

        tree_b = cKDTree(pts_b)
        dists, idxs = tree_b.query(pts_a)
        start_a = int(np.argmin(dists))
        start_b = int(idxs[start_a])

        # 6b. Orient loops to run in opposite directions
        axis = centroids[li_b] - centroids[li_a]
        axis_len = np.linalg.norm(axis)
        if axis_len > 1e-10:
            axis = axis / axis_len
        else:
            axis = np.array([0.0, 0.0, 1.0])

        def _winding(pts_loop: np.ndarray, ax: np.ndarray) -> float:
            c = pts_loop.mean(axis=0)
            total = 0.0
            for k in range(len(pts_loop)):
                e1 = pts_loop[k] - c
                e2 = pts_loop[(k + 1) % len(pts_loop)] - c
                total += float(np.dot(np.cross(e1, e2), ax))
            return total

        wa = _winding(pts_a, axis)
        wb = _winding(pts_b, axis)
        if wa * wb > 0:
            loop_b = loop_b[::-1]
            start_b = len(loop_b) - 1 - start_b

        # 6c. Rotate so starting indices are at position 0
        loop_a = loop_a[start_a:] + loop_a[:start_a]
        loop_b = loop_b[start_b:] + loop_b[:start_b]

        # 6d. Zipper walk
        na, nb = len(loop_a), len(loop_b)
        triangles: list[list[int]] = []
        ia, ib = 0, 0
        steps_a, steps_b = 0, 0

        while steps_a < na or steps_b < nb:
            ia_next = (ia + 1) % na
            ib_next = (ib + 1) % nb
            can_a = steps_a < na
            can_b = steps_b < nb

            if can_a and can_b:
                diag_a = float(
                    np.linalg.norm(
                        mesh.vertices[loop_a[ia_next]] - mesh.vertices[loop_b[ib]]
                    )
                )
                diag_b = float(
                    np.linalg.norm(
                        mesh.vertices[loop_a[ia]] - mesh.vertices[loop_b[ib_next]]
                    )
                )
                advance_a = diag_a <= diag_b
            elif can_a:
                advance_a = True
            else:
                advance_a = False

            if advance_a:
                triangles.append([loop_a[ia], loop_a[ia_next], loop_b[ib]])
                ia = ia_next
                steps_a += 1
            else:
                triangles.append([loop_a[ia], loop_b[ib_next], loop_b[ib]])
                ib = ib_next
                steps_b += 1

        # 6e. Orient triangles consistently with the surrounding mesh
        ref_fis: list[int] = []
        for vi in loop_a[:5]:
            ref_fis.extend(vert_to_faces[vi][:5])
        if ref_fis:
            ref_n = mesh.face_normals[ref_fis].mean(axis=0)
            tri_normals = []
            for t in triangles:
                v0 = mesh.vertices[t[0]]
                v1 = mesh.vertices[t[1]]
                v2 = mesh.vertices[t[2]]
                tri_normals.append(np.cross(v1 - v0, v2 - v0))
            mean_n = np.mean(tri_normals, axis=0)
            if np.dot(mean_n, ref_n) < 0:
                triangles = [[t[0], t[2], t[1]] for t in triangles]

        all_stitch.extend(triangles)

        if verbose:
            print(
                f"[skeliner.pre]   Pair ({len(loop_a)}v + {len(loop_b)}v): "
                f"{len(triangles)} stitch faces"
            )

    # ── 7. Assemble final mesh ──────────────────────────────────────
    keep = np.ones(len(mesh.faces), dtype=bool)
    for fi in sel:
        keep[fi] = False
    kept_faces = mesh.faces[keep]
    stitch_faces = np.array(all_stitch, dtype=np.int64)
    all_faces = np.vstack([kept_faces, stitch_faces])
    result = trimesh.Trimesh(
        vertices=mesh.vertices.copy(),
        faces=all_faces,
        process=False,
    )
    result.remove_unreferenced_vertices()

    if verbose:
        print(
            f"[skeliner.pre] Merge result: {len(result.faces):,} faces "
            f"({len(sel)} removed, {len(all_stitch)} stitched)"
        )

    return result


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


def find_soma(
    mesh: trimesh.Trimesh,
    *,
    max_fragment_faces: int = 50,
    density_cutoff: float = 0.50,
    verbose: bool = False,
) -> Soma | None:
    """Estimate soma from the spatial clustering of leftover small fragments.

    After organelle removal, leftover internal fragments (failed removals)
    cluster densely inside the soma.  Their median gives a robust centre.
    A BFS flood on the main-component surface grows outward from that
    centre; the soma boundary is where the frontier ring density (verts
    per unit distance from centre) drops to *density_cutoff* × peak.

    The returned :class:`Soma` is fit to the identified surface vertices.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Input mesh (should already have organelles removed).
    max_fragment_faces : int, default 50
        Components with at most this many faces are treated as the
        indicator fragments whose clustering reveals the soma.
    density_cutoff : float, default 0.30
        Fraction of peak ring-density at which the soma boundary is set.
    verbose : bool, default False
        Print summary.

    Returns
    -------
    Soma or None
        Fitted ellipsoidal soma, or *None* if too few indicators exist.
    """
    from collections import deque

    labels, main = _face_edge_components(mesh)

    # ── 1. Locate soma centre from small-fragment clustering ──────────
    centroids = []
    for cid in np.unique(labels):
        cid = int(cid)
        if cid == main:
            continue
        fi = np.where(labels == cid)[0]
        if len(fi) > max_fragment_faces:
            continue
        verts = np.unique(mesh.faces[fi])
        centroids.append(mesh.vertices[verts].mean(axis=0))

    centroids = np.asarray(centroids)
    if len(centroids) < 3:
        if verbose:
            print("[skeliner.pre] Soma: too few indicator fragments")
        return None

    center = np.median(centroids, axis=0)
    dists = np.linalg.norm(centroids - center, axis=1)
    core = centroids[dists < 2 * np.median(dists)]
    if len(core) < 3:
        core = centroids
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

    # ── 3. BFS from nearest vertex to centre ─────────────────────────
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
        if lv > 500:
            break
        for nv in adj[v]:
            if nv not in ring_level:
                ring_level[nv] = lv + 1
                queue.append(nv)
                ring_verts[lv + 1].append(nv)

    # ── 4. Ring-density analysis to find soma boundary ───────────────
    max_ring = max(ring_verts.keys())
    densities = np.zeros(max_ring + 1)
    for lv in range(max_ring + 1):
        verts_in_ring = ring_verts[lv]
        n = len(verts_in_ring)
        if n == 0:
            continue
        mean_r = float(
            np.linalg.norm(mesh.vertices[verts_in_ring] - center, axis=1).mean()
        )
        densities[lv] = n / max(mean_r, 1.0)

    # Smooth and find peak
    window = 5
    smoothed = np.convolve(densities, np.ones(window) / window, mode="same")
    peak_ring = int(np.argmax(smoothed))
    peak_val = smoothed[peak_ring]

    # Cutoff: first ring past the peak where density < threshold × peak
    cutoff = peak_ring
    for lv in range(peak_ring, len(smoothed)):
        if smoothed[lv] < density_cutoff * peak_val:
            cutoff = lv
            break

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

    # ── 7. Absorb pockets that are trapped between the ellipsoid
    #       boundary and the actual soma surface.  After removing soma
    #       verts, each remaining connected component is either a
    #       neurite branch (extends far from soma) or a pocket (stays
    #       close).  Only pockets are absorbed.
    all_main_set = set(all_main_verts.tolist())
    for _iteration in range(10):
        outside = all_main_set - soma_set
        visited: set[int] = set()
        absorbed = 0
        for start in outside:
            if start in visited:
                continue
            comp: list[int] = []
            queue = deque([start])
            while queue:
                v = queue.popleft()
                if v in visited:
                    continue
                visited.add(v)
                comp.append(v)
                for nv in adj[v]:
                    if nv in outside and nv not in visited:
                        queue.append(nv)
            # Check if this component extends into the neurites:
            # compute max body-coord distance from ellipsoid centre.
            # If all verts are within the ellipsoid boundary (body
            # dist < pocket_frac), it is a pocket → absorb.
            coords = mesh.vertices[comp]
            body = soma._body_coords(coords)
            max_body_dist = float(np.sqrt((body**2).sum(axis=1)).max())
            if max_body_dist < 1.5:
                soma_set.update(comp)
                absorbed += len(comp)
        if absorbed == 0:
            break

    soma.verts = np.fromiter(sorted(soma_set), dtype=np.intp)

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


def find_gaps(
    mesh: trimesh.Trimesh,
    *,
    min_faces: int = 100,
    verbose: bool = False,
    _precomputed_soma: Soma | None = None,
) -> list[list[int]]:
    """Detect disconnected mesh components that represent segmentation gaps.

    A "gap" is a substantial disconnected component — a neurite segment
    broken off from the main mesh due to segmentation / proofreading
    errors.  Tiny fragments and soma-region debris are excluded.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Input mesh.
    min_faces : int, default 100
        Minimum face count for a component to be considered a gap.
        Smaller components are treated as fragments, not gaps.
    verbose : bool, default False
        Print summary.
    _precomputed_soma : Soma or None
        Pre-computed soma from :func:`find_soma`.  When *None* (default),
        ``find_soma`` is called internally.

    Returns
    -------
    list[list[int]]
        Each element is a list of face indices for one gap component,
        sorted largest-first.
    """
    labels, main = _face_edge_components(mesh)
    n_faces = len(mesh.faces)

    # Locate soma so we can exclude components inside it
    soma = _precomputed_soma if _precomputed_soma is not None else find_soma(mesh, verbose=verbose)

    # KD-tree of main-component vertices for proximity checks
    main_verts = np.unique(mesh.faces[np.where(labels == main)[0]])
    main_tree = KDTree(mesh.vertices[main_verts])

    # Collect non-main components that meet the size threshold
    comp_faces: dict[int, list[int]] = {}
    for fi in range(n_faces):
        cid = int(labels[fi])
        if cid == main:
            continue
        comp_faces.setdefault(cid, []).append(fi)

    gaps = []
    n_soma_excluded = 0
    n_organelle_excluded = 0
    for cid, fis in comp_faces.items():
        if len(fis) < min_faces:
            continue
        verts = np.unique(mesh.faces[fis])
        coords = mesh.vertices[verts]
        centroid = coords.mean(axis=0)

        # Exclude components whose centroid falls inside the soma ellipsoid
        if soma is not None:
            if soma.contains(centroid.reshape(1, -1))[0]:
                n_soma_excluded += 1
                continue

        # Exclude organelle-like components: blob-shaped (PCA < 5) AND
        # touching or very close to the main mesh (likely enclosed).
        # Real gap fragments are isolated in space (far from main).
        if len(verts) >= 4:
            centered = coords - centroid
            evals = np.linalg.eigh(np.cov(centered.T))[0]
            pca_ratio = evals.max() / (np.sort(evals)[-2] + 1e-10)
            if pca_ratio < 5.0:
                dists_to_main, _ = main_tree.query(coords)
                if dists_to_main.min() < 1000:
                    n_organelle_excluded += 1
                    continue

        gaps.append(fis)

    # Sort largest-first
    gaps.sort(key=len, reverse=True)

    if verbose:
        total = sum(len(g) for g in gaps)
        excluded = []
        if n_soma_excluded:
            excluded.append(f"{n_soma_excluded} in soma")
        if n_organelle_excluded:
            excluded.append(f"{n_organelle_excluded} organelle-like")
        soma_msg = f", {', '.join(excluded)} excluded" if excluded else ""
        print(
            f"[skeliner.pre] Gaps: {len(gaps)} components, "
            f"{total:,} faces (min_faces={min_faces}{soma_msg})"
        )
        for i, g in enumerate(gaps):
            verts = np.unique(mesh.faces[g])
            coords = mesh.vertices[verts]
            centroid = coords.mean(axis=0)
            extent = coords.max(axis=0) - coords.min(axis=0)
            print(
                f"  gap {i}: {len(g):,} faces, "
                f"centroid=[{centroid[0]:.0f}, {centroid[1]:.0f}, {centroid[2]:.0f}], "
                f"extent=[{extent[0]:.0f}, {extent[1]:.0f}, {extent[2]:.0f}]"
            )

    return gaps


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
) -> np.ndarray:
    """Per-face outward score: dot(face_normal, direction_from_local_COM).

    For each face, finds all vertices within *radius* of its centroid,
    computes their center of mass, and dots the face normal against
    the direction from that COM to the face centroid.

    Surface faces point outward (positive), internal faces point
    inward (negative).

    Returns
    -------
    np.ndarray
        (nFaces,) float64 array of outward dot products.
    """
    verts = mesh.vertices
    face_centers = mesh.triangles_center
    face_normals = mesh.face_normals
    vtree = KDTree(verts)

    outward_dots = np.zeros(len(mesh.faces), dtype=np.float64)
    for fi in range(len(mesh.faces)):
        fc = face_centers[fi]
        idx = vtree.query_ball_point(fc, radius)
        if len(idx) < 4:
            continue
        local_com = verts[idx].mean(axis=0)
        outward_dir = fc - local_com
        norm = np.linalg.norm(outward_dir)
        if norm < 1e-10:
            continue
        outward_dir /= norm
        outward_dots[fi] = np.dot(face_normals[fi], outward_dir)

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

    outward_dots = _outward_dot(mesh, radius)

    if verbose:
        raw_count = (outward_dots < 0).sum()
        print(
            f"[skeliner.pre] Raw internal faces: {raw_count:,} "
            f"({100 * raw_count / len(mesh.faces):.1f}%)"
        )

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
    min_fold_ratio: float = 2.0,
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

        # Count boundary loops (connected components of boundary edges)
        n_loops = _count_edge_loops(boundary_edges)
        if n_loops <= 1:
            continue  # single loop = flat patch, not a pocket

        # Fold ratio: pocket surface area / rim enclosed planar area.
        # A real pocket folds inward through a small opening (high ratio).
        # A flat patch has ratio ≈ 1 (same area inside and outside).
        pocket_area = float(mesh.area_faces[cluster].sum())
        opening_area = _rim_enclosed_area(boundary_edges, mesh.vertices)
        if opening_area > 0:
            fold_ratio = pocket_area / opening_area
            if fold_ratio < min_fold_ratio:
                continue

        rims.append(boundary_edges)

    if verbose:
        print(
            f"[skeliner.pre] Rims: {len(rims)} pockets "
            f"(multi-loop, fold >= {min_fold_ratio}), "
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
    min_fold_ratio: float = 2.0,
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

    if verbose:
        print(
            f"[skeliner.pre] Pocket organelles (>= {min_cluster_size}): "
            f"{pocket.sum():,} faces "
            f"(initial {initial_count:,}, "
            f"bridged +{bridge_count:,}, "
            f"holes +{hole_count:,})"
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

    for ci in range(n_comps):
        if ci == main_ci:
            continue
        comp_face_idx = np.where(face_comp == ci)[0]
        if len(comp_face_idx) == 0:
            continue
        mean_dot = outward_dots[comp_face_idx].mean()

        # Clearly inward-pointing → organelle
        is_internal = mean_dot < 0

        # Ambiguous normals on blob-shaped components: small fragments
        # with mixed face winding can have mean_dot near zero even when
        # enclosed.  Use PCA aspect ratio to identify blobs (< 3) and
        # apply a relaxed threshold.
        if not is_internal and mean_dot < 0.5:
            comp_verts = np.unique(mesh.faces[comp_face_idx])
            if len(comp_verts) >= 4:
                coords = mesh.vertices[comp_verts]
                centered = coords - coords.mean(axis=0)
                evals = np.linalg.eigh(np.cov(centered.T))[0]
                pca_ratio = evals.max() / (np.sort(evals)[-2] + 1e-10)
                if pca_ratio < 3.0:
                    is_internal = True

        if is_internal:
            isolated[comp_face_idx] = True
            n_internal_frags += 1
            n_internal_frag_faces += len(comp_face_idx)
        else:
            n_kept_frags += 1

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
    precomputed = _organelle_precompute(
        mesh,
        radius,
        radius_multiplier,
        verbose,
    )

    pocket = find_pocket_organelles(
        mesh,
        radius=radius,
        radius_multiplier=radius_multiplier,
        min_cluster_size=min_cluster_size,
        verbose=verbose,
        _precomputed=precomputed,
    )
    isolated = find_isolated_organelles(
        mesh,
        radius=radius,
        radius_multiplier=radius_multiplier,
        verbose=verbose,
        _precomputed=precomputed,
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
