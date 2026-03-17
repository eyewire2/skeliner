"""skeliner.pre – mesh preprocessing utilities."""

from collections import defaultdict

import igraph as ig
import numpy as np
import trimesh
from scipy.spatial import KDTree

__all__ = [
    "ensure_watertight",
    "fill_holes",
    "find_holes",
    "remove_avocados",
    "remove_fragments",
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
            e = (min(int(face[i]), int(face[(i + 1) % 3])),
                 max(int(face[i]), int(face[(i + 1) % 3])))
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
                e = (min(int(face[i]), int(face[(i + 1) % 3])),
                     max(int(face[i]), int(face[(i + 1) % 3])))
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
            e = (min(int(face[i]), int(face[(i + 1) % 3])),
                 max(int(face[i]), int(face[(i + 1) % 3])))
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
    tree = KDTree(verts)
    nearby: set[int] = set()
    for vi in loop:
        nearby.update(tree.query_ball_point(verts[vi], 3.0 * target_edge))
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

    rbf = RBFInterpolator(rbf_pts, rbf_vals, kernel="thin_plate_spline")

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
        new_h = np.clip(rbf(new_2d), bnd_h.min(), bnd_h.max())
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
    # Use local indices (before remapping) so vertex lookup is always valid
    local_v = np.vstack([boundary_pts, new_3d]) if n_new > 0 else boundary_pts
    tri_normals = np.cross(
        local_v[faces_arr[:, 1]] - local_v[faces_arr[:, 0]],
        local_v[faces_arr[:, 2]] - local_v[faces_arr[:, 0]],
    )
    loop_set = set(loop)
    adj_fi = [
        fi for fi, face in enumerate(mesh.faces)
        if set(int(v) for v in face) & loop_set
    ]
    if adj_fi:
        ref_n = mesh.face_normals[adj_fi].mean(axis=0)
        if np.dot(tri_normals.mean(axis=0), ref_n) < 0:
            remapped = remapped[:, ::-1]

    return new_3d, remapped


def fill_holes(
    mesh: trimesh.Trimesh,
    *,
    verbose: bool = False,
) -> trimesh.Trimesh:
    """Fill holes in the mesh with curvature-aware tessellation.

    New faces match the local edge-length statistics and follow the
    surface curvature (via RBF interpolation of nearby geometry)
    rather than spanning the hole with oversized flat faces.

    Algorithm:

    1. **find_holes** – detect boundary loops in the main component.
    2. **Ear-clip** each loop projected to its best-fit 2D plane.
    3. **Refine** — centroid-split faces whose longest edge exceeds
       1.5x the mesh median edge length.
    4. **RBF lift** – position new interior vertices on an RBF surface
       fitted to boundary + nearby mesh vertices, so the fill follows
       local curvature.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Input mesh (typically after ``remove_avocados``).
    verbose : bool, default False
        Print progress.

    Returns
    -------
    trimesh.Trimesh
        Mesh with holes filled.
    """
    loops = find_holes(mesh, verbose=verbose)
    if not loops:
        return mesh

    target_edge = float(np.median(mesh.edges_unique_length))
    if verbose:
        print(f"[skeliner.pre] Target edge length: {target_edge:.1f}")

    new_verts: list[np.ndarray] = []
    new_faces: list[np.ndarray] = []
    offset = len(mesh.vertices)

    for i, loop in enumerate(loops):
        nv, nf = _fill_single_hole(mesh, loop, target_edge, offset)
        if verbose:
            print(
                f"  Hole {i}: {len(loop)} bnd verts "
                f"→ +{len(nv)} verts, +{len(nf)} faces"
            )
        new_verts.append(nv)
        new_faces.append(nf)
        offset += len(nv)

    v_parts = [mesh.vertices] + [v for v in new_verts if len(v) > 0]
    f_parts = [mesh.faces] + [f for f in new_faces if len(f) > 0]

    result = trimesh.Trimesh(
        vertices=np.vstack(v_parts),
        faces=np.vstack(f_parts),
        process=False,
    )
    if verbose:
        print(
            f"[skeliner.pre] Result: {len(result.vertices):,} verts, "
            f"{len(result.faces):,} faces"
        )
    return result


def remove_fragments(
    mesh: trimesh.Trimesh,
    *,
    verbose: bool = False,
) -> trimesh.Trimesh:
    """Remove faces not edge-connected to the main surface component.

    Floating fragments — isolated faces or tiny clusters connected to
    the main mesh by at most a single vertex — are stripped.  This is
    useful after ``remove_avocados`` to clean up debris.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Input mesh.
    verbose : bool, default False
        Print summary.

    Returns
    -------
    trimesh.Trimesh
        Mesh with only the main edge-adjacency component retained.
    """
    labels, main = _face_edge_components(mesh)
    keep_mask = labels == main
    n_removed = int((~keep_mask).sum())

    if n_removed == 0:
        if verbose:
            print("[skeliner.pre] No fragments to remove")
        return mesh

    clean = mesh.submesh([np.where(keep_mask)[0]], append=True)
    clean.remove_unreferenced_vertices()

    if verbose:
        print(
            f"[skeliner.pre] Removed {n_removed} fragment faces "
            f"({len(mesh.vertices) - len(clean.vertices)} verts), "
            f"result: {len(clean.vertices):,} verts, "
            f"{len(clean.faces):,} faces"
        )
    return clean


def _remove_fins(
    mesh: trimesh.Trimesh,
    *,
    verbose: bool = False,
) -> trimesh.Trimesh:
    """Remove "fin" faces — faces with 2+ boundary edges.

    These are peninsula faces hanging off the surface by a single edge.
    They cause non-manifold edges when holes are filled.  Iterates
    until no more fins remain (removing a fin can expose new fins).
    """
    result = mesh
    total_removed = 0

    for _ in range(100):
        edge_count: dict[tuple[int, int], int] = defaultdict(int)
        for face in result.faces:
            for i in range(3):
                e = (min(int(face[i]), int(face[(i + 1) % 3])),
                     max(int(face[i]), int(face[(i + 1) % 3])))
                edge_count[e] += 1

        fin_mask = np.zeros(len(result.faces), dtype=bool)
        for fi, face in enumerate(result.faces):
            n_bnd = 0
            for i in range(3):
                e = (min(int(face[i]), int(face[(i + 1) % 3])),
                     max(int(face[i]), int(face[(i + 1) % 3])))
                if edge_count[e] == 1:
                    n_bnd += 1
            if n_bnd >= 2:
                fin_mask[fi] = True

        n_fins = int(fin_mask.sum())
        if n_fins == 0:
            break

        total_removed += n_fins
        keep = ~fin_mask
        result = result.submesh([np.where(keep)[0]], append=True)
        result.remove_unreferenced_vertices()

    if verbose and total_removed > 0:
        print(f"[skeliner.pre] Removed {total_removed} fin faces")

    return result


def ensure_watertight(
    mesh: trimesh.Trimesh,
    *,
    verbose: bool = False,
) -> trimesh.Trimesh:
    """Remove fragments and fins, fill holes, and verify watertightness.

    Chains ``remove_fragments`` → fin removal → ``fill_holes``.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Input mesh (typically after ``remove_avocados``).
    verbose : bool, default False
        Print progress.

    Returns
    -------
    trimesh.Trimesh
        Watertight mesh (or best-effort if non-manifold edges remain).
    """
    result = remove_fragments(mesh, verbose=verbose)
    result = _remove_fins(result, verbose=verbose)
    result = remove_fragments(result, verbose=verbose)
    result = fill_holes(result, verbose=verbose)

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
            e = (min(int(f[i]), int(f[(i + 1) % 3])),
                 max(int(f[i]), int(f[(i + 1) % 3])))
            edge_to_faces[e].append(fi)

    int_list = sorted(flagged)
    int_remap = {fi: i for i, fi in enumerate(int_list)}
    edges = set()
    for fi in int_list:
        f = mesh.faces[fi]
        for i in range(3):
            e = (min(int(f[i]), int(f[(i + 1) % 3])),
                 max(int(f[i]), int(f[(i + 1) % 3])))
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


def remove_avocados(
    mesh: trimesh.Trimesh,
    *,
    radius: float | None = None,
    radius_multiplier: float = 5.0,
    min_cluster_size: int = 5,
    verbose: bool = False,
) -> trimesh.Trimesh:
    """Remove internal mesh fragments ("avocado" artifacts) from a neuron mesh.

    Organelle membranes (mitochondria, ER, etc.) often appear as
    connected or semi-connected components sitting *inside* the
    neuron body.  They bias skeleton-node positions and radius estimates
    and should be removed before skeletonisation.

    The algorithm works in four steps:

    1. **Local outward scoring** – for each face, compute the dot product
       between its normal and the direction from the local center of mass
       (vertices within *radius*) to the face centroid.  Surface faces
       point outward (positive), internal faces point inward (negative).
    2. **Cluster filtering** – discard isolated small groups of flagged
       faces (< *min_cluster_size*) to avoid removing surface faces at
       local concavities.
    3. **Face removal** – flagged faces are dropped and unreferenced
       vertices are cleaned up.
    4. **Fragment removal** – drop small disconnected components.

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
    trimesh.Trimesh
        Cleaned mesh with internal fragments removed.
    """
    n_faces = len(mesh.faces)

    # ------------------------------------------------------------------
    # Step 0 – auto-compute radius from mesh edge lengths
    # ------------------------------------------------------------------
    if radius is None:
        median_edge = float(np.median(mesh.edges_unique_length))
        radius = radius_multiplier * median_edge
        if verbose:
            print(
                f"[skeliner.pre] Auto radius: {radius:.1f} "
                f"({radius_multiplier}x median edge {median_edge:.1f})"
            )

    # ------------------------------------------------------------------
    # Step 1 – local outward scoring
    # ------------------------------------------------------------------
    outward_dots = _outward_dot(mesh, radius)
    avocado_raw = outward_dots < 0

    if verbose:
        print(
            f"[skeliner.pre] Raw internal faces: {avocado_raw.sum():,} "
            f"({100 * avocado_raw.mean():.1f}% of {n_faces:,})"
        )

    # ------------------------------------------------------------------
    # Step 2 – cluster filtering
    # ------------------------------------------------------------------
    avocado = _filter_small_clusters(mesh, avocado_raw, min_cluster_size)

    if verbose:
        print(
            f"[skeliner.pre] After cluster filter (>= {min_cluster_size}): "
            f"{avocado.sum():,} faces"
        )

    # ------------------------------------------------------------------
    # Step 3 – flag internal disconnected fragments
    #          (reuse outward_dots from step 1, no recomputation)
    # ------------------------------------------------------------------
    edge_list = set()
    for face in mesh.faces:
        for i in range(3):
            a, b = int(face[i]), int(face[(i + 1) % 3])
            edge_list.add((min(a, b), max(a, b)))

    g = ig.Graph(n=len(mesh.vertices), edges=list(edge_list), directed=False)
    comps = g.connected_components()
    main_ci = max(range(len(comps)), key=lambda i: len(comps[i]))

    n_internal_frags = 0
    n_internal_frag_faces = 0
    n_kept_frags = 0

    if len(comps) > 1:
        # Map each vertex to its component
        vert_comp = np.full(len(mesh.vertices), -1, dtype=np.intp)
        for ci, cl in enumerate(comps):
            for v in cl:
                vert_comp[v] = ci

        # Assign each face to its component (by first vertex)
        face_comp = vert_comp[mesh.faces[:, 0]]

        for ci in range(len(comps)):
            if ci == main_ci:
                continue
            comp_face_mask = face_comp == ci
            comp_face_idx = np.where(comp_face_mask)[0]
            if len(comp_face_idx) == 0:
                continue
            mean_dot = outward_dots[comp_face_idx].mean()
            if mean_dot < 0:
                # Internal fragment — flag for removal
                avocado[comp_face_idx] = True
                n_internal_frags += 1
                n_internal_frag_faces += len(comp_face_idx)
            else:
                n_kept_frags += 1

    if verbose:
        print(
            f"[skeliner.pre] Fragments: {n_internal_frags:,} internal "
            f"({n_internal_frag_faces:,} faces removed), "
            f"{n_kept_frags:,} external (kept)"
        )

    # ------------------------------------------------------------------
    # Step 4 – remove all flagged faces (avocados + internal fragments)
    # ------------------------------------------------------------------
    if not avocado.any():
        if verbose:
            print("[skeliner.pre] Nothing to remove")
        return mesh

    keep = ~avocado
    clean = mesh.submesh([np.where(keep)[0]], append=True)
    clean.remove_unreferenced_vertices()

    if verbose:
        print(
            f"[skeliner.pre] Result: {len(clean.vertices):,} verts, "
            f"{len(clean.faces):,} faces "
            f"(removed {avocado.sum():,} faces, "
            f"{len(mesh.vertices) - len(clean.vertices):,} verts)"
        )

    return clean
