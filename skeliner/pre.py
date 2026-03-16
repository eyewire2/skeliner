"""skeliner.pre – mesh preprocessing utilities."""

from collections import defaultdict

import igraph as ig
import numpy as np
import trimesh
from scipy.spatial import KDTree

__all__ = [
    "remove_avocados",
]


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

    if not avocado.any():
        if verbose:
            print("[skeliner.pre] Nothing to remove")
        return mesh

    # ------------------------------------------------------------------
    # Step 3 – remove flagged faces
    # ------------------------------------------------------------------
    keep = ~avocado
    clean = mesh.submesh([np.where(keep)[0]], append=True)
    clean.remove_unreferenced_vertices()

    if verbose:
        print(
            f"[skeliner.pre] After avocado removal: {len(clean.vertices):,} verts, "
            f"{len(clean.faces):,} faces "
            f"(removed {avocado.sum():,} faces, "
            f"{len(mesh.vertices) - len(clean.vertices):,} verts)"
        )

    # ------------------------------------------------------------------
    # Step 4 – drop small disconnected components (fragments)
    # ------------------------------------------------------------------
    edges = set()
    for face in clean.faces:
        for i in range(3):
            a, b = int(face[i]), int(face[(i + 1) % 3])
            edges.add((min(a, b), max(a, b)))

    g = ig.Graph(n=len(clean.vertices), edges=list(edges), directed=False)
    comps = g.connected_components()
    main_comp = set(max(comps, key=len))

    if len(comps) > 1:
        # Keep only faces whose vertices are all in the main component
        main_face_mask = np.array([
            all(int(v) in main_comp for v in face)
            for face in clean.faces
        ])
        n_fragment_faces = (~main_face_mask).sum()
        n_fragments = len(comps) - 1

        if n_fragment_faces > 0:
            clean = clean.submesh(
                [np.where(main_face_mask)[0]], append=True
            )
            clean.remove_unreferenced_vertices()

            if verbose:
                print(
                    f"[skeliner.pre] Removed {n_fragments:,} small fragments "
                    f"({n_fragment_faces:,} faces). "
                    f"Result: {len(clean.vertices):,} verts, "
                    f"{len(clean.faces):,} faces"
                )

    return clean
