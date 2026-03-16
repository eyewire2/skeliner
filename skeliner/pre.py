"""skeliner.pre – mesh preprocessing utilities."""

import igraph as ig
import numpy as np
import trimesh

__all__ = [
    "remove_avocados",
]


def _outer_shell_faces(
    mesh: trimesh.Trimesh,
    main_face_idx: np.ndarray,
) -> np.ndarray:
    """Return indices of faces belonging to the smooth outer shell.

    Builds a face-adjacency graph over the main vertex-component faces
    and cuts edges where neighbouring normals are sharply opposing
    (dot < 0).  The largest resulting component is the outer shell.
    """
    normals = mesh.face_normals
    face_adj = mesh.face_adjacency

    # restrict to adjacencies within the main component
    main_mask = np.zeros(len(mesh.faces), dtype=bool)
    main_mask[main_face_idx] = True
    in_main = main_mask[face_adj[:, 0]] & main_mask[face_adj[:, 1]]
    main_adj = face_adj[in_main]

    # cut at sharp normal transitions
    adj_dots = np.einsum(
        "ij,ij->i",
        normals[main_adj[:, 0]],
        normals[main_adj[:, 1]],
    )
    smooth = adj_dots >= 0.0

    # remap to [0, n_main)
    n_main = len(main_face_idx)
    face_remap = np.full(len(mesh.faces), -1, dtype=np.intp)
    face_remap[main_face_idx] = np.arange(n_main)

    smooth_adj = main_adj[smooth]
    remapped = np.stack(
        [face_remap[smooth_adj[:, 0]], face_remap[smooth_adj[:, 1]]], axis=1
    )
    fg = ig.Graph(n=n_main, edges=remapped.tolist(), directed=False)
    face_comps = fg.components()
    shell_local = max(face_comps, key=len)

    return main_face_idx[list(shell_local)]


def remove_avocados(
    mesh: trimesh.Trimesh,
    *,
    normal_offset: float = 5.0,
    winding_threshold: float = 0.5,
    verbose: bool = False,
) -> trimesh.Trimesh:
    """Remove internal mesh fragments ("avocado" artifacts) from a neuron mesh.

    Organelle membranes (mitochondria, ER, etc.) often appear as
    disconnected or semi-connected components sitting *inside* the
    neuron body.  They bias skeleton-node positions and radius estimates
    and should be removed before skeletonisation.

    The algorithm works in three steps:

    1. **Identify the outer shell** – extract the main vertex-connected
       component, then cut its face-adjacency graph at sharp normal
       transitions (dot < 0) to isolate the smooth outer surface.
    2. **Winding-number classification** – evaluate the generalised
       winding number (via ``igl.fast_winding_number``) of the outer
       shell at points offset slightly outward *and* inward from every
       face centroid.  Faces whose **both** offsets lie inside the shell
       (winding number > *winding_threshold*) are internal membranes.
    3. **Face removal** – flagged faces are dropped and unreferenced
       vertices are cleaned up.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Input neuron mesh (may have multiple connected components).
    normal_offset : float, default 5.0
        Distance (in mesh units, typically nm) to offset query points
        along the face normal on each side.
    winding_threshold : float, default 0.5
        Winding-number threshold above which a query point is considered
        inside the outer shell.
    verbose : bool, default False
        Print summary statistics.

    Returns
    -------
    trimesh.Trimesh
        Cleaned mesh with internal fragments removed.
    """
    import igl

    V = mesh.vertices.astype(np.float64)
    F = mesh.faces.astype(np.int64)
    centroids = mesh.triangles_center.astype(np.float64)
    normals = mesh.face_normals.astype(np.float64)
    n_faces = len(F)

    # ------------------------------------------------------------------
    # Step 1 – find the outer shell
    # ------------------------------------------------------------------
    edges = [(int(a), int(b)) for a, b in mesh.edges_unique]
    g = ig.Graph(n=len(V), edges=edges, directed=False)
    comps = g.components()
    main_comp = max(comps, key=len)
    main_set = set(main_comp)
    main_face_mask = np.all(np.isin(F, list(main_set)), axis=1)
    main_face_idx = np.where(main_face_mask)[0]

    shell_face_idx = _outer_shell_faces(mesh, main_face_idx)
    shell_F = F[shell_face_idx]

    if verbose:
        print(
            f"[skeliner.pre] Outer shell: {len(shell_face_idx):,} faces "
            f"(of {n_faces:,} total)"
        )

    # ------------------------------------------------------------------
    # Step 2 – winding-number classification
    # ------------------------------------------------------------------
    query_out = centroids + normal_offset * normals
    query_in = centroids - normal_offset * normals

    wn_out = igl.fast_winding_number(V, shell_F, query_out)
    wn_in = igl.fast_winding_number(V, shell_F, query_in)

    # A face is an avocado if BOTH sides are inside the outer shell
    avocado = (wn_out > winding_threshold) & (wn_in > winding_threshold)

    # Never remove outer-shell faces (guard against rare edge cases)
    shell_mask = np.zeros(n_faces, dtype=bool)
    shell_mask[shell_face_idx] = True
    avocado &= ~shell_mask

    if verbose:
        print(
            f"[skeliner.pre] Avocado faces: {avocado.sum():,} "
            f"({100 * avocado.mean():.1f}%)"
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
            f"[skeliner.pre] Result: {len(clean.vertices):,} verts, "
            f"{len(clean.faces):,} faces "
            f"(removed {avocado.sum():,} faces, "
            f"{len(mesh.vertices) - len(clean.vertices):,} verts)"
        )

    return clean
