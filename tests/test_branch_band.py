"""Cutting a bin that wraps a branch point into one piece per tube.

A geodesic shell that lands on a branch point wraps the parent tube and both
children in a single connected band.  It passes the ring test — removing it
still separates the surface — so it becomes a bin, and its node is the mean of
points on two diverging tubes, which lands between them instead of inside one.
``_neighbour_groups`` counts how many separate neighbourhoods a bin touches
(two for a cross-section, three at a branch) and ``_split_branch_band`` cuts on
that count.
"""

import igraph as ig
import numpy as np

from skeliner.skeletonize import _neighbour_groups, _split_branch_band


def _ring(start, n):
    """Vertex ids of a cycle plus its edges, as (ids, edges)."""
    ids = list(range(start, start + n))
    edges = [(ids[i], ids[(i + 1) % n]) for i in range(n)]
    return ids, edges


def _pants():
    """Three rings around a fourth: the parent, the band, and two children.

    ``band`` touches all three of the others, so it is the bin a branch point
    produces.  Returns the graph, the band's vertex ids, and the ownership map.
    """
    n = 8
    parent, e_parent = _ring(0, n)
    band, e_band = _ring(n, 2 * n)          # wide enough to feed two children
    childA, e_a = _ring(3 * n, n)
    childB, e_b = _ring(4 * n, n)

    edges = e_parent + e_band + e_a + e_b
    # parent sits under the first half of the band, the children over the
    # second half, one on each quarter
    for i in range(n):
        edges.append((parent[i], band[i]))
    for i in range(n):
        edges.append((childA[i], band[n + (i % (n // 2))]))
        edges.append((childB[i], band[n + n // 2 + (i % (n // 2))]))

    g = ig.Graph(n=5 * n, edges=edges)
    owner = {}
    for v in parent:
        owner[v] = (0, 0)
    for v in band:
        owner[v] = (1, 0)
    for v in childA:
        owner[v] = (2, 0)
    for v in childB:
        owner[v] = (3, 0)
    return g, np.asarray(band, dtype=np.int64), owner


def test_branch_band_sees_three_neighbourhoods():
    g, band, owner = _pants()
    groups = _neighbour_groups(band, g, owner, (1, 0))
    assert len(groups) == 3, "parent and both children must count separately"


def test_two_patches_of_one_bin_are_one_neighbourhood():
    """Dipping into the same neighbour twice is one tube, not two.

    Counting vertex patches instead of bins made a plain band look like a
    branch point and cut it into slivers.
    """
    g, band, owner = _pants()
    # merge both children into a single bin: the band now touches two
    # bins (parent, child) but along three separate patches
    for v in list(owner):
        if owner[v] == (3, 0):
            owner[v] = (2, 0)
    groups = _neighbour_groups(band, g, owner, (1, 0))
    assert len(groups) == 2, "same bin reached twice must count once"


def test_cross_section_sees_two():
    """The parent ring touches only the band and nothing else: two is a ring."""
    g, band, owner = _pants()
    parent = np.arange(0, 8, dtype=np.int64)
    groups = _neighbour_groups(parent, g, owner, (0, 0))
    assert len(groups) == 1, "an end ring touches one neighbourhood only"


def test_split_yields_one_piece_per_tube():
    g, band, owner = _pants()
    groups = _neighbour_groups(band, g, owner, (1, 0))
    parts = _split_branch_band(band, groups, g)

    assert len(parts) == 3
    # every vertex kept exactly once
    allv = np.concatenate(parts)
    assert sorted(allv.tolist()) == sorted(band.tolist())
    assert len(set(allv.tolist())) == len(allv), "a vertex landed in two pieces"
    for p in parts:
        assert len(p) > 0


def test_pieces_are_connected():
    """Each piece must be one connected patch, or its node has no place to be."""
    g, band, owner = _pants()
    groups = _neighbour_groups(band, g, owner, (1, 0))
    for part in _split_branch_band(band, groups, g):
        sub = g.induced_subgraph([int(v) for v in part])
        assert len(sub.components()) == 1


def test_split_is_deterministic():
    g, band, owner = _pants()
    groups = _neighbour_groups(band, g, owner, (1, 0))
    first = [p.tolist() for p in _split_branch_band(band, groups, g)]
    second = [p.tolist() for p in _split_branch_band(band, groups, g)]
    assert first == second
