"""Weighing a severing stitch against what it rescues.

``_removal_would_sever`` reports that a removal patch is an annulus and so
*could* cut the surface. That is not enough to decide with: on 564241053 such
a patch stranded 13,438 faces to rescue 54, while on 554656742 a patch of the
same shape stranded **4** and would have rescued 4,160. ``_sever_cost``
supplies the missing number — how much is actually stranded — capped by a
budget so the work is bounded by the answer rather than by the mesh.
"""

import numpy as np
import trimesh

from skeliner.pre import _edge_to_faces, _face_adjacency, _sever_cost


def _tube(sections=16, height=1000.0):
    """A cylinder subdivided so there are several face rings along z.

    The raw trimesh cylinder puts its whole side wall in one ring of
    triangles, so there is no band to cut across.
    """
    c = trimesh.creation.cylinder(
        radius=50.0, height=height, sections=sections
    )
    return c.subdivide().subdivide()


def _band(mesh, lo, hi):
    """Faces whose centroid z falls in [lo, hi] — an annulus around the tube."""
    z = mesh.vertices[mesh.faces].mean(axis=1)[:, 2]
    return {int(i) for i in np.nonzero((z >= lo) & (z <= hi))[0]}


def _adj(mesh):
    return _face_adjacency(mesh, _edge_to_faces(mesh))


def test_band_across_a_tube_strands_the_shorter_side():
    """Cutting a tube off-centre strands the short end, not the long one."""
    mesh = _tube()
    adj = _adj(mesh)
    sel = _band(mesh, 100.0, 300.0)
    assert sel, "no band selected"

    cost = _sever_cost(mesh, sel, adj, budget=len(mesh.faces))
    # everything above the band is stranded; it must be the smaller side
    assert 0 < cost < len(mesh.faces) - len(sel) - cost + 1


def test_cap_at_the_end_strands_nothing():
    """A patch at a tube's end is a cap: removing it leaves one piece."""
    mesh = _tube()
    adj = _adj(mesh)
    z = mesh.vertices[mesh.faces].mean(axis=1)[:, 2]
    sel = {int(i) for i in np.nonzero(z >= z.max() - 1e-6)[0]}
    assert sel
    assert _sever_cost(mesh, sel, adj, budget=len(mesh.faces)) == 0


def test_budget_caps_the_work_and_the_answer():
    """Past the budget the exact cost does not matter, only that it is over."""
    mesh = _tube()
    adj = _adj(mesh)
    sel = _band(mesh, 100.0, 300.0)
    full = _sever_cost(mesh, sel, adj, budget=len(mesh.faces))
    assert full > 2
    tiny = _sever_cost(mesh, sel, adj, budget=2)
    assert tiny > 2, "an over-budget cost must still compare as larger"


def test_cost_is_zero_when_the_patch_is_everything():
    mesh = _tube()
    adj = _adj(mesh)
    sel = set(range(len(mesh.faces)))
    assert _sever_cost(mesh, sel, adj, budget=10) == 0


def test_decision_prefers_the_cheaper_loss():
    """The rule the guard applies: skip only when severing costs more.

    Mirrors the two real cells — a big stranded piece against a small
    rescue must skip, a tiny stranded piece against a big rescue must not.
    """
    mesh = _tube()
    adj = _adj(mesh)
    sel = _band(mesh, 100.0, 300.0)
    stranded = _sever_cost(mesh, sel, adj, budget=len(mesh.faces))

    assert _sever_cost(mesh, sel, adj, budget=stranded - 1) > stranded - 1
    assert _sever_cost(mesh, sel, adj, budget=stranded + 5) <= stranded + 5
