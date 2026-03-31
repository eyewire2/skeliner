from .vis2d import details, node_details, projection, threeviews
from .vis3d import view3d, view_contacts

__all__ = [
    "details",
    "node_details",
    "projection",
    "threeviews",
    "view3d",
    "view_contacts",
]


def view(mesh_path, **kwargs):
    """Launch interactive mesh viewer. Lazy import to avoid heavy deps."""
    from .viewer import view as _view

    return _view(mesh_path, **kwargs)
