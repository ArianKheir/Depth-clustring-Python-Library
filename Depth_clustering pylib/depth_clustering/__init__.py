# depth_clustering/__init__.py
__version__ = "1.0.0"
__author__  = "Depth Clustering Team"

# ------------------------------------------------------------------
# 1.  Let Python find the extension the standard way.
# ------------------------------------------------------------------
try:
    from depth_clustering._depth_clustering import *
    _extension_imported = True
except ModuleNotFoundError as e:          # real missing module
    raise ImportError(
        "Cannot import the native _depth_clustering extension.  "
        "Re-build and install the package (pip install -v .) and "
        "make sure all native libraries are available at run-time."
    ) from e

__all__ = ["__version__", "__author__"]