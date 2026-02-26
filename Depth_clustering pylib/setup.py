#!/usr/bin/env python3
# Import necessary modules for the build process.

import os
import sys
import subprocess
from pathlib import Path
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

# --- Project Paths ---
project_root = Path(__file__).parent
src_dir      = project_root / "src"

# --- Conda Environment Detection ---
def get_conda_prefix():
    """Get the conda environment prefix."""
    # Check if we're in a conda environment
    conda_prefix = os.environ.get('CONDA_PREFIX')
    if conda_prefix:
        return Path(conda_prefix)
    
    # Fallback: try to detect from Python executable path
    python_path = Path(sys.executable)
    if 'envs' in python_path.parts or 'conda' in str(python_path):
        # Typically: /path/to/conda/envs/myenv/bin/python
        return python_path.parent.parent
    
    return None

conda_prefix = get_conda_prefix()

# ---------- 1. Source Files ----------
binder = "src/python_bindings.cpp"
cpp_sources = [binder] + [
    str(p.relative_to(project_root))
    for p in src_dir.rglob("*.cpp")
    if str(p.relative_to(project_root)) != binder and
       not p.is_relative_to(src_dir / "qt") and
       not p.is_relative_to(src_dir / "ros_bridge") and
       p.name not in {"main.cpp", "visualizer.cpp"} and
       "ui_" not in p.name and
       "moc_" not in p.name
]

# ---------- 2. Include Directories ----------
include_dirs = [str(src_dir)]

if conda_prefix:
    # Conda environment paths
    conda_include = conda_prefix / "include"
    include_dirs.extend([
        str(conda_include),
        str(conda_include / "eigen3"),
        str(conda_include / "opencv4"),
    ])
    
    # Try to find PCL version dynamically
    pcl_dirs = list(conda_include.glob("pcl-*"))
    if pcl_dirs:
        include_dirs.append(str(pcl_dirs[0]))
else:
    # Fallback to system paths
    include_dirs.extend([
        "/usr/include/eigen3",
        "/usr/include/opencv4",
        "/usr/include/pcl-1.14",
        "/usr/include/boost",
    ])

# ---------- 3. Libraries ----------
libraries = [
    "opencv_core", "opencv_imgproc", "opencv_imgcodecs", "opencv_highgui",
    "pcl_common", "pcl_io", "pcl_kdtree", "pcl_search", "pcl_segmentation",
    "boost_system", "boost_filesystem", "boost_regex", "boost_program_options",
]

# ---------- 4. Library Directories ----------
library_dirs = []

if conda_prefix:
    # Conda library paths
    conda_lib = conda_prefix / "lib"
    library_dirs.append(str(conda_lib))
    
    # Add RPATH so the shared library can find conda libraries at runtime
    extra_link_args = [
        f"-Wl,-rpath,{conda_lib}",
        f"-L{conda_lib}",
    ]
else:
    extra_link_args = []

# ---------- 5. Compile Flags ----------
extra_compile_args = [
    "-std=c++17",
    "-O2",
    "-fPIC",
]

define_macros = [
    ("ROS_NOT_AVAILABLE", "1"),
    ("PCL_FOUND", "1"),
]

# ---------- 6. Extension Module ----------
ext_modules = [
    Pybind11Extension(
        "depth_clustering._depth_clustering",
        sources=cpp_sources,
        include_dirs=include_dirs,
        libraries=libraries,
        library_dirs=library_dirs,
        define_macros=define_macros,
        cxx_std=17,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    )
]

# ---------- 7. Package Metadata ----------
setup(
    name="depth_clustering",
    version="1.0.0",
    author="Depth Clustering Team + Arian Kheriandish (AISL)",
    description="Python wrapper for depth clustering library (no Qt / no ROS)",
    packages=["depth_clustering"],
    package_dir={"depth_clustering": "depth_clustering"},
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.7",
)