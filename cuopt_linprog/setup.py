"""Build the cuopt_lp_op PyTorch extension against a real cuOpt install.

    CUOPT_ROOT=/path/to/cuopt pip install .

Expects $CUOPT_ROOT/include/cuopt/linear_programming/cuopt_c.h and
$CUOPT_ROOT/lib/libcuopt.so (or cuopt_c.h/libcuopt.so on default system
paths if CUOPT_ROOT is unset). cuOpt is Apache-2.0:
https://github.com/NVIDIA/cuopt — install via pip (nvidia-cuopt-cu12),
conda, or a source build.
"""
import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

include_dirs, library_dirs, rpaths = [], [], []
root = os.environ.get("CUOPT_ROOT")
if root:
    include_dirs.append(os.path.join(root, "include"))
    library_dirs.append(os.path.join(root, "lib"))
    rpaths.append(os.path.join(root, "lib"))

setup(
    name="cuopt-lp-op",
    version="0.1.0",
    description="PyTorch custom op wrapping NVIDIA cuOpt's LP solver",
    py_modules=["cuopt_linprog"],
    ext_modules=[
        CppExtension(
            name="cuopt_lp_op",
            sources=["cuopt_lp_op.cpp"],
            include_dirs=include_dirs,
            library_dirs=library_dirs,
            libraries=["cuopt"],
            extra_link_args=[f"-Wl,-rpath,{p}" for p in rpaths],
            extra_compile_args=["-O2"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
