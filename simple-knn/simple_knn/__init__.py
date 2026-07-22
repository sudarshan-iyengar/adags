"""Python package wrapper for the simple-knn CUDA extension."""

from ._C import distCUDA2

__all__ = ["distCUDA2"]
