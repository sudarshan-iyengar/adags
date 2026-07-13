# Shim package so `import simple_knn` works.
# The compiled extension is expected to be `simple_knn._C`.
from . import _C  # noqa: F401
