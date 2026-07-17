"""Deterministic foundations for CSVL-ISR Slice A.

The package import surface intentionally has no Torch, CUDA, trainer, or DA3
side effects.  Scientific modules import the constants below when validating
frozen manifests.
"""

METHOD_ID = "csvl-isr-v1"
CONFIG_SCHEMA_VERSION = "csvl-isr-config-v1"
PAYLOAD_ENCODER_VERSION = "csvl-cjson-v1"
__version__ = "0.1.0"

__all__ = [
    "METHOD_ID",
    "CONFIG_SCHEMA_VERSION",
    "PAYLOAD_ENCODER_VERSION",
    "__version__",
]
