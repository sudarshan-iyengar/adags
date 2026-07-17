"""Typed fail-closed exceptions for the CSVL-ISR pipeline."""


class ContractError(ValueError):
    """A frozen scientific contract was violated."""


class SchemaError(ContractError):
    """An artifact or configuration does not match its declared schema."""


class ProvenanceError(ContractError):
    """Required provenance is missing, inconsistent, or contaminated."""


class CameraConventionError(ContractError):
    """Camera metadata or a geometric operation violates camera conventions."""


class FlowSemanticsError(ContractError):
    """Flow direction, units, or provenance are ambiguous."""


class NonFiniteError(ContractError):
    """A scientific payload contains NaN or infinity."""


class ArtifactError(ContractError):
    """A transactional artifact is missing, corrupt, or incomplete."""


__all__ = [
    "ArtifactError",
    "CameraConventionError",
    "ContractError",
    "FlowSemanticsError",
    "NonFiniteError",
    "ProvenanceError",
    "SchemaError",
]
