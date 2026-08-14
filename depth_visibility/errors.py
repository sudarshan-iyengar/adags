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


class DetInvocationError(ContractError):
    """A `det` (Determined CLI) invocation failed: nonzero exit, an OSError
    raised by the process launcher, or a timeout. Never raised for
    non-empty stderr alone -- the live CLI writes a version-skew warning to
    stderr on every successful call, so stderr content is not a failure
    signal by itself."""


class DetParseError(ContractError):
    """`det` CLI output could not be parsed into the expected shape (not
    JSON/YAML, or JSON/YAML lacking a required field such as a state)."""


class DetUnknownStateError(ContractError):
    """A state string reported by the `det` CLI is not a member of the
    known terminal/nonterminal state sets. Never defaulted to either
    classification -- an unrecognised state is always an error."""


__all__ = [
    "ArtifactError",
    "CameraConventionError",
    "ContractError",
    "DetInvocationError",
    "DetParseError",
    "DetUnknownStateError",
    "FlowSemanticsError",
    "NonFiniteError",
    "ProvenanceError",
    "SchemaError",
]
