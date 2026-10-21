"""Focused, user-reportable framework exceptions."""


class XevolutionError(Exception):
    """Base class for expected framework failures."""


class ConfigurationError(XevolutionError):
    """Invalid configuration, component composition, or JSON value."""


class EvaluationError(XevolutionError):
    """An evaluator returned an invalid or incomplete result batch."""


class SelectionError(XevolutionError):
    """A selector cannot select valid candidates from the supplied population."""


class OperatorInapplicable(XevolutionError):
    """A recoverable variation failure for the selected parent data."""


class ReproductionError(XevolutionError):
    """A complete offspring population could not be produced."""


class CheckpointError(XevolutionError):
    """A checkpoint or its committed history cannot be saved or restored."""


class CheckpointCompatibilityError(CheckpointError):
    """A checkpoint requires incompatible code, codecs, or runtime state."""


class RunStateError(XevolutionError):
    """An operation is inconsistent with the current lifecycle state."""
