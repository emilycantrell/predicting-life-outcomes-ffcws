#!/usr/bin/env python3
from schema import SchemaError


class MetadataEditError(Exception):
    pass


class PipelineException(Exception):
    """Base class for exceptions that arise during pipeline stages"""


class TrainTestSplitException(PipelineException):
    pass


class FeatureSelectionException(PipelineException):
    pass


class DataException(Exception):
    """Base class for exceptions during data processing"""


class NotEnoughValidOutcomeDataException(DataException):
    """Exception class for outcomes without enough data for processing"""


class UsingColumbiaFollowupDataException(DataException):
    """Exception class for using data from Columbia followup"""


class InvalidVariableDataException(SchemaError):
    """Exception class for providing invalid variable names in the job file"""


def raise_invalid_variable(varnames, location):
    msg = f"Invalid variable(s) in the job file. The variables [{varnames}] in {location} are invalid."
    raise InvalidVariableDataException(msg)


def raise_metadata_edit_error(msg):
    raise MetadataEditError(msg)


def raise_not_enough_valid_outcome_data_exception(
    outcome, min_valid_outcomes, num_samples_in_data
):
    msg = f"Not enough outcome data to run piplines for the outcome *{outcome}*. Minimum number of outcome samples = {min_valid_outcomes}. There are {num_samples_in_data} samples in the data."
    raise NotEnoughValidOutcomeDataException(msg)


def raise_invalid_concept(concept_file, concept_vars, init_columns):
    msg = f"Metadata concept file {concept_file} has variables ( {concept_vars.difference(init_columns)}) not present in the data"
    raise SchemaError(msg)


class ExportError(Exception):
    pass


class ExportFileError(ExportError):
    pass


class ParsingError(ExportError):
    pass


class IdLinkError(ExportError):
    pass


class CVFoldError(ExportError):
    pass


class PipelineError(ExportError):
    pass


class MetricsError(ExportError):
    pass


class StaleCodeError(Exception):
    pass
