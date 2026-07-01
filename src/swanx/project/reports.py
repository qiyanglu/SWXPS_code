"""Compatibility facade for YAML project report writers.

The maintained implementations live in ``swanx.project.reporting``. This
module re-exports the existing report writer functions so older imports such as
``from swanx.project.reports import write_fit_files`` keep working.
"""

from .reporting.csv_outputs import (
    write_experimental_files,
    write_fit_files,
    write_fit_summary,
    write_input_files,
    write_resolved_files,
    write_simulation_files,
)
from .reporting.markdown import write_markdown_report
from .reporting.identifiability import write_identifiability_outputs
from .reporting.optimizer_outputs import write_method_outputs
from .reporting.paths import prepare_output_dir
from .reporting.plots import write_plots

__all__ = [
    "prepare_output_dir",
    "write_experimental_files",
    "write_fit_files",
    "write_fit_summary",
    "write_identifiability_outputs",
    "write_input_files",
    "write_markdown_report",
    "write_method_outputs",
    "write_plots",
    "write_resolved_files",
    "write_simulation_files",
]
