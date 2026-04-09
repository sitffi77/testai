"""
UI package for Pain Diagnosis Application.

Contains all user interface components following MVC pattern.
"""

from ui.main_window import MainWindow
from ui.input_form import ClinicalInputForm
from ui.results_view import ResultsView

__all__ = [
    "MainWindow",
    "ClinicalInputForm",
    "ResultsView",
]
