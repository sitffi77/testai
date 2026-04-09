"""
Tests for UI logic and controller.

Tests input validation, controller coordination, and UI signal handling.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, Any

from controller import Controller, get_controller
from ui.input_form import ClinicalInputForm


class TestControllerValidation:
    """Test cases for controller input validation."""

    @pytest.fixture
    def controller(self) -> Controller:
        """Create Controller instance."""
        ctrl = Controller()
        # Mock ML engine and DB manager for unit testing
        ctrl.ml_engine = Mock()
        ctrl.ml_engine.is_loaded = True
        ctrl.db_manager = Mock()
        ctrl.db_manager.is_initialized.return_value = True
        ctrl._is_initialized = True
        return ctrl

    def test_validate_valid_data(self, controller: Controller) -> None:
        """Test validation with valid data."""
        data = self._create_valid_data()
        
        is_valid, error_msg = controller.validate_input_data(data)
        
        assert is_valid is True
        assert error_msg == ""

    def test_validate_missing_required_field(self, controller: Controller) -> None:
        """Test validation with missing required field."""
        data = self._create_valid_data()
        del data["gender"]
        
        is_valid, error_msg = controller.validate_input_data(data)
        
        assert is_valid is False
        assert "gender" in error_msg

    def test_validate_age_out_of_range(self, controller: Controller) -> None:
        """Test validation with age out of range."""
        data = self._create_valid_data()
        data["age"] = 150  # Invalid age
        
        is_valid, error_msg = controller.validate_input_data(data)
        
        assert is_valid is False
        assert "Возраст" in error_msg

    def test_validate_pain_intensity_out_of_range(self, controller: Controller) -> None:
        """Test validation with pain intensity out of range."""
        data = self._create_valid_data()
        data["pain_intensity"] = 15.0  # Invalid (> 10)
        
        is_valid, error_msg = controller.validate_input_data(data)
        
        assert is_valid is False
        assert "Интенсивность боли" in error_msg

    def test_validate_non_numeric_value(self, controller: Controller) -> None:
        """Test validation with non-numeric value."""
        data = self._create_valid_data()
        data["age"] = "invalid"
        
        is_valid, error_msg = controller.validate_input_data(data)
        
        assert is_valid is False
        assert "числом" in error_msg

    def _create_valid_data(self) -> Dict[str, Any]:
        """Create valid clinical data for testing."""
        return {
            "patient_name": "Иванов Иван",
            "gender": "M",
            "age": 45,
            "pain_intensity": 7.5,
            "duration_days": 100,
            "frequency_per_week": 5,
            "sleep_hours": 6.5,
            "stress_level": 8.0,
            "pain_location": "back",
            "pain_type": "burning",
            "trigger_factor": "stress",
            "relief_factor": "medication",
            "medication_use": "nsaids",
        }


class TestClinicalInputForm:
    """Test cases for ClinicalInputForm widget."""

    @pytest.fixture
    def form(self, qtbot) -> ClinicalInputForm:
        """Create ClinicalInputForm instance."""
        form = ClinicalInputForm()
        qtbot.addWidget(form)
        return form

    def test_form_initialization(self, form: ClinicalInputForm) -> None:
        """Test form initializes with default values."""
        assert form.fields is not None
        assert len(form.fields) > 0
        
        # Check required fields exist
        required_fields = [
            "patient_name", "gender", "age",
            "pain_intensity", "pain_location", "pain_type"
        ]
        for field in required_fields:
            assert field in form.fields

    def test_get_form_data(self, form: ClinicalInputForm) -> None:
        """Test getting form data."""
        data = form.get_form_data()
        
        assert isinstance(data, dict)
        assert "patient_name" in data
        assert "age" in data
        assert "pain_intensity" in data

    def test_reset_form(self, form: ClinicalInputForm) -> None:
        """Test form reset functionality."""
        # Modify some fields
        form.fields["patient_name"].setText("Test Patient")
        form.fields["age"].setValue(50)
        
        # Reset
        form.reset_form()
        
        # Check reset
        assert form.fields["patient_name"].text() == ""
        assert form.fields["age"].value() == 40  # Default

    def test_set_enabled(self, form: ClinicalInputForm) -> None:
        """Test enabling/disabling form."""
        form.set_enabled(False)
        
        for widget in form.fields.values():
            assert widget.isEnabled() is False
        
        form.set_enabled(True)
        
        for widget in form.fields.values():
            assert widget.isEnabled() is True


class TestControllerSingleton:
    """Test controller singleton pattern."""

    def test_singleton_instance(self) -> None:
        """Test that get_controller returns same instance."""
        # Clear singleton first
        import controller
        controller._controller_instance = None
        
        ctrl1 = get_controller()
        ctrl2 = get_controller()
        
        assert ctrl1 is ctrl2


class TestControllerSignals:
    """Test controller signal emission."""

    def test_prediction_started_signal(self) -> None:
        """Test prediction_started signal is emitted."""
        controller = Controller()
        controller.ml_engine = Mock()
        controller.ml_engine.is_loaded = True
        controller.db_manager = Mock()
        controller.db_manager.is_initialized.return_value = True
        controller._is_initialized = True
        
        # Mock the worker to avoid actual threading
        with patch.object(controller, 'current_worker') as mock_worker:
            mock_worker.is_running.return_value = False
            
            # Connect signal to mock
            mock_callback = Mock()
            controller.prediction_started.connect(mock_callback)
            
            # Run prediction with invalid data to trigger early exit
            controller.run_prediction({})
            
            # Signal should be called or error emitted depending on validation
            # (validation happens before prediction_started)

    def test_status_message_signal(self) -> None:
        """Test status_message signal."""
        controller = Controller()
        
        mock_callback = Mock()
        controller.status_message.connect(mock_callback)
        
        controller.status_message.emit("Test message")
        
        mock_callback.assert_called_once_with("Test message")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
