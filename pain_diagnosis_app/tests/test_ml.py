"""
Tests for ML engine module.

Tests model loading, preprocessing, prediction, and SHAP calculation.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from models.ml_engine import MLEngine, get_ml_engine
from config import NUMERICAL_FEATURES, CATEGORICAL_FEATURES, PAIN_CLASSES


class TestMLEngine:
    """Test cases for MLEngine class."""

    @pytest.fixture
    def ml_engine(self) -> MLEngine:
        """Create MLEngine instance with default model."""
        engine = MLEngine()
        # Create default model for testing
        engine._create_default_model()
        return engine

    def test_initialization(self, ml_engine: MLEngine) -> None:
        """Test MLEngine initialization."""
        assert ml_engine.model is not None
        assert ml_engine.is_loaded is True
        assert ml_engine.explainer is not None

    def test_preprocess_numerical_features(self, ml_engine: MLEngine) -> None:
        """Test preprocessing of numerical features."""
        data = {
            "age": 45,
            "pain_intensity": 7.5,
            "duration_days": 100,
            "frequency_per_week": 5,
            "sleep_hours": 6.5,
            "stress_level": 8.0,
        }
        
        df = ml_engine.preprocess(data)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        
        # Check numerical values are present
        assert df.iloc[0][0] == 45.0  # age
        assert df.iloc[0][1] == 7.5  # pain_intensity

    def test_preprocess_categorical_features(self, ml_engine: MLEngine) -> None:
        """Test preprocessing of categorical features."""
        data = {
            "gender": "F",
            "pain_location": "back",
            "pain_type": "burning",
            "trigger_factor": "stress",
            "relief_factor": "medication",
            "medication_use": "nsaids",
        }
        
        df = ml_engine.preprocess(data)
        
        assert isinstance(df, pd.DataFrame)
        # Should have all features (numerical + one-hot encoded categorical)
        expected_features = len(NUMERICAL_FEATURES) + sum(
            len(ml_engine._categorical_maps[feat]) 
            for feat in CATEGORICAL_FEATURES
        )
        assert df.shape[1] == expected_features

    def test_preprocess_missing_values(self, ml_engine: MLEngine) -> None:
        """Test preprocessing handles missing values."""
        data = {}  # Empty data
        
        df = ml_engine.preprocess(data)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        # Should not raise exception

    def test_predict_returns_correct_structure(self, ml_engine: MLEngine) -> None:
        """Test prediction returns correct dictionary structure."""
        data = self._create_sample_data()
        
        result = ml_engine.predict(data)
        
        assert isinstance(result, dict)
        assert "class" in result
        assert "probabilities" in result
        assert "shap_values" in result
        assert "shap_plot_base64" in result
        
        assert result["class"] in PAIN_CLASSES
        assert isinstance(result["probabilities"], dict)
        assert isinstance(result["shap_values"], dict)
        assert isinstance(result["shap_plot_base64"], str)

    def test_predict_probabilities_sum_to_one(self, ml_engine: MLEngine) -> None:
        """Test that predicted probabilities sum to approximately 1."""
        data = self._create_sample_data()
        
        result = ml_engine.predict(data)
        
        prob_sum = sum(result["probabilities"].values())
        assert abs(prob_sum - 1.0) < 1e-6

    def test_shap_values_structure(self, ml_engine: MLEngine) -> None:
        """Test SHAP values have correct structure."""
        data = self._create_sample_data()
        
        result = ml_engine.predict(data)
        
        shap_values = result["shap_values"]
        assert isinstance(shap_values, dict)
        
        # Should have values for all features
        for feature in NUMERICAL_FEATURES:
            assert feature in shap_values
            assert isinstance(shap_values[feature], float)

    def test_get_interpretation(self, ml_engine: MLEngine) -> None:
        """Test interpretation generation."""
        data = self._create_sample_data()
        result = ml_engine.predict(data)
        
        interpretation = ml_engine.get_interpretation(result)
        
        assert isinstance(interpretation, str)
        assert len(interpretation) > 0
        assert "Предсказанный тип боли" in interpretation
        assert "Уверенность модели" in interpretation

    def test_translate_class(self, ml_engine: MLEngine) -> None:
        """Test class name translation to Russian."""
        translations = {
            "nociceptive": "Ноцицептивная (тканевая)",
            "neuropathic": "Нейропатическая",
            "nociplastic": "Ноципластическая",
            "mixed": "Смешанная",
        }
        
        for en_name, ru_name in translations.items():
            assert ml_engine._translate_class(en_name) == ru_name

    def _create_sample_data(self) -> dict:
        """Create sample clinical data for testing."""
        return {
            "age": 45,
            "pain_intensity": 7.5,
            "duration_days": 100,
            "frequency_per_week": 5,
            "sleep_hours": 6.5,
            "stress_level": 8.0,
            "gender": "F",
            "pain_location": "back",
            "pain_type": "burning",
            "trigger_factor": "stress",
            "relief_factor": "medication",
            "medication_use": "nsaids",
        }


class TestGetMLEngine:
    """Test singleton ML engine getter."""

    def test_singleton_instance(self) -> None:
        """Test that get_ml_engine returns same instance."""
        engine1 = get_ml_engine()
        engine2 = get_ml_engine()
        
        assert engine1 is engine2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
