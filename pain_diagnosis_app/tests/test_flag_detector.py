"""Tests for flag detector module."""

import pytest
from pathlib import Path

from nlp.flag_detector import FlagDetector, flag_type_from_id
from nlp.models import DocumentText, FlagsSummary
from nlp.context_analyzer import ContextAnalyzer


@pytest.fixture
def flag_detector():
    """Fixture providing initialized FlagDetector."""
    config_path = Path(__file__).parent.parent / 'config' / 'flags_config.yaml'
    return FlagDetector(config_path=str(config_path))


@pytest.fixture
def sample_document():
    """Sample document text with various flags."""
    text = """
    Пациент: Иванов Иван Иванович, 62 года.
    
    Жалобы: Боль в грудном отделе позвоночника, прогрессирующая боль.
    Температура тела 37.5°C.
    
    Анамнез: Онкология в анамнезе (рак лёгкого, 2022). 
    Длительный приём кортикостероидов (преднизолон).
    Потеря массы тела на 10 кг за 3 месяца.
    
    Психологическое состояние: Депрессия, катастрофизация боли.
    Отмечает: "это никогда не пройдёт", страх движения (кинезиофобия).
    
    Социальные факторы: Неудовлетворённость работой, связывание боли с работой.
    Сомневается что сможет вернуться к работе.
    """
    return DocumentText(content=text, paragraphs=text.split('\n'))


class TestFlagDetector:
    """Tests for FlagDetector class."""
    
    def test_initialization(self, flag_detector):
        """Test detector initializes correctly."""
        assert flag_detector is not None
        assert len(flag_detector.flags_config) > 0
    
    def test_detect_red_flags(self, flag_detector, sample_document):
        """Test detection of red flags."""
        summary = flag_detector.detect_flags(sample_document)
        
        # Should detect multiple red flags
        assert len(summary.red_flags) >= 3
        
        # Check specific flags
        flag_ids = [f.id for f in summary.red_flags]
        assert 'R04' in flag_ids  # Age > 55
        assert 'R17' in flag_ids  # Temperature > 37
    
    def test_detect_yellow_flags(self, flag_detector, sample_document):
        """Test detection of yellow flags."""
        summary = flag_detector.detect_flags(sample_document)
        
        # Should detect yellow flags
        assert len(summary.yellow_flags) >= 2
        
        flag_ids = [f.id for f in summary.yellow_flags]
        assert 'Y02' in flag_ids  # Depression
        assert 'Y05' in flag_ids  # Kinesiophobia
    
    def test_detect_blue_flags(self, flag_detector, sample_document):
        """Test detection of blue flags."""
        summary = flag_detector.detect_flags(sample_document)
        
        # Should detect blue flags
        assert len(summary.blue_flags) >= 2
        
        flag_ids = [f.id for f in summary.blue_flags]
        assert 'B05' in flag_ids  # Job dissatisfaction
        assert 'B07' in flag_ids  # Work-related pain
    
    def test_threshold_detection(self, flag_detector, sample_document):
        """Test threshold met flags."""
        summary = flag_detector.detect_flags(sample_document)
        
        # Red threshold should be met (>=1)
        assert summary.red_threshold_met == True
        
        # Yellow threshold should be met (>=2)
        assert summary.yellow_threshold_met == True
        
        # Blue threshold should be met (>=2)
        assert summary.blue_threshold_met == True
    
    def test_ml_features_generation(self, flag_detector, sample_document):
        """Test ML features are generated correctly."""
        summary = flag_detector.detect_flags(sample_document)
        
        features = summary.ml_features
        
        # Check required features exist
        assert 'red_flags_count' in features
        assert 'yellow_flags_count' in features
        assert 'has_kinesiophobia' in features
        assert 'has_depression' in features
        assert 'total_severity_score' in features
        
        # Check values are reasonable
        assert features['red_flags_count'] >= 0
        assert features['total_severity_score'] >= 0
    
    def test_age_extraction(self, flag_detector, sample_document):
        """Test age extraction from document."""
        assert sample_document.age == 62
    
    def test_temperature_extraction(self, flag_detector, sample_document):
        """Test temperature extraction from document."""
        assert sample_document.temperature == 37.5
    
    def test_confidence_grading(self, flag_detector, sample_document):
        """Test confidence levels are assigned."""
        summary = flag_detector.detect_flags(sample_document)
        
        for flag in summary.red_flags + summary.yellow_flags:
            assert flag.confidence in ['High', 'Medium', 'Low']


class TestNegationHandling:
    """Tests for negation handling in flag detection."""
    
    def test_negation_excludes_flag(self):
        """Test that negated symptoms don't trigger flags."""
        analyzer = ContextAnalyzer(
            negation_patterns=['нет', 'исключено', 'отрицает', 'в норме'],
            exclusion_patterns=[],
            window_size=5
        )
        
        text_with_negation = "Пациент отрицает боль в спине, нет лихорадки"
        result = analyzer.analyze_keyword_context(text_with_negation, "боль")
        
        assert result.has_negation == True
    
    def test_positive_detection_without_negation(self):
        """Test that positive symptoms are detected without negation."""
        analyzer = ContextAnalyzer(
            negation_patterns=['нет', 'исключено', 'отрицает', 'в норме'],
            exclusion_patterns=[],
            window_size=5
        )
        
        text_positive = "Пациент отмечает сильную боль в спине"
        result = analyzer.analyze_keyword_context(text_positive, "боль")
        
        assert result.has_negation == False


class TestAgeTrigger:
    """Tests for age-based red flag triggers."""
    
    def test_age_under_20_triggers(self):
        """Test age < 20 triggers red flag with pain mention."""
        analyzer = ContextAnalyzer([], [])
        
        # Young patient with pain
        assert analyzer.check_age_trigger(age=18, has_pain_mention=True) == True
        assert analyzer.check_age_trigger(age=19, has_pain_mention=True) == True
        
        # Young patient without pain
        assert analyzer.check_age_trigger(age=18, has_pain_mention=False) == False
    
    def test_age_over_55_triggers(self):
        """Test age > 55 triggers red flag with pain mention."""
        analyzer = ContextAnalyzer([], [])
        
        # Older patient with pain
        assert analyzer.check_age_trigger(age=56, has_pain_mention=True) == True
        assert analyzer.check_age_trigger(age=70, has_pain_mention=True) == True
        
        # Older patient without pain
        assert analyzer.check_age_trigger(age=60, has_pain_mention=False) == False
    
    def test_age_20_to_55_no_trigger(self):
        """Test age 20-55 does not trigger red flag."""
        analyzer = ContextAnalyzer([], [])
        
        for age in range(20, 56):
            assert analyzer.check_age_trigger(age=age, has_pain_mention=True) == False


class TestTemperatureTrigger:
    """Tests for temperature-based red flag triggers."""
    
    def test_fever_triggers(self):
        """Test temperature > 37.0 triggers red flag."""
        analyzer = ContextAnalyzer([], [])
        
        assert analyzer.check_temperature_trigger(37.1) == True
        assert analyzer.check_temperature_trigger(38.5) == True
        assert analyzer.check_temperature_trigger(39.0) == True
    
    def test_normal_temp_no_trigger(self):
        """Test normal temperature does not trigger."""
        analyzer = ContextAnalyzer([], [])
        
        assert analyzer.check_temperature_trigger(36.6) == False
        assert analyzer.check_temperature_trigger(37.0) == False
        assert analyzer.check_temperature_trigger(36.9) == False
    
    def test_none_no_trigger(self):
        """Test None temperature does not trigger."""
        analyzer = ContextAnalyzer([], [])
        assert analyzer.check_temperature_trigger(None) == False


class TestFlagTypeFromId:
    """Tests for flag type identification from ID."""
    
    def test_red_flags(self):
        """Test red flag IDs."""
        assert flag_type_from_id('R01') == 'red'
        assert flag_type_from_id('R17') == 'red'
    
    def test_yellow_flags(self):
        """Test yellow flag IDs."""
        assert flag_type_from_id('Y01') == 'yellow'
        assert flag_type_from_id('Y10') == 'yellow'
    
    def test_blue_flags(self):
        """Test blue flag IDs."""
        assert flag_type_from_id('B01') == 'blue'
        assert flag_type_from_id('B08') == 'blue'
    
    def test_black_flags(self):
        """Test black flag IDs."""
        assert flag_type_from_id('BK01') == 'black'
        assert flag_type_from_id('BK04') == 'black'


class TestExclusionPatterns:
    """Tests for exclusion pattern handling."""
    
    def test_prophylactic_excluded(self):
        """Test prophylactic mentions are excluded."""
        analyzer = ContextAnalyzer(
            negation_patterns=[],
            exclusion_patterns=['профилактически', 'однократно'],
            window_size=5
        )
        
        text = "Принимает обезболивающие профилактически"
        result = analyzer.analyze_keyword_context(text, "обезболивающие")
        
        assert result.has_exclusion == True


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
