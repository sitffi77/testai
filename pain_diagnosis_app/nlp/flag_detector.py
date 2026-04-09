"""Clinical flag detection engine.

Implements rule-based detection of red, yellow, blue, and black flags
with context analysis, confidence grading, and priority-based processing.
"""

import re
import logging
import yaml
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass

from .models import FlagResult, DetectionContext, FlagsSummary, DocumentText
from .context_analyzer import ContextAnalyzer

logger = logging.getLogger(__name__)


@dataclass
class FlagConfig:
    """Конфигурация отдельного флага."""
    id: str
    name: str
    keywords: List[str]
    priority: int
    age_trigger: bool = False
    temperature_trigger: bool = False


class FlagDetector:
    """Детектор клинических флагов.
    
    Реализует:
    - Загрузку конфигурации из YAML
    - Приоритетную обработку: Красные → Чёрные → Жёлтые → Голубые
    - Пороги активации: 🔴≥1, ⚫≥1, 🟡≥2, 🔵≥2
    - Контекстуальный анализ (отрицание, исключения)
    - Специальные триггеры (возраст, температура)
    - Градацию уверенности (High/Medium/Low)
    - Генерацию вектора признаков для ML
    """
    
    # Priority mapping
    PRIORITY_MAP = {
        'red': 1,
        'black': 2,
        'yellow': 3,
        'blue': 4
    }
    
    # Thresholds for flag activation
    THRESHOLDS = {
        'red': 1,
        'black': 1,
        'yellow': 2,
        'blue': 2
    }
    
    def __init__(self, config_path: Optional[str] = None):
        """Инициализация детектора флагов.
        
        Args:
            config_path: Путь к YAML конфигурации. Если None, используется путь по умолчанию.
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent / 'config' / 'flags_config.yaml'
        
        self.config_path = Path(config_path)
        self.flags_config: Dict[str, List[FlagConfig]] = {}
        self.negation_patterns: List[str] = []
        self.exclusion_patterns: List[str] = []
        self.confidence_rules: Dict[str, List[str]] = {}
        self.context_window: int = 5
        
        self._load_config()
        
        # Инициализация анализатора контекста
        self.context_analyzer = ContextAnalyzer(
            negation_patterns=self.negation_patterns,
            exclusion_patterns=self.exclusion_patterns,
            window_size=self.context_window
        )
        
        # Паттерны для упоминания боли
        self._pain_patterns = [
            r'боль', r'болевой', r'болит', r'болезненность',
            r'pain', r'ache', r'sore'
        ]
        self._pain_regex = re.compile('|'.join(self._pain_patterns), re.IGNORECASE | re.UNICODE)
        
        logger.info(f"FlagDetector инициализирован с конфигурацией: {self.config_path}")
    
    def _load_config(self) -> None:
        """Загрузка конфигурации из YAML файла."""
        if not self.config_path.exists():
            logger.warning(f"Конфигурация не найдена: {self.config_path}. Используются значения по умолчанию.")
            self._set_default_config()
            return
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # Парсим флаги по типам
            self.flags_config = {}
            for flag_type in ['red', 'yellow', 'blue', 'black']:
                if flag_type in config.get('flags', {}):
                    self.flags_config[flag_type] = [
                        FlagConfig(
                            id=flag['id'],
                            name=flag['name'],
                            keywords=flag['keywords'],
                            priority=flag['priority'],
                            age_trigger=flag.get('age_trigger', False),
                            temperature_trigger=flag.get('temperature_trigger', False)
                        )
                        for flag in config['flags'][flag_type]
                    ]
            
            # Паттерны отрицания и исключения
            self.negation_patterns = config.get('negation_patterns', [])
            self.exclusion_patterns = config.get('exclusion_patterns', [])
            self.context_window = config.get('context_window', 5)
            
            # Правила уверенности
            self.confidence_rules = config.get('confidence_rules', {
                'high_keywords': [],
                'medium_keywords': [],
                'low_keywords': []
            })
            
            logger.info(f"Загружено конфигураций флагов: {sum(len(v) for v in self.flags_config.values())}")
            
        except Exception as e:
            logger.error(f"Ошибка загрузки конфигурации: {e}")
            self._set_default_config()
    
    def _set_default_config(self) -> None:
        """Установка конфигурации по умолчанию."""
        self.flags_config = {
            'red': [],
            'yellow': [],
            'blue': [],
            'black': []
        }
        self.negation_patterns = ['нет', 'исключено', 'отрицает', 'в норме']
        self.exclusion_patterns = ['профилактически', 'однократно']
        self.confidence_rules = {
            'high_keywords': ['диагноз', 'МРТ', 'КТ', 'подтверждено'],
            'medium_keywords': ['жалобы', 'анамнез', 'отмечает'],
            'low_keywords': ['возможно', 'вероятно', 'предположительно']
        }
    
    def _compile_keyword_patterns(self, keywords: List[str]) -> List[tuple]:
        """Компиляция списка ключевых слов в regex паттерны.
        
        Args:
            keywords: Список ключевых слов.
            
        Returns:
            Список кортежей (pattern, keyword).
        """
        patterns = []
        for keyword in keywords:
            pattern = re.compile(re.escape(keyword), re.IGNORECASE | re.UNICODE)
            patterns.append((pattern, keyword))
        return patterns
    
    def _determine_confidence(self, context_snippet: str) -> str:
        """Определение уровня уверенности на основе контекста.
        
        Args:
            context_snippet: Фрагмент текста вокруг ключевого слова.
            
        Returns:
            'High', 'Medium', или 'Low'.
        """
        text_lower = context_snippet.lower()
        
        # Проверяем High confidence keywords
        high_patterns = self.confidence_rules.get('high_keywords', [])
        for keyword in high_patterns:
            if keyword.startswith('\\'):  # regex pattern
                if re.search(keyword, context_snippet, re.IGNORECASE):
                    return 'High'
            elif keyword in text_lower:
                return 'High'
        
        # Проверяем Medium confidence keywords
        medium_patterns = self.confidence_rules.get('medium_keywords', [])
        for keyword in medium_patterns:
            if keyword in text_lower:
                return 'Medium'
        
        # Проверяем Low confidence keywords
        low_patterns = self.confidence_rules.get('low_keywords', [])
        for keyword in low_patterns:
            if keyword in text_lower:
                return 'Low'
        
        # По умолчанию Medium
        return 'Medium'
    
    def _has_pain_mention(self, text: str) -> bool:
        """Проверка наличия упоминания боли в тексте.
        
        Args:
            text: Текст для проверки.
            
        Returns:
            True, если боль упомянута.
        """
        return bool(self._pain_regex.search(text))
    
    def detect_flags(
        self, 
        document: DocumentText,
        patient_age: Optional[int] = None,
        patient_temperature: Optional[float] = None
    ) -> FlagsSummary:
        """Детекция всех клинических флагов в документе.
        
        Args:
            document: Документ с текстом и метаданными.
            patient_age: Возраст пациента (переопределяет извлечённый из текста).
            patient_temperature: Температура пациента (переопределяет извлечённую из текста).
            
        Returns:
            FlagsSummary с результатами детекции.
        """
        # Используем возраст и температуру из документа или переданные явно
        age = patient_age if patient_age is not None else document.age
        temperature = patient_temperature if patient_temperature is not None else document.temperature
        
        # Если не извлечены, пытаемся извлечь из текста
        if age is None:
            age = self.context_analyzer.extract_age(document.content)
        if temperature is None:
            temperature = self.context_analyzer.extract_temperature(document.content)
        
        logger.info(f"Детекция флагов: возраст={age}, температура={temperature}")
        
        # Создаём контекст для анализа
        context = DetectionContext(
            text=document.content,
            age=age,
            temperature=temperature,
            window_size=self.context_window,
            negation_patterns=self.negation_patterns,
            exclusion_patterns=self.exclusion_patterns
        )
        
        # Результаты по типам
        all_flags: Dict[str, List[FlagResult]] = {
            'red': [],
            'yellow': [],
            'blue': [],
            'black': []
        }
        
        # Приоритет обработки: red > black > yellow > blue
        processing_order = ['red', 'black', 'yellow', 'blue']
        
        for flag_type in processing_order:
            flags = self._detect_flags_by_type(context, flag_type, age, temperature)
            all_flags[flag_type] = flags
        
        # Проверяем пороги
        summary = FlagsSummary(
            red_flags=all_flags['red'],
            yellow_flags=all_flags['yellow'],
            blue_flags=all_flags['blue'],
            black_flags=all_flags['black'],
            red_threshold_met=len(all_flags['red']) >= self.THRESHOLDS['red'],
            yellow_threshold_met=len(all_flags['yellow']) >= self.THRESHOLDS['yellow'],
            blue_threshold_met=len(all_flags['blue']) >= self.THRESHOLDS['blue'],
            black_threshold_met=len(all_flags['black']) >= self.THRESHOLDS['black']
        )
        
        # Генерируем вектор признаков для ML
        summary.ml_features = self._generate_ml_features(summary)
        
        logger.info(
            f"Детекция завершена: "
            f"🔴{len(summary.red_flags)} ⚫{len(summary.black_flags)} "
            f"🟡{len(summary.yellow_flags)} 🔵{len(summary.blue_flags)}"
        )
        
        return summary
    
    def _detect_flags_by_type(
        self,
        context: DetectionContext,
        flag_type: str,
        age: Optional[int],
        temperature: Optional[float]
    ) -> List[FlagResult]:
        """Детекция флагов конкретного типа.
        
        Args:
            context: Контекст для анализа.
            flag_type: Тип флага (red/yellow/blue/black).
            age: Возраст пациента.
            temperature: Температура пациента.
            
        Returns:
            Список обнаруженных флагов.
        """
        detected = []
        flag_configs = self.flags_config.get(flag_type, [])
        
        for flag_config in flag_configs:
            flag_results = self._check_single_flag(context, flag_config, age, temperature)
            detected.extend(flag_results)
        
        # Обновляем threshold_met для каждого флага
        threshold = self.THRESHOLDS.get(flag_type, 1)
        for flag in detected:
            flag.threshold_met = len([f for f in detected if f.type == flag_type]) >= threshold
        
        return detected
    
    def _check_single_flag(
        self,
        context: DetectionContext,
        flag_config: FlagConfig,
        age: Optional[int],
        temperature: Optional[float]
    ) -> List[FlagResult]:
        """Проверка одного флага по всем ключевым словам.
        
        Args:
            context: Контекст для анализа.
            flag_config: Конфигурация флага.
            age: Возраст пациента.
            temperature: Температура пациента.
            
        Returns:
            Список результатов детекции.
        """
        results = []
        patterns = self._compile_keyword_patterns(flag_config.keywords)
        
        for pattern, keyword in patterns:
            match = pattern.search(context.text)
            if not match:
                continue
            
            # Анализируем контекст
            window_analysis = self.context_analyzer.analyze_keyword_context(
                context.text, keyword
            )
            
            # Проверяем отрицание
            if window_analysis.has_negation:
                logger.debug(f"Флаг {flag_config.id} исключён (отрицание): {keyword}")
                continue
            
            # Проверяем исключение
            if window_analysis.has_exclusion:
                logger.debug(f"Флаг {flag_config.id} исключён (паттерн исключения): {keyword}")
                continue
            
            # Специальная логика для возрастных флагов
            if flag_config.age_trigger:
                has_pain = self._has_pain_mention(context.text)
                if not self.context_analyzer.check_age_trigger(age, has_pain):
                    logger.debug(f"Флаг {flag_config.id} не активен (возрастной триггер)")
                    continue
            
            # Специальная логика для температурных флагов
            if flag_config.temperature_trigger:
                if not self.context_analyzer.check_temperature_trigger(temperature):
                    logger.debug(f"Флаг {flag_config.id} не активен (температурный триггер)")
                    continue
            
            # Получаем фрагмент контекста
            snippet = self.context_analyzer.get_context_snippet(context.text, keyword)
            
            # Определяем уверенность
            confidence = self._determine_confidence(snippet)
            
            result = FlagResult(
                type=flag_config.id[0].lower(),  # R->red, Y->yellow, etc.
                id=flag_config.id,
                name=flag_config.name,
                keyword_matched=keyword,
                confidence=confidence,
                context_snippet=snippet,
                priority_rank=self.PRIORITY_MAP.get(
                    flag_config.id[:-2].lower() if flag_config.id[-2:].isdigit() 
                    else flag_config.id[0].lower(),
                    self.PRIORITY_MAP.get(flag_type_from_id(flag_config.id), 4)
                ),
                excluded=False
            )
            
            # Добавляем специальные значения
            if flag_config.age_trigger and age:
                result.age_value = age
            if flag_config.temperature_trigger and temperature:
                result.temperature_value = temperature
            
            results.append(result)
            logger.info(f"Обнаружен флаг: {flag_config.id} ({flag_config.name})")
        
        return results
    
    def _generate_ml_features(self, summary: FlagsSummary) -> Dict[str, Any]:
        """Генерация вектора признаков для ML-модели.
        
        Args:
            summary: Сводка по обнаруженным флагам.
            
        Returns:
            Словарь с признаками.
        """
        features = {
            # Counts
            'red_flags_count': len(summary.red_flags),
            'yellow_flags_count': len(summary.yellow_flags),
            'blue_flags_count': len(summary.blue_flags),
            'black_flags_count': len(summary.black_flags),
            
            # Thresholds met
            'red_threshold_met': int(summary.red_threshold_met),
            'yellow_threshold_met': int(summary.yellow_threshold_met),
            'blue_threshold_met': int(summary.blue_threshold_met),
            'black_threshold_met': int(summary.black_threshold_met),
            
            # Specific flags (one-hot)
            'has_kinesiophobia': int(any(f.id == 'Y05' for f in summary.yellow_flags)),
            'has_depression': int(any(f.id == 'Y02' for f in summary.yellow_flags)),
            'has_catastrophizing': int(any(f.id == 'Y01' for f in summary.yellow_flags)),
            'has_oncology_history': int(any(f.id == 'R09' for f in summary.red_flags)),
            'has_trauma': int(any(f.id == 'R14' for f in summary.red_flags)),
            'has_fever': int(any(f.id == 'R17' for f in summary.red_flags)),
            'has_unemployment': int(any(f.id == 'BK02' for f in summary.black_flags)),
            
            # Total severity score
            'total_severity_score': (
                len(summary.red_flags) * 4 +
                len(summary.black_flags) * 3 +
                len(summary.yellow_flags) * 2 +
                len(summary.blue_flags) * 1
            )
        }
        
        return features


def flag_type_from_id(flag_id: str) -> str:
    """Определение типа флага по ID.
    
    Args:
        flag_id: Идентификатор флага (например, 'R01', 'Y05').
        
    Returns:
        Тип флага ('red', 'yellow', 'blue', 'black').
    """
    prefix = flag_id[0].upper()
    mapping = {
        'R': 'red',
        'Y': 'yellow',
        'B': 'blue',
        'K': 'black'  # BK -> black
    }
    
    if flag_id.startswith('BK'):
        return 'black'
    
    return mapping.get(prefix, 'blue')
