"""Data models for NLP flag detection system."""

from dataclasses import dataclass, field
from typing import List, Optional, Literal
from datetime import datetime


@dataclass
class FlagResult:
    """Результат детекции клинического флага.
    
    Attributes:
        type: Тип флага (red, yellow, blue, black).
        id: Идентификатор флага из конфигурации.
        name: Человекочитаемое название флага.
        keyword_matched: Ключевое слово, по которому сработал флаг.
        confidence: Уровень уверенности (High, Medium, Low).
        context_snippet: Фрагмент текста вокруг найденного ключа.
        threshold_met: Преодолён ли порог активации для этого типа флагов.
        priority_rank: Ранг приоритета (1=red, 2=black, 3=yellow, 4=blue).
        excluded: Исключён ли флаг из-за контекста отрицания.
        age_value: Извлечённое значение возраста (если применимо).
        temperature_value: Извлечённое значение температуры (если применимо).
    """
    type: Literal["red", "yellow", "blue", "black"]
    id: str
    name: str
    keyword_matched: str
    confidence: Literal["High", "Medium", "Low"]
    context_snippet: str
    threshold_met: bool = False
    priority_rank: int = 0
    excluded: bool = False
    age_value: Optional[int] = None
    temperature_value: Optional[float] = None


@dataclass
class DocumentText:
    """Извлечённый текст документа с метаданными.
    
    Attributes:
        content: Полный текст документа.
        paragraphs: Список абзацев текста.
        age: Возраст пациента, извлечённый из текста (если найден).
        temperature: Температура тела, извлечённая из текста (если найдена).
        source_file: Имя исходного файла.
        extracted_at: Время извлечения текста.
    """
    content: str
    paragraphs: List[str] = field(default_factory=list)
    age: Optional[int] = None
    temperature: Optional[float] = None
    source_file: str = ""
    extracted_at: datetime = field(default_factory=datetime.now)


@dataclass
class DetectionContext:
    """Контекст для анализа флагов.
    
    Attributes:
        text: Текст для анализа.
        age: Возраст пациента (из документа или ввода).
        temperature: Температура тела (из документа или ввода).
        window_size: Размер окна контекста в словах.
        negation_patterns: Паттерны отрицания.
        exclusion_patterns: Паттерны исключения.
    """
    text: str
    age: Optional[int] = None
    temperature: Optional[float] = None
    window_size: int = 5
    negation_patterns: List[str] = field(default_factory=list)
    exclusion_patterns: List[str] = field(default_factory=list)


@dataclass
class FlagsSummary:
    """Сводка по всем обнаруженным флагам.
    
    Attributes:
        red_flags: Список красных флагов.
        yellow_flags: Список жёлтых флагов.
        blue_flags: Список голубых флагов.
        black_flags: Список чёрных флагов.
        red_threshold_met: Преодолён ли порог красных флагов.
        yellow_threshold_met: Преодолён ли порог жёлтых флагов.
        blue_threshold_met: Преодолён ли порог голубых флагов.
        black_threshold_met: Преодолён ли порог чёрных флагов.
        ml_features: Вектор признаков для ML-модели.
    """
    red_flags: List[FlagResult] = field(default_factory=list)
    yellow_flags: List[FlagResult] = field(default_factory=list)
    blue_flags: List[FlagResult] = field(default_factory=list)
    black_flags: List[FlagResult] = field(default_factory=list)
    red_threshold_met: bool = False
    yellow_threshold_met: bool = False
    blue_threshold_met: bool = False
    black_threshold_met: bool = False
    ml_features: dict = field(default_factory=dict)
