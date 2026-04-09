"""Context analyzer for clinical flag detection.

Implements negation detection, temporal analysis, and special trigger evaluation
(age, temperature) using rule-based window analysis.
"""

import re
import logging
from typing import List, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class WindowAnalysis:
    """Результат анализа окна контекста."""
    has_negation: bool
    has_exclusion: bool
    context_words_before: List[str]
    context_words_after: List[str]


class ContextAnalyzer:
    """Анализатор контекста для клинических флагов.
    
    Реализует:
    - Обнаружение отрицания в окне ±N слов от ключевого маркера
    - Обнаружение паттернов исключения (профилактически, однократно и т.д.)
    - Анализ временных маркеров (настоящее/прошедшее время)
    - Специальные триггеры: возраст, температура
    """
    
    def __init__(
        self,
        negation_patterns: List[str],
        exclusion_patterns: List[str],
        window_size: int = 5
    ):
        """Инициализация анализатора контекста.
        
        Args:
            negation_patterns: Список паттернов отрицания.
            exclusion_patterns: Список паттернов исключения.
            window_size: Размер окна контекста в словах.
        """
        self.negation_patterns = negation_patterns
        self.exclusion_patterns = exclusion_patterns
        self.window_size = window_size
        
        # Компилируем regex для эффективности
        self._negation_regex = self._compile_patterns(negation_patterns)
        self._exclusion_regex = self._compile_patterns(exclusion_patterns)
        
        # Паттерны для извлечения возраста
        self._age_patterns = [
            r'(\d{1,2})\s*(?:лет|года|год|y\.?o\.?|years?\s*old)',
            r'возраст\s*(\d+)',
            r'пациент\s*(\d+)\s*лет',
            r'(\d{1,2})-летний',
            r'(\d{1,2})\s*года\s*жизни',
        ]
        
        # Паттерны для извлечения температуры
        self._temp_patterns = [
            r'(?:температура|t°|t)\s*[:=]?\s*(\d+[.,]\d+)\s*°?[cC]',
            r'(\d+[.,]\d+)\s*°[cC]',
            r'субфебрилитет',
            r'лихорадка',
            r'гипертермия',
        ]
        
        # Паттерны для нормализации температуры
        self._subfebrile_keywords = ['субфебрилитет', 'лихорадка', 'гипертермия', 'fever']
    
    def _compile_patterns(self, patterns: List[str]) -> re.Pattern:
        """Компиляция списка паттернов в единый regex."""
        escaped = [re.escape(p) for p in patterns if p]
        pattern = '|'.join(escaped)
        return re.compile(pattern, re.IGNORECASE | re.UNICODE)
    
    def tokenize(self, text: str) -> List[str]:
        """Токенизация текста на слова.
        
        Args:
            text: Входной текст.
            
        Returns:
            Список токенов (слов).
        """
        # Разделяем по пробелам и знакам препинания, сохраняя структуру
        tokens = re.findall(r'\b\w+\b|[^\w\s]', text, re.UNICODE)
        return tokens
    
    def get_context_window(
        self, 
        text: str, 
        keyword_position: int,
        keyword_length: int = 1
    ) -> Tuple[List[str], List[str]]:
        """Получение окна контекста вокруг ключевой позиции.
        
        Args:
            text: Исходный текст.
            keyword_position: Позиция ключевого слова в токенах.
            keyword_length: Длина ключевого слова в токенах.
            
        Returns:
            Кортеж (слова_до, слова_после).
        """
        tokens = self.tokenize(text)
        
        start_idx = max(0, keyword_position - self.window_size)
        end_idx = min(len(tokens), keyword_position + keyword_length + self.window_size)
        
        words_before = tokens[start_idx:keyword_position]
        words_after = tokens[keyword_position + keyword_length:end_idx]
        
        return words_before, words_after
    
    def check_negation_in_window(
        self, 
        words_before: List[str], 
        words_after: List[str]
    ) -> bool:
        """Проверка наличия отрицания в окне контекста.
        
        Args:
            words_before: Слова до ключевого маркера.
            words_after: Слова после ключевого маркера.
            
        Returns:
            True, если обнаружено отрицание.
        """
        context = ' '.join(words_before + words_after)
        match = self._negation_regex.search(context)
        
        if match:
            logger.debug(f"Обнаружено отрицание: '{match.group()}'")
            return True
        
        return False
    
    def check_exclusion_in_window(
        self, 
        words_before: List[str], 
        words_after: List[str]
    ) -> bool:
        """Проверка наличия паттерна исключения в окне контекста.
        
        Args:
            words_before: Слова до ключевого маркера.
            words_after: Слова после ключевого маркера.
            
        Returns:
            True, если обнаружен паттерн исключения.
        """
        context = ' '.join(words_before + words_after)
        match = self._exclusion_regex.search(context)
        
        if match:
            logger.debug(f"Обнаружен паттерн исключения: '{match.group()}'")
            return True
        
        return False
    
    def analyze_keyword_context(
        self, 
        text: str, 
        keyword: str
    ) -> WindowAnalysis:
        """Полный анализ контекста вокруг ключевого слова.
        
        Args:
            text: Исходный текст.
            keyword: Ключевое слово для анализа.
            
        Returns:
            WindowAnalysis с результатами проверки.
        """
        text_lower = text.lower()
        keyword_lower = keyword.lower()
        
        # Находим позицию ключевого слова
        position = text_lower.find(keyword_lower)
        if position == -1:
            return WindowAnalysis(
                has_negation=False,
                has_exclusion=False,
                context_words_before=[],
                context_words_after=[]
            )
        
        # Конвертируем позицию символов в позицию токенов
        tokens = self.tokenize(text[:position + len(keyword)])
        token_position = len(tokens) - 1  # Позиция последнего токена ключа
        
        words_before, words_after = self.get_context_window(text, token_position, 1)
        
        has_negation = self.check_negation_in_window(words_before, words_after)
        has_exclusion = self.check_exclusion_in_window(words_before, words_after)
        
        return WindowAnalysis(
            has_negation=has_negation,
            has_exclusion=has_exclusion,
            context_words_before=words_before,
            context_words_after=words_after
        )
    
    def extract_age(self, text: str) -> Optional[int]:
        """Извлечение возраста пациента из текста.
        
        Args:
            text: Текст документа.
            
        Returns:
            Возраст в годах или None.
        """
        for pattern in self._age_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.UNICODE)
            if match:
                try:
                    age = int(match.group(1))
                    if 0 < age < 120:  # sanity check
                        logger.info(f"Извлечён возраст: {age}")
                        return age
                except (ValueError, IndexError):
                    continue
        
        return None
    
    def extract_temperature(self, text: str) -> Optional[float]:
        """Извлечение температуры тела из текста.
        
        Args:
            text: Текст документа.
            
        Returns:
            Температура в °C или None.
        """
        # Сначала ищем числовые значения
        for pattern in self._temp_patterns:
            if 'субфебрилитет' in pattern or 'лихорадка' in pattern:
                continue  # Пропускаем текстовые паттерны в этом цикле
            
            match = re.search(pattern, text, re.IGNORECASE | re.UNICODE)
            if match:
                try:
                    temp_str = match.group(1).replace(',', '.')
                    temp = float(temp_str)
                    if 35.0 <= temp <= 42.0:  # sanity check
                        logger.info(f"Извлечена температура: {temp}°C")
                        return temp
                except (ValueError, IndexError):
                    continue
        
        # Проверяем текстовые маркеры субфебрилитета
        text_lower = text.lower()
        for keyword in self._subfebrile_keywords:
            if keyword in text_lower:
                logger.info("Обнаружен субфебрилитет по текстовому маркеру")
                return 37.1  # Дефолтное значение для субфебрилитета
        
        return None
    
    def check_age_trigger(self, age: Optional[int], has_pain_mention: bool) -> bool:
        """Проверка возрастного триггера для красного флага.
        
        Красный флаг активен если:
        - age < 20 И есть упоминание боли, ИЛИ
        - age > 55 И есть упоминание боли
        
        Args:
            age: Возраст пациента.
            has_pain_mention: Есть ли упоминание боли в тексте.
            
        Returns:
            True, если возрастной триггер активен.
        """
        if age is None or not has_pain_mention:
            return False
        
        return age < 20 or age > 55
    
    def check_temperature_trigger(self, temperature: Optional[float]) -> bool:
        """Проверка температурного триггера для красного флага.
        
        Красный флаг активен если температура > 37.0°C.
        
        Args:
            temperature: Температура тела.
            
        Returns:
            True, если температурный триггер активен.
        """
        if temperature is None:
            return False
        
        return temperature > 37.0
    
    def get_context_snippet(
        self, 
        text: str, 
        keyword: str, 
        max_length: int = 100
    ) -> str:
        """Получение фрагмента текста вокруг ключевого слова.
        
        Args:
            text: Исходный текст.
            keyword: Ключевое слово.
            max_length: Максимальная длина фрагмента.
            
        Returns:
            Фрагмент текста с многоточием по краям.
        """
        text_lower = text.lower()
        keyword_lower = keyword.lower()
        
        position = text_lower.find(keyword_lower)
        if position == -1:
            return keyword
        
        start = max(0, position - max_length // 2)
        end = min(len(text), position + len(keyword) + max_length // 2)
        
        snippet = text[start:end].strip()
        
        # Добавляем многоточие если обрезали
        if start > 0:
            snippet = "..." + snippet
        if end < len(text):
            snippet = snippet + "..."
        
        return snippet
