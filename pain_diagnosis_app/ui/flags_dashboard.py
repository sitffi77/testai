"""Flags dashboard widget for displaying clinical flags with confidence levels."""

import logging
from typing import List, Optional, Dict

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QScrollArea, 
    QFrame, QGroupBox, QGridLayout
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QColor, QPalette

from nlp.models import FlagResult, FlagsSummary

logger = logging.getLogger(__name__)


# Цветовые схемы для типов флагов
FLAG_COLORS = {
    'red': {
        'bg': '#ffebee',
        'border': '#f44336',
        'text': '#c62828',
        'icon': '🔴'
    },
    'yellow': {
        'bg': '#fff9c4',
        'border': '#fbc02d',
        'text': '#f57f17',
        'icon': '🟡'
    },
    'blue': {
        'bg': '#e3f2fd',
        'border': '#2196f3',
        'text': '#1565c0',
        'icon': '🔵'
    },
    'black': {
        'bg': '#eceff1',
        'border': '#607d8b',
        'text': '#37474f',
        'icon': '⚫'
    }
}

# Цвета для уровней уверенности
CONFIDENCE_COLORS = {
    'High': '#4caf50',      # Green
    'Medium': '#ff9800',    # Orange
    'Low': '#9e9e9e'        # Gray
}


class FlagCard(QFrame):
    """Карточка отдельного клинического флага."""
    
    def __init__(self, flag: FlagResult, parent: Optional[QWidget] = None):
        """Инициализация карточки флага.
        
        Args:
            flag: Результат детекции флага.
            parent: Родительский виджет.
        """
        super().__init__(parent)
        self.flag = flag
        self._setup_ui()
    
    def _setup_ui(self) -> None:
        """Настройка интерфейса карточки."""
        colors = FLAG_COLORS.get(self.flag.type, FLAG_COLORS['blue'])
        
        # Стили
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {colors['bg']};
                border: 2px solid {colors['border']};
                border-radius: 8px;
                padding: 8px;
            }}
        """)
        
        layout = QVBoxLayout(self)
        layout.setSpacing(6)
        
        # Заголовок с иконкой и названием
        header_layout = QHBoxLayout()
        
        icon_label = QLabel(colors['icon'])
        icon_label.setFont(QFont("Segoe UI Emoji", 16))
        header_layout.addWidget(icon_label)
        
        title_label = QLabel(self.flag.name)
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(10)
        title_label.setFont(title_font)
        title_label.setStyleSheet(f"color: {colors['text']};")
        header_layout.addWidget(title_label)
        
        header_layout.addStretch()
        
        # Бейдж уверенности
        confidence_badge = QLabel(self.flag.confidence)
        confidence_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        confidence_badge.setMinimumWidth(60)
        confidence_badge.setMaximumHeight(20)
        confidence_badge.setStyleSheet(f"""
            QLabel {{
                background-color: {CONFIDENCE_COLORS.get(self.flag.confidence, '#999')};
                color: white;
                border-radius: 4px;
                padding: 2px 6px;
                font-weight: bold;
                font-size: 10px;
            }}
        """)
        header_layout.addWidget(confidence_badge)
        
        layout.addLayout(header_layout)
        
        # ID флага
        id_label = QLabel(f"ID: {self.flag.id}")
        id_label.setStyleSheet("color: #666; font-size: 9px;")
        layout.addWidget(id_label)
        
        # Ключевое слово
        keyword_label = QLabel(f"Ключ: \"{self.flag.keyword_matched}\"")
        keyword_label.setStyleSheet("color: #444; font-style: italic;")
        layout.addWidget(keyword_label)
        
        # Контекст
        if self.flag.context_snippet:
            context_group = QGroupBox("Контекст:")
            context_layout = QVBoxLayout(context_group)
            
            context_text = QLabel(self.flag.context_snippet)
            context_text.setWordWrap(True)
            context_text.setStyleSheet("color: #333; font-size: 9px;")
            context_layout.addWidget(context_text)
            
            layout.addWidget(context_group)
        
        # Специальные значения
        special_info = []
        if self.flag.age_value is not None:
            special_info.append(f"Возраст: {self.flag.age_value}")
        if self.flag.temperature_value is not None:
            special_info.append(f"Температура: {self.flag.temperature_value}°C")
        
        if special_info:
            info_label = QLabel(" | ".join(special_info))
            info_label.setStyleSheet("color: #1976d2; font-size: 9px; font-weight: bold;")
            layout.addWidget(info_label)


class FlagsDashboard(QWidget):
    """Панель отображения клинических флагов.
    
    Отображает:
    - Сводку по типам флагов (количество, пороги)
    - Карточки флагов с цветовой кодировкой
    - Сортировку по приоритету (Red → Black → Yellow → Blue)
    - Уровни уверенности для каждого флага
    """
    
    def __init__(self, parent: Optional[QWidget] = None):
        """Инициализация панели флагов.
        
        Args:
            parent: Родительский виджет.
        """
        super().__init__(parent)
        self._summary: Optional[FlagsSummary] = None
        self._setup_ui()
        
        logger.info("FlagsDashboard инициализирован")
    
    def _setup_ui(self) -> None:
        """Настройка пользовательского интерфейса."""
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(10)
        
        # Заголовок
        title_label = QLabel("🚩 Клинические флаги")
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(12)
        title_label.setFont(title_font)
        main_layout.addWidget(title_label)
        
        # Панель сводки
        self.summary_frame = QFrame()
        self.summary_frame.setStyleSheet("""
            QFrame {
                background-color: #f5f5f5;
                border-radius: 8px;
                padding: 8px;
            }
        """)
        
        summary_layout = QHBoxLayout(self.summary_frame)
        
        # Счётчики по типам
        self.red_count_label = QLabel("🔴 0")
        self.red_count_label.setStyleSheet("font-weight: bold; color: #c62828;")
        summary_layout.addWidget(self.red_count_label)
        
        self.black_count_label = QLabel("⚫ 0")
        self.black_count_label.setStyleSheet("font-weight: bold; color: #37474f;")
        summary_layout.addWidget(self.black_count_label)
        
        self.yellow_count_label = QLabel("🟡 0")
        self.yellow_count_label.setStyleSheet("font-weight: bold; color: #f57f17;")
        summary_layout.addWidget(self.yellow_count_label)
        
        self.blue_count_label = QLabel("🔵 0")
        self.blue_count_label.setStyleSheet("font-weight: bold; color: #1565c0;")
        summary_layout.addWidget(self.blue_count_label)
        
        summary_layout.addStretch()
        
        # Статус порогов
        self.threshold_status_label = QLabel("")
        self.threshold_status_label.setStyleSheet("font-size: 10px;")
        summary_layout.addWidget(self.threshold_status_label)
        
        main_layout.addWidget(self.summary_frame)
        
        # Scroll area для карточек
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        self.cards_container = QWidget()
        self.cards_layout = QGridLayout(self.cards_container)
        self.cards_layout.setSpacing(10)
        
        scroll_area.setWidget(self.cards_container)
        main_layout.addWidget(scroll_area, 1)
        
        # Пустое состояние
        self._show_empty_state()
    
    def _show_empty_state(self) -> None:
        """Отображение пустого состояния."""
        self._clear_cards()
        
        empty_label = QLabel("Нет обнаруженных флагов")
        empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        empty_label.setStyleSheet("color: #999; font-size: 14px; padding: 40px;")
        self.cards_layout.addWidget(empty_label, 0, 0)
    
    def _clear_cards(self) -> None:
        """Очистка всех карточек."""
        while self.cards_layout.count():
            item = self.cards_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
    
    def set_flags_summary(self, summary: FlagsSummary) -> None:
        """Установка сводки флагов для отображения.
        
        Args:
            summary: Сводка по обнаруженным флагам.
        """
        self._summary = summary
        
        # Обновляем счётчики
        self.red_count_label.setText(f"🔴 {len(summary.red_flags)}")
        self.black_count_label.setText(f"⚫ {len(summary.black_flags)}")
        self.yellow_count_label.setText(f"🟡 {len(summary.yellow_flags)}")
        self.blue_count_label.setText(f"🔵 {len(summary.blue_flags)}")
        
        # Обновляем статус порогов
        threshold_messages = []
        if summary.red_threshold_met:
            threshold_messages.append("🔴 Порог красных флагов превышен!")
        if summary.black_threshold_met:
            threshold_messages.append("⚫ Порог чёрных флагов превышен!")
        if summary.yellow_threshold_met:
            threshold_messages.append("🟡 Порог жёлтых флагов превышен!")
        if summary.blue_threshold_met:
            threshold_messages.append("🔵 Порог голубых флагов превышен!")
        
        if threshold_messages:
            self.threshold_status_label.setText(" | ".join(threshold_messages))
            self.threshold_status_label.setStyleSheet(
                "font-size: 10px; color: #d32f2f; font-weight: bold;"
            )
        else:
            self.threshold_status_label.setText("Все пороги в норме")
            self.threshold_status_label.setStyleSheet(
                "font-size: 10px; color: #388e3c;"
            )
        
        # Очищаем и заполняем карточки
        self._clear_cards()
        
        # Собираем все флаги с приоритетами
        all_flags: List[tuple] = []
        
        for flag in summary.red_flags:
            all_flags.append((1, flag))  # Priority 1
        for flag in summary.black_flags:
            all_flags.append((2, flag))  # Priority 2
        for flag in summary.yellow_flags:
            all_flags.append((3, flag))  # Priority 3
        for flag in summary.blue_flags:
            all_flags.append((4, flag))  # Priority 4
        
        # Сортируем по приоритету
        all_flags.sort(key=lambda x: x[0])
        
        if not all_flags:
            self._show_empty_state()
            return
        
        # Добавляем карточки
        row = 0
        col = 0
        max_cols = 2  # Две колонки
        
        for _, flag in all_flags:
            card = FlagCard(flag)
            card.setMinimumWidth(280)
            card.setMaximumWidth(400)
            
            self.cards_layout.addWidget(card, row, col)
            
            col += 1
            if col >= max_cols:
                col = 0
                row += 1
        
        logger.info(f"FlagsDashboard обновлён: {len(all_flags)} флагов")
    
    def get_ml_features(self) -> Dict[str, any]:
        """Получение вектора признаков для ML.
        
        Returns:
            Словарь с признаками из сводки флагов.
        """
        if self._summary is None:
            return {}
        
        return self._summary.ml_features
    
    def clear(self) -> None:
        """Очистка панели."""
        self._summary = None
        self._show_empty_state()
        
        self.red_count_label.setText("🔴 0")
        self.black_count_label.setText("⚫ 0")
        self.yellow_count_label.setText("🟡 0")
        self.blue_count_label.setText("🔵 0")
        self.threshold_status_label.setText("")
