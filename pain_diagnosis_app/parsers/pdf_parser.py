"""PDF document parser using pdfplumber."""

import logging
from pathlib import Path
from typing import List, Optional

try:
    import pdfplumber
except ImportError:
    pdfplumber = None
    logging.warning("pdfplumber not installed. PDF parsing will be unavailable.")

from nlp.models import DocumentText

logger = logging.getLogger(__name__)


class PDFParser:
    """Парсер PDF документов.
    
    Использует pdfplumber для извлечения текста из PDF файлов.
    Поддерживает извлечение метаданных (возраст, температура) из текста.
    """
    
    def __init__(self):
        """Инициализация PDF парсера."""
        if pdfplumber is None:
            raise ImportError(
                "pdfplumber не установлен. Установите: pip install pdfplumber"
            )
        
        logger.info("PDFParser инициализирован")
    
    def parse(self, file_path: str) -> DocumentText:
        """Извлечение текста из PDF файла.
        
        Args:
            file_path: Путь к PDF файлу.
            
        Returns:
            DocumentText с извлечённым текстом и метаданными.
            
        Raises:
            FileNotFoundError: Если файл не найден.
            ValueError: Если файл не является валидным PDF.
        """
        path = Path(file_path)
        
        if not path.exists():
            logger.error(f"Файл не найден: {file_path}")
            raise FileNotFoundError(f"Файл не найден: {file_path}")
        
        if not path.suffix.lower() == '.pdf':
            logger.error(f"Неверный формат файла: {file_path}")
            raise ValueError(f"Ожидается PDF файл, получен: {path.suffix}")
        
        logger.info(f"Парсинг PDF: {file_path}")
        
        try:
            paragraphs = []
            full_text = ""
            
            with pdfplumber.open(path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    text = page.extract_text() or ""
                    
                    if text.strip():
                        # Разбиваем на абзацы
                        page_paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
                        paragraphs.extend(page_paragraphs)
                        
                        if full_text:
                            full_text += "\n\n" + text
                        else:
                            full_text = text
                    
                    logger.debug(f"Страница {page_num + 1}: извлечено {len(text)} символов")
            
            if not full_text.strip():
                logger.warning(f"PDF файл пуст или содержит только изображения: {file_path}")
            
            # Создаём DocumentText
            doc_text = DocumentText(
                content=full_text,
                paragraphs=paragraphs,
                source_file=path.name
            )
            
            # Извлекаем возраст и температуру из текста
            from nlp.context_analyzer import ContextAnalyzer
            
            analyzer = ContextAnalyzer(
                negation_patterns=[],
                exclusion_patterns=[],
                window_size=5
            )
            
            doc_text.age = analyzer.extract_age(full_text)
            doc_text.temperature = analyzer.extract_temperature(full_text)
            
            logger.info(
                f"PDF распарсен: {len(paragraphs)} абзацев, "
                f"возраст={doc_text.age}, температура={doc_text.temperature}"
            )
            
            return doc_text
            
        except Exception as e:
            logger.error(f"Ошибка парсинга PDF {file_path}: {e}")
            raise ValueError(f"Ошибка парсинга PDF: {e}")
    
    def parse_bytes(self, data: bytes, filename: str = "document.pdf") -> DocumentText:
        """Извлечение текста из PDF данных в памяти.
        
        Args:
            data: Байты PDF файла.
            filename: Имя файла (для логирования).
            
        Returns:
            DocumentText с извлечённым текстом.
        """
        import io
        
        logger.info(f"Парсинг PDF из памяти: {filename}")
        
        try:
            paragraphs = []
            full_text = ""
            
            with pdfplumber.open(io.BytesIO(data)) as pdf:
                for page in pdf.pages:
                    text = page.extract_text() or ""
                    
                    if text.strip():
                        page_paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
                        paragraphs.extend(page_paragraphs)
                        
                        if full_text:
                            full_text += "\n\n" + text
                        else:
                            full_text = text
            
            doc_text = DocumentText(
                content=full_text,
                paragraphs=paragraphs,
                source_file=filename
            )
            
            # Извлекаем метаданные
            from nlp.context_analyzer import ContextAnalyzer
            
            analyzer = ContextAnalyzer([], [])
            doc_text.age = analyzer.extract_age(full_text)
            doc_text.temperature = analyzer.extract_temperature(full_text)
            
            return doc_text
            
        except Exception as e:
            logger.error(f"Ошибка парсинга PDF из памяти: {e}")
            raise ValueError(f"Ошибка парсинга PDF: {e}")
