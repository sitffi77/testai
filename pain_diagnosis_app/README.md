# Pain Diagnosis Application

Десктопное приложение поддержки диагностирования болевых синдромов на основе ML.

## Описание

Приложение предназначено для врачей и позволяет:
- Вводить клинические данные пациента
- Классифицировать тип боли с помощью ML-модели (XGBoost)
- Получать интерпретацию результатов (SHAP)
- Сохранять результаты в базу данных MySQL

## Архитектура

Приложение следует паттерну MVC:
- **Model**: `models/ml_engine.py`, `models/db_models.py`
- **View**: `ui/main_window.py`, `ui/input_form.py`, `ui/results_view.py`
- **Controller**: `controller.py`

Асинхронный инференс модели выполняется в отдельном потоке (`worker.py`).

## Установка

### 1. Создайте виртуальное окружение

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

### 2. Установите зависимости

```bash
pip install -r requirements.txt
```

### 3. Настройте базу данных

Создайте файл `.env` на основе `.env.example`:

```bash
cp .env.example .env
```

Отредактируйте `.env` и укажите параметры подключения к MySQL:

```
DB_HOST=localhost
DB_PORT=3306
DB_NAME=pain_diagnosis
DB_USER=root
DB_PASSWORD=your_password
```

Создайте базу данных:

```sql
CREATE DATABASE pain_diagnosis CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
```

### 4. Запустите приложение

```bash
python main.py
```

## Структура проекта

```
pain_diagnosis_app/
├── config.py              # Конфигурация
├── requirements.txt       # Зависимости
├── main.py               # Точка входа
├── controller.py         # Координатор MVC
├── worker.py             # QThread для ML-инференса
├── models/
│   ├── ml_engine.py      # ML-движок (XGBoost + SHAP)
│   └── db_models.py      # SQLAlchemy модели
├── database/
│   └── connection.py     # Подключение к БД, CRUD
├── ui/
│   ├── main_window.py    # Главное окно
│   ├── input_form.py     # Форма ввода данных
│   ├── results_view.py   # Отображение результатов
│   └── widgets/          # Кастомные виджеты
└── tests/
    ├── test_ml.py        # Тесты ML
    └── test_ui_logic.py  # Тесты UI логики
```

## Функциональность

### Ввод данных
- Демографические данные (ФИО, пол, возраст)
- Числовые показатели:
  - Интенсивность боли (0-10)
  - Длительность (дни)
  - Частота (в неделю)
  - Часы сна
  - Уровень стресса (0-10)
- Категориальные признаки:
  - Локализация боли
  - Тип боли
  - Провоцирующие факторы
  - Облегчающие факторы
  - Прием лекарств

### Результаты
- Предсказанный тип боли (ноцицептивная, нейропатическая, ноципластическая, смешанная)
- Вероятности для каждого класса
- График важности признаков (SHAP)
- Текстовая интерпретация

### Горячие клавиши
- `Ctrl+N` — Новая диагностика
- `Ctrl+1` — Перейти к вводу данных
- `Ctrl+2` — Перейти к результатам
- `Ctrl+Q` — Выход
- `F1` — Помощь

## Тестирование

```bash
pytest tests/ -v
```

## Требования

- Python 3.10+
- PyQt6 >= 6.4.0
- XGBoost >= 1.7.0
- SHAP >= 0.42.0
- SQLAlchemy >= 2.0.0
- MySQL 8.0+ (опционально, для сохранения результатов)

## Лицензия

© 2024 Medical AI Lab
