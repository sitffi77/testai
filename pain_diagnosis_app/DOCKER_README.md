# Запуск приложения в Docker

Этот проект поддерживает запуск через Docker и Docker Compose для упрощения развёртывания.

## Быстрый старт

### 1. Подготовка переменных окружения

```bash
# Скопируйте шаблон и задайте пароль для БД
cp .env.docker .env
# Отредактируйте .env и установите DB_PASSWORD
```

### 2. Запуск с базой данных (headless режим, без GUI)

```bash
# Запуск всех сервисов кроме GUI
docker-compose --profile headless up -d

# Просмотр логов приложения
docker-compose logs -f app-headless
```

### 3. Запуск с GUI (только Linux с X11)

```bash
# Разрешить доступ к X-серверу
xhost +local:docker

# Запуск с GUI
docker-compose --profile gui up -d

# Запретить доступ после завершения
xhost -local:docker
```

### 4. Доступ к phpMyAdmin (опционально)

```bash
# Запуск phpMyAdmin
docker-compose --profile dev up -d phpmyadmin

# Открыть в браузере: http://localhost:8080
# Логин: root, Пароль: из .env
```

## Режимы запуска

### Headless режим (рекомендуется для сервера)

```bash
docker-compose --profile headless up -d
```

Приложение работает в фоновом режиме без графического интерфейса. Подходит для:
- Серверного развёртывания
- Автоматизированной обработки документов
- API-подобного использования

### GUI режим (Linux)

```bash
docker-compose --profile gui up -d
```

Требует:
- Linux с работающим X-сервером
- Переменную окружения `DISPLAY`
- Команду `xhost +local:docker` перед запуском

### Только база данных

```bash
docker-compose up -d db
```

## Управление контейнерами

```bash
# Остановить все сервисы
docker-compose down

# Остановить и удалить тома (данные БД будут удалены!)
docker-compose down -v

# Пересобрать образ
docker-compose build

# Запустить тесты
docker-compose run --rm app-headless pytest tests/

# Войти в контейнер приложения
docker-compose exec app-headless bash

# Посмотреть логи
docker-compose logs -f app-headless
docker-compose logs -f db
```

## Структура томов

- `db-data` — постоянные данные MySQL
- `app-logs` — логи приложения
- `app-models` — сохранённые ML-модели

## Настройка для Windows/macOS

Для работы GUI на Windows/macOS потребуется дополнительный X-сервер:

### Windows
1. Установите [VcXsrv](https://sourceforge.net/projects/vcxsrv/) или Xming
2. Запустите XLaunch с настройками по умолчанию
3. В `.env` установите `DISPLAY=host.docker.internal:0`

### macOS
1. Установите [XQuartz](https://www.xquartz.org/)
2. Запустите XQuartz
3. В терминале: `xhost +localhost`
4. В `.env` установите `DISPLAY=host.docker.internal:0`

## Проблемы и решения

### Ошибка подключения к БД
```bash
# Дождитесь готовности БД (healthcheck)
docker-compose logs db
# Проверка подключения
docker-compose exec db mysql -u root -p -e "SHOW DATABASES;"
```

### Ошибка PyQt6 (GUI)
```bash
# Убедитесь, что X-сервер запущен
echo $DISPLAY
# Проверьте права доступа
xhost +local:docker
```

### Медленная первая сборка
Первая сборка может занять 5-10 минут из-за установки зависимостей. Последующие сборки используют кэш.

## Production развёртывание

Для production рекомендуется:
1. Изменить пароли в `.env`
2. Использовать external volumes для данных
3. Настроить бэкапы БД
4. Использовать secrets для чувствительных данных
5. Отключить phpMyAdmin

Пример production docker-compose:

```yaml
# docker-compose.prod.yml
version: '3.8'
services:
  app:
    image: pain-diagnosis-app:latest
    environment:
      - DB_HOST=db
      - DB_PASSWORD=${DB_PASSWORD}
    volumes:
      - app-logs:/app/logs
    restart: unless-stopped
  
  db:
    image: mysql:8.0
    environment:
      - MYSQL_ROOT_PASSWORD=${DB_PASSWORD}
    volumes:
      - db-data:/var/lib/mysql
    restart: unless-stopped

volumes:
  db-data:
    external: true
  app-logs:
    external: true
```
