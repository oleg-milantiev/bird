# Bird Detect Pipeline

Пайплайн обнаружения птиц на видео с помощью YOLO11 и генерации таймлайна для DaVinci Resolve.

## Компоненты

| Файл | Назначение |
|---|---|
| `bird.py` | Запуск YOLO11l на видео, сохранение кадров с обнаружением в `bird.csv` |
| `bird.resolve.py` | Чтение CSV, группировка в серии, генерация FCPXML v1.9 для Resolve |
| `requirements.txt` | Зависимости Python |
| `install-cuda.bat` | Установка PyTorch (CUDA 12.1) + все зависимости |

## Быстрый старт

### 1. Установка

```bash
# Вариант А — полная установка с нуля
E:/yolo/install-cuda.bat

# Вариант Б — только зависимости (если torch уже установлен)
E:/yolo/Scripts/pip.exe install -r requirements.txt
```

### 2. Обнаружение птиц

```bash
E:/yolo/Scripts/python.exe bird.py "C:/PATH/TO/video.mp4"
```

Результат — `bird.csv` в той же папке, что видео:

```
frame_number,video_path
120,C:\PATH\TO\video.mp4
121,C:\PATH\TO\video.mp4
...
```

### 3. Генерация таймлайна для Resolve

```bash
E:/yolo/Scripts/python.exe bird.resolve.py bird.csv
```

Результат — `timeline.fcpxml` в папке со скриптом.

## Конфигурация

Константы в `bird.resolve.py`:

| Параметр | По умолчанию | Описание |
|---|---|---|
| `MIN_FRAMES` | 10 | Минимум кадров с птицей для серии |
| `SERIES_GAP` | 60 | Разрыв между сериями (в кадрах) |
| `PRE_ROLL` | 2.0 сек | Кадров до обнаружения |
| `POST_ROLL` | 2.0 сек | Кадров после обнаружения |

## Как работает

### bird.py

1. Открывает видео через OpenCV
2. На каждом кадре запускает YOLO11l (`class=14` — "bird" в COCO)
3. Сохраняет номера кадров с обнаружением в CSV

### bird.resolve.py

1. Читает CSV (номера кадров + пути к видео)
2. Группирует обнаружения по файлам и разбивает на серии (разрыв > 60 кадров)
3. Добавляет PRE_ROLL и POST_ROLL, объединяет перекрывающиеся диапазоны
4. Через ffprobe получает FPS и длительность каждого файла
5. Вычисляет временные метки в кадрах таймлайна
6. Генерирует FCPXML v1.9:
   - `<format>` — стандартный SMPTE FPS (59.94, 30, 24...)
   - `<asset>` — ссылки на файлы (без `%20`, без `start="0s"`)
   - `<asset-clip>` — фрагменты с `offset`, `start`, `duration` в формате `N/fpsNum s`

## Требования

- **Python 3.11+**
- **FFmpeg** — портативная версия в `E:/PORTABLE/ffmpeg/bin/`
- **NVIDIA GPU** — PyTorch CUDA 12.1 (RTX 3060)
- **DaVinci Resolve** — для импорта FCPXML

## Примечания

- Если в Resolve возникает **Media Offline** — импортируйте файлы вручную в Media Pool, затем импортируйте XML
- При обработке нескольких видео в один CSV — `bird.resolve.py` обрабатывает все записи разом
- FPS таймлайна берётся из первого файла и нормализуется до ближайшего стандарта SMPTE

## Структура проекта

```
E:/yolo/
├── bird.py              # Обнаружение птиц
├── bird.resolve.py      # Генерация FCPXML
├── requirements.txt     # Зависимости (с CUDA)
├── install-cuda.bat     # Скрипт установки
├── Scripts/             # Venv (Python 3.11)
└── *.pt                 # Модели YOLO (автоскачивание)
```
