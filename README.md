# Распознавание автомобильных номеров

Небольшой проект на PyTorch с двумя этапами:

- **OCR / Detection** — находит координаты автомобильного номера на изображении.
- **REC / Recognition** — распознаёт символы на вырезанном изображении номера.

Архитектуры моделей и вычисления обучения сохранены. Проект приведён к более понятной структуре, исправлены пути и ошибки организации.

## Структура

```text
project/
├── plate_recognition/
│   ├── config.py          # Пути к данным и весам
│   ├── data.py            # Dataset, transforms и DataLoader
│   ├── decoder.py         # Кодирование/декодирование номеров
│   ├── early_stopping.py  # Early stopping
│   └── models.py          # OCR- и REC-модели
├── scripts/
│   ├── train_ocr.py       # Первичное обучение OCR
│   ├── train_rec.py       # Первичное обучение REC
│   ├── model_manager.py   # Загрузка, дообучение и тестирование
│   └── crop_images.py     # Обрезка номеров по OCR-координатам
├── data/                  # Датасеты (добавляются отдельно)
├── models/checkpoints/    # Веса и checkpoint-файлы
├── requirements.txt
└── README.md
```

## Установка

Рекомендуется Python 3.10 или 3.11.

```bash
python -m venv .venv
```

Windows:

```powershell
.venv\Scripts\activate
```

Linux / macOS:

```bash
source .venv/bin/activate
```

Установите PyTorch и torchvision подходящей сборки (особенно если используете CUDA), затем:

```bash
pip install -r requirements.txt
```

## Данные

По умолчанию ожидаются такие папки:

```text
data/
├── OCR_data/
│   ├── train/
│   │   ├── coords.json
│   │   └── изображения...
│   ├── val/
│   │   ├── coords.json
│   │   └── изображения...
│   └── test/
│       ├── coords.json
│       └── изображения...
└── REC_data/
    ├── train/
    │   ├── data.json
    │   └── изображения...
    ├── val/
    │   ├── data.json
    │   └── изображения...
    └── test/
        ├── data.json
        └── изображения...
```

`coords.json` должен содержать координаты для каждого изображения OCR, `data.json` — текст номера для каждого изображения REC. Формат значений должен соответствовать исходному датасету и коду `Dataset`.

Если данные лежат в другом месте, пути можно задать переменными окружения: `OCR_TRAIN_DIR`, `OCR_VALID_DIR`, `OCR_TEST_DIR`, `REC_TRAIN_DIR`, `REC_VALID_DIR`, `REC_TEST_DIR`.

Папка весов по умолчанию — `models/checkpoints`. Её можно переопределить через `MODEL_DIR`.

## Запуск

Команды запускаются из корневой папки проекта.

Первичное обучение OCR:

```bash
python -m scripts.train_ocr
```

Первичное обучение REC:

```bash
python -m scripts.train_rec
```

Менеджер моделей (дообучение, тестирование, просмотр изображений):

```bash
python -m scripts.model_manager
```

Пакетная обрезка номеров:

```bash
python -m scripts.crop_images
```

Для обрезки по умолчанию используются `data/to_crop`, `data/cropped` и веса `models/checkpoints/OCR_Detection_NN.pt`. Пути можно переопределить через `CROP_INPUT_DIR`, `CROP_OUTPUT_DIR` и `OCR_WEIGHTS`.

## Важные замечания

- Датасеты и файлы весов в исходном архиве отсутствовали. Поэтому проверить фактическое обучение и загрузку checkpoint на реальных данных невозможно.
- В `model_manager.py` для OCR и REC используются разные checkpoint-файлы: `Dict_OCR_Detection_NN_5.pt` и `Dict_REC_Detection_NN_5.pt`. Если у вас сохранены файлы с другими номерами эпох, измените константы `OCR_CHECKPOINT` и `REC_CHECKPOINT` в начале этого файла.
- Формат checkpoint сохранён совместимым с исходным кодом.
- `Recognized_NN.py` был пустым и удалён.
