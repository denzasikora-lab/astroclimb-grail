# GRAIL для AstroCLIMB

Реализация системы из статьи *GRAIL: Compact Multimodal Similarities for Astronomy Citation Linking*, принятой в трек AstroCLIMB на WASP 2026. Автор статьи: Egor Fedosov, Ural Federal University.

Система классифицирует пары изображений и подписей по четырем отношениям: `same_figure`, `same_paper`, `related_papers`, `unrelated_papers`. Основная модель использует 190 симметричных признаков из замороженных CLIP ViT-B/16, CLIP ViT-B/32, SciNCL, OCR, лексических совпадений и геометрии изображений. Классификаторы обучаются отдельно для каждой модальности.

## Подтвержденные результаты

| Вариант | Локальный macro-F1 | Kaggle |
| --- | ---: | ---: |
| Фиксированная факторизованная модель | 0.48018 | 0.48335 |
| Вложенный выбор модели и поправок решений | 0.47692 | 0.48264 |
| Фиксированная модель без OCR | 0.47212 | не измерялось |
| Фиксированная модель без TF-IDF и редкости терминов | 0.46743 | не измерялось |

Локальные оценки рассчитаны по пяти внешним разбиениям, разделяющим компоненты графа точных объектов. Это результаты разработки, а не независимый тест. Разделение по исходным научным статьям не гарантируется. Kaggle-оценки относятся к двум отправкам от 19 сентября 2026 года. Сравнения по замечаниям рецензентов выполнены отдельно 8 октября 2026 года и приведены ниже; исходные оценки отправленных систем сохранены.

### Выполненные cosine-сравнения и CLIP-абляции

| Вариант | Область | Macro-F1 |
| --- | --- | ---: |
| CLIP B/16 cosine, логистические головы | 10 000 пар, 4 класса | 0.40436 |
| CLIP B/32 cosine, логистические головы | 10 000 пар, 4 класса | 0.39016 |
| SciNCL cosine, логистические головы | 3000 пар подписей, 3 класса | 0.50052 |
| CLIP B/16 + SciNCL cosine, логистические головы | 10 000 пар, 4 класса | 0.42432 |
| Два CLIP + SciNCL cosine, логистические головы | 10 000 пар, 4 класса | 0.42133 |
| Два CLIP + SciNCL cosine, фиксированные boosted-головы | 10 000 пар, 4 класса | 0.41223 |
| Полная модель, повтор с восстановленными представлениями | 10 000 пар, 4 класса | 0.48040 |
| Полная модель без блока B/16 | 10 000 пар, 4 класса | 0.47077 |
| Полная модель без блока B/32 | 10 000 пар, 4 класса | 0.47280 |

При одинаковом boosted-классификаторе полные 190 признаков дают абсолютный прирост 0.06817 относительно трех cosine-признаков. Удаление B/16 снижает оценку на 0.00963, удаление B/32 - на 0.00760. Все остальные признаки и настройки сохраняются, каждая CLIP-абляция использует 173 признака.

Исходный train.csv полностью сверяется с сохраненными хешами, типами, длинами объектов и метками всех 10 000 пар. Используются прежние пять внешних разбиений; точные объекты обучения и проверки не пересекаются. Представления восстановлены по фиксированным версиям моделей, OCR повторно использован по хешам. Тестовые данные, поправки решений и выбор модели не используются.

Оценка повторной полной модели отличается от сохраненной исходной на 0.00022. Она служит контролем для новых сравнений и не заменяет результаты уже отправленной системы. Полные оценки, отчеты по разбиениям, происхождение представлений, контрольные суммы и OOF-предсказания находятся в [reports/review_experiments](reports/review_experiments).

## Установка

Требуется Python 3.10 или новее. Версии библиотек исходного локального эксперимента сохранены в `reports/environment.json`.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[ml,dev]'
python -m pytest
```

Все регрессионные проверки находятся в `tests/test_all_regressions.py`.

## Данные

Данные получают отдельно со [страницы AstroCLIMB](https://www.kaggle.com/competitions/astroclimb). Положите разрешенные файлы `train.csv`, `test.csv` и `sample_submission.csv` в каталог `astroclimb/`. Исходные изображения, подписи, веса моделей и учетные данные в репозитории отсутствуют.

Файл `solution.csv` не используется: код запрещает его чтение. Идентификаторы пар и порядок строк не являются признаками. Словари, IDF и статистики эмбеддингов подгоняются только на объектах обучающей части каждого разбиения.

## Восстановление признаков и исходного эксперимента

Команды выполняются из корня репозитория.

```bash
export PYTHONPATH=src
python scripts/audit_data.py --data-dir astroclimb --output-dir artifacts/audit
python scripts/build_object_cache.py --data-dir astroclimb --output-dir artifacts/object_cache --side 384 --quality 88 --minimum-free-gib 8

python scripts/extract_embeddings.py --cache-dir artifacts/object_cache --output-dir artifacts/embeddings --kind joint --name clip_b16 --model openai/clip-vit-base-patch16 --revision 57c216476eefef5ab752ec549e440a49ae4ae5f3 --batch-size 32 --max-length 77 --image-preprocess letterbox --device auto
python scripts/extract_embeddings.py --cache-dir artifacts/object_cache --output-dir artifacts/embeddings --kind joint --name clip_b32 --model openai/clip-vit-base-patch32 --revision 3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268 --batch-size 32 --max-length 77 --image-preprocess letterbox --device auto
python scripts/extract_embeddings.py --cache-dir artifacts/object_cache --output-dir artifacts/embeddings --kind text --name scincl --model malteos/scincl --revision ebc5348d184ba2fc9beee69b4e394263fce57b2e --batch-size 32 --pooling cls --max-length 512 --device auto
python scripts/extract_layout_descriptors.py --cache-dir artifacts/object_cache --output-dir artifacts/embeddings --name layout
```

Для точного воспроизведения локального OCR нужен macOS Vision:

```bash
swiftc -module-cache-path artifacts/swift_module_cache scripts/ocr_macos.swift -o artifacts/ocr_macos
artifacts/ocr_macos artifacts/object_cache/images artifacts/ocr_fast.jsonl 0 fast
python scripts/train_rich_nested.py --project "$PWD" --output-dir artifacts/run_rich_nested --threads 2
```

Для Linux предусмотрен `scripts/ocr_tesseract.py`. Замена движка OCR может изменить оценки; такой результат следует описывать как отдельный эксперимент.

## Эксперименты по замечаниям рецензентов

```bash
PYTHONPATH=src python scripts/review_experiments.py --project "$PWD" --output-dir artifacts/review_experiments --threads 2
```

Скрипт сравнивает простые классификаторы над косинусными сходствами CLIP/SciNCL и факторизованную модель с теми же настройками, что в статье. Абляции финальной модели удаляют только блок B/16 или только блок B/32, сохраняя остальные признаки. Используются пять разбиений по компонентам объектов без поправок решений и выбора модели по проверочным меткам. SciNCL отдельно оценивается только на парах подписей, где он доступен; его трехклассовый результат нельзя напрямую сравнивать с общим четырехклассовым macro-F1.

Для сравнения OCR можно передать `--full-resolution-ocr путь/к/ocr.jsonl`: файл должен содержать распознавание исходных изображений тем же движком и с теми же настройками. Скрипт сравнит два OCR-варианта при одинаковых классификаторах и разбиениях. Для точного совпадения с сохраненным экспериментом используйте `--fold-ids artifacts/run_rich_nested/fold_ids.npy`.

### Повторные сравнения без тестового набора

Для этих сравнений достаточно `train.csv`, сохраненных обучающих метаданных, внешних разбиений и OCR. Полный архив соревнования, `test.csv` и `sample_submission.csv` не нужны. Подготовка читает CSV построчно, проверяет каждый объект по прежним хешам и создает только компактный кеш; исходный CSV не копируется. Параметр `--minimum-free-gib 6` сохраняет запас места на диске.

В примере сохраненные метаданные, разбиения и OCR находятся в `saved/`. Каждый повтор запускайте в новом каталоге, поскольку завершенные кеши и отчеты защищены от перезаписи.

```bash
review_root=artifacts/review_reproduction
PYTHONPATH=src python scripts/prepare_review_data.py --train train.csv --metadata saved/meta_train.parquet --fold-ids saved/fold_ids.npy --project "$review_root" --minimum-free-gib 6

python scripts/extract_embeddings.py --cache-dir "$review_root/artifacts/object_cache" --output-dir "$review_root/artifacts/embeddings" --kind joint --name clip_b16 --model openai/clip-vit-base-patch16 --revision 57c216476eefef5ab752ec549e440a49ae4ae5f3 --batch-size 16 --max-length 77 --image-preprocess letterbox --device auto
python scripts/extract_embeddings.py --cache-dir "$review_root/artifacts/object_cache" --output-dir "$review_root/artifacts/embeddings" --kind joint --name clip_b32 --model openai/clip-vit-base-patch32 --revision 3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268 --batch-size 16 --max-length 77 --image-preprocess letterbox --device auto
python scripts/extract_embeddings.py --cache-dir "$review_root/artifacts/object_cache" --output-dir "$review_root/artifacts/embeddings" --kind text --name scincl --model malteos/scincl --revision ebc5348d184ba2fc9beee69b4e394263fce57b2e --batch-size 16 --pooling cls --max-length 512 --device auto
PYTHONPATH=src python scripts/extract_layout_descriptors.py --cache-dir "$review_root/artifacts/object_cache" --output-dir "$review_root/artifacts/embeddings" --name layout

PYTHONPATH=src python scripts/review_experiments.py --project "$review_root" --output-dir "$review_root/artifacts/review_experiments" --fold-ids "$review_root/artifacts/audit/fold_ids.npy" --ocr-jsonl saved/ocr_fast.jsonl --threads 2
```

CLIP и SciNCL выполняются по очереди. При локально сохраненных весах можно отключить сеть переменными `HF_HUB_OFFLINE=1` и `TRANSFORMERS_OFFLINE=1`. Для отдельной полностью сохранившейся группы предусмотрен `--modalities` в подготовке; такой результат относится только к явно выбранным модальностям, а не ко всему набору.

## Структура

- `src/astroclimb/`: обработка объектов, признаки, модели, разбиения и проверки входных данных.
- `scripts/`: извлечение представлений, обучение и анализ результатов.
- `tests/test_all_regressions.py`: единый набор регрессионных проверок.
- `reports/`: небольшие отчеты уже выполненных экспериментов.
- `configs/`: настройки исходного эксперимента.

## Лицензия и источники

Код распространяется под MIT, см. `LICENSE`. Данные Kaggle и предварительно обученные модели имеют собственные условия использования.

- [AstroCLIMB](https://www.kaggle.com/competitions/astroclimb)
- [CLIP](https://proceedings.mlr.press/v139/radford21a.html)
- [SciNCL](https://aclanthology.org/2022.emnlp-main.802/)
