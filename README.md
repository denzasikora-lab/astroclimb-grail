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

Локальные оценки рассчитаны по пяти внешним разбиениям, разделяющим компоненты графа точных объектов. Это результаты разработки, а не независимый тест. Разделение по исходным научным статьям не гарантируется. Kaggle-оценки относятся к двум отправкам от 19 сентября 2026 года. Дополнительные эксперименты по замечаниям рецензентов запускаются отдельно; подготовленный скрипт не означает, что они уже выполнены.

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
