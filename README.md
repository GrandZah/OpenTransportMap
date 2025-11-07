# GeoTransportMap

Генератор постерных карт для автобусных остановок: базовые слои из OpenStreetMap. Выход — статичные PNG для печати.

## Примеры изображений

**Подробная схематика**  
![Detail schematic](docs/images/detail_schematic.png)

**Как может выглядеть в реальном мире**
<details>
  <summary>Посмотреть изображение</summary>
  
  ![Overview map](docs/images/overview_map.png)
</details>

## Быстрый старт

```bash
poetry run transport-posters --limit 1 --render-map --render-routes --area-id 1614795
```

### Параметры

| Параметр          | Описание                                                                     |
|-------------------|------------------------------------------------------------------------------|
| `--render-map`    | Флаг для рендеринга подложки карт (фоновые карты для маршрутов)              |
| `--render-routes` | Флаг для рендеринга самих маршрутов                                          |
| `--limit`         | Количество остановок, для которых будут сгенерированы плакаты                |
| `--area-id`       | ID Relation в OpenStreetMap (OSM), внутри которого ищутся маршруты и границы |

**Примеры ID городов:**

- [`1614795` — Вологда](https://www.openstreetmap.org/relation/1614795)
- [`2206549` — Люблин](https://www.openstreetmap.org/relation/2206549)


### Результаты рендера

- `output/maps/` — итоговые PNG-карты (при запуске `generate_maps`)
- `output/logs/` — логи запуска
- `output/composed_img/` — составные постеры (при запуске `generate_tree_in_one`)

## Структура проекта

```
GeoTransportMap/
├─ cache/                         # Runtime cache
├─ data/
│  ├─ cache/                      # Кэш данных
│  ├─ processed/                  # Обработанные данные от OSM (формат Parquet)
│  ├─ raw/                        # Сырые данные о транспорте из OSM
│  └─ map/                        # Кэш тайлов по слоям (tile=zoom-x-y)
├─ docs/
│  ├─ images/                     # Изображения для README.md
├─ output/
│  ├─ composed_img/               # Составные постеры
│  ├─ logs/                       # Логи запуска
│  ├─ maps/                       # Итоговые PNG-карты
├─ src/
│  ├─ transport_posters/
│  │  ├─ data_map/                # Получение данных о карте от OSM + стилизация
│  │  ├─ data_transport/          # Получение маршрутов от OSM и построение CityRouteDatabase
│  │  ├─ generate_maps/           # Базовый генератор плакатов
│  │  ├─ generate_three_in_one/   # Несколько детализаций карты в один постер
│  │  ├─ render_map/              # Отображение карты на фигуре
│  │  ├─ render_transport/        # Отображение транспорта на фигуре
│  │  ├─ utils/                   # Вспомогательные модули
│  │  ├─ __init__.py
│  │  ├─ load_configs.py          # Загрузка конфигов проекта
│  │  ├─ logger.py                # Логирование
│  │  ├─ main.py                  # CLI-вход
├─ style/
│  ├─ assets/
│  │  ├─ fonts/                   # Шрифты
│  │  ├─ patterns/                # Паттерны для карт
│  ├─ detailed_city_layers.json
│  ├─ detailed_city_layers_small.json
│  ├─ detailed_city_layers_v2.json
│  ├─ far_city_layers.json
│  ├─ far_city_layers_labeled.json
│  ├─ general_city_layers.json
│  ├─ general_city_layers_small.json
```


## Как установить и запустить проект

<details>
  <summary>Подробнее...</summary>
  
### 0) Требования

- **OS**: Linux, macOS или Windows 10/11
- **Python**: 3.11 или 3.12 (рекомендуется 3.12)
- **Poetry**: ≥ 1.6
- **Git**: для клонирования репозитория
- Интернет-доступ для загрузки данных OSM

### 1) Клонируем репозиторий

```bash
git clone https://github.com/GrandZah/OpenTransportMap
cd OpenTransportMap
```

### 2) Устанавливаем Poetry (если еще не установлен)

- **macOS / Linux**:
  ```bash
  curl -sSL https://install.python-poetry.org | python3 -
  ```
  После установки добавьте Poetry в `PATH`, если требуется.

- **Windows (PowerShell)**:
  ```powershell
  (Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -
  ```
  Перезапустите терминал.

Проверка:
```bash
poetry --version
```

### 3) Устанавливаем зависимости проекта

В корне репозитория:
```bash
poetry install
```

Poetry создаст виртуальное окружение и установит все зависимости из `pyproject.toml`.

### 4) Запускаем из виртуального окружения

```bash
poetry run transport-posters --limit 1 --render-map --render-routes --area-id 1614795
```

Через несколько минут в `output/` появятся результаты (см. раздел ниже).

Чтобы посмотреть предварительно, то можно не генерировать карту. (Пару десятков секунд)

```bash
poetry run transport-posters --limit 1 --render-routes --area-id 1614795
```

</details>

