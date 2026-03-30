# Тестовое

Локальный RAG-чатбот для Moodle docs:
- парсинг офлайн HTML зеркала Moodle docs;
- индексация в Chroma;
- retrieval + генерация ответов через Ollama (`qwen3:1.7b`);
- REST API на FastAPI.

## 1) Что в репозитории

- `scripts/parse_moodle_docs_offline_pages.py` — HTML -> `pages.jsonl`
- `scripts/index_pages_jsonl_to_chroma.py` — `pages.jsonl` -> Chroma index
- `scripts/retrieve_chroma.py` — локальная проверка retrieval
- `backend/app/main.py` — чат API
- `configs/config.yaml` — единый конфиг
- `docker-compose.yml` — запуск `backend + ollama + ollama-pull`



## 2) Подготовка данных 

### 2.1 Подготовка Python окружения

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2.2 Парсинг офлайн HTML -> `pages.jsonl`

```bash
python3 scripts/parse_moodle_docs_offline_pages.py --write-errors
```

Выход:
- `data/moodle_docs_501/pages.jsonl`
- `data/moodle_docs_501/errors.jsonl`

### 2.3 Индексация в Chroma

```bash
python3 scripts/index_pages_jsonl_to_chroma.py --recreate
```

Проверка retrieval:

```bash
python3 scripts/retrieve_chroma.py --query "Как создать новый курс в Moodle?" --k 5
```

После индексации перезапусти backend-контейнер:

```bash
sudo docker compose restart backend
```

## 2) Быстрый запуск через Docker

```bash
sudo docker compose up --build -d
```

Проверка статуса:

```bash
sudo docker compose ps
```

Ожидаемо:
- `local-assist-ollama` -> `healthy`
- `local-assist-ollama-pull` -> `exited (0)`
- `local-assist-backend` -> `running`

Проверка API:

```bash
curl -s http://localhost:8061/api/v1/health
```

Остановка:

```bash
sudo docker compose down
```
## 3) REST API: smoke test

Создать сессию:

```bash
SESSION_ID=$(curl -s -X POST http://localhost:8061/api/v1/chat/sessions | python3 -c "import sys,json; print(json.load(sys.stdin)['session_id'])")
echo "$SESSION_ID"
```

Отправить сообщение:

```bash
curl -s -X POST "http://localhost:8061/api/v1/chat/sessions/${SESSION_ID}/messages" \
  -H "Content-Type: application/json" \
  -d '{"content":"Как создать новый курс в Moodle?"}'
```

Эндпоинты:
- `GET /api/v1/health`
- `POST /api/v1/chat/sessions`
- `POST /api/v1/chat/sessions/{session_id}/messages`

## 4) Быстрый запуск через Docker

```bash
sudo docker compose up --build -d
```

Проверка статуса:

```bash
sudo docker compose ps
```

Ожидаемо:
- `local-assist-ollama` -> `healthy`
- `local-assist-ollama-pull` -> `exited (0)`
- `local-assist-backend` -> `running`

Проверка API:

```bash
curl -s http://localhost:8061/api/v1/health
```

Остановка:

```bash
sudo docker compose down
```

## 5) Конфиг и воспроизводимость

Все дефолты в `configs/config.yaml`:
- chunking, embedding model, retrieval top-k;
- параметры chat/history;
- параметры Ollama.

`llm.random_seed` передаётся в Ollama как `options.seed` 

### Как работает `configs/config.yaml`

- Скрипты из `scripts/` читают конфиг через `--config` (по умолчанию `configs/config.yaml`).
- `backend` тоже читает `configs/config.yaml` при старте.

Ключевые секции:
- `paths` — где входные/выходные данные и Chroma;
- `parse_html` — фильтры парсинга (`min_chars`, `max_chars`, `limit`);
- `chunking` — размер и overlap чанков;
- `embeddings` — embedding model;
- `indexing` — batch/limit индексации;
- `retrieval` — `top_k`, вывод retrieval-скрипта;
- `chat` — глубина истории;
- `llm` — `ollama_base_url`, `ollama_model`, `timeout_seconds`, `num_gpu`, `random_seed`, `system_prompt_template`.

Пример: сменить модель и top-k

```yaml
embeddings:
  model_name: "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

retrieval:
  top_k: 7

llm:
  ollama_model: "qwen3:1.7b"
```

Пример: использовать другой путь к данным

```yaml
paths:
  pages_jsonl_path: "data/my_docs/pages.jsonl"
  chroma_persist_dir: "data/chroma_my_docs"
  chroma_collection: "my_docs_collection"
```

## 6) логи Docker

```bash
sudo docker compose logs -f backend
sudo docker compose logs -f ollama
```
