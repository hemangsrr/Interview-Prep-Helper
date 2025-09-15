# Interview Helper

A Flask + Socket.IO app for mock interviews with multiple SME agents, streaming questions, voice (TTS/ASR), and feedback export. Panels are generated from a JD via an LLM and reused via semantic matching. State is persisted to MongoDB so sessions can resume after reloads.

## Features
- SME panel generation from JD (Markdown-rendered feedback)
- Streaming SME questions with sentence-level TTS
- User voice input via Web Speech (ASR) with long-silence stop
- Auto-advance interview flow, per-agent colored chat bubbles with animations
- Feedback page with PDF export
- Dark mode toggle (persisted)
- MongoDB persistence: panels, JD embeddings, and per-session interview state
- LLM provider abstraction in `llm.py` for easy swaps

## Tech stack
- Backend: Flask, Flask-SocketIO, PyMongo
- LLM: OpenAI via `llm.py` abstraction
- Frontend: Bootstrap 5, Bootstrap Icons, Web Speech API (TTS/ASR)
- PDF: ReportLab

## Project layout
- `app.py` — Flask app and Socket.IO endpoints
- `agents/`
  - `panel_builder.py` — builds panel from JD using `LLM`
  - `langgraph_orchestrator.py` — interview flow using LangGraph and `LLM`
  - `sme_react.py` — SME nodes (ask/feedback)
- `services/`
  - `db.py` — Mongo helpers (state save/rehydrate, JD embedding reuse)
  - `pdf_utils.py` — PDF text extraction
  - `pdf_report.py` — build feedback PDF
  - `openai_client.py` — OpenAI client construction used by `llm.py`
- `llm.py` — LLM abstraction with `invoke()` and `embed()`
- `templates/` — Jinja2 templates (Bootstrap UI, dark mode)
- `static/main.css` — custom styles, chat bubbles, dark mode
- `.env.example` — environment variable example

## Prerequisites
- Python 3.10+
- MongoDB 6+
- LLM deployed Locally or an API Key (By Deafult set up to use OpenAI API Call)

## Setup
1) Clone and install Python deps
```bash
pip install -r requirements.txt
```

2) Configure environment
- Copy `.env.example` to `.env` and fill values according to your setup:
```
OPENAI_API_KEY="sk-..."
SECRET_KEY=some-secret
MONGO_URI=mongodb://<user>:<password>@localhost:27017/<database>?authSource=<database>
MONGO_DB=<database>
```

3) (Optional) Create Mongo user/db. Skip if you already have a user/db setup
```js
// In Mongo shell
use <database>
// Create a user with readWrite on <database>
// Adjust credentials to match MONGO_URI
```

4) Run the app
```bash
python app.py
```
Visit http://localhost:5000

## How it works
- JD analysis: `POST /analyze_jd`
  - Extracts text if PDF is uploaded (`services/pdf_utils.py`).
  - Embeds JD via `LLM.embed()` and checks Mongo for similar JD via cosine (`services/db.find_similar_jd_panel`).
  - If similarity >= 0.9, reuses saved panel to save tokens; else, generates panel with `LLM.invoke()`.
  - Saves JD + panel (+ embedding) to `panels` collection.

- Interview flow:
  - Panel is editable on the UI; saving updates DB panel for the session id.
  - Start interview initializes `LangGraphOrchestrator` (with `LLM`) and persists initial state.
  - Streaming questions use Socket.IO events: `question_start`, `question_chunk`, `question_end`.
  - On each step, state is upserted into `interviews` by `save_interview_state()`.
  - If the page reloads, `/interview` attempts to rehydrate from Mongo state.

- Voice UX:
  - TTS uses `window.speechSynthesis` to speak sentence-by-sentence as chunks arrive.
  - ASR uses `webkitSpeechRecognition`/`SpeechRecognition` with continuous mode and long-silence stop (2.5s).

- Feedback:
  - `feedback.html` renders Markdown summary to HTML via `Markdown` package.
  - PDF download streams a generated ReportLab PDF.

## Environment variables
- `OPENAI_API_KEY` — OpenAI key for completions and embeddings
- `SECRET_KEY` — Flask secret key
- `MONGO_URI` — MongoDB connection string
- `MONGO_DB` — Mongo database name

## Common commands
- Install deps: `pip install -r requirements.txt`
- Run dev server: `python app.py`

## Switching LLM providers
All LLM calls go through `llm.py`:
- `LLM.invoke(system, user, stream=False, json=False)`
- `LLM.embed(text, model=...)`

To switch providers:
- Replace OpenAI client creation in `services/openai_client.py` and calls in `llm.py`, or create a new backend and branch logic by env vars.

## Use a custom LLM via the wrapper
There are two easy paths to use your own/local LLM without touching the rest of the code.

1) OpenAI-compatible HTTP API (recommended)

Many local servers (LM Studio, llama.cpp server, vLLM, Ollama OpenAI‑compat mode) expose an OpenAI-compatible REST API. In that case you only need to point the OpenAI SDK to a different base URL and model name.

- Set env vars (adjust to your server):
```
OPENAI_API_KEY="sk-local"              # any non-empty string
OPENAI_BASE_URL="http://localhost:8000/v1"  # your server's base URL
MODEL="my-local-model-name"            # e.g., "lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF"
```

- Update `services/openai_client.py` to pass `base_url` to the client:
```python
from openai import OpenAI
import os

_client = None

def get_openai_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set.")
        base_url = os.getenv("OPENAI_BASE_URL")  # optional
        if base_url:
            _client = OpenAI(api_key=api_key, base_url=base_url)
        else:
            _client = OpenAI(api_key=api_key)
    return _client
```

- Optionally set the default model via env and read it in `llm.py`:
```python
import os
self._model = model or os.getenv("MODEL", "gpt-4o")
```

With this, `LLM.invoke()` and `LLM.embed()` will route to your local server seamlessly.

2) Fully custom provider (minimal edits)

If your provider is not OpenAI-compatible, implement the two methods in `llm.py`:
```python
class LLM:
    def invoke(self, system_prompt, user_prompt, stream=False, json=False):
        if stream:
            # yield partial strings (chunks)
            yield from my_provider_stream(system_prompt, user_prompt)
        else:
            return my_provider_complete(system_prompt, user_prompt, json=json)

    def embed(self, text: str, model: str | None = None) -> list[float]:
        return my_provider_embed(text, model)
```

Keep the same signature so agents/orchestrator continue working without changes. If your provider lacks an embeddings API, you can disable JD reuse (skip `LLM.embed()` usage) or plug a separate embedding library.

Tip: Use an env flag (e.g., `LLM_PROVIDER=openai|custom`) to switch implementations at runtime inside `llm.py`.

## Notes and limits
- JD semantic reuse uses brute-force cosine across all stored embeddings. For larger datasets, consider a vector DB (FAISS/Pinecone/Atlas Vector Search).
- Web Speech APIs work best on Chromium-based browsers; behavior varies by platform.

## Troubleshooting
- Socket.IO client errors: ensure matching versions (Flask-SocketIO v5 with Socket.IO v4 client). We include the CDN v4 client.
- OpenAI httpx proxy error: `httpx==0.27.2` is pinned in `requirements.txt`.
- PDF parsing: some PDFs have no extractable text; provide JD text instead.

## License
MIT (or your preferred license)
