# ElevenLabs Knowledge Agent

An open-source starter kit for connecting ElevenLabs Conversational AI to a custom retrieval pipeline. Instead of relying on the built-in Knowledge Base, this project intercepts the Custom LLM webhook, runs your own RAG step, and streams the grounded answer back in the OpenAI chat completion format that ElevenLabs expects.

## Architecture

ElevenLabs offers a built-in Knowledge Base, but it is a black box: you cannot control chunking, swap embedding models, or implement hybrid retrieval. This starter bypasses it entirely using the Custom LLM webhook, giving you full ownership of the retrieval pipeline.

```text
User speaks into browser
        |
        v
[Next.js + @elevenlabs/react]
        |
        v
[ElevenLabs Conversational AI]
VAD + STT
        |
        v
[POST /v1/chat/completions]       <-- Custom LLM webhook (bypasses built-in KB)
FastAPI webhook
  -> extract latest user message
  -> embed query (OpenAI text-embedding-3-small)
  -> retrieve top chunks from Qdrant
  -> inject context into system prompt
  -> stream LLM tokens back as SSE
        |
        v
[ElevenLabs TTS]
        |
        v
User hears grounded response
```

## When to use this vs the built-in Knowledge Base

Use ElevenLabs' native Knowledge Base when you want the fastest possible setup and do not need to control retrieval internals.

Use this starter when you need:

- custom chunking
- a different embedding model
- hybrid or multi-source retrieval
- custom prompt injection
- OpenAI-compatible provider swapping

This repo exists because the Custom LLM path is the right integration point for enterprise teams that need retrieval control, but there is no official starter that demonstrates it end to end.

## Project layout

```text
backend/
  app/
    main.py
    rag.py
    ingest.py
    config.py
frontend/
  src/app/
data/sample_docs/
.qdrant/              <-- Local Qdrant vector DB (created on first ingest)
.env.example
README.md
```

## Local deployment

Everything runs on your machine. Qdrant stores vectors locally in the `.qdrant` folder at the repo root.
The default demo dataset is [`data/sample_docs/the-adventure-of-the-speckled-band.md`](data/sample_docs/the-adventure-of-the-speckled-band.md), the Sherlock Holmes story from _The Adventures of Sherlock Holmes_.

### Prerequisites

- Python 3.11+
- Node.js 20+
- OpenAI API key
- ElevenLabs Agent ID

### 1. Configure environment

Copy `.env.example` to `.env` and fill in your keys:

```bash
cp .env.example .env
```

Edit `.env` and set:

```text
OPENAI_API_KEY=your_openai_api_key
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_CHAT_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
QDRANT_IN_MEMORY=true
QDRANT_LOCAL_PATH=.qdrant
QDRANT_COLLECTION_NAME=knowledge_base
ENABLE_DEBUG_RETRIEVAL=true
```

For the frontend, add to `.env` or `frontend/.env.local`:

```text
NEXT_PUBLIC_ELEVENLABS_AGENT_ID=your_elevenlabs_agent_id
```

### 2. Backend setup

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Ingest the default sample document

Qdrant stores data in `.qdrant` at the repo root. Ingest the Speckled Band sample story:

```bash
cd backend
source .venv/bin/activate
python -m app.ingest ../data/sample_docs/the-adventure-of-the-speckled-band.md --recreate-collection
```

### 4. Start the backend

```bash
cd backend
source .venv/bin/activate
uvicorn app.main:app --reload --port 8000
```

Health check:

```bash
curl http://localhost:8000/health
```

### 5. Configure your ElevenLabs agent

- Open the [ElevenLabs app](https://elevenlabs.io/app) → **Configure** → **Agents**.
- Select your agent and open the **Agent** tab.
- In the **LLM** section (bottom right), click the model row (e.g. "Gemini 2.5 Flash").
- For a deployed setup, choose **Custom LLM** and set the Server URL to your Railway backend URL. This project accepts both `https://your-backend.up.railway.app/v1` and `https://your-backend.up.railway.app`.
- Model ID can be `custom` or any label. API Key can stay "None" for a self-hosted endpoint.
- Test with the text playground first.

### 6. Run the frontend

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:3000` and test voice. The browser connects to ElevenLabs directly with your public agent ID, so add `localhost:3000` and your deployed frontend hostname to the agent allowlist first.

### 7. Test the webhook with curl

Before connecting ElevenLabs, verify the SSE format:

```bash
curl -N http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "custom",
    "stream": true,
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Who hired Sherlock Holmes and what was their problem?"}
    ]
  }'
```

You should see OpenAI-style `data:` SSE chunks followed by `data: [DONE]`.

To verify retrieval specifically during local development, hit the backend debug route:

```bash
curl "http://localhost:8000/debug/retrieval?q=What%20was%20the%20speckled%20band%3F"
```

If retrieval is working, the response will include matching chunks from `data/sample_docs/the-adventure-of-the-speckled-band.md`.

Suggested demo questions:

- Who hired Sherlock Holmes and what was their problem?
- What was the speckled band?
- Who is Dr. Roylott and what did he do?
- How did Holmes solve the mystery?
- What happened to Julia Stoner?

---

## Railway deployment

For a production deployment with backend, frontend, and Qdrant all on Railway, use the steps below.

### Prerequisites

- Railway account
- OpenAI API key
- ElevenLabs Agent ID

### Architecture on Railway

```text
User speaks into browser
        |
        v
[Next.js + @elevenlabs/react on Railway]
        |
        v
[ElevenLabs Conversational AI]
        |
        v
[POST /v1/chat/completions]
FastAPI webhook on Railway
  -> Qdrant (Railway service, private network)
  -> stream LLM tokens back as SSE
        |
        v
[ElevenLabs TTS]
        |
        v
User hears grounded response
```

### Step 1: Create Railway services

Create one Railway project with three services from the same repo. Each service uses a different **Root Directory** so Railway builds and deploys the correct subfolder.

**Backend service**

1. Add a new service and connect the GitHub repo.
2. Under **Source** → **Add Root Directory**, set `backend`.
3. Generate a public domain for the backend (required for ElevenLabs webhook).

**Frontend service**

1. Add another service in the same project.
2. Connect the same repo.
3. Under **Source** → **Add Root Directory**, set `frontend`.
4. Generate a public domain for the frontend.

**Qdrant service**

1. Add a third service.
2. Choose **Deploy from Docker image** (not GitHub).
3. Use image `qdrant/qdrant:latest`.
4. Attach a persistent volume: mount path `/qdrant/storage`.
5. Enable a public domain only if you want to ingest from your local machine; otherwise use private networking.

| Service    | Root Directory | Notes                                                            |
| ---------- | -------------- | ---------------------------------------------------------------- |
| `backend`  | `backend`      | Uses `Procfile`: binds to `$PORT` (Railway sets 8080 by default) |
| `frontend` | `frontend`     | Uses `Procfile` for Next.js                                      |
| `qdrant`   | (Docker image) | `qdrant/qdrant:latest`, volume at `/qdrant/storage`              |

### Step 2: Backend environment variables

Set these on the `backend` service:

```text
OPENAI_API_KEY=...
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_CHAT_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
QDRANT_URL=http://qdrant.railway.internal:6333
QDRANT_COLLECTION_NAME=knowledge_base
QDRANT_IN_MEMORY=false
BACKEND_PUBLIC_URL=https://your-backend.up.railway.app
ENABLE_DEBUG_RETRIEVAL=false
```

Use `http://qdrant.railway.internal:6333` so the backend reaches Qdrant over Railway's private network. Replace `qdrant` with your Qdrant service name if different.

### Step 3: Frontend environment variables

Set these on the `frontend` service:

```text
NEXT_PUBLIC_ELEVENLABS_AGENT_ID=your_elevenlabs_agent_id
```

### Step 4: Ingest documents into Railway Qdrant

**Option A: Ingest from local machine** (Qdrant has public domain)

```bash
cd backend
source .venv/bin/activate
QDRANT_URL=https://your-qdrant-public-domain.up.railway.app \
python -m app.ingest ../data/sample_docs/the-adventure-of-the-speckled-band.md --recreate-collection
```

**Option B: Ingest from a one-off Railway service** (Qdrant has no public domain)

1. Add a new service, connect the same repo, leave **Root Directory** empty.
2. Set `RAILWAY_DOCKERFILE_PATH=Dockerfile.ingest`.
3. Add the same variables as the backend (including `QDRANT_URL=http://qdrant.railway.internal:6333`).
4. Deploy. The service runs the ingest, then exits. Delete the service after successful ingest.

### Step 5: Deploy and health check

Push to your connected repo or trigger a deploy. Verify:

```bash
curl https://your-backend.up.railway.app/health
```

Expected response includes `"status": "ok"`, `"qdrant_in_memory": false`, and a non-zero `"qdrant_points_count"` after ingestion.

### Step 6: Configure ElevenLabs

1. Open the [ElevenLabs agent dashboard](https://elevenlabs.io/app/conversational-ai).
2. Enable **Custom LLM** and set the Server URL to `https://your-backend.up.railway.app/v1` or `https://your-backend.up.railway.app`.
3. Test with the text playground first.
4. Open the Railway frontend and test voice. The frontend connects directly to ElevenLabs with `NEXT_PUBLIC_ELEVENLABS_AGENT_ID`, so the agent allowlist should include your Railway frontend hostname.

### Step 7: Test the webhook

```bash
curl -N https://your-backend.up.railway.app/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "custom",
    "stream": true,
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Who hired Sherlock Holmes and what was their problem?"}
    ]
  }'
```

To test retrieval directly in a non-production environment where diagnostics are enabled:

```bash
curl "https://your-backend.up.railway.app/debug/retrieval?q=What%20was%20the%20speckled%20band%3F"
```

### Troubleshooting "Application failed to respond"

- **Port binding** – Procfile must use `web: uvicorn app.main:app --host 0.0.0.0 --port $PORT`
- **Qdrant connection** – Confirm `QDRANT_URL=http://qdrant.railway.internal:6333` and the Qdrant service name matches.
- **Missing env vars** – Ensure `OPENAI_API_KEY` is set on the backend and `NEXT_PUBLIC_ELEVENLABS_AGENT_ID` is set on the frontend.
- **Agent allowlist** – Add your deployed frontend origin and `localhost:3000` to the ElevenLabs agent allowlist.
- **Root directory** – Backend service must have Root Directory = `backend`.

---

## Swap your LLM provider

The webhook contract is OpenAI-compatible by design. To use a different compatible provider (Ollama, Together, Groq, etc.), change `OPENAI_BASE_URL` and set matching chat and embedding model names. No provider-specific adapter is required.

## Implementation notes

- The backend supports `.md` and `.txt` ingestion in v1.
- The backend loads the repo-root `.env` automatically even when you run commands from `backend/`.
- If Qdrant returns no matches, the webhook still forwards the request to the base model.
- The frontend is intentionally minimal. The webhook is the core value of the project.
- The frontend only needs `NEXT_PUBLIC_ELEVENLABS_AGENT_ID` for voice session startup.
- `/debug/retrieval` is intended for local or explicitly enabled environments, not for a default public production deployment.
- Qdrant runs in embedded mode with data stored in `.qdrant` at the repo root. No Docker or separate Qdrant server needed.

## About the Author

Built by [GT](https://gaurangtorvekar.com) - 3x technical co-founder, 12+ years building developer tools and infrastructure. London-based. [LinkedIn](https://www.linkedin.com) [Twitter/X](https://x.com)
