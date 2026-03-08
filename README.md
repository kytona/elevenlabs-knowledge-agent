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
docs/
  railway.md          <-- Railway deployment guide
.qdrant/              <-- Local Qdrant vector DB (created on first ingest)
.env.example
README.md
```

## Prerequisites (local setup)

- Python 3.11+
- Node.js 20+
- OpenAI API key
- ElevenLabs API key and Agent ID
- [ngrok](https://ngrok.com) account (to expose your local backend to ElevenLabs)

## Local setup

Everything runs on your machine. Qdrant stores vectors locally in the `.qdrant` folder at the repo root—no Docker, no separate server.

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
```

For the frontend, add to `.env` or `frontend/.env.local`:

```text
NEXT_PUBLIC_ELEVENLABS_AGENT_ID=your_agent_id
```

### 2. Backend setup

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Ingest the sample documents

Qdrant stores data in `.qdrant` at the repo root. Ingest the sample docs:

```bash
cd backend
source .venv/bin/activate
python -m app.ingest ../data/sample_docs --recreate-collection
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

### 5. Expose the webhook with ngrok

ElevenLabs needs a public URL to call your webhook. Run ngrok in a separate terminal:

```bash
ngrok http 8000
```

Copy the HTTPS URL (e.g. `https://abc123.ngrok-free.app`).

### 6. Configure your ElevenLabs agent

- Open the [ElevenLabs app](https://elevenlabs.io/app) → **Configure** → **Agents**.
- Select your agent and open the **Agent** tab.
- In the **LLM** section (bottom right), click the model row (e.g. "Gemini 2.5 Flash").
- Choose **Custom LLM** and set the Server URL to: `https://your-ngrok-url.ngrok-free.app/v1` (include `/v1`; ElevenLabs appends `/chat/completions`)
- Model ID can be `custom` or any label. API Key can stay "None" for a self-hosted endpoint.
- Test with the text playground first.

**If you see "Server URL must use a secure HTTPS connection":** Ensure the URL includes `/v1` (e.g. `https://xxx.ngrok-free.app/v1`). If it still fails, ngrok's free-tier interstitial may block ElevenLabs' validation—try adding the header `ngrok-skip-browser-warning: 1` under **Request headers**, or use a paid ngrok plan.

### 7. Run the frontend

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:3000` and test voice.

## Testing the webhook with curl

Before connecting ElevenLabs, verify the SSE format:

```bash
curl -N http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "custom",
    "stream": true,
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "When should I use a custom LLM instead of the built-in knowledge base?"}
    ]
  }'
```

You should see OpenAI-style `data:` SSE chunks followed by `data: [DONE]`.

## Swap your LLM provider

The webhook contract is OpenAI-compatible by design. To use a different compatible provider (Ollama, Together, Groq, etc.), change `OPENAI_BASE_URL` and set matching chat and embedding model names. No provider-specific adapter is required.

## Deploy to Railway

For a production deployment with backend, frontend, and Qdrant all on Railway, see **[docs/railway.md](docs/railway.md)**.

## Implementation notes

- The backend supports `.md` and `.txt` ingestion in v1.
- The backend loads the repo-root `.env` automatically even when you run commands from `backend/`.
- If Qdrant returns no matches, the webhook still forwards the request to the base model.
- The frontend is intentionally minimal. The webhook is the core value of the project.
- Qdrant runs in embedded mode with data stored in `.qdrant` at the repo root. No Docker or separate Qdrant server needed.

## About the Author

Built by [GT](https://gaurangtorvekar.com) - 3x technical co-founder, 12+ years building developer tools and infrastructure. London-based. [LinkedIn](https://www.linkedin.com) [Twitter/X](https://x.com)
