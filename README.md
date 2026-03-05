# ElevenLabs Knowledge Agent

An open-source starter kit for connecting ElevenLabs Conversational AI to a custom retrieval pipeline. Instead of relying on the built-in Knowledge Base, this project intercepts the Custom LLM webhook, runs your own RAG step, and streams the grounded answer back in the OpenAI chat completion format that ElevenLabs expects.

## Architecture

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
[POST /v1/chat/completions]
FastAPI webhook
  -> extract latest user message
  -> embed query
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
docker-compose.yml
.env.example
README.md
```

## Prerequisites

- Python 3.11+
- Node.js 20+
- OpenAI API key
- ElevenLabs API key and Agent ID
- ngrok account for local webhook testing

Docker is optional. If you do not want to run Qdrant in Docker, set `QDRANT_IN_MEMORY=true`.

## Setup

### 1. Configure environment

Copy `.env.example` to `.env` and fill in your keys.

Frontend-only env values can also live in `frontend/.env.local`:

```bash
NEXT_PUBLIC_ELEVENLABS_AGENT_ID=your_agent_id
```

### 2. Backend setup

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Start Qdrant

Docker path:

```bash
docker-compose up -d
```

No-Docker path:

```bash
export QDRANT_IN_MEMORY=true
export QDRANT_LOCAL_PATH=../.qdrant
```

This no-Docker mode uses Qdrant's embedded local storage on disk, not a process-local in-memory store, so the ingest CLI and API server can share the same collection across separate commands.

### 4. Ingest the sample document

From the repo root:

```bash
cd backend
python -m app.ingest ../data/sample_docs --recreate-collection
```

### 5. Run the backend

```bash
cd backend
uvicorn app.main:app --reload --port 8000
```

Health check:

```bash
curl http://localhost:8000/health
```

### 6. Expose the webhook with ngrok

```bash
ngrok http 8000
```

Set the ElevenLabs Custom LLM URL to:

```text
https://your-ngrok-subdomain.ngrok.app/v1/chat/completions
```

### 7. Configure your ElevenLabs agent

- Open the ElevenLabs agent dashboard.
- Enable the Custom LLM option.
- Point it at your ngrok URL.
- Test with the text playground first.
- After the text path works, open the local frontend and test voice.

### 8. Run the frontend

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:3000`.

## Testing the webhook with curl first

Before connecting ElevenLabs, verify the SSE format directly:

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

The webhook contract is OpenAI-compatible by design. To use a different compatible provider such as Ollama, Together, or Groq, change `OPENAI_BASE_URL` and set matching chat and embedding model names.

No provider-specific adapter is required in this starter.

## Demo proof checklist

- Screenshot of successful `curl` streaming test
- Screenshot of the ElevenLabs text playground returning grounded answers
- Screenshot or short recording of the browser voice client

## Implementation notes

- The backend supports `.md` and `.txt` ingestion in v1.
- The backend loads the repo-root `.env` automatically even when you run commands from `backend/`.
- If Qdrant returns no matches, the webhook still forwards the request to the base model.
- The frontend is intentionally minimal. The webhook is the core value of the project.
- ngrok is for local development only. Deploying the backend publicly removes that requirement.

## About the Author

Built by [GT](https://gaurangtorvekar.com) - 3x technical co-founder, 12+ years building developer tools and infrastructure. London-based. [LinkedIn](https://www.linkedin.com) [Twitter/X](https://x.com)
