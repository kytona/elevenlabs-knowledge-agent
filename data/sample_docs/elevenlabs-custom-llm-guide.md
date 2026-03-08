# ElevenLabs Custom LLM Quick Reference

ElevenLabs Conversational AI can either use the built-in Knowledge Base feature or route each turn to a Custom LLM endpoint that follows the OpenAI chat completion schema.

Use the built-in Knowledge Base when you want quick setup and you do not need to control chunking, embeddings, or retrieval ranking. Use a Custom LLM endpoint when you need full ownership of retrieval logic, prompt construction, or model routing.

In the Custom LLM flow, ElevenLabs performs voice activity detection and speech-to-text first. It then sends a POST request to your backend with a JSON body that looks like a chat completion request. The key fields are the message list and the `stream` flag.

Your backend should extract the latest user message, embed it, search your vector store, and add the retrieved context to the prompt before calling the actual language model. The backend then streams text tokens back using OpenAI-style server-sent events. ElevenLabs converts those tokens to speech in real time.

For this starter, the backend is deployed on Railway so ElevenLabs can reach the webhook over HTTPS without any tunnel. Qdrant can also run as a Railway service, which keeps the retrieval path public for ingestion and private for backend traffic.

Qdrant is a good vector database for this pattern because it can run as embedded local storage during development or as a Railway service for deployment. The local embedded mode is useful for quick tests because the ingest CLI and API server can share the same on-disk collection.

This starter kit uses OpenAI `text-embedding-3-small` for embeddings and `gpt-4o-mini` for generation by default. Any OpenAI-compatible provider can be swapped in by changing the base URL and model settings.

Three common reasons to choose the custom architecture are:

1. You need a different chunking strategy than the platform default.
2. You want to use a different embedding model or reranking step.
3. You want the same webhook to pull from multiple data sources instead of a single hosted knowledge base.

The fastest way to verify the integration is to test the backend with curl first, then configure the same endpoint inside the ElevenLabs agent dashboard, test with the text playground, and only then switch to the browser voice client.
