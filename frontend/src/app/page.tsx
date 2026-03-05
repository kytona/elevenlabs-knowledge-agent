"use client";

import { useConversation } from "@elevenlabs/react";
import { useState } from "react";

type ActivityLog = {
  label: string;
  detail: string;
};

const agentId = process.env.NEXT_PUBLIC_ELEVENLABS_AGENT_ID;

export default function Home() {
  const [logs, setLogs] = useState<ActivityLog[]>([]);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const conversation = useConversation({
    onConnect: () => appendLog("Connected", "Microphone session started."),
    onDisconnect: () => appendLog("Disconnected", "The voice session has ended."),
    onMessage: (message) => {
      const source = typeof message === "object" && message && "source" in message ? String(message.source) : "agent";
      const messageText =
        typeof message === "object" && message && "message" in message ? String(message.message) : JSON.stringify(message);
      appendLog(source, messageText);
    },
    onError: (error: unknown) => {
      const detail = error instanceof Error ? error.message : "Unknown ElevenLabs client error.";
      setErrorMessage(detail);
      appendLog("Error", detail);
    },
  });

  function appendLog(label: string, detail: string) {
    setLogs((current) => [{ label, detail }, ...current].slice(0, 8));
  }

  async function startConversation() {
    if (!agentId) {
      setErrorMessage("Missing NEXT_PUBLIC_ELEVENLABS_AGENT_ID in the frontend environment.");
      return;
    }

    setErrorMessage(null);
    try {
      await conversation.startSession({ agentId, connectionType: "webrtc" });
    } catch (error) {
      const detail = error instanceof Error ? error.message : "Could not start the voice session.";
      setErrorMessage(detail);
      appendLog("Error", detail);
    }
  }

  async function endConversation() {
    try {
      await conversation.endSession();
    } catch (error) {
      const detail = error instanceof Error ? error.message : "Could not stop the voice session.";
      setErrorMessage(detail);
      appendLog("Error", detail);
    }
  }

  const statusText = conversation.status ?? "disconnected";

  return (
    <main className="page-shell">
      <div className="page-grid">
        <section className="card hero">
          <span className="eyebrow">Custom LLM Starter</span>
          <h1>Voice agents with your own retrieval pipeline.</h1>
          <p className="hero-copy">
            This starter bypasses the ElevenLabs Knowledge Base and routes each turn through your FastAPI webhook,
            where you can run retrieval, prompt augmentation, and model routing before streaming the answer back for
            text-to-speech.
          </p>
          <div className="stack">
            <span>Next.js + @elevenlabs/react</span>
            <span>FastAPI webhook</span>
            <span>Qdrant retrieval</span>
            <span>OpenAI-compatible streaming</span>
          </div>
        </section>

        <section className="card panel">
          <h2>Conversation Controls</h2>
          <p>Use your configured ElevenLabs agent to test the custom LLM path against the sample knowledge base.</p>
          <div className="status">
            <strong>Status</strong>
            <span>{statusText}</span>
          </div>
          {errorMessage ? (
            <div className="status">
              <strong>Issue</strong>
              <span>{errorMessage}</span>
            </div>
          ) : null}
          <div className="actions">
            <button
              type="button"
              className="primary"
              disabled={statusText === "connected" || statusText === "connecting"}
              onClick={startConversation}
            >
              Start conversation
            </button>
            <button
              type="button"
              className="secondary"
              disabled={statusText !== "connected"}
              onClick={endConversation}
            >
              Stop conversation
            </button>
          </div>
          <div className="logs" aria-live="polite">
            {logs.length === 0 ? (
              <div className="log">
                <strong>Ready</strong>
                Waiting for the first voice session.
              </div>
            ) : (
              logs.map((entry, index) => (
                <div className="log" key={`${entry.label}-${index}`}>
                  <strong>{entry.label}</strong>
                  {entry.detail}
                </div>
              ))
            )}
          </div>
        </section>

        <section className="card panel notes">
          <h3>How it works</h3>
          <p>
            ElevenLabs performs VAD and speech-to-text, then posts an OpenAI-style chat completion request to your
            backend. The backend embeds the latest user utterance, retrieves the top matching chunks from Qdrant, and
            injects them into the system prompt before forwarding the request to the configured LLM.
          </p>
        </section>

        <section className="card panel notes">
          <h3>Troubleshooting</h3>
          <ul>
            <li>Confirm the Custom LLM URL in ElevenLabs points to your ngrok-backed `/v1/chat/completions` endpoint.</li>
            <li>Run the ingestion command before starting the voice client so Qdrant contains the sample chunks.</li>
            <li>Set `NEXT_PUBLIC_ELEVENLABS_AGENT_ID` in `frontend/.env.local` or the repo root `.env` file.</li>
          </ul>
        </section>
      </div>
    </main>
  );
}
