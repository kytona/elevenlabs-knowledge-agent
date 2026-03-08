"use client";

import Image from "next/image";
import { useConversation } from "@elevenlabs/react";
import { useState, useEffect, useRef } from "react";

type ActivityLog = {
  label: string;
  detail: string;
};

const backendBaseUrl = process.env.NEXT_PUBLIC_BACKEND_URL?.replace(/\/$/, "");

function getStatusDisplay(status: string): { label: string; dotClass: string } {
  switch (status) {
    case "connected":
      return { label: "Connected", dotClass: "status-dot status-dot--connected" };
    case "connecting":
      return { label: "Connecting...", dotClass: "status-dot status-dot--connecting" };
    default:
      return { label: "Ready", dotClass: "status-dot" };
  }
}

function classifyLog(entry: ActivityLog): "message-agent" | "message-user" | "event" | "error" {
  const lower = entry.label.toLowerCase();
  if (lower === "error") return "error";
  if (lower === "connected" || lower === "disconnected") return "event";
  if (lower === "user") return "message-user";
  return "message-agent";
}

export default function Home() {
  const [logs, setLogs] = useState<ActivityLog[]>([]);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const transcriptRef = useRef<HTMLDivElement>(null);
  const currentYear = new Date().getFullYear();

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
    setLogs((current) => [...current, { label, detail }].slice(-8));
  }

  useEffect(() => {
    if (transcriptRef.current) {
      transcriptRef.current.scrollTop = transcriptRef.current.scrollHeight;
    }
  }, [logs]);

  async function startConversation() {
    if (!backendBaseUrl) {
      setErrorMessage("Missing NEXT_PUBLIC_BACKEND_URL in the frontend environment.");
      return;
    }

    setErrorMessage(null);
    try {
      const response = await fetch(`${backendBaseUrl}/v1/elevenlabs/conversation-token`, {
        method: "POST",
      });

      if (!response.ok) {
        let detail = "Could not start the voice session.";
        try {
          const payload = await response.json();
          if (payload && typeof payload.detail === "string") {
            detail = payload.detail;
          }
        } catch {
          // Ignore non-JSON token errors and use the fallback message.
        }
        throw new Error(detail);
      }

      const payload = await response.json();
      const conversationToken =
        payload && typeof payload.token === "string" && payload.token.trim() ? payload.token : null;
      if (!conversationToken) {
        throw new Error("Backend did not return an ElevenLabs conversation token.");
      }

      await conversation.startSession({ conversationToken, connectionType: "webrtc" });
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
  const statusDisplay = getStatusDisplay(statusText);

  return (
    <>
      <header className="topbar">
        <div className="topbar-inner">
          <span className="topbar-brand">ElevenLabs Knowledge Agent</span>
          <a
            className="topbar-github"
            href="https://github.com/elevenlabs/elevenlabs-examples"
            target="_blank"
            rel="noopener noreferrer"
          >
            GitHub &rarr;
          </a>
        </div>
      </header>

      <main className="hero">
        <div className="hero-grid">
          <div className="hero-left">
            <span className="eyebrow">Custom LLM Starter</span>
            <h1>Voice agents with your own retrieval pipeline.</h1>
            <p className="subtitle">
              Route each turn through your FastAPI webhook for retrieval,
              prompt augmentation, and model routing before streaming TTS.
            </p>
            <div className="stack">
              <span>elevenlabs/react</span>
              <span>FastAPI + Qdrant</span>
              <span>OpenAI streaming</span>
            </div>
          </div>

          <div className="console">
            <div className="console-header">
              <div className="console-status">
                <span className={statusDisplay.dotClass}></span>
                <span>{statusDisplay.label}</span>
              </div>
              <div className="console-actions">
                <button
                  type="button"
                  className="btn-start"
                  disabled={statusText === "connected" || statusText === "connecting"}
                  onClick={startConversation}
                >
                  Start
                </button>
                <button
                  type="button"
                  className="btn-stop"
                  disabled={statusText !== "connected"}
                  onClick={endConversation}
                >
                  Stop
                </button>
              </div>
            </div>

            {errorMessage && (
              <div className="error-banner">{errorMessage}</div>
            )}

            <div className="console-transcript" ref={transcriptRef} aria-live="polite">
              {logs.length === 0 ? (
                <div className="transcript-placeholder">
                  Ask about The Adventure of the Speckled Band — try &quot;Who hired Holmes?&quot; or &quot;What was the speckled band?&quot;
                </div>
              ) : (
                logs.map((entry, index) => {
                  const type = classifyLog(entry);
                  switch (type) {
                    case "message-agent":
                      return (
                        <div className="msg--agent" key={`${entry.label}-${index}`}>
                          {entry.detail}
                        </div>
                      );
                    case "message-user":
                      return (
                        <div className="msg--user" key={`${entry.label}-${index}`}>
                          {entry.detail}
                        </div>
                      );
                    case "event":
                      return (
                        <div className="msg--event" key={`${entry.label}-${index}`}>
                          &mdash; {entry.label}: {entry.detail}
                        </div>
                      );
                    case "error":
                      return (
                        <div className="msg--error" key={`${entry.label}-${index}`}>
                          {entry.detail}
                        </div>
                      );
                  }
                })
              )}
            </div>
          </div>
        </div>
      </main>

      <section className="section">
        <div className="architecture-grid">
          <div className="architecture-visual">
            <Image
              src="/railway_canvas.png"
              alt="Railway deployment canvas showing frontend, backend, ingest service, and Qdrant connections."
              width={688}
              height={647}
              className="architecture-image"
              priority
            />
          </div>

          <div className="architecture-copy">
            <h2 className="section-label">Architecture</h2>
            <div className="pipeline">
              <span className="pipeline-step">Speech Input</span>
              <span className="pipeline-arrow">&rarr;</span>
              <span className="pipeline-step">FastAPI Webhook</span>
              <span className="pipeline-arrow">&rarr;</span>
              <span className="pipeline-step">Qdrant Retrieval</span>
              <span className="pipeline-arrow">&rarr;</span>
              <span className="pipeline-step">LLM</span>
              <span className="pipeline-arrow">&rarr;</span>
              <span className="pipeline-step">TTS Output</span>
            </div>
            <p className="section-body">
              ElevenLabs handles VAD and speech-to-text, then posts an OpenAI-style
              chat completion request to your Railway-hosted backend. The backend
              embeds the latest user utterance, retrieves the top matching chunks
              from Qdrant, and injects them into the system prompt before forwarding
              the request to the configured LLM.
            </p>
          </div>
        </div>
      </section>

      <section className="section section--secondary">
        <h2 className="section-label">Troubleshooting</h2>
        <ul className="troubleshoot-list">
          <li>Set the Custom LLM Server URL in ElevenLabs to your backend URL (e.g. <code>https://your-backend.up.railway.app</code>).</li>
          <li>Run the ingestion command before starting the voice client so Qdrant contains the Speckled Band sample story.</li>
          <li>Set <code>NEXT_PUBLIC_BACKEND_URL</code> in the Railway frontend service or locally for preview.</li>
        </ul>
      </section>

      <footer className="footer">
        <p>
          Built with ElevenLabs &middot; FastAPI &middot; Qdrant &middot; &copy; {currentYear}{" "}
          <a href="https://kytona.com" target="_blank" rel="noopener noreferrer">
            Kytona Limited
          </a>
        </p>
        <p className="disclaimer">
          This project is not affiliated with, endorsed by, or associated with ElevenLabs in any way. It is an independent demo.
        </p>
      </footer>
    </>
  );
}
