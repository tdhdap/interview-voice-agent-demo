import { useEffect, useRef, useState, useCallback } from "react";

const WS_URL = "ws://localhost:7000/ws";

interface Transcript {
  role: "user" | "alex";
  text: string;
}

interface InterviewRoomProps {
  onEnd: () => void;
}

// --- Audio utilities ---

function downsample(buffer: Float32Array, fromRate: number, toRate = 16000): Float32Array {
  if (fromRate === toRate) return buffer;
  const ratio = fromRate / toRate;
  const newLength = Math.round(buffer.length / ratio);
  const result = new Float32Array(newLength);
  for (let i = 0; i < newLength; i++) {
    const start = Math.round(i * ratio);
    const end = Math.round((i + 1) * ratio);
    let sum = 0, count = 0;
    for (let j = start; j < end && j < buffer.length; j++) { sum += buffer[j]; count++; }
    result[i] = count > 0 ? sum / count : 0;
  }
  return result;
}

function float32ToInt16(buffer: Float32Array): ArrayBuffer {
  const out = new Int16Array(buffer.length);
  for (let i = 0; i < buffer.length; i++)
    out[i] = Math.min(1, Math.max(-1, buffer[i])) * 0x7fff;
  return out.buffer;
}

// --- Component ---

export default function InterviewRoom({ onEnd }: InterviewRoomProps) {
  const [agentSpeaking, setAgentSpeaking] = useState(false);
  const [isMute, setIsMute] = useState(false);
  const [transcripts, setTranscripts] = useState<Transcript[]>([]);
  const [error, setError] = useState<string | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const captureCtxRef = useRef<AudioContext | null>(null);
  const playCtxRef = useRef<AudioContext | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const workletRef = useRef<AudioWorkletNode | null>(null);
  const nextPlayTimeRef = useRef(0);
  const scheduledSourcesRef = useRef<AudioBufferSourceNode[]>([]);
  const muteRef = useRef(false);
  const speakingTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const transcriptEndRef = useRef<HTMLDivElement | null>(null);

  // Bug 2 fix: track turn boundaries so each Alex response is a separate bubble
  const newTurnRef = useRef(true);

  // Bug 3 fix: gate mic while agent audio is playing (prevents echo self-interruption)
  const agentPlayingRef = useRef(false);
  const playingTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    transcriptEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [transcripts]);

  useEffect(() => {
    // Bug 1 fix: local variable (not a ref) — each effect invocation gets its own
    // closure, so React StrictMode's double-mount can't reset a shared ref mid-flight.
    let cancelled = false;

    async function setup() {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          audio: {
            echoCancellation: true,
            noiseSuppression: true,
            autoGainControl: true,
          },
          video: false,
        });
        if (cancelled) { stream.getTracks().forEach(t => t.stop()); return; }
        streamRef.current = stream;

        const captureCtx = new AudioContext();
        captureCtxRef.current = captureCtx;
        await captureCtx.audioWorklet.addModule("/pcm-processor.js");
        if (cancelled) { captureCtx.close(); stream.getTracks().forEach(t => t.stop()); return; }

        const playCtx = new AudioContext({ sampleRate: 24000 });
        playCtxRef.current = playCtx;
        nextPlayTimeRef.current = 0;

        const ws = new WebSocket(WS_URL);
        ws.binaryType = "arraybuffer";
        wsRef.current = ws;
        if (cancelled) { ws.close(); return; }

        let wsErrored = false;

        ws.onerror = () => {
          wsErrored = true;
          if (!cancelled)
            setError("Cannot connect to server. Make sure `python main.py` is running on port 7000.");
        };

        ws.onclose = (e) => {
          if (!cancelled && !e.wasClean && !wsErrored)
            setError("Connection lost. Please try again.");
        };

        ws.onopen = () => {
          if (cancelled) { ws.close(); return; }
          playCtx.resume();

          const source = captureCtx.createMediaStreamSource(stream);
          const worklet = new AudioWorkletNode(captureCtx, "pcm-processor");
          workletRef.current = worklet;

          worklet.port.onmessage = (e: MessageEvent<Float32Array>) => {
            // Bug 3: gate mic when agent is playing audio — prevents echo self-interruption
            if (ws.readyState !== WebSocket.OPEN || muteRef.current || agentPlayingRef.current) return;
            const downsampled = downsample(e.data, captureCtx.sampleRate, 16000);
            const pcm16 = float32ToInt16(downsampled);
            ws.send(pcm16);
          };

          const muteGain = captureCtx.createGain();
          muteGain.gain.value = 0;
          source.connect(worklet);
          worklet.connect(muteGain);
          muteGain.connect(captureCtx.destination);
        };

        ws.onmessage = (event: MessageEvent) => {
          if (event.data instanceof ArrayBuffer) {
            playAudio(event.data, playCtx);

            // Bug 3: mark agent as playing; debounce-clear 600ms after last chunk
            agentPlayingRef.current = true;
            setAgentSpeaking(true);
            if (speakingTimerRef.current) clearTimeout(speakingTimerRef.current);
            if (playingTimerRef.current) clearTimeout(playingTimerRef.current);
            speakingTimerRef.current = setTimeout(() => setAgentSpeaking(false), 800);
            playingTimerRef.current = setTimeout(() => { agentPlayingRef.current = false; }, 600);
          } else {
            try {
              handleJsonMessage(JSON.parse(event.data as string), playCtx);
            } catch { /* ignore malformed */ }
          }
        };
      } catch (err) {
        if (!cancelled)
          setError(err instanceof Error ? err.message : "Failed to access microphone.");
      }
    }

    function playAudio(arrayBuffer: ArrayBuffer, playCtx: AudioContext) {
      const int16 = new Int16Array(arrayBuffer);
      const float32 = new Float32Array(int16.length);
      for (let i = 0; i < int16.length; i++) float32[i] = int16[i] / 32768.0;

      const buffer = playCtx.createBuffer(1, float32.length, 24000);
      buffer.copyToChannel(float32, 0);
      const src = playCtx.createBufferSource();
      src.buffer = buffer;
      src.connect(playCtx.destination);

      const now = playCtx.currentTime;
      nextPlayTimeRef.current = Math.max(now, nextPlayTimeRef.current);
      src.start(nextPlayTimeRef.current);
      nextPlayTimeRef.current += buffer.duration;

      scheduledSourcesRef.current.push(src);
      src.onended = () => {
        scheduledSourcesRef.current = scheduledSourcesRef.current.filter(s => s !== src);
      };
    }

    function stopPlayback(playCtx: AudioContext) {
      scheduledSourcesRef.current.forEach(s => { try { s.stop(); } catch {} });
      scheduledSourcesRef.current = [];
      nextPlayTimeRef.current = playCtx.currentTime;
      agentPlayingRef.current = false;
    }

    function handleJsonMessage(msg: { type: string; text?: string }, playCtx: AudioContext) {
      if (msg.type === "interrupted") {
        stopPlayback(playCtx);
        setAgentSpeaking(false);
        if (speakingTimerRef.current) clearTimeout(speakingTimerRef.current);
        if (playingTimerRef.current) clearTimeout(playingTimerRef.current);
        // Bug 2: interruption ends the current turn — next Alex chunk starts fresh
        newTurnRef.current = true;
      } else if (msg.type === "turn_complete") {
        setAgentSpeaking(false);
        // Bug 2: mark that next Alex chunk belongs to a new turn
        newTurnRef.current = true;
      } else if (msg.type === "user" && msg.text) {
        setTranscripts(prev => {
          const last = prev[prev.length - 1];
          if (last?.role === "user") {
            return [...prev.slice(0, -1), { role: "user", text: last.text + msg.text }];
          }
          return [...prev, { role: "user", text: msg.text! }];
        });
      } else if (msg.type === "gemini" && msg.text) {
        setTranscripts(prev => {
          const last = prev[prev.length - 1];
          // Bug 2: start a new bubble after every turn boundary
          if (!newTurnRef.current && last?.role === "alex") {
            return [...prev.slice(0, -1), { role: "alex", text: last.text + msg.text }];
          }
          newTurnRef.current = false;
          return [...prev, { role: "alex", text: msg.text! }];
        });
      }
    }

    setup();

    return () => {
      cancelled = true;
      if (speakingTimerRef.current) clearTimeout(speakingTimerRef.current);
      if (playingTimerRef.current) clearTimeout(playingTimerRef.current);
      workletRef.current?.disconnect();
      wsRef.current?.close();
      captureCtxRef.current?.close();
      playCtxRef.current?.close();
      streamRef.current?.getTracks().forEach(t => t.stop());
    };
  }, []);

  const toggleMic = useCallback(() => {
    const next = !isMute;
    setIsMute(next);
    muteRef.current = next;
  }, [isMute]);

  const endInterview = useCallback(() => {
    wsRef.current?.close();
    onEnd();
  }, [onEnd]);

  if (error) {
    return (
      <div style={styles.center}>
        <div style={styles.card}>
          <p style={styles.errorText}>{error}</p>
          <button onClick={onEnd} style={styles.button}>Back</button>
        </div>
      </div>
    );
  }

  return (
    <div style={styles.page}>
      <div style={styles.avatarWrapper}>
        <div style={{
          ...styles.avatar,
          borderColor: agentSpeaking ? "#22c55e" : "#e5e7eb",
          background: agentSpeaking ? "#f0fdf4" : "#f3f4f6",
          boxShadow: agentSpeaking ? "0 0 0 6px #bbf7d0" : "none",
        }}>
          🎙
        </div>
        <p style={styles.agentLabel}>
          {agentSpeaking ? "Alex is speaking..." : "Alex is listening"}
        </p>
      </div>

      {transcripts.length > 0 && (
        <div style={styles.transcriptBox}>
          {transcripts.map((t, i) => (
            <div key={i} style={{
              ...styles.transcriptLine,
              textAlign: t.role === "user" ? "right" : "left",
              color: t.role === "user" ? "#374151" : "#1d4ed8",
            }}>
              <span style={styles.transcriptLabel}>{t.role === "user" ? "You" : "Alex"}</span>
              <span>{t.text}</span>
            </div>
          ))}
          <div ref={transcriptEndRef} />
        </div>
      )}

      <p style={styles.micLabel}>
        {isMute ? "🔇 Microphone muted" : "🎤 Microphone active"}
      </p>

      <div style={styles.controls}>
        <button
          onClick={toggleMic}
          title={isMute ? "Unmute" : "Mute"}
          style={{
            ...styles.iconBtn,
            background: isMute ? "#fee2e2" : "#ffffff",
            borderColor: isMute ? "#fca5a5" : "#e5e7eb",
            color: isMute ? "#dc2626" : "#374151",
          }}
        >
          {isMute ? "🔇" : "🎤"}
        </button>
        <button onClick={endInterview} style={styles.endBtn}>
          📵 End Interview
        </button>
      </div>
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  page: {
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    justifyContent: "center",
    minHeight: "100vh",
    gap: "24px",
    padding: "24px",
    background: "#f9fafb",
    fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
  },
  center: {
    display: "flex",
    minHeight: "100vh",
    alignItems: "center",
    justifyContent: "center",
    padding: "24px",
    background: "#f9fafb",
    fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
  },
  card: {
    background: "#ffffff",
    border: "1px solid #e5e7eb",
    borderRadius: "12px",
    padding: "40px",
    width: "100%",
    maxWidth: "420px",
  },
  avatarWrapper: {
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    gap: "12px",
  },
  avatar: {
    width: "96px",
    height: "96px",
    borderRadius: "50%",
    border: "4px solid",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    fontSize: "36px",
    transition: "all 0.3s ease",
  },
  agentLabel: {
    fontSize: "14px",
    fontWeight: 500,
    color: "#6b7280",
    margin: 0,
  },
  transcriptBox: {
    width: "100%",
    maxWidth: "560px",
    maxHeight: "220px",
    overflowY: "auto",
    background: "#ffffff",
    border: "1px solid #e5e7eb",
    borderRadius: "10px",
    padding: "12px 16px",
    display: "flex",
    flexDirection: "column",
    gap: "8px",
  },
  transcriptLine: {
    display: "flex",
    flexDirection: "column",
    gap: "2px",
    fontSize: "13px",
    lineHeight: 1.5,
  },
  transcriptLabel: {
    fontSize: "11px",
    fontWeight: 600,
    color: "#9ca3af",
    textTransform: "uppercase" as const,
    letterSpacing: "0.05em",
  },
  micLabel: {
    fontSize: "13px",
    color: "#9ca3af",
    margin: 0,
  },
  controls: {
    display: "flex",
    alignItems: "center",
    gap: "16px",
  },
  iconBtn: {
    width: "48px",
    height: "48px",
    borderRadius: "50%",
    border: "1px solid",
    fontSize: "20px",
    cursor: "pointer",
    transition: "opacity 0.15s",
  },
  endBtn: {
    background: "#dc2626",
    color: "#ffffff",
    border: "none",
    borderRadius: "8px",
    padding: "12px 20px",
    fontSize: "14px",
    fontWeight: 500,
    cursor: "pointer",
    transition: "opacity 0.15s",
  },
  errorText: {
    fontSize: "13px",
    color: "#dc2626",
    background: "#fef2f2",
    border: "1px solid #fecaca",
    borderRadius: "6px",
    padding: "10px 12px",
    marginBottom: "16px",
  },
  button: {
    width: "100%",
    background: "#111827",
    color: "#ffffff",
    border: "none",
    borderRadius: "8px",
    padding: "12px 20px",
    fontSize: "14px",
    fontWeight: 500,
    cursor: "pointer",
  },
};
