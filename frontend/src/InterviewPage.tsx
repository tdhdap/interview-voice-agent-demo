import { useState } from "react";
import InterviewRoom from "./components/InterviewRoom";

type Phase = "lobby" | "active";

export default function InterviewPage() {
  const [phase, setPhase] = useState<Phase>("lobby");

  if (phase === "active") {
    return <InterviewRoom onEnd={() => setPhase("lobby")} />;
  }

  return (
    <div style={styles.center}>
      <div style={styles.card}>
        <h1 style={styles.heading}>Mock Interview with Alex</h1>
        <p style={styles.body}>
          You'll be interviewed by Alex, an AI Staff Engineer. Make sure your
          microphone is working before you begin.
        </p>
        <button onClick={() => setPhase("active")} style={styles.button}>
          Start Interview
        </button>
      </div>
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
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
    boxShadow: "0 1px 3px rgba(0,0,0,0.08)",
  },
  heading: {
    fontSize: "22px",
    fontWeight: 600,
    color: "#111827",
    margin: "0 0 10px",
  },
  body: {
    fontSize: "14px",
    color: "#6b7280",
    lineHeight: 1.6,
    margin: "0 0 24px",
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
    transition: "opacity 0.15s",
  },
};
