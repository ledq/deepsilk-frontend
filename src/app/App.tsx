// src/app/App.tsx
import { useEffect, useRef, useState } from "react";
import Controls from "../components/Controls";
import Preview from "../components/Preview";
import ProgressBar from "../components/ProgressBar";
import { useAnnotator } from "../hooks/useAnnotator";
import { annotateOnServer } from "../services/api";

export default function App() {
  // UI state
  const [videoURL, setVideoURL] = useState<string | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [fps, setFps] = useState(10);
  const [conf, setConf] = useState(0.25);
  const [iou, setIou] = useState(0.45);

  // DOM refs for preview
  const localVideoRef = useRef<HTMLVideoElement>(null);
  const localOverlayRef = useRef<HTMLCanvasElement>(null);

  // Annotator hook (handles client ONNX loop + progress)
  const {
    runClient,
    running,
    progress,
    setRunning,
    setProgress,
    videoRef,
    overlayRef
  } = useAnnotator();

  // Wire our local refs into the hook's refs once
  useEffect(() => {
    (videoRef as any).current = localVideoRef.current;
    (overlayRef as any).current = localOverlayRef.current;
  }, [videoRef, overlayRef]);

  // File picked
  function onPick(f: File) {
    setFile(f);
    const url = URL.createObjectURL(f);
    if (videoURL) URL.revokeObjectURL(videoURL);
    setVideoURL(url);
  }

  // Run in browser (ONNX Runtime Web)
  async function onLocal() {
    await runClient(fps, conf, iou /*, optional names array */);
  }

  // Run on server (FastAPI /annotate)
  async function onServer() {
    if (!file) return;
    setRunning(true);
    setProgress(15);
    try {
      const res = await annotateOnServer(file, fps, conf, iou, 640);
      setProgress(70);

      if (res instanceof Blob) {
        // Direct MP4 stream
        const url = URL.createObjectURL(res);
        const a = document.createElement("a");
        a.href = url;
        a.download = "annotated.mp4";
        a.click();
        setTimeout(() => URL.revokeObjectURL(url), 1500);
      } else {
        // JSON response with URLs or payload
        if (res.video_url) window.open(res.video_url, "_blank");
        if (res.json_url) window.open(res.json_url, "_blank");
      }
      setProgress(100);
    } catch (e) {
      console.error(e);
      alert("Server processing failed. Check backend or VITE_API_BASE.");
    } finally {
      setRunning(false);
      setTimeout(() => setProgress(0), 800);
    }
  }

  // Clear overlay
  function onClear() {
    setProgress(0);
    setRunning(false);
    const ctx = localOverlayRef.current?.getContext("2d");
    if (ctx) ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
  }

  return (
    <div className="container">
      <div className="hero">
        <span className="badge">DeepSilk • Silksong video annotator</span>
        <h1>Upload clip → annotate → export</h1>
        <div className="sub">
          Client ONNX with optional FastAPI fallback. Fully responsive layout.
        </div>
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--gap)' }}>
        <section className="card panel">
          <div className="label">1. Controls</div>
          <Controls
            fps={fps}
            setFps={setFps}
            conf={conf}
            setConf={setConf}
            iou={iou}
            setIou={setIou}
            onLocal={onLocal}
            onServer={onServer}
            onClear={onClear}
            disabled={!videoURL || running}
          />
          <hr className="line" />
          <ProgressBar value={progress} />
          <div style={{ fontSize: 12, color: "var(--muted)", marginTop: 6 }}>
            {running ? "Processing…" : "Idle"} • API base:{" "}
            <code>{import.meta.env.VITE_API_BASE ?? "(none)"}</code>
          </div>
        </section>

        <section className="card panel">
          <div className="label">2. Preview</div>
          <Preview
            videoURL={videoURL}
            onPick={onPick}
            videoRef={localVideoRef}
            overlayRef={localOverlayRef}
          />
        </section>
      </div>
    </div>
  );
}
