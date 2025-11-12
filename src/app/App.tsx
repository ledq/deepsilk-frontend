// src/app/App.tsx
import { useEffect, useRef, useState } from "react";
import Controls from "../components/Controls";
import Preview from "../components/Preview";
// import ProgressBar from "../components/ProgressBar";
import { useAnnotator } from "../hooks/useAnnotator";
import { annotateOnServer } from "../services/api";

export default function App() {
  // UI state
  const [videoURL, setVideoURL] = useState<string | null>(null);
  const [file, setFile] = useState<File | null>(null);

  // Detection mode: 'browser' or 'server'
  const [detectionMode, setDetectionMode] = useState<'browser' | 'server'>('server');
  const [annotatedVideoURL, setAnnotatedVideoURL] = useState<string | null>(null);

  // DOM refs for preview
  const localVideoRef = useRef<HTMLVideoElement>(null);
  const localOverlayRef = useRef<HTMLCanvasElement>(null);

  // Annotator hook (handles client ONNX loop + progress)
  const {
    runClient,
    running,
    setRunning,
    videoRef,
    overlayRef,
    cleanup
  } = useAnnotator();

  // Wire our local refs into the hook's refs once
  useEffect(() => {
    (videoRef as any).current = localVideoRef.current;
    (overlayRef as any).current = localOverlayRef.current;
  }, [videoRef, overlayRef]);

  // File picked
  function onPick(f: File) {
  setFile(f);
  if (videoURL) URL.revokeObjectURL(videoURL);
  setVideoURL(URL.createObjectURL(f));
  }

  // Run in browser (ONNX Runtime Web)
  async function onLocal() {
  await runClient(10, 0.25, 0.45 /*, optional names array */);
  }

  // Run on server (FastAPI /annotate)
  async function onServer() {
    if (!file) return;
    setRunning(true);
  // setProgress(15); // Progress bar removed
    setAnnotatedVideoURL(null);
    try {
  const res = await annotateOnServer(file, 10, 0.25, 0.45, 640);
  // setProgress(70); // Progress bar removed

      if (res instanceof Blob) {
        // Direct MP4 stream
        const url = URL.createObjectURL(res);
        setAnnotatedVideoURL(url);
        // Download is now manual via button
      } else {
        // JSON response with URLs or payload
        if (res.video_url) {
          setAnnotatedVideoURL(res.video_url);
        }
        if (res.json_url) window.open(res.json_url, "_blank");
      }
  // setProgress(100); // Progress bar removed
    } catch (e) {
      console.error(e);
      alert("Server processing failed. Check backend or VITE_API_BASE.");
    } finally {
      setRunning(false);
    }
  }

  // Clear overlay
  function onClear() {
  // setProgress(0); // Progress bar removed
    setRunning(false);
    cleanup(); // Clean up any ongoing animations and cached data
    const ctx = localOverlayRef.current?.getContext("2d");
    if (ctx) ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
  }

  // Unified detection handler
  function onRunDetection() {
    if (detectionMode === 'browser') {
      onLocal();
    } else {
      onServer();
    }
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
        {/* Controls and Preview side by side */}
        <div style={{ display: 'flex', flexDirection: 'row', gap: 'var(--gap)' }}>
          <section className="card panel" style={{ flex: 1 }}>
            <div className="label">1. Controls</div>
            <Controls
              detectionMode={detectionMode}
              setDetectionMode={setDetectionMode}
              onRunDetection={onRunDetection}
              onClear={onClear}
              disabled={!videoURL || running}
              videoUploaded={!!videoURL}
            />
            <hr className="line" />
          </section>
          <section className="card panel" style={{ width: 'auto', flex: 'none', padding: '12px 12px 0 12px' }}>
            <div className="label">2. Preview</div>
            <Preview
              videoURL={videoURL}
              onPick={onPick}
              videoRef={localVideoRef}
              overlayRef={localOverlayRef}
            />
          </section>
        </div>
        {/* Detection panel below */}
        <section className="card panel" style={{ marginTop: 'var(--gap)' }}>
          <div className="label">3. Detection</div>
          <div style={{ width: '100%', minHeight: 480, display: 'flex', alignItems: 'center', justifyContent: 'center', background: 'var(--panel)', borderRadius: 12, color: 'var(--muted)', flexDirection: 'column', boxShadow: '0 4px 32px #0003', padding: 16 }}>
            {running && detectionMode === 'server' ? (
              <>
                <div style={{marginBottom: 8}}>Processing on server…</div>
                <div className="loader" style={{width: 32, height: 32, border: '4px solid #ccc', borderTop: '4px solid var(--primary)', borderRadius: '50%', animation: 'spin 1s linear infinite'}} />
              </>
            ) : annotatedVideoURL && detectionMode === 'server' ? (
              <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', width: '100%' }}>
                <video
                  key={annotatedVideoURL}
                  src={annotatedVideoURL}
                  controls
                  style={{ width: '90vw', maxWidth: 960, maxHeight: 540, borderRadius: 12, background: '#000', boxShadow: '0 2px 16px #0006' }}
                />
              </div>
            ) : (
              <span>Detection video will appear here</span>
            )}
          </div>
        </section>
      </div>
    </div>
  );
}
