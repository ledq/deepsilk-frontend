// src/app/App.tsx
import { useRef, useState } from "react";
import Controls from "../components/Controls";
import Preview from "../components/Preview";
import { useAnnotator } from "../hooks/useAnnotator";
import { annotateOnServer } from "../services/api";

export default function App() {
  // UI state
  const [videoURL, setVideoURL] = useState<string | null>(null);
  const [file, setFile] = useState<File | null>(null);

  // Detection mode: 'browser' or 'server'
  const [detectionMode, setDetectionMode] = useState<"browser" | "server">(
    "server"
  );
  const [annotatedVideoURL, setAnnotatedVideoURL] = useState<string | null>(
    null
  );

  // Preview refs (used only inside Preview panel)
  const previewVideoRef = useRef<HTMLVideoElement>(null);
  const previewOverlayRef = useRef<HTMLCanvasElement>(null);

  // Annotator hook (for browser mode, detection panel)
  const {
    runClient,
    running,
    setRunning,
    videoRef,    // used in Detection panel
    overlayRef,  // used in Detection panel
    cleanup,
  } = useAnnotator();

  // File picked
  function onPick(f: File) {
    setFile(f);
    if (videoURL) URL.revokeObjectURL(videoURL);
    setAnnotatedVideoURL(null);
    setVideoURL(URL.createObjectURL(f));
  }

  // Run in browser (ONNX Runtime Web)
  async function onLocal() {
    if (!videoURL) return;

    // Ensure detection video actually plays so frames are available
    if (videoRef.current && videoRef.current.paused) {
      try {
        await videoRef.current.play();
      } catch (err) {
        console.warn("Autoplay failed, user interaction may be required:", err);
      }
    }

    // Slightly lower conf since NMS is baked into the ONNX
    await runClient(10, 0.1, 0.45);
  }

  // Run on server (FastAPI /annotate)
  async function onServer() {
    if (!file) return;
    setRunning(true);
    setAnnotatedVideoURL(null);
    try {
      const res = await annotateOnServer(file, 10, 0.25, 0.45, 640);

      if (res instanceof Blob) {
        const url = URL.createObjectURL(res);
        setAnnotatedVideoURL(url);
      } else {
        if (res.video_url) {
          setAnnotatedVideoURL(res.video_url);
        }
        if (res.json_url) window.open(res.json_url, "_blank");
      }
    } catch (e) {
      console.error(e);
      alert("Server processing failed. Check backend or VITE_API_BASE.");
    } finally {
      setRunning(false);
    }
  }

  // Clear overlay and state
  function onClear() {
    setRunning(false);
    cleanup();               // stop browser-mode intervals & clear overlay
    setAnnotatedVideoURL(null);
  }

  // Unified detection handler
  function onRunDetection() {
    if (detectionMode === "browser") {
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

      <div
        style={{
          display: "flex",
          flexDirection: "column",
          gap: "var(--gap)",
        }}
      >
        {/* Controls and Preview side by side */}
        <div
          style={{
            display: "flex",
            flexDirection: "row",
            gap: "var(--gap)",
          }}
        >
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

          <section
            className="card panel"
            style={{
              width: "auto",
              flex: "none",
              padding: "12px 12px 0 12px",
            }}
          >
            <div className="label">2. Preview</div>
            <Preview
              videoURL={videoURL}
              onPick={onPick}
              videoRef={previewVideoRef}
              overlayRef={previewOverlayRef}
            />
          </section>
        </div>

        {/* Detection panel below */}
        <section className="card panel" style={{ marginTop: "var(--gap)" }}>
          <div className="label">3. Detection</div>
          <div
            style={{
              width: "100%",
              minHeight: 480,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              background: "var(--panel)",
              borderRadius: 12,
              color: "var(--muted)",
              flexDirection: "column",
              boxShadow: "0 4px 32px #0003",
              padding: 16,
            }}
          >
            {detectionMode === "browser" && videoURL ? (
              <div
                style={{
                  position: "relative",
                  width: 640,
                  height: 360,
                  background: "#000",
                  borderRadius: 8,
                  overflow: "hidden",
                  boxShadow: "0 2px 8px #0002",
                  margin: "0 auto",
                }}
              >
                <video
                  ref={videoRef}
                  src={videoURL}
                  style={{
                    width: "100%",
                    height: "100%",
                    objectFit: "cover",
                    borderRadius: 8,
                    background: "#000",
                    display: "block",
                  }}
                  muted
                  playsInline
                  controls
                />
                <canvas
                  ref={overlayRef}
                  style={{
                    position: "absolute",
                    top: 0,
                    left: 0,
                    width: "100%",
                    height: "100%",
                    pointerEvents: "none",
                    border: "2px solid #444",
                  }}
                />
                {!running && (
                  <div
                    style={{
                      position: "absolute",
                      bottom: 8,
                      left: 8,
                      padding: "4px 8px",
                      background: "#000a",
                      color: "#fff",
                      fontSize: 12,
                      borderRadius: 4,
                    }}
                  >
                    Click “Run detection” to start.
                  </div>
                )}
              </div>
            ) : running && detectionMode === "server" ? (
              <>
                <div style={{ marginBottom: 8 }}>Processing…</div>
                <div
                  className="loader"
                  style={{
                    width: 32,
                    height: 32,
                    border: "4px solid #ccc",
                    borderTop: "4px solid var(--primary)",
                    borderRadius: "50%",
                    animation: "spin 1s linear infinite",
                  }}
                />
              </>
            ) : annotatedVideoURL && detectionMode === "server" ? (
              <div
                style={{
                  display: "flex",
                  flexDirection: "column",
                  alignItems: "center",
                  width: "100%",
                }}
              >
                <video
                  key={annotatedVideoURL}
                  src={annotatedVideoURL}
                  controls
                  style={{
                    width: "90vw",
                    maxWidth: 960,
                    maxHeight: 540,
                    borderRadius: 12,
                    background: "#000",
                    boxShadow: "0 2px 16px #0006",
                  }}
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
