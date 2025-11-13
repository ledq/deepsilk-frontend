import React, { useEffect, useRef, useState } from "react";
import * as ort from "onnxruntime-web";

// Type for detection box
export type YoloBox = {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  score: number;
  classId: number;
};

const MODEL_URL = "/models/best.onnx";
const INPUT_WIDTH = 640;
const INPUT_HEIGHT = 640;
const SCORE_THRESH = 0.25;
const IOU_THRESH = 0.45;

function iou(a: YoloBox, b: YoloBox): number {
  const x1 = Math.max(a.x1, b.x1);
  const y1 = Math.max(a.y1, b.y1);
  const x2 = Math.min(a.x2, b.x2);
  const y2 = Math.min(a.y2, b.y2);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
  const areaB = (b.x2 - b.x1) * (b.y2 - b.y1);
  const union = areaA + areaB - inter;
  return union <= 0 ? 0 : inter / union;
}

function nms(boxes: YoloBox[], iouThresh: number): YoloBox[] {
  boxes.sort((a, b) => b.score - a.score);
  const selected: YoloBox[] = [];
  for (const box of boxes) {
    let keep = true;
    for (const sel of selected) {
      if (iou(box, sel) > iouThresh) {
        keep = false;
        break;
      }
    }
    if (keep) selected.push(box);
  }
  return selected;
}

export const YoloVideoAnnotator: React.FC = () => {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [session, setSession] = useState<ort.InferenceSession | null>(null);
  const [inputName, setInputName] = useState<string | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [status, setStatus] = useState("Loading model...");

  // Load ONNX model once
  useEffect(() => {
    let cancelled = false;
    async function loadModel() {
      try {
        ort.env.wasm = {
          ...ort.env.wasm,
          numThreads: navigator.hardwareConcurrency || 4,
          simd: true,
        };
        const sess = await ort.InferenceSession.create(MODEL_URL, {
          executionProviders: ["wasm"],
        });
        if (cancelled) return;
        setSession(sess);
        setInputName(sess.inputNames[0]);
        setStatus("Model loaded. Choose a video.");
      } catch (err) {
        console.error("Error loading model", err);
        setStatus("Failed to load model");
      }
    }
    loadModel();
    return () => {
      cancelled = true;
    };
  }, []);

  // Handle video file selection
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file || !videoRef.current) return;
    const url = URL.createObjectURL(file);
    videoRef.current.src = url;
    videoRef.current.onloadedmetadata = () => {
      videoRef.current?.play();
      setStatus("Video loaded. Click Run to start detection.");
    };
  };

  // Main detection loop
  const handleRun = () => {
    if (!session || !inputName || !videoRef.current || !canvasRef.current) return;
    if (isRunning) return;
    setIsRunning(true);
    setStatus("Running detection...");
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const offscreen = document.createElement("canvas");
    offscreen.width = INPUT_WIDTH;
    offscreen.height = INPUT_HEIGHT;
    const offctx = offscreen.getContext("2d");
    let frameCount = 0;
    const loop = async () => {
      if (!isRunning || video.paused || video.ended) {
        setIsRunning(false);
        setStatus("Stopped.");
        return;
      }
      frameCount += 1;
      const PROCESS_EVERY = 2;
      if (frameCount % PROCESS_EVERY !== 0) {
        requestAnimationFrame(loop);
        return;
      }
      try {
        if (!offctx || !ctx) return;
        offctx.drawImage(video, 0, 0, INPUT_WIDTH, INPUT_HEIGHT);
        const imageData = offctx.getImageData(0, 0, INPUT_WIDTH, INPUT_HEIGHT);
        const { data } = imageData;
        const tensorData = new Float32Array(1 * 3 * INPUT_HEIGHT * INPUT_WIDTH);
        const hw = INPUT_HEIGHT * INPUT_WIDTH;
        for (let i = 0; i < hw; i++) {
          const r = data[i * 4] / 255;
          const g = data[i * 4 + 1] / 255;
          const b = data[i * 4 + 2] / 255;
          tensorData[i] = r;
          tensorData[i + hw] = g;
          tensorData[i + 2 * hw] = b;
        }
        const inputTensor = new ort.Tensor("float32", tensorData, [1, 3, INPUT_HEIGHT, INPUT_WIDTH]);
        const feeds: Record<string, ort.Tensor> = {};
        feeds[inputName] = inputTensor;
        const results = await session.run(feeds);
        const outputName = session.outputNames[0];
        const out = results[outputName] as ort.Tensor;
        const outData = out.data as Float32Array;
        const numDet = out.dims[1];
        const numVals = out.dims[2];
        const boxes: YoloBox[] = [];
        for (let i = 0; i < numDet; i++) {
          const offset = i * numVals;
          const x = outData[offset + 0];
          const y = outData[offset + 1];
          const w = outData[offset + 2];
          const h = outData[offset + 3];
          const score = outData[offset + 4];
          if (score < SCORE_THRESH) continue;
          let bestClass = 0;
          let bestScore = 0;
          for (let c = 5; c < numVals; c++) {
            const clsScore = outData[offset + c];
            if (clsScore > bestScore) {
              bestScore = clsScore;
              bestClass = c - 5;
            }
          }
          const finalScore = score * bestScore;
          if (finalScore < SCORE_THRESH) continue;
          const cx = x * video.videoWidth;
          const cy = y * video.videoHeight;
          const bw = w * video.videoWidth;
          const bh = h * video.videoHeight;
          boxes.push({
            x1: cx - bw / 2,
            y1: cy - bh / 2,
            x2: cx + bw / 2,
            y2: cy + bh / 2,
            score: finalScore,
            classId: bestClass,
          });
        }
        const finalBoxes = nms(boxes, IOU_THRESH);
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        ctx.lineWidth = 2;
        ctx.font = "14px sans-serif";
        ctx.strokeStyle = "red";
        ctx.fillStyle = "red";
        for (const b of finalBoxes) {
          ctx.strokeRect(b.x1, b.y1, b.x2 - b.x1, b.y2 - b.y1);
          const label = `${b.classId} ${(b.score * 100).toFixed(1)}%`;
          ctx.fillText(label, b.x1 + 4, b.y1 + 14);
        }
        setStatus(`Running detection... found ${finalBoxes.length} boxes`);
      } catch (err) {
        console.error("Inference error:", err);
        setStatus("Error during detection (see console).");
        setIsRunning(false);
        return;
      }
      requestAnimationFrame(loop);
    };
    requestAnimationFrame(loop);
  };

  const handleStop = () => {
    setIsRunning(false);
    setStatus("Stopped.");
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
      <div>
        <input type="file" accept="video/*" onChange={handleFileChange} />
      </div>
      <div>
        <button onClick={handleRun} disabled={!session || isRunning}>
          Run
        </button>
        <button onClick={handleStop} disabled={!isRunning}>
          Stop
        </button>
      </div>
      <div>{status}</div>
      <div style={{ position: "relative", width: "640px", height: "360px" }}>
        <video
          ref={videoRef}
          style={{ width: "100%", height: "100%", display: "none" }}
          muted
          playsInline
        />
        <canvas
          ref={canvasRef}
          style={{
            width: "100%",
            height: "100%",
            border: "1px solid #444",
          }}
        />
      </div>
    </div>
  );
};
