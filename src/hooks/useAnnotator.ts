// src/hooks/useAnnotator.ts
import { useCallback, useRef, useState } from "react";
import { loadSession, runOnnx, type Detection } from "../services/inference";
import { drawDetections } from "../utils/draw";
import { SILKSONG_CLASS_NAMES } from "../data/classNames";

export function useAnnotator() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const overlayRef = useRef<HTMLCanvasElement | null>(null);

  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState(0);

  const isRunningRef = useRef(false);
  const isProcessingRef = useRef(false);

  // Keep last detections for drawing (optional, but handy if you want to reuse)
  const lastDetectionsRef = useRef<Detection[]>([]);

  // We need to remember the seeked handler to detach it in cleanup
  const seekHandlerRef = useRef<((ev: Event) => void) | null>(null);

  const runClient = useCallback(
    async (fps: number, conf: number, iou: number, names?: string[]) => {
      const video = videoRef.current;
      const overlay = overlayRef.current;
      if (!video || !overlay) return;

      // Clean up any previous run
      isRunningRef.current = false;
      isProcessingRef.current = false;
      lastDetectionsRef.current = [];
      setProgress(0);

      // Remove old seek handler if any
      if (seekHandlerRef.current) {
        video.removeEventListener("seeked", seekHandlerRef.current);
        seekHandlerRef.current = null;
      }

      // Load ONNX session
      await loadSession();

      const ctx = overlay.getContext("2d");
      if (!ctx) return;

      // Match overlay size to rendered video
      const rect = video.getBoundingClientRect();
      overlay.width = rect.width;
      overlay.height = rect.height;
      overlay.style.width = `${rect.width}px`;
      overlay.style.height = `${rect.height}px`;

      const MODEL_SIZE = 320; // must match your ONNX export imgsz
      const processCanvas = document.createElement("canvas");
      processCanvas.width = MODEL_SIZE;
      processCanvas.height = MODEL_SIZE;
      const processCtx = processCanvas.getContext("2d");
      if (!processCtx) return;

      isRunningRef.current = true;
      setRunning(true);
      lastDetectionsRef.current = [];

      // Core routine: run detection on the current frame
      const processCurrentFrame = async () => {
        const v = videoRef.current;
        const o = overlayRef.current;
        if (!v || !o) return;
        if (!isRunningRef.current) return;
        if (v.duration === 0) return;

        // Stop when we reach the end
        if (v.currentTime >= v.duration) {
          isRunningRef.current = false;
          setRunning(false);
          return;
        }

        // Don't start another inference if one is still going
        if (isProcessingRef.current) return;
        isProcessingRef.current = true;

        try {
          // Draw current video frame into model-sized canvas
          processCtx.drawImage(v, 0, 0, MODEL_SIZE, MODEL_SIZE);

          // Run ONNX inference
          const detections = await runOnnx(processCanvas, {
            conf,
            iou,
            names: SILKSONG_CLASS_NAMES,
          });

          // Scale boxes from model space to overlay space
          const scaled = detections.map((det) => {
            const [x1, y1, x2, y2] = det.xyxy;
            const sx1 = (x1 / MODEL_SIZE) * o.width;
            const sy1 = (y1 / MODEL_SIZE) * o.height;
            const sx2 = (x2 / MODEL_SIZE) * o.width;
            const sy2 = (y2 / MODEL_SIZE) * o.height;
            return {
              ...det,
              xyxy: [sx1, sy1, sx2, sy2] as [number, number, number, number],
            };
          });

          lastDetectionsRef.current = scaled;

          // Draw detections for *this* frame
          ctx.clearRect(0, 0, o.width, o.height);
          if (scaled.length > 0) {
            drawDetections(ctx, scaled);
          }

          // Update progress based on current time
          const pct =
            v.duration > 0
              ? Math.min(100, Math.round((v.currentTime / v.duration) * 100))
              : 0;
          setProgress(pct);

          // Advance to next frame time
          const frameStep = 1 / fps; // seconds per "step"
          const nextTime = v.currentTime + frameStep;
          if (nextTime < v.duration && isRunningRef.current) {
            v.currentTime = nextTime; // triggers "seeked" → process next frame
          } else {
            // End
            isRunningRef.current = false;
            setRunning(false);
          }
        } catch (err) {
          console.error("Frame processing error:", err);
          isRunningRef.current = false;
          setRunning(false);
        } finally {
          isProcessingRef.current = false;
        }
      };

      // When the video has finished seeking to a new time, process that frame
      const handleSeeked = () => {
        if (!isRunningRef.current) return;
        // Process the frame at the newly-seeked currentTime
        void processCurrentFrame();
      };

      seekHandlerRef.current = handleSeeked;
      video.addEventListener("seeked", handleSeeked);

      // Start from the beginning
      try {
        video.pause();
        video.currentTime = 0;
      } catch {
        // ignore
      }

      // If metadata is loaded, process immediately once;
      // otherwise wait for loadeddata and then process.
      if (video.readyState >= 2) {
        // We have enough data to grab a frame
        void processCurrentFrame();
      } else {
        const handleLoadedData = () => {
          video.removeEventListener("loadeddata", handleLoadedData);
          if (!isRunningRef.current) return;
          void processCurrentFrame();
        };
        video.addEventListener("loadeddata", handleLoadedData);
      }
    },
    []
  );

  const cleanup = useCallback(() => {
    isRunningRef.current = false;
    isProcessingRef.current = false;

    const video = videoRef.current;
    if (video && seekHandlerRef.current) {
      video.removeEventListener("seeked", seekHandlerRef.current);
      seekHandlerRef.current = null;
    }

    setRunning(false);
    setProgress(0);
    lastDetectionsRef.current = [];

    const overlay = overlayRef.current;
    if (overlay) {
      const ctx = overlay.getContext("2d");
      if (ctx) {
        ctx.clearRect(0, 0, overlay.width, overlay.height);
      }
    }
  }, []);

  return {
    videoRef,
    overlayRef,
    running,
    progress,
    runClient,
    setRunning,
    setProgress,
    cleanup,
  };
}
