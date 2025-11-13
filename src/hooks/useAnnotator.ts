// src/hooks/useAnnotator.ts
import { useCallback, useRef, useState } from "react";
import { loadSession, runOnnx, type Detection } from "../services/inference";
import { drawDetections } from "../utils/draw";

export function useAnnotator() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const overlayRef = useRef<HTMLCanvasElement | null>(null);

  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState(0);

  const rafIdRef = useRef<number | null>(null);
  const isProcessingRef = useRef(false);
  const isRunningRef = useRef(false);

  // Keep last non-empty detections so boxes stay visible
  const lastDetectionsRef = useRef<Detection[]>([]);

  const runClient = useCallback(
    async (fps: number, conf: number, iou: number, names?: string[]) => {
      const video = videoRef.current;
      const overlay = overlayRef.current;
      if (!video || !overlay) return;

      // Stop any previous run
      if (rafIdRef.current !== null) {
        window.cancelAnimationFrame(rafIdRef.current);
        rafIdRef.current = null;
      }
      isRunningRef.current = false;
      isProcessingRef.current = false;
      lastDetectionsRef.current = [];

      // Ensure ONNX session loaded
      await loadSession();

      const ctx = overlay.getContext("2d");
      if (!ctx) return;

      // Match overlay size to the rendered video box
      const rect = video.getBoundingClientRect();
      overlay.width = rect.width;
      overlay.height = rect.height;
      overlay.style.width = `${rect.width}px`;
      overlay.style.height = `${rect.height}px`;

      setRunning(true);
      setProgress(0);
      isRunningRef.current = true;

      // Offscreen canvas for model input
      const MODEL_SIZE = 320; // must match imgsz used in ONNX export
      const processCanvas = document.createElement("canvas");
      processCanvas.width = MODEL_SIZE;
      processCanvas.height = MODEL_SIZE;
      const processCtx = processCanvas.getContext("2d");
      if (!processCtx) return;

      // Prepare video playback
      try {
        video.currentTime = 0;
      } catch {
        // ignore seek errors
      }
      video.muted = true;
      video.playsInline = true;

      try {
        await video.play();
      } catch (err) {
        console.warn(
          "Video autoplay failed, user interaction may be required:",
          err
        );
      }

      const processFrame = async () => {
        if (!video || video.ended || !isRunningRef.current) {
          return;
        }

        // Only process if video is playing and we are not already in a call
        if (video.paused || isProcessingRef.current) {
          return;
        }
        isProcessingRef.current = true;

        try {
          // Draw current frame into model-sized canvas
          processCtx.drawImage(video, 0, 0, MODEL_SIZE, MODEL_SIZE);

          // Run ONNX inference -> model-space boxes
          const detections = await runOnnx(processCanvas, {
            conf,
            iou,
            names,
          });

          // Scale boxes to overlay size
          const scaled = detections.map((det) => {
            const [x1, y1, x2, y2] = det.xyxy;
            const sx1 = (x1 / MODEL_SIZE) * overlay.width;
            const sy1 = (y1 / MODEL_SIZE) * overlay.height;
            const sx2 = (x2 / MODEL_SIZE) * overlay.width;
            const sy2 = (y2 / MODEL_SIZE) * overlay.height;
            return {
              ...det,
              xyxy: [sx1, sy1, sx2, sy2] as [number, number, number, number],
            };
          });

          // Only overwrite last detections if we have any
          if (scaled.length > 0) {
            lastDetectionsRef.current = scaled;
          }

          // Clear overlay and draw the last known detections
          ctx.clearRect(0, 0, overlay.width, overlay.height);
          if (lastDetectionsRef.current.length > 0) {
            drawDetections(ctx, lastDetectionsRef.current);
          }

          // Simple progress: based on video time
          if (video.duration) {
            const percent = Math.min(
              100,
              Math.round((video.currentTime / video.duration) * 100)
            );
            setProgress(percent);
          }
        } catch (error) {
          console.error("Frame processing error:", error);
        } finally {
          isProcessingRef.current = false;
        }
      };

      const loop = async () => {
        if (!isRunningRef.current || !video) {
          return;
        }

        if (video.ended) {
          isRunningRef.current = false;
          setRunning(false);
          return;
        }

        await processFrame();
        if (isRunningRef.current) {
          rafIdRef.current = window.requestAnimationFrame(loop);
        }
      };

      // Start animation loop
      rafIdRef.current = window.requestAnimationFrame(loop);
    },
    []
  );

  const cleanup = useCallback(() => {
    isRunningRef.current = false;
    isProcessingRef.current = false;

    if (rafIdRef.current !== null) {
      window.cancelAnimationFrame(rafIdRef.current);
      rafIdRef.current = null;
    }

    setRunning(false);
    setProgress(0);
    lastDetectionsRef.current = [];

    if (overlayRef.current) {
      const ctx = overlayRef.current.getContext("2d");
      if (ctx) {
        ctx.clearRect(
          0,
          0,
          overlayRef.current.width,
          overlayRef.current.height
        );
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
