// src/hooks/useAnnotator.ts
import { useCallback, useRef, useState } from "react";
import { loadSession, runOnnx, type Detection } from "../services/inference";

type TimestampedDetections = {
  [timestamp: number]: Detection[];
};

export function useAnnotator() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const overlayRef = useRef<HTMLCanvasElement | null>(null);

  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState(0);

  const detectionsCache = useRef<TimestampedDetections>({});
  const processingIntervalRef = useRef<number | undefined>(undefined);
  const isProcessingRef = useRef(false);
  const cleanupRef = useRef<(() => void) | null>(null);

  // Main browser-side detection loop
  const runClient = useCallback(
    async (fps: number, conf: number, iou: number, names?: string[]) => {
      if (!videoRef.current || !overlayRef.current) return;

      // Make sure ONNX session is ready
      await loadSession();

      const video = videoRef.current;
      const overlay = overlayRef.current;
      const ctx = overlay.getContext("2d");
      if (!ctx) return;

      // Match overlay canvas size to rendered video size
      const rect = video.getBoundingClientRect();
      overlay.width = rect.width;
      overlay.height = rect.height;
      overlay.style.width = `${rect.width}px`;
      overlay.style.height = `${rect.height}px`;

      setRunning(true);
      setProgress(0);
      detectionsCache.current = {};

      // Offscreen canvas for model input
      const MODEL_SIZE = 320; // must match imgsz used in export
      const processCanvas = document.createElement("canvas");
      processCanvas.width = MODEL_SIZE;
      processCanvas.height = MODEL_SIZE;

      const processCtx = processCanvas.getContext("2d");
      if (!processCtx) return;

      const captureAndProcess = async () => {
        if (!video || video.paused || video.ended) return;
        if (isProcessingRef.current) return;

        isProcessingRef.current = true;
        try {
          // Draw current video frame into model resolution
          processCtx.drawImage(video, 0, 0, MODEL_SIZE, MODEL_SIZE);

          // Run ONNX inference - returns boxes in model space
          const detections = await runOnnx(processCanvas, {
            conf,
            iou,
            names,
          });

          // Scale boxes from [0, MODEL_SIZE] to overlay size
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

          // Cache by rounded timestamp for progress tracking
          const timestamp = Math.round(video.currentTime * fps) / fps;
          detectionsCache.current[timestamp] = scaled;

          // Clear overlay and draw current frame detections
          ctx.clearRect(0, 0, overlay.width, overlay.height);
          const { drawDetections } = await import("../utils/draw");
          drawDetections(ctx, scaled);

          // Simple progress estimate: processed timestamps / duration
          const processedDuration =
            Object.keys(detectionsCache.current).length / fps;
          if (video.duration) {
            const percent = Math.min(
              100,
              Math.round((processedDuration / video.duration) * 100)
            );
            setProgress(percent);
          }
        } catch (err) {
          console.error("Frame processing error:", err);
        } finally {
          isProcessingRef.current = false;
        }
      };

      // Clear any old interval before starting a new run
      if (processingIntervalRef.current !== undefined) {
        window.clearInterval(processingIntervalRef.current);
      }
      processingIntervalRef.current = window.setInterval(
        captureAndProcess,
        1000 / fps
      );

      const handleSeeked = () => {
        // After seeking, process one fresh frame
        setTimeout(() => {
          if (!video.paused && !video.ended) {
            captureAndProcess();
          }
        }, 100);
      };

      const handlePlay = () => {
        if (processingIntervalRef.current === undefined) {
          processingIntervalRef.current = window.setInterval(
            captureAndProcess,
            1000 / fps
          );
        }
      };

      const handlePause = () => {
        if (processingIntervalRef.current !== undefined) {
          window.clearInterval(processingIntervalRef.current);
          processingIntervalRef.current = undefined;
        }
      };

      video.addEventListener("seeked", handleSeeked);
      video.addEventListener("play", handlePlay);
      video.addEventListener("pause", handlePause);

      // Store how to undo all of this
      cleanupRef.current = () => {
        if (processingIntervalRef.current !== undefined) {
          window.clearInterval(processingIntervalRef.current);
          processingIntervalRef.current = undefined;
        }
        video.removeEventListener("seeked", handleSeeked);
        video.removeEventListener("play", handlePlay);
        video.removeEventListener("pause", handlePause);
      };

      // Kick off first frame if video is already loaded
      if (video.readyState >= 2) {
        captureAndProcess();
      }
    },
    []
  );

  // Cleanup you can call from App (onClear, unmount, etc.)
  const cleanup = useCallback(() => {
    setRunning(false);
    setProgress(0);
    isProcessingRef.current = false;

    if (processingIntervalRef.current !== undefined) {
      window.clearInterval(processingIntervalRef.current);
      processingIntervalRef.current = undefined;
    }

    if (cleanupRef.current) {
      cleanupRef.current();
      cleanupRef.current = null;
    }

    detectionsCache.current = {};

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
