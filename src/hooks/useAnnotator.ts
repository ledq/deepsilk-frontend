import { useCallback, useRef, useState } from "react";
import { runOnnx, loadSession, type Detection } from "../services/inference";

// Store detections by timestamp for playback synchronization
type TimestampedDetections = {
  [timestamp: number]: Detection[];
};

export function useAnnotator() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const overlayRef = useRef<HTMLCanvasElement | null>(null);
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState(0);
  
  // Store processed detections by timestamp
  const detectionsCache = useRef<TimestampedDetections>({});
  const processingIntervalRef = useRef<number | undefined>(undefined);
  const isProcessingRef = useRef(false);

  const runClient = useCallback(async (fps: number, conf: number, iou: number, names?: string[]) => {
    if (!videoRef.current || !overlayRef.current) return;
    
    await loadSession();
    const video = videoRef.current;
    const overlay = overlayRef.current;
    const ctx = overlay.getContext("2d")!;
    
    // Setup canvas sizing
    const videoRect = video.getBoundingClientRect();
    overlay.width = videoRect.width;
    overlay.height = videoRect.height;
    overlay.style.width = videoRect.width + 'px';
    overlay.style.height = videoRect.height + 'px';

    setRunning(true);
    setProgress(0);
    
    // Clear previous detections
    detectionsCache.current = {};
    
    // Create processing canvas
    const processCanvas = document.createElement("canvas");
    const processCtx = processCanvas.getContext("2d")!;
    
    // Real-time frame capture and processing
    const captureAndProcess = async () => {
      if (video.paused || video.ended || isProcessingRef.current) return;
      
      isProcessingRef.current = true;
      
      try {
        // Use direct scaling without letterboxing - some models expect this
        const modelSize = 640;
        processCanvas.width = modelSize;
        processCanvas.height = modelSize;
        
        // Draw video frame stretched to fill entire canvas (no letterboxing)
        processCtx.drawImage(video, 0, 0, modelSize, modelSize);
        
        // Run inference on letterboxed frame
        const detections = await runOnnx(processCanvas, { conf, iou, names });
        
        // Transform detections from stretched model space to display space
        const scaledDetections = detections.map((det, i) => {
          const originalBox = det.xyxy;
          const scaledBox = [
            Math.max(0, det.xyxy[0] * overlay.width / modelSize),
            Math.max(0, det.xyxy[1] * overlay.height / modelSize),
            Math.max(0, det.xyxy[2] * overlay.width / modelSize),
            Math.max(0, det.xyxy[3] * overlay.height / modelSize)
          ] as [number, number, number, number];
          
          console.log(`Transform detection ${i}: model(${originalBox[0].toFixed(1)}, ${originalBox[1].toFixed(1)}, ${originalBox[2].toFixed(1)}, ${originalBox[3].toFixed(1)}) -> display(${scaledBox[0].toFixed(1)}, ${scaledBox[1].toFixed(1)}, ${scaledBox[2].toFixed(1)}, ${scaledBox[3].toFixed(1)}) [overlay=${overlay.width}x${overlay.height}, model=${modelSize}]`);
          
          return {
            ...det,
            xyxy: scaledBox
          };
        });
        
        // Cache detections with rounded timestamp for consistency
        const timestamp = Math.round(video.currentTime * fps) / fps;
        detectionsCache.current[timestamp] = scaledDetections;
        
        // Draw detections immediately
        const { drawDetections } = await import("../utils/draw");
        drawDetections(ctx, scaledDetections);
        
        // Update progress based on video duration coverage
        const processedDuration = Object.keys(detectionsCache.current).length / fps;
        const progressPercent = Math.min(100, Math.round((processedDuration / video.duration) * 100));
        setProgress(progressPercent);
        
      } catch (error) {
        console.error("Frame processing error:", error);
      } finally {
        isProcessingRef.current = false;
      }
    };
    
    // Process frames at specified FPS
    processingIntervalRef.current = window.setInterval(captureAndProcess, 1000 / fps);
    
    // Handle video seeking - process new frame after seek
    const handleSeeked = () => {
      setTimeout(captureAndProcess, 100); // Small delay for seek completion
    };
    
    // Handle video play/pause
    const handlePlay = () => {
      if (!processingIntervalRef.current) {
        processingIntervalRef.current = window.setInterval(captureAndProcess, 1000 / fps);
      }
    };
    
    const handlePause = () => {
      if (processingIntervalRef.current) {
        clearInterval(processingIntervalRef.current);
        processingIntervalRef.current = undefined;
      }
    };
    
    // Add event listeners
    video.addEventListener('seeked', handleSeeked);
    video.addEventListener('play', handlePlay);
    video.addEventListener('pause', handlePause);
    
    // Setup cleanup function
    const cleanup = () => {
      if (processingIntervalRef.current) {
        clearInterval(processingIntervalRef.current);
        processingIntervalRef.current = undefined;
      }
      video.removeEventListener('seeked', handleSeeked);
      video.removeEventListener('play', handlePlay);
      video.removeEventListener('pause', handlePause);
    };
    
    // Store cleanup for external access
    cleanupRef.current = cleanup;
    
    // Initial frame processing if video is ready
    if (video.readyState >= 2) {
      captureAndProcess();
    }
    
  }, []);

  // Cleanup function reference
  const cleanupRef = useRef<(() => void) | null>(null);
  
  const cleanup = useCallback(() => {
    setRunning(false);
    setProgress(0);
    isProcessingRef.current = false;
    
    if (processingIntervalRef.current) {
      clearInterval(processingIntervalRef.current);
      processingIntervalRef.current = undefined;
    }
    
    if (cleanupRef.current) {
      cleanupRef.current();
      cleanupRef.current = null;
    }
    
    detectionsCache.current = {};
    
    // Clear overlay
    if (overlayRef.current) {
      const ctx = overlayRef.current.getContext("2d");
      if (ctx) {
        ctx.clearRect(0, 0, overlayRef.current.width, overlayRef.current.height);
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
    cleanup 
  };
}
