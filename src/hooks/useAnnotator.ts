import { useCallback, useRef, useState } from "react";
import { runOnnx, loadSession, type Detection } from "../services/inference";

export function useAnnotator() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const overlayRef = useRef<HTMLCanvasElement | null>(null);
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState(0);

  const runClient = useCallback(async (fps:number, conf:number, iou:number, names?:string[])=>{
    if (!videoRef.current || !overlayRef.current) return;
    await loadSession(); // ONNX session once (will gracefully fail if model not found)
    const video = videoRef.current;
    const overlay = overlayRef.current;
    const ctx = overlay.getContext("2d")!;
    
    // Set canvas size to match the displayed video size, not native size
    const videoRect = video.getBoundingClientRect();
    overlay.width = videoRect.width;
    overlay.height = videoRect.height;
    
    // Also set CSS size to match (in case it differs)
    overlay.style.width = videoRect.width + 'px';
    overlay.style.height = videoRect.height + 'px';

    setRunning(true); setProgress(0);
    const dur = video.duration || 1;
    const totalSteps = Math.max(1, Math.floor(dur * fps));
    let frameIdx = 0;
    video.currentTime = 0;
    await video.play();

    const tick = async () => {
      if (video.paused || video.ended) { setRunning(false); return; }
      // draw current frame to a temp canvas for ONNX - use native video size for processing
      const tmp = document.createElement("canvas");
      tmp.width = video.videoWidth || 1280;
      tmp.height = video.videoHeight || 720;
      tmp.getContext("2d")!.drawImage(video,0,0,tmp.width,tmp.height);
      
      // Add a small delay in demo mode to simulate processing
      const dets: Detection[] = await runOnnx(tmp, { conf, iou, names });

      // Scale detection coordinates from native video size to displayed size
      const scaleX = overlay.width / tmp.width;
      const scaleY = overlay.height / tmp.height;
      const scaledDets = dets.map(det => ({
        ...det,
        xyxy: [
          det.xyxy[0] * scaleX,
          det.xyxy[1] * scaleY,
          det.xyxy[2] * scaleX,
          det.xyxy[3] * scaleY
        ] as [number, number, number, number]
      }));

      // draw overlay
      const { drawDetections } = await import("../utils/draw");
      drawDetections(ctx, scaledDets);

      frameIdx += 1;
      setProgress(Math.min(100, Math.round((frameIdx/totalSteps)*100)));
      setTimeout(tick, 1000 / Math.max(1, fps));
    };
    tick();
  }, []);

  return { videoRef, overlayRef, running, progress, runClient, setRunning, setProgress };
}
