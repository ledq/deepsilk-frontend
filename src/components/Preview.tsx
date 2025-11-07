import React from "react";

type Props = {
  videoURL: string | null;
  onPick: (f: File) => void;
  videoRef: React.RefObject<HTMLVideoElement | null>;
  overlayRef: React.RefObject<HTMLCanvasElement | null>;
};

export default function Preview({ videoURL, onPick, videoRef, overlayRef }: Props) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
      {/* Main video container - full width */}
      <div className="card panel">
        <div className="label">Video & Overlay</div>
        <div style={{ position: 'relative', marginBottom: '12px' }}>
          <video
            ref={videoRef}
            src={videoURL ?? undefined}
            controls
            className="w-full"
            style={{ width: "100%", borderRadius: "12px", background: "#000", display: "block" }}
          />
          <canvas ref={overlayRef} className="overlay" />
        </div>
        <div style={{ marginTop: 10 }}>
          <label className="input" htmlFor="file" style={{ cursor: "pointer" }}>
            <span>Choose video…</span>
            <input
              id="file"
              type="file"
              accept="video/*"
              style={{ display: "none" }}
              onChange={(e) => {
                const f = e.target.files?.[0];
                if (f) onPick(f);
              }}
            />
          </label>
        </div>
      </div>

      {/* Detection info - compact horizontal layout */}
      <div className="card panel" style={{ padding: '12px 16px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px', fontSize: '12px', color: 'var(--muted)' }}>
          <span style={{ 
            display: 'inline-flex', 
            alignItems: 'center', 
            gap: '6px',
            padding: '4px 8px',
            background: 'rgba(110,231,183,.15)',
            borderRadius: '6px',
            border: '1px solid rgba(110,231,183,.25)'
          }}>
            Real-Time Detection
          </span>
          <span>Live processing as video plays • Full video control maintained</span>
        </div>
      </div>
    </div>
  );
}
