import React, { useState } from "react";
import { createPortal } from "react-dom";

type Props = {
  videoURL: string | null;
  onPick: (f: File) => void;
  videoRef: React.RefObject<HTMLVideoElement | null>;
  overlayRef: React.RefObject<HTMLCanvasElement | null>;
};

export default function Preview({ videoURL, onPick, videoRef, overlayRef }: Props) {
  const [showModal, setShowModal] = useState(false);
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '12px', alignItems: 'flex-start', width: '100%' }}>
      {/* Mini preview frame or placeholder */}
      {videoURL ? (
        <div style={{ position: 'relative', width: 180, height: 100, marginBottom: 0, background: '#000', borderRadius: 8, overflow: 'hidden', boxShadow: '0 2px 8px #0002' }}>
          <video
            ref={videoRef}
            src={videoURL}
            style={{ width: '100%', height: '100%', objectFit: 'cover', borderRadius: 8, background: '#000', display: 'block' }}
            muted
            playsInline
          />
          <button
            onClick={() => setShowModal(true)}
            title="Preview full video"
            style={{
              position: 'absolute',
              top: 8,
              right: 8,
              background: 'rgba(0,0,0,0.6)',
              border: 'none',
              borderRadius: '50%',
              width: 32,
              height: 32,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              cursor: 'pointer',
              color: '#fff',
              zIndex: 2
            }}
          >
            <svg width="20" height="20" viewBox="0 0 20 20" fill="none" xmlns="http://www.w3.org/2000/svg">
              <path d="M10 4C5 4 1.73 8.11 1.13 8.93a1 1 0 000 1.14C1.73 11.89 5 16 10 16s8.27-4.11 8.87-4.93a1 1 0 000-1.14C18.27 8.11 15 4 10 4zm0 10c-3.31 0-6.13-2.91-7.19-4C3.87 8.91 6.69 6 10 6s6.13 2.91 7.19 4C16.13 11.09 13.31 14 10 14zm0-7a3 3 0 100 6 3 3 0 000-6zm0 4a1 1 0 110-2 1 1 0 010 2z" fill="currentColor"/>
            </svg>
          </button>
        </div>
      ) : (
        <div style={{ width: 180, height: 100, marginBottom: 0, background: '#222', borderRadius: 8, display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#888', fontSize: 14, boxShadow: '0 2px 8px #0002' }}>
          Please upload a video
        </div>
      )}
      {/* Modal for full preview */}
      {showModal && createPortal(
        <div style={{
          position: 'fixed',
          top: 0,
          left: 0,
          width: '100vw',
          height: '100vh',
          background: 'rgba(0,0,0,0.7)',
          zIndex: 1000,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
        }}
          onClick={() => setShowModal(false)}
        >
          <div style={{ background: '#222', borderRadius: 12, padding: 16, maxWidth: '90vw', maxHeight: '90vh', boxShadow: '0 4px 32px #0008', display: 'flex', flexDirection: 'column', alignItems: 'center', position: 'relative' }} onClick={e => e.stopPropagation()}>
            <button
              onClick={() => setShowModal(false)}
              style={{ position: 'absolute', top: 8, right: 8, background: 'none', border: 'none', color: '#fff', fontSize: 24, cursor: 'pointer', zIndex: 2 }}
              title="Close preview"
            >
              ×
            </button>
            <video
              src={videoURL ?? undefined}
              controls
              autoPlay
              style={{ width: '70vw', maxWidth: 900, maxHeight: '70vh', borderRadius: 8, background: '#000' }}
            />
          </div>
        </div>,
        document.body
      )}
      {/* File input */}
      <div style={{ marginTop: 6 }}>
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
  );
}
