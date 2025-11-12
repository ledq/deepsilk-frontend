type Props = {
  detectionMode: 'browser' | 'server';
  setDetectionMode: (mode: 'browser' | 'server') => void;
  onRunDetection: () => void;
  onClear: ()=>void;
  disabled?: boolean;
  videoUploaded?: boolean;
};
export default function Controls(p: Props){
  return (
    <div className="row" style={{alignItems:"center", gap: 16}}>
      {/* Detection mode toggle */}
      <div className="input" style={{marginRight: 16}}>
        <span>Mode:</span>
        <label style={{marginLeft: 8}}>
          <input
            type="radio"
            name="detectionMode"
            value="browser"
            checked={p.detectionMode === 'browser'}
            onChange={() => p.setDetectionMode('browser')}
            // disabled={!!p.videoUploaded}
          />
          Browser
        </label>
        <label style={{marginLeft: 8}}>
          <input
            type="radio"
            name="detectionMode"
            value="server"
            checked={p.detectionMode === 'server'}
            onChange={() => p.setDetectionMode('server')}
            // disabled={!!p.videoUploaded}
          />
          Server
        </label>
      </div>


      <button className="button primary" onClick={p.onRunDetection} disabled={p.disabled}>▶ Run detection</button>
      <button className="button danger" onClick={p.onClear} disabled={p.disabled}>✕ Clear</button>
    </div>
  );
}
