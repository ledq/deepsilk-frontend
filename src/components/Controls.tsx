type Props = {
  fps: number; setFps: (v:number)=>void;
  conf: number; setConf: (v:number)=>void;
  iou: number;  setIou:  (v:number)=>void;
  onLocal: ()=>void; onServer: ()=>void; onClear: ()=>void;
  disabled?: boolean;
};
export default function Controls(p: Props){
  return (
    <div className="row" style={{alignItems:"center"}}>
      <div className="input">
        <span>FPS</span>
        <input type="number" min={2} max={30} value={p.fps}
          onChange={(e)=>p.setFps(Number(e.target.value))}
          style={{width:64, background:"transparent", border:"none", color:"var(--text)"}}/>
      </div>
      <div className="input" title="Confidence threshold">
        <span>Conf</span>
        <input type="number" min={0.05} max={0.95} step={0.05} value={p.conf}
          onChange={(e)=>p.setConf(Number(e.target.value))}
          style={{width:74, background:"transparent", border:"none", color:"var(--text)"}}/>
      </div>
      <div className="input" title="IoU for NMS">
        <span>IoU</span>
        <input type="number" min={0.10} max={0.90} step={0.05} value={p.iou}
          onChange={(e)=>p.setIou(Number(e.target.value))}
          style={{width:74, background:"transparent", border:"none", color:"var(--text)"}}/>
      </div>

      <button className="button primary" onClick={p.onLocal} disabled={p.disabled}>▶ Run in browser</button>
      <button className="button alt" onClick={p.onServer} disabled={p.disabled}>☁ Run on server</button>
      <button className="button danger" onClick={p.onClear} disabled={p.disabled}>✕ Clear</button>
    </div>
  );
}
