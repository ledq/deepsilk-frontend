import type { Detection } from "../services/inference";

export function drawDetections(
  ctx: CanvasRenderingContext2D, boxes: Detection[]
) {
  ctx.clearRect(0,0,ctx.canvas.width, ctx.canvas.height);
  ctx.lineWidth = 2;
  ctx.font = "12px ui-sans-serif, system-ui";
  for (const d of boxes){
    const [x1,y1,x2,y2] = d.xyxy;
    ctx.strokeStyle = "rgba(110,231,183,1)";
    ctx.fillStyle = "rgba(110,231,183,.24)";
    ctx.fillRect(x1,y1,x2-x1,y2-y1);
    ctx.strokeRect(x1,y1,x2-x1,y2-y1);
    const label = `${d.name ?? d.cls} ${d.conf.toFixed(2)}`;
    const pad = 4, w = ctx.measureText(label).width + pad*2, h=16;
    ctx.fillStyle = "rgba(15,18,26,.9)";
    ctx.fillRect(x1, Math.max(0,y1-h), w, h);
    ctx.fillStyle = "#fff";
    ctx.fillText(label, x1+pad, Math.max(10, y1-4));
  }
}
