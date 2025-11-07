// Simple NMS for YOLO-style [x1,y1,x2,y2, score, class]
export type BBox = [number, number, number, number];
export function iou(a: BBox, b: BBox): number {
  const x1 = Math.max(a[0], b[0]);
  const y1 = Math.max(a[1], b[1]);
  const x2 = Math.min(a[2], b[2]);
  const y2 = Math.min(a[3], b[3]);
  const w = Math.max(0, x2 - x1), h = Math.max(0, y2 - y1);
  const inter = w * h;
  const areaA = (a[2]-a[0])*(a[3]-a[1]);
  const areaB = (b[2]-b[0])*(b[3]-b[1]);
  const union = areaA + areaB - inter + 1e-6;
  return inter / union;
}

export function nms(
  boxes: BBox[], scores: number[], classes: number[], iouThresh: number
) {
  const idxs = scores.map((s,i)=>i).sort((a,b)=>scores[b]-scores[a]);
  const kept: number[] = [];
  while (idxs.length) {
    const i = idxs.shift()!;
    kept.push(i);
    const rest: number[] = [];
    for (const j of idxs) {
      if (classes[i] !== classes[j]) { rest.push(j); continue; }
      if (iou(boxes[i], boxes[j]) <= iouThresh) rest.push(j);
    }
    idxs.splice(0, idxs.length, ...rest);
  }
  return kept;
}
