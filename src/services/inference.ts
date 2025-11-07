import * as ort from "onnxruntime-web";
import { nms } from "../utils/nms";

// Detections
export type Detection = {
  cls: number;
  name?: string;
  conf: number;
  xyxy: [number, number, number, number];
};

// Session singleton
let session: ort.InferenceSession | null = null;
let inputName = "images";
let inputSize = 640;

export async function loadSession(modelUrl = "/models/yolo.onnx") {
  if (session) return;
  
  try {
    const providers = ((navigator as any).gpu ? ["webgpu"] : ["wasm"]) as any;
    session = await ort.InferenceSession.create(modelUrl, { executionProviders: providers });
    // Try to detect input tensor name & size
    const input = session.inputNames[0];
    inputName = input;
    const meta = session.inputMetadata[inputName];
    const shape = meta?.dimensions ?? [1,3,640,640];
    inputSize = Number(shape[2] || 640);
    console.log("ONNX model loaded successfully");
  } catch (error) {
    console.warn("Failed to load ONNX model:", error);
    console.log("Running in demo mode - will generate mock detections");
    session = null; // Explicitly set to null to indicate demo mode
  }
}

// Letterbox to square with padding info
function letterbox(
  src: HTMLCanvasElement | ImageData, size: number
): { data: Float32Array; ratio: number; dx: number; dy: number; w: number; h: number } {
  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d")!;
  const w = (src as HTMLCanvasElement).width ?? (src as ImageData).width;
  const h = (src as HTMLCanvasElement).height ?? (src as ImageData).height;
  const r = Math.min(size / w, size / h);
  const nw = Math.round(w * r), nh = Math.round(h * r);
  const dx = Math.floor((size - nw) / 2), dy = Math.floor((size - nh) / 2);

  canvas.width = size; canvas.height = size;
  ctx.fillStyle = "black"; ctx.fillRect(0,0,size,size);
  if (src instanceof HTMLCanvasElement) ctx.drawImage(src, 0,0,w,h, dx,dy,nw,nh);
  else ctx.putImageData(src, dx,dy);

  const img = ctx.getImageData(0,0,size,size).data;
  // NHWC uint8 -> NCHW float32(0..1)
  const data = new Float32Array(1*3*size*size);
  let p = 0, c0 = 0, c1 = size*size, c2 = c1*2;
  for (let y=0; y<size; y++){
    for (let x=0; x<size; x++){
      const r8 = img[p], g8 = img[p+1], b8 = img[p+2];
      data[c0++] = r8/255; data[c1++] = g8/255; data[c2++] = b8/255;
      p += 4;
    }
  }
  return { data, ratio: r, dx, dy, w, h };
}

function scaleBack(xyxy: [number,number,number,number], ratio:number, dx:number, dy:number) {
  // undo letterbox: from model-space back to original frame coords
  const [x1,y1,x2,y2]=xyxy;
  return [
    (x1 - dx) / ratio,
    (y1 - dy) / ratio,
    (x2 - dx) / ratio,
    (y2 - dy) / ratio
  ] as [number,number,number,number];
}

// Parse YOLOv5/8-like output: [1, N, 85] = [x,y,w,h,obj,80 classes...]
function parseOutput(
  out: ort.TypedTensor, confThresh: number, iouThresh: number,
  ratio: number, dx: number, dy: number, names?: string[]
): Detection[] {
  const arr = out.data as Float32Array;
  const dims = out.dims; // [1,N,85]
  const N = dims[dims.length-2];
  const C = dims[dims.length-1];

  const boxes: [number,number,number,number][] = [];
  const scores: number[] = [];
  const classes: number[] = [];

  for (let i=0;i<N;i++){
    const off = i*C;
    const x = arr[off+0], y = arr[off+1], w = arr[off+2], h = arr[off+3];
    const obj = arr[off+4];
    if (obj <= 0) continue;

    // best class
    let bestC = -1, bestP = 0;
    for (let c=5;c<C;c++){
      const p = arr[off+c];
      if (p>bestP){ bestP = p; bestC = c-5; }
    }
    const score = obj * bestP;
    if (score < confThresh) continue;

    // xywh -> xyxy in model-space
    const x1 = x - w/2, y1 = y - h/2, x2 = x + w/2, y2 = y + h/2;
    boxes.push([x1,y1,x2,y2]);
    scores.push(score);
    classes.push(bestC);
  }

  // NMS in model space
  const keep = nms(boxes, scores, classes, iouThresh);
  const outDet: Detection[] = [];
  for (const k of keep){
    const xyxyScaled = scaleBack(boxes[k], ratio, dx, dy);
    outDet.push({
      cls: classes[k],
      name: names?.[classes[k]],
      conf: scores[k],
      xyxy: [
        Math.max(0, xyxyScaled[0]),
        Math.max(0, xyxyScaled[1]),
        Math.max(0, xyxyScaled[2]),
        Math.max(0, xyxyScaled[3]),
      ]
    });
  }
  return outDet;
}

/** Generate mock detections for demo mode */
function generateMockDetections(
  frame: HTMLCanvasElement | ImageData,
  opts: { conf: number; iou: number; names?: string[] }
): Detection[] {
  const w = (frame as HTMLCanvasElement).width ?? (frame as ImageData).width;
  const h = (frame as HTMLCanvasElement).height ?? (frame as ImageData).height;
  
  const mockDetections: Detection[] = [];
  const numDetections = Math.floor(Math.random() * 3) + 1; // 1-3 detections
  
  for (let i = 0; i < numDetections; i++) {
    const x1 = Math.random() * (w * 0.6);
    const y1 = Math.random() * (h * 0.6);
    const x2 = x1 + Math.random() * (w * 0.3) + 50;
    const y2 = y1 + Math.random() * (h * 0.3) + 50;
    
    const mockClasses = opts.names || ['person', 'car', 'dog', 'cat', 'bird'];
    const cls = Math.floor(Math.random() * mockClasses.length);
    
    mockDetections.push({
      cls,
      name: mockClasses[cls] || `class_${cls}`,
      conf: opts.conf + Math.random() * (1 - opts.conf), // Random conf above threshold
      xyxy: [
        Math.max(0, x1),
        Math.max(0, y1),
        Math.min(w, x2),
        Math.min(h, y2)
      ]
    });
  }
  
  return mockDetections;
}

/** Run ONNX inference on a frame canvas/imageData */
export async function runOnnx(
  frame: HTMLCanvasElement | ImageData,
  opts: { conf: number; iou: number; names?: string[] }
): Promise<Detection[]> {
  // If no session (demo mode), return mock detections
  if (!session) {
    console.log("Running in demo mode - generating mock detections");
    return generateMockDetections(frame, opts);
  }
  
  const { data, ratio, dx, dy } = letterbox(frame, inputSize);
  const input = new ort.Tensor("float32", data, [1,3,inputSize,inputSize]);
  const outputs = await session!.run({ [inputName]: input });
  // assume single output
  const outName = Object.keys(outputs)[0];
  const out = outputs[outName] as ort.TypedTensor;
  return parseOutput(out, opts.conf, opts.iou, ratio, dx, dy, opts.names);
}
