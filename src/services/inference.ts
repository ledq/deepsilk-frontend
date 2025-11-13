// src/services/inference.ts
import * as ort from "onnxruntime-web";
import { env } from "onnxruntime-web";

export type Detection = {
  cls: number;
  name?: string;
  conf: number;
  xyxy: [number, number, number, number];
};

// Singleton session + input metadata
let session: ort.InferenceSession | null = null;
let inputName = "images";
let inputSize = 320; // will be overwritten from metadata if possible

export async function loadSession(modelUrl = "/models/best.onnx") {
  if (session) return;

  console.log(`Attempting to load ONNX model from: ${modelUrl}`);

  try {
    // Match your installed onnxruntime-web version
    env.wasm.wasmPaths =
      "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.23.2/dist/";
    env.wasm.numThreads = 1;
    env.wasm.simd = true;

    console.log("ONNX Runtime Web environment configured");

    // Fetch model bytes explicitly (easier to debug than letting ORT fetch)
    const response = await fetch(modelUrl);
    if (!response.ok) {
      throw new Error(
        `Failed to fetch model: ${response.status} ${response.statusText}`
      );
    }

    const contentType = response.headers.get("content-type");
    console.log(`Model file content-type: ${contentType}`);
    console.log(
      `Model file size: ${response.headers.get("content-length")} bytes`
    );

    const modelArrayBuffer = await response.arrayBuffer();
    const firstFourBytes = new Uint8Array(modelArrayBuffer.slice(0, 4));
    const signature = Array.from(firstFourBytes)
      .map((b) => b.toString(16).padStart(2, "0"))
      .join(" ");
    console.log(`Model file signature (first 4 bytes): ${signature}`);

    // Guard against HTML error pages etc.
    if (firstFourBytes[0] === 0x3c && firstFourBytes[1] === 0x21) {
      throw new Error(
        `Received HTML instead of ONNX model. Check if file exists at ${modelUrl}`
      );
    }
    if (modelArrayBuffer.byteLength < 100) {
      throw new Error(
        `Model file too small (${modelArrayBuffer.byteLength} bytes).`
      );
    }

    console.log("Creating ONNX inference session…");

    const sessionOptions = {
      executionProviders: ["webgl", "wasm"],
      enableMemPattern: false,
      enableCpuMemArena: false,
      enableProfiling: false,
    };

    session = await ort.InferenceSession.create(
      modelArrayBuffer,
      sessionOptions
    );

    // Detect input name & size
    inputName = session.inputNames[0];
    const meta = (session.inputMetadata as Record<string, any>)[inputName];
    const dims = meta?.dimensions ?? [1, 3, 320, 320];

    // dims is usually [1, 3, H, W]
    if (dims.length === 4) {
      inputSize = Number(dims[2] || dims[3] || 320);
    } else {
      inputSize = 320;
    }

    console.log("ONNX model loaded successfully");
    console.log(`Input name: ${inputName}, inputSize: ${inputSize}`);
    console.log(`Model input names: ${session.inputNames.join(", ")}`);
    console.log(`Model output names: ${session.outputNames.join(", ")}`);
  } catch (error) {
    console.error("Failed to load ONNX model:", error);
    // If this fails, we just keep session = null and runOnnx will return []
    session = null;
  }
}

/**
 * Run ONNX inference on one frame.
 * Assumes NMS is baked into the ONNX graph:
 *   each detection = [x1, y1, x2, y2, score, label]
 */
export async function runOnnx(
  frame: HTMLCanvasElement | ImageData,
  opts: { conf: number; iou: number; names?: string[] }
): Promise<Detection[]> {
  if (!session) {
    console.warn("ONNX session not loaded, returning no detections.");
    return [];
  }

  console.log("Running real ONNX inference");

  // Prepare NCHW float32 input [1, 3, H, W]
  const size = inputSize;
  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d");
  if (!ctx) {
    console.error("Could not get 2D context");
    return [];
  }

  canvas.width = size;
  canvas.height = size;

  if (frame instanceof HTMLCanvasElement) {
    // Scale to model size
    ctx.drawImage(frame, 0, 0, size, size);
  } else {
    // If ImageData, just draw as-is (assuming already correct size)
    ctx.putImageData(frame, 0, 0);
  }

  const imageData = ctx.getImageData(0, 0, size, size);

  const data = new Float32Array(1 * 3 * size * size);
  let p = 0,
    c0 = 0,
    c1 = size * size,
    c2 = c1 * 2;
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const r8 = imageData.data[p],
        g8 = imageData.data[p + 1],
        b8 = imageData.data[p + 2];
      data[c0++] = r8 / 255;
      data[c1++] = g8 / 255;
      data[c2++] = b8 / 255;
      p += 4;
    }
  }

  const input = new ort.Tensor("float32", data, [1, 3, size, size]);
  const outputs = await session.run({ [inputName]: input });

  console.log("Output names:", Object.keys(outputs));

  // Find a tensor that looks like [1, numDet, 6] or [numDet, 6]
  let detTensor: ort.TypedTensor<"float32"> | null = null;
  for (const [name, tensor] of Object.entries(outputs)) {
    const t = tensor as ort.TypedTensor<"float32">;
    const dims = t.dims;
    console.log(`Output '${name}': dims=[${dims.join(", ")}]`);

    if (dims.length === 3 && dims[0] === 1 && dims[2] >= 6) {
      detTensor = t;
      break;
    }
    if (dims.length === 2 && dims[1] >= 6) {
      detTensor = t;
      break;
    }
  }

  if (!detTensor) {
    console.warn("No suitable detection output tensor found");
    return [];
  }

  const dims = detTensor.dims;
  const arr = detTensor.data as Float32Array;

  // Normalize dims to [numDet, feat]
  let numDet = 0;
  let feat = 0;

  if (dims.length === 3) {
    // [1, numDet, feat]
    numDet = dims[1];
    feat = dims[2];
  } else {
    // [numDet, feat]
    numDet = dims[0];
    feat = dims[1];
  }

  console.log(`Using detection tensor with numDet=${numDet}, feat=${feat}`);

  const detections: Detection[] = [];
  const confThresh = opts.conf;

  for (let i = 0; i < numDet; i++) {
    const off = i * feat;

    const x1 = arr[off + 0];
    const y1 = arr[off + 1];
    const x2 = arr[off + 2];
    const y2 = arr[off + 3];
    const score = arr[off + 4];
    const clsRaw = arr[off + 5];

    if (score < confThresh) continue;
    if (!isFinite(x1) || !isFinite(y1) || !isFinite(x2) || !isFinite(y2))
      continue;
    if (x2 <= x1 || y2 <= y1) continue;

    const cls = Math.max(0, Math.round(clsRaw));
    const name = opts.names?.[cls];

    detections.push({
      cls,
      name,
      conf: score,
      xyxy: [x1, y1, x2, y2],
    });
  }

  console.log(`Found ${detections.length} detections`);
  return detections;
}
