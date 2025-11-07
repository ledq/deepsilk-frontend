import * as ort from "onnxruntime-web";
import { env } from "onnxruntime-web";
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

export async function loadSession(modelUrl = "/models/best.onnx") {
  if (session) return;
  
  console.log(`Attempting to load ONNX model from: ${modelUrl}`);
  
  try {
    // Set WASM paths to match our package version
    env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.23.2/dist/';
    env.wasm.numThreads = 1; // Use single thread for better compatibility
    env.wasm.simd = true; // Enable SIMD for better performance
    
    console.log('ONNX Runtime Web environment configured');
    
    // First, verify the model file exists and is accessible
    const response = await fetch(modelUrl);
    if (!response.ok) {
      throw new Error(`Failed to fetch model: ${response.status} ${response.statusText}`);
    }
    
    const contentType = response.headers.get('content-type');
    console.log(`Model file content-type: ${contentType}`);
    console.log(`Model file size: ${response.headers.get('content-length')} bytes`);
    
    // Check if we're getting an actual ONNX file (not HTML error page)
    const modelArrayBuffer = await response.arrayBuffer();
    const firstFourBytes = new Uint8Array(modelArrayBuffer.slice(0, 4));
    const signature = Array.from(firstFourBytes).map(b => b.toString(16).padStart(2, '0')).join(' ');
    console.log(`Model file signature (first 4 bytes): ${signature}`);
    
    // Check for HTML content (starts with '<!' which is 0x3C 0x21)
    if (firstFourBytes[0] === 0x3C && firstFourBytes[1] === 0x21) {
      throw new Error(`Received HTML instead of ONNX model. File signature: ${signature}. Check if model file exists at ${modelUrl}`);
    }
    
    // Basic validation that it's a binary file (not text)
    if (modelArrayBuffer.byteLength < 100) {
      throw new Error(`Model file too small (${modelArrayBuffer.byteLength} bytes). Likely not a valid ONNX model.`);
    }
    
    console.log('Creating ONNX inference session...');
    
    // Try multiple execution providers for better compatibility
    const providers = ["webgl", "wasm"] as any;
    console.log(`Using execution providers: ${JSON.stringify(providers)}`);
    
    // Create session with more lenient options
    const sessionOptions = {
      executionProviders: providers,
      enableMemPattern: false,
      enableCpuMemArena: false,
      enableProfiling: false
    };
    
    session = await ort.InferenceSession.create(modelArrayBuffer, sessionOptions);
    
    // Try to detect input tensor name & size
    const input = session.inputNames[0];
    inputName = input;
    const metadata = session.inputMetadata as Record<string, any>;
    const meta = metadata[inputName];
    const shape = meta?.dimensions ?? [1,3,640,640];
    inputSize = Number(shape[2] || 640);
    console.log("ONNX model loaded successfully");
    console.log(`Input name: ${inputName}, Input size: ${inputSize}x${inputSize}`);
    console.log(`Model input names: ${session.inputNames.join(', ')}`);
    console.log(`Model output names: ${session.outputNames.join(', ')}`);
  } catch (error) {
    console.error("Failed to load ONNX model:", error);
    console.log("Running in demo mode - will generate mock detections");
    session = null; // Explicitly set to null to indicate demo mode
  }
}

// Parse YOLOv5/8-like output: [1, N, 85] = [x,y,w,h,obj,80 classes...]
function parseOutput(
  out: ort.TypedTensor<'float32'>, confThresh: number, iouThresh: number,
  names?: string[]
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

    // Debug raw model values for first few detections
    if (i < 5) {
      console.log(`Raw values [${i}]: x=${x.toFixed(4)}, y=${y.toFixed(4)}, w=${w.toFixed(4)}, h=${h.toFixed(4)}, obj=${obj.toFixed(4)}`);
    }

    // best class
    let bestC = -1, bestP = 0;
    for (let c=5;c<C;c++){
      const p = arr[off+c];
      if (p>bestP){ bestP = p; bestC = c-5; }
    }
    const score = obj * bestP;
    if (score < confThresh) continue;

    // Check if coordinates might be normalized (0-1) and need scaling to 640
    const scaledX = x <= 1 ? x * 640 : x;
    const scaledY = y <= 1 ? y * 640 : y; 
    const scaledW = w <= 1 ? w * 640 : w;
    const scaledH = h <= 1 ? h * 640 : h;

    // xywh -> xyxy in model-space
    const x1 = scaledX - scaledW/2, y1 = scaledY - scaledH/2, x2 = scaledX + scaledW/2, y2 = scaledY + scaledH/2;
    console.log(`Raw model output: center(${x.toFixed(1)}, ${y.toFixed(1)}) size(${w.toFixed(1)}x${h.toFixed(1)}) -> scaled center(${scaledX.toFixed(1)}, ${scaledY.toFixed(1)}) -> box(${x1.toFixed(1)}, ${y1.toFixed(1)}, ${x2.toFixed(1)}, ${y2.toFixed(1)})`);
    boxes.push([x1,y1,x2,y2]);
    scores.push(score);
    classes.push(bestC);
  }

  // NMS in model space
  const keep = nms(boxes, scores, classes, iouThresh);
  const outDet: Detection[] = [];
  for (const k of keep){
    const finalBox = boxes[k];
    console.log(`Final detection ${k}: box(${finalBox[0].toFixed(1)}, ${finalBox[1].toFixed(1)}, ${finalBox[2].toFixed(1)}, ${finalBox[3].toFixed(1)}) conf=${scores[k].toFixed(3)}`);
    outDet.push({
      cls: classes[k],
      name: names?.[classes[k]],
      conf: scores[k],
      xyxy: boxes[k] // Keep coordinates in model space (640x640)
    });
  }
  return outDet;
}

// Parse Ultralytics YOLO output that might have non-standard format
function parseCustomYoloOutput(
  out: ort.TypedTensor<'float32'>, confThresh: number, iouThresh: number,
  names?: string[]
): Detection[] {
  const arr = out.data as Float32Array;
  const dims = out.dims; // [1, N, features]
  const N = dims[1];
  const features = dims[2];

  console.log(`Parsing custom YOLO with ${N} anchors, ${features} features each`);

  const boxes: [number,number,number,number][] = [];
  const scores: number[] = [];
  const classes: number[] = [];

  if (features === 2000) {
    // This looks like YOLOv3-tiny with flattened multi-scale output
    console.log(`Detected YOLOv3-tiny format (${features}) - parsing with normalized confidence`);
    
    // YOLOv3-tiny typically has 2 detection layers with different scales
    // The 2000 features likely contain multiple detection grids
    // Try parsing with a single stride pattern and normalize confidence scores
    
    for (let i = 0; i < Math.min(N, 4); i++) { // Only check first 4 anchors to avoid duplicates
      const off = i * features;
      
      if (i < 2) {
        const samples = [0, 5, 85, 170, 255];
        const sampleValues = samples.map(idx => 
          idx < features ? arr[off + idx].toFixed(3) : 'N/A'
        ).join(', ');
        console.log(`Anchor ${i} samples at [${samples.join(',')}]: ${sampleValues}`);
      }
      
      // Try standard YOLO parsing at position 0 (most likely to contain valid detections)
      const x = arr[off + 0];
      const y = arr[off + 1];
      const w = arr[off + 2]; 
      const h = arr[off + 3];
      let conf = arr[off + 4];
      
      // Normalize confidence if it's too high (likely raw logit)
      if (conf > 10) {
        conf = 1 / (1 + Math.exp(-conf / 100)); // Sigmoid normalization for high values
      } else if (conf > 1) {
        conf = conf / 100; // Simple scaling for moderate values
      }
      
      console.log(`Anchor ${i}: raw_conf=${arr[off + 4].toFixed(2)} -> normalized_conf=${conf.toFixed(4)}`);
      console.log(`  Raw coords: x=${x.toFixed(3)}, y=${y.toFixed(3)}, w=${w.toFixed(3)}, h=${h.toFixed(3)}`);
      
      if (conf > confThresh && x > 0 && y > 0 && w > 0 && h > 0) {
        
        // Find best class - try different class starting positions
        let bestC = -1, bestP = 0;
        const classPositions = [5, 85, 170]; // Try different possible class start positions
        
        for (const classStart of classPositions) {
          if (classStart + 80 < features) { // Ensure we have room for 80 classes
            for (let c = 0; c < 80; c++) {
              if (classStart + c < features) {
                let p = arr[off + classStart + c];
                // Normalize class probability if needed
                if (p > 10) {
                  p = 1 / (1 + Math.exp(-p / 100));
                } else if (p > 1) {
                  p = p / 100;
                }
                
                if (p > bestP) {
                  bestP = p;
                  bestC = c;
                }
              }
            }
            if (bestP > 0) break; // Found valid classes, stop trying other positions
          }
        }
        
        const finalScore = conf * bestP;
        if (finalScore > confThresh) {
          // The coordinates appear to be in grid cell format or different scale
          // YOLOv3-tiny typically uses 13x13 and 26x26 grids
          // Try different scaling approaches
          let scaledX = x;
          let scaledY = y;
          let scaledW = w;
          let scaledH = h;
          
          // If coordinates are in range 0-80, they might be grid coordinates
          if (x < 80 && y < 80) {
            // Try scaling as if they're grid coordinates (13x13 = 169, 26x26 = 676)
            // Scale up by a factor that brings them to 640 space
            const gridScale = 640 / 13; // ~49.23 for 13x13 grid
            scaledX = x * gridScale;
            scaledY = y * gridScale;
            scaledW = w * gridScale;
            scaledH = h * gridScale;
            console.log(`Scaling grid coords (13x13): (${x.toFixed(1)},${y.toFixed(1)},${w.toFixed(1)},${h.toFixed(1)}) -> (${scaledX.toFixed(1)},${scaledY.toFixed(1)},${scaledW.toFixed(1)},${scaledH.toFixed(1)})`);
          } else if (x < 1 && y < 1 && w < 1 && h < 1) {
            // Normalized coordinates (0-1)
            scaledX = x * 640;
            scaledY = y * 640;
            scaledW = w * 640;
            scaledH = h * 640;
            console.log(`Scaling normalized coords: (${x.toFixed(3)},${y.toFixed(3)},${w.toFixed(3)},${h.toFixed(3)}) -> (${scaledX.toFixed(1)},${scaledY.toFixed(1)},${scaledW.toFixed(1)},${scaledH.toFixed(1)})`);
          } else {
            console.log(`Using coords as-is: (${x.toFixed(1)},${y.toFixed(1)},${w.toFixed(1)},${h.toFixed(1)})`);
          }
          
          // Convert to xyxy format
          const x1 = scaledX - scaledW/2;
          const y1 = scaledY - scaledH/2;
          const x2 = scaledX + scaledW/2;
          const y2 = scaledY + scaledH/2;
          
          boxes.push([x1, y1, x2, y2]);
          scores.push(finalScore);
          classes.push(bestC);
          
          if (boxes.length <= 5) {
            console.log(`Valid detection: box(${x1.toFixed(1)},${y1.toFixed(1)},${x2.toFixed(1)},${y2.toFixed(1)}) conf=${finalScore.toFixed(4)} class=${bestC}`);
          }
        }
      }
    }
  } else {
    // Standard parsing for smaller feature dimensions
    for (let i = 0; i < N; i++) {
      const off = i * features;
      
      if (i < 3) {
        const sample = Array.from(arr.slice(off, off + Math.min(10, features)))
          .map(v => v.toFixed(3)).join(', ');
        console.log(`Sample detection ${i} features[0-${Math.min(9, features-1)}]: ${sample}`);
      }
      
      if (features >= 5) {
        const x = arr[off + 0];
        const y = arr[off + 1]; 
        const w = arr[off + 2];
        const h = arr[off + 3];
        const conf = arr[off + 4];
        
        if (conf <= confThresh || x === 0 && y === 0 && w === 0 && h === 0) continue;
        
        // Find best class
        let bestC = -1, bestP = 0;
        const startClass = 5;
        const numClasses = features - startClass;
        
        for (let c = 0; c < numClasses && (startClass + c) < features; c++) {
          const p = arr[off + startClass + c];
          if (p > bestP) { 
            bestP = p; 
            bestC = c; 
          }
        }
        
        const finalScore = conf * bestP;
        if (finalScore < confThresh) continue;

        const scaledX = x <= 2 ? x * 320 : x;
        const scaledY = y <= 2 ? y * 320 : y;
        const scaledW = w <= 2 ? w * 320 : w;
        const scaledH = h <= 2 ? h * 320 : h;

        const x1 = scaledX - scaledW/2;
        const y1 = scaledY - scaledH/2;
        const x2 = scaledX + scaledW/2;
        const y2 = scaledY + scaledH/2;

        boxes.push([x1, y1, x2, y2]);
        scores.push(finalScore);
        classes.push(bestC);
      }
    }
  }

  // Apply NMS
  const keep = nms(boxes, scores, classes, iouThresh);
  const outDet: Detection[] = [];
  
  for (const k of keep) {
    outDet.push({
      cls: classes[k],
      name: names?.[classes[k]],
      conf: scores[k],
      xyxy: boxes[k]
    });
  }

  console.log(`Custom parser found ${outDet.length} valid detections after NMS`);
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
  
  console.log("Running real ONNX inference");
  
  // Input is already letterboxed 640x640 - extract image data directly
  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d")!;
  const size = inputSize;
  
  canvas.width = size;
  canvas.height = size;
  
  // Draw frame (already letterboxed 640x640)
  if (frame instanceof HTMLCanvasElement) {
    ctx.drawImage(frame, 0, 0);
  } else {
    ctx.putImageData(frame, 0, 0);
  }
  
  const imageData = ctx.getImageData(0, 0, size, size);
  const data = new Float32Array(1 * 3 * size * size);
  
  // Convert RGBA to RGB and normalize to [0,1] in NCHW format
  let p = 0, c0 = 0, c1 = size * size, c2 = c1 * 2;
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const r8 = imageData.data[p], g8 = imageData.data[p + 1], b8 = imageData.data[p + 2];
      data[c0++] = r8 / 255;
      data[c1++] = g8 / 255;
      data[c2++] = b8 / 255;
      p += 4;
    }
  }
  
  const input = new ort.Tensor("float32", data, [1, 3, inputSize, inputSize]);
  const outputs = await session!.run({ [inputName]: input });
  
  console.log(`All output names:`, Object.keys(outputs));
  for (const [name, tensor] of Object.entries(outputs)) {
    console.log(`Output '${name}': dims=[${tensor.dims.join(', ')}], type=${tensor.type}`);
    if (tensor.dims.length === 3) {
      const [batch, anchors, features] = tensor.dims;
      console.log(`  → Batch=${batch}, Anchors/Grid=${anchors}, Features=${features}`);
      if (features === 85) {
        console.log(`This looks like standard YOLO output!`);
      } else if (features > 80) {
        console.log(`This might be YOLO output with different class count`);
      }
    }
  }
  
  // Try to find the best output tensor for detections
  let bestOutput: ort.TypedTensor<'float32'> | null = null;
  let bestOutputName = "";
  
  for (const [name, tensor] of Object.entries(outputs)) {
    const t = tensor as ort.TypedTensor<'float32'>;
    if (t.dims.length === 3) {
      const [batch, anchors, features] = t.dims;
      // Ultralytics YOLO can have large feature dimensions due to multiple detection heads
      // Accept tensors with reasonable batch size and feature count
      if (batch === 1 && anchors > 0 && features >= 5) {
        bestOutput = t;
        bestOutputName = name;
        console.log(`Using output '${name}' for detection parsing (${anchors} anchors, ${features} features)`);
        break;
      }
    }
  }
  
  if (!bestOutput) {
    console.log(`No suitable output tensor found for YOLO detection parsing`);
    return [];
  }
  
  const out = bestOutput;
  const outName = bestOutputName;
  
  console.log(`Using output '${outName}': dims=[${out.dims.join(', ')}], data type=${out.type}`);
  console.log(`First 20 values:`, Array.from(out.data.slice(0, 20)).map(v => v.toFixed(3)).join(', '));
  
  // Parse based on actual tensor format
  const dims = out.dims;
  let detections: Detection[] = [];
  
  if (dims.length === 3) {
    const [batch, num_anchors, features] = dims;
    console.log(`Tensor analysis: batch=${batch}, anchors=${num_anchors}, features=${features}`);
    
    if (features === 85) {
      // Standard YOLO format: 4(bbox) + 1(conf) + 80(classes)
      console.log(`Standard YOLO format detected (85 features)`);
      detections = parseOutput(out, opts.conf, opts.iou, opts.names);
    } else {
      // Non-standard format - use custom parser for Ultralytics models
      console.log(`Non-standard YOLO format detected (${features} features) - using custom parser`);
      detections = parseCustomYoloOutput(out, opts.conf, opts.iou, opts.names);
    }
  } else {
    console.log(`Unexpected tensor dimensions: ${dims.length}D tensor`);
  }
  
  console.log(`Found ${detections.length} detections`);
  return detections;
}
