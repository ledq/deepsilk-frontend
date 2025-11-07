const API_BASE = import.meta.env.VITE_API_BASE as string | undefined;

export async function annotateOnServer(
  file: File, fps: number, conf: number, iou: number, imgsz = 640
): Promise<Blob | { video_url?: string; json_url?: string; frames?: unknown }> {
  if (!API_BASE) throw new Error("VITE_API_BASE not set");
  const form = new FormData();
  form.append("file", file);
  form.append("fps", String(fps));
  form.append("conf", String(conf));
  form.append("iou", String(iou));
  form.append("imgsz", String(imgsz));

  const res = await fetch(`${API_BASE}/annotate`, { method: "POST", body: form });
  const ct = res.headers.get("content-type") || "";
  if (ct.includes("video/")) return await res.blob();
  return await res.json();
}
