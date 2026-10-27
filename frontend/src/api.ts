import type { OptimizationResult } from "./types";

interface StreamCallbacks<T> {
  onStatus: (message: string) => void;
  onDone?: (data: T) => void;
  onResult?: (data: T) => void;
  onError: (error: string) => void;
}

async function consumeSSE<T>(
  response: Response,
  callbacks: StreamCallbacks<T>,
  terminalEventType: "done" | "result"
) {
  if (!response.ok) {
    let errorText = `Request failed with status ${response.status}`;
    try {
      const json = await response.json();
      if (json.detail) errorText = json.detail;
    } catch {
      // ignore json parse error
    }
    callbacks.onError(errorText);
    return;
  }

  const reader = response.body?.getReader();
  if (!reader) {
    callbacks.onError("No response body received from server");
    return;
  }

  const decoder = new TextDecoder();
  let buffer = "";

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const blocks = buffer.split("\n\n");
      buffer = blocks.pop() || "";

      for (const block of blocks) {
        if (!block.trim()) continue;

        let eventType = "message";
        let dataStr = "";

        const lines = block.split("\n");
        for (const line of lines) {
          if (line.startsWith("event:")) {
            eventType = line.slice(6).trim();
          } else if (line.startsWith("data:")) {
            dataStr = line.slice(5).trim();
          }
        }

        if (!dataStr) continue;

        try {
          const payload = JSON.parse(dataStr);
          if (eventType === "status" && payload.message) {
            callbacks.onStatus(payload.message);
          } else if (eventType === "error" && payload.message) {
            callbacks.onError(payload.message);
            return;
          } else if (eventType === terminalEventType) {
            callbacks.onDone?.(payload as T);
            callbacks.onResult?.(payload as T);
            return;
          }
        } catch {
          // Ignore invalid JSON chunk
        }
      }
    }
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : "Stream processing error";
    callbacks.onError(message);
  } finally {
    reader.releaseLock();
  }
}

export async function runStep1(
  formData: FormData,
  callbacks: StreamCallbacks<{ session: string; missing_skills: string }>
) {
  try {
    const response = await fetch("/api/v1/step1", {
      method: "POST",
      body: formData,
    });
    await consumeSSE(response, callbacks, "done");
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : "Failed to connect to server";
    callbacks.onError(message);
  }
}

export async function runStep2(
  formData: FormData,
  callbacks: StreamCallbacks<OptimizationResult>
) {
  try {
    const response = await fetch("/api/v1/step2", {
      method: "POST",
      body: formData,
    });
    await consumeSSE(response, callbacks, "result");
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : "Failed to connect to server";
    callbacks.onError(message);
  }
}

export async function checkProviderHealth(): Promise<{ ok: boolean; message: string }> {
  try {
    const res = await fetch("/api/v1/health");
    if (!res.ok) return { ok: false, message: `HTTP ${res.status}` };
    return await res.json();
  } catch (err: unknown) {
    return { ok: false, message: "Server offline" };
  }
}
