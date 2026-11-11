import type {
  ApplicationDetail,
  ApplicationSummary,
  OptimizationResult,
} from "./types";
interface StreamCallbacks<T> {
  onStatus: (message: string, node?: string) => void;
  onWarning?: (message: string) => void;
  onSession?: (session: string) => void;
  onDone?: (data: T) => void;
  onResult?: (data: T) => void;
  onError: (error: string) => void;
}

export interface ExtensionSettings {
  backendUrl: string;
  appAuthUser: string;
  appAuthPass: string;
  customGroqKey: string;
}

const DEFAULT_SETTINGS: ExtensionSettings = {
  backendUrl: typeof chrome !== "undefined" && chrome.runtime?.id ? "http://localhost:8000" : "",
  appAuthUser: "",
  appAuthPass: "",
  customGroqKey: "",
};

export async function getSettings(): Promise<ExtensionSettings> {
  if (typeof chrome !== "undefined" && chrome.storage && chrome.storage.local) {
    try {
      const data = (await chrome.storage.local.get([
        "backendUrl",
        "appAuthUser",
        "appAuthPass",
        "customGroqKey",
      ])) as Record<string, string | undefined>;

      return {
        backendUrl: typeof data.backendUrl === "string" ? data.backendUrl : DEFAULT_SETTINGS.backendUrl,
        appAuthUser: typeof data.appAuthUser === "string" ? data.appAuthUser : "",
        appAuthPass: typeof data.appAuthPass === "string" ? data.appAuthPass : "",
        customGroqKey: typeof data.customGroqKey === "string" ? data.customGroqKey : "",
      };
    } catch {
      // fallback
    }
  }

  // Web browser fallback to localStorage
  try {
    const raw = localStorage.getItem("resume_optimizer_settings");
    if (raw) {
      return { ...DEFAULT_SETTINGS, ...JSON.parse(raw) };
    }
  } catch {
    // fallback
  }

  return DEFAULT_SETTINGS;
}

export async function saveSettings(settings: Partial<ExtensionSettings>): Promise<void> {
  const current = await getSettings();
  const updated = { ...current, ...settings };

  if (typeof chrome !== "undefined" && chrome.storage && chrome.storage.local) {
    await chrome.storage.local.set(updated);
  } else {
    try {
      localStorage.setItem("resume_optimizer_settings", JSON.stringify(updated));
    } catch {
      // ignore
    }
  }
}

export async function getBaseUrl(): Promise<string> {
  const settings = await getSettings();
  return (settings.backendUrl || "").replace(/\/+$/, "");
}

export async function getAuthHeaders(): Promise<Record<string, string>> {
  const settings = await getSettings();
  const headers: Record<string, string> = {};

  if (settings.appAuthUser && settings.appAuthPass) {
    const encoded = btoa(`${settings.appAuthUser}:${settings.appAuthPass}`);
    headers["Authorization"] = `Basic ${encoded}`;
  }

  if (settings.customGroqKey) {
    headers["X-Groq-Api-Key"] = settings.customGroqKey.trim();
  }

  return headers;
}

export async function resolveArtifactUrl(relativePath: string): Promise<string> {
  if (relativePath.startsWith("http://") || relativePath.startsWith("https://")) {
    return relativePath;
  }
  const baseUrl = await getBaseUrl();
  const cleanPath = relativePath.startsWith("/") ? relativePath : `/${relativePath}`;
  return `${baseUrl}${cleanPath}`;
}

// BB1: application history (persistent profile store).

export async function fetchApplications(limit = 50): Promise<ApplicationSummary[]> {
  const baseUrl = await getBaseUrl();
  const headers = await getAuthHeaders();
  const res = await fetch(`${baseUrl}/api/v1/applications?limit=${limit}`, { headers });
  if (!res.ok) throw new Error(`Request failed with status ${res.status}`);
  const data = await res.json();
  return (data.applications ?? []) as ApplicationSummary[];
}

export async function fetchApplication(id: number): Promise<ApplicationDetail> {
  const baseUrl = await getBaseUrl();
  const headers = await getAuthHeaders();
  const res = await fetch(`${baseUrl}/api/v1/applications/${id}`, { headers });
  if (!res.ok) throw new Error(`Request failed with status ${res.status}`);
  return (await res.json()) as ApplicationDetail;
}

export async function deleteApplication(id: number): Promise<void> {
  const baseUrl = await getBaseUrl();
  const headers = await getAuthHeaders();
  const res = await fetch(`${baseUrl}/api/v1/applications/${id}`, {
    method: "DELETE",
    headers,
  });
  if (!res.ok) throw new Error(`Request failed with status ${res.status}`);
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
            callbacks.onStatus(payload.message, payload.node);
          } else if (eventType === "warning" && payload.message) {
            callbacks.onWarning?.(payload.message);
          } else if (eventType === "session" && payload.session) {
            callbacks.onSession?.(payload.session);
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
    const baseUrl = await getBaseUrl();
    const authHeaders = await getAuthHeaders();

    const response = await fetch(`${baseUrl}/api/v1/step1`, {
      method: "POST",
      headers: authHeaders,
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
    const baseUrl = await getBaseUrl();
    const authHeaders = await getAuthHeaders();

    const response = await fetch(`${baseUrl}/api/v1/step2`, {
      method: "POST",
      headers: authHeaders,
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
    const baseUrl = await getBaseUrl();
    const authHeaders = await getAuthHeaders();

    const res = await fetch(`${baseUrl}/api/v1/health`, {
      headers: authHeaders,
    });
    if (!res.ok) return { ok: false, message: `HTTP ${res.status}` };
    return await res.json();
  } catch {
    return { ok: false, message: "Backend offline (check URL or run server)" };
  }
}
