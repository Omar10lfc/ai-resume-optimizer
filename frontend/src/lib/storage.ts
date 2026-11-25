/**
 * Unified local persistence across Chrome Extension (`chrome.storage.local`)
 * and Web SPA (`localStorage`). Never throws when storage is restricted.
 */

export function isExtension(): boolean {
  return typeof chrome !== "undefined" && !!chrome.storage?.local;
}

/** Synchronous web-only read for initial React state (no-op in extension mode). */
export function readWebValue(webKey: string): string {
  try {
    if (isExtension()) return "";
    return localStorage.getItem(webKey) || "";
  } catch {
    return "";
  }
}

export async function writeStorageValue(
  extKey: string,
  webKey: string,
  value: string,
): Promise<void> {
  try {
    if (isExtension()) {
      await chrome.storage.local.set({ [extKey]: value });
    } else {
      localStorage.setItem(webKey, value);
    }
  } catch {
    // storage unavailable or quota exceeded
  }
}

export async function removeStorageValues(
  extKeys: string[],
  webKeys: string[],
): Promise<void> {
  try {
    if (isExtension()) {
      await chrome.storage.local.remove(extKeys);
    } else {
      for (const key of webKeys) {
        localStorage.removeItem(key);
      }
    }
  } catch {
    // ignore
  }
}
