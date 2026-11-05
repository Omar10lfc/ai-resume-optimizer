import { isExtension } from "./theme";

/**
 * First-run onboarding flag. Lives with the other local-only data
 * (chrome.storage.local in the extension / localStorage on the web) so
 * "forget my data" also resets the tour.
 */

export const ONBOARD_KEY = "onboardedAt";

export async function isOnboarded(): Promise<boolean> {
  try {
    if (isExtension()) {
      const data = await chrome.storage.local.get([ONBOARD_KEY]);
      return data[ONBOARD_KEY] !== undefined;
    }
    return localStorage.getItem(ONBOARD_KEY) !== null;
  } catch {
    return true; // fail closed: don't nag if storage is broken
  }
}

export async function writeOnboarded(): Promise<void> {
  try {
    const value = { at: new Date().toISOString() };
    if (isExtension()) {
      await chrome.storage.local.set({ [ONBOARD_KEY]: value });
    } else {
      localStorage.setItem(ONBOARD_KEY, JSON.stringify(value));
    }
  } catch {
    // ignore
  }
}
