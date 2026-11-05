import { useCallback, useEffect, useState } from "react";

/**
 * Theme controller.
 *
 * Follows the OS color scheme by default, with a manual override. The choice
 * lives in chrome.storage.local ("uiTheme") inside the extension and in
 * localStorage ("ui_theme") on the web — no new permissions needed, and no
 * inline scripts (this module is bundled code run before first render).
 *
 * Mechanism stays in CSS: index.css defines light tokens on :root and dark
 * tokens under [data-theme="dark"] plus prefers-color-scheme. Setting the
 * data-theme attribute explicitly to "light" pins light mode even when the
 * OS is dark; clearing the attribute returns control to the system.
 */

export type ThemePref = "system" | "light" | "dark";

export const THEME_STORAGE_KEY = "uiTheme";
const WEB_STORAGE_KEY = "ui_theme";

export function isExtension(): boolean {
  return typeof chrome !== "undefined" && !!chrome.storage?.local;
}

function readStoredPrefSync(): ThemePref | null {
  try {
    if (!isExtension()) {
      const raw = localStorage.getItem(WEB_STORAGE_KEY);
      return raw === "light" || raw === "dark" || raw === "system" ? raw : null;
    }
  } catch {
    // storage unavailable — fall through to system
  }
  return null; // extension path is async-only; paint system first
}

function readStoredPrefAsync(): Promise<ThemePref | null> {
  if (isExtension()) {
    return chrome.storage.local
      .get([THEME_STORAGE_KEY])
      .then((data: Record<string, unknown>) => {
        const raw = data[THEME_STORAGE_KEY];
        return raw === "light" || raw === "dark" || raw === "system" ? raw : null;
      })
      .catch(() => null);
  }
  return Promise.resolve(readStoredPrefSync());
}

async function writeStoredPref(pref: ThemePref): Promise<void> {
  try {
    if (isExtension()) {
      await chrome.storage.local.set({ [THEME_STORAGE_KEY]: pref });
    } else {
      localStorage.setItem(WEB_STORAGE_KEY, pref);
    }
  } catch {
    // ignore — theme is cosmetic, never worth failing a render for
  }
}

export function applyThemePref(pref: ThemePref): void {
  const root = document.documentElement;
  // "system" clears the attribute so the prefers-color-scheme media query in
  // index.css takes over. "light"/"dark" pin the theme explicitly.
  if (pref === "system") {
    root.removeAttribute("data-theme");
  } else {
    root.setAttribute("data-theme", pref);
  }
}

/** Runs once at startup (main.tsx), before React mounts, to avoid a theme flash. */
export function initTheme(): void {
  applyThemePref(readStoredPrefSync() ?? "system");
  void readStoredPrefAsync().then((pref) => {
    if (pref) applyThemePref(pref);
  });

  // Another context (e.g. the popup vs side panel) changed the theme — follow it.
  if (isExtension() && chrome.storage.onChanged) {
    chrome.storage.onChanged.addListener((changes, area) => {
      if (area === "local" && changes[THEME_STORAGE_KEY]) {
        const value = changes[THEME_STORAGE_KEY].newValue;
        if (value === "light" || value === "dark" || value === "system") {
          applyThemePref(value);
        }
      }
    });
  }
}

interface ThemeState {
  pref: ThemePref;
  resolved: "light" | "dark";
  setPref: (pref: ThemePref) => void;
}

/** Live theme state for UI controls (Header toggle). */
export function useTheme(): ThemeState {
  const [pref, setPrefState] = useState<ThemePref>(() => readStoredPrefSync() ?? "system");
  const [systemDark, setSystemDark] = useState<boolean>(
    () => window.matchMedia?.("(prefers-color-scheme: dark)").matches ?? false
  );

  useEffect(() => {
    void readStoredPrefAsync().then((stored) => {
      if (stored) setPrefState(stored);
    });
  }, []);

  useEffect(() => {
    const mql = window.matchMedia?.("(prefers-color-scheme: dark)");
    if (!mql) return;
    const onChange = (e: MediaQueryListEvent) => setSystemDark(e.matches);
    mql.addEventListener("change", onChange);
    return () => mql.removeEventListener("change", onChange);
  }, []);

  const setPref = useCallback((next: ThemePref) => {
    setPrefState(next);
    applyThemePref(next);
    void writeStoredPref(next);
  }, []);

  // Resolved value mirrors the CSS media-query logic in index.css:
  // explicit data-theme wins, otherwise the system preference applies.
  const resolved: "light" | "dark" =
    pref === "system" ? (systemDark ? "dark" : "light") : pref;

  return { pref, resolved, setPref };
}
