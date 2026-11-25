import React, { useEffect, useRef, useState } from "react";
import { X, Check, PlugsConnected, ArrowClockwise, Key, Globe, Shield, WarningCircle, Trash } from "@phosphor-icons/react";
import { getSettings, saveSettings, checkProviderHealth, deleteAllServerData, type ExtensionSettings, type ProviderHealth } from "../api";
import { removeStorageValues } from "../lib/storage";

/**
 * Connection settings. Same storage contract as before (api.ts getSettings /
 * saveSettings); the additions are presentation: proper labels, Escape to
 * close, focus recovery, and a local-data eraser:
 *   - saved resume text (chrome.storage.local.savedResumeText / web
 *     localStorage.saved_resume_text)
 *   - pending scraped job payload
 *   - onboarding completion flag (so "forget my data" is a full reset)
 * Backend URL and credentials are NOT touched by the eraser — they belong to
 * the connection, not to personal data. That split is stated in the UI copy.
 */

interface SettingsModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSettingsSaved?: () => void;
}

export const SettingsModal: React.FC<SettingsModalProps> = ({ isOpen, onClose, onSettingsSaved }) => {
  const [settings, setSettings] = useState<ExtensionSettings>({
    backendUrl: "",
    appAuthUser: "",
    appAuthPass: "",
    customGroqKey: "",
  });
  const [testing, setTesting] = useState(false);
  const [testResult, setTestResult] = useState<ProviderHealth | null>(null);
  const [savedSuccess, setSavedSuccess] = useState(false);
  const [eraseArmed, setEraseArmed] = useState(false);
  const [eraseDone, setEraseDone] = useState(false);
  const [deleteAllText, setDeleteAllText] = useState("");
  const [deleteAllBusy, setDeleteAllBusy] = useState(false);
  const [deleteAllDone, setDeleteAllDone] = useState(false);
  const [deleteAllError, setDeleteAllError] = useState<string | null>(null);
  const dialogRef = useRef<HTMLDivElement>(null);
  const previouslyFocused = useRef<HTMLElement | null>(null);

  useEffect(() => {
    if (!isOpen) return;
    previouslyFocused.current = document.activeElement as HTMLElement | null;
    dialogRef.current?.querySelector<HTMLElement>("input")?.focus();
    void getSettings().then((s) => {
      setSettings(s);
      setTestResult(null);
      setSavedSuccess(false);
      setEraseArmed(false);
      setEraseDone(false);
    });
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") handleClose();
    };
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("keydown", onKey);
      previouslyFocused.current?.focus();
    };
    function handleClose() {
      onClose();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isOpen]);

  if (!isOpen) return null;

  const handleTestConnection = async () => {
    setTesting(true);
    setTestResult(null);
    // Temporarily persist to storage to test this exact configuration
    await saveSettings(settings);
    const result = await checkProviderHealth();
    setTestResult(result);
    setTesting(false);
  };

  const handleSave = async (e: React.FormEvent) => {
    e.preventDefault();
    await saveSettings(settings);
    setSavedSuccess(true);
    if (onSettingsSaved) onSettingsSaved();
    setTimeout(() => {
      setSavedSuccess(false);
      onClose();
    }, 800);
  };

  const handleDeleteAll = async () => {
    if (deleteAllText.trim().toUpperCase() !== "DELETE ALL") {
      setDeleteAllError("Type DELETE ALL to confirm this destructive action.");
      return;
    }
    setDeleteAllBusy(true);
    setDeleteAllError(null);
    try {
      const outcome = await deleteAllServerData();
      if (!outcome.ok) {
        setDeleteAllError(outcome.detail);
        return;
      }
      setDeleteAllDone(true);
      onSettingsSaved?.();
    } catch {
      setDeleteAllError("Delete failed. Is the backend running?");
    } finally {
      setDeleteAllBusy(false);
    }
  };

  const handleErase = async () => {
    if (!eraseArmed) {
      setEraseArmed(true);
      return;
    }
    await removeStorageValues(
      ["savedResumeText", "pendingScrapedJob", "onboardedAt", "pendingUserNotes"],
      ["saved_resume_text", "pending_optimizer_notes", "onboardedAt"],
    );
    setEraseDone(true);
    setEraseArmed(false);
    setTimeout(() => setEraseDone(false), 4000);
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/50 backdrop-blur-sm animate-in"
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div
        ref={dialogRef}
        role="dialog"
        aria-modal="true"
        aria-labelledby="settings-title"
        className="panel w-full max-w-md p-5 shadow-pop max-h-[90dvh] overflow-y-auto"
      >
        <div className="flex items-center justify-between pb-3 border-b border-line mb-4">
          <div className="flex items-center gap-2.5">
            <div className="p-2 rounded-md2 bg-accent-tint text-accent border border-accent/20">
              <PlugsConnected weight="bold" className="w-4.5 h-4.5" />
            </div>
            <div>
              <h2 id="settings-title" className="text-sm font-semibold text-ink">
                Optimizer settings
              </h2>
              <p className="text-2xs text-ink-3">Backend connection and credentials</p>
            </div>
          </div>
          <button
            type="button"
            onClick={onClose}
            aria-label="Close settings"
            className="p-1 rounded-sm2 text-ink-3 hover:text-ink hover:bg-raised-hover transition-colors"
          >
            <X weight="bold" className="w-4 h-4" />
          </button>
        </div>

        <form onSubmit={handleSave} className="space-y-4">
          <div>
            <label htmlFor="settings-backend-url" className="field-label flex items-center gap-1.5">
              <Globe weight="bold" className="w-3.5 h-3.5 text-accent" />
              Backend server URL
            </label>
            <input
              id="settings-backend-url"
              type="text"
              inputMode="url"
              value={settings.backendUrl}
              onChange={(e) => setSettings({ ...settings, backendUrl: e.target.value })}
              placeholder="http://localhost:8000 (or cloud endpoint)"
              className="input font-mono input-sm"
            />
            <p className="text-2xs text-ink-3 mt-1">
              Leave blank on the web to use same-origin /api. In the extension, set it to your server's URL.
            </p>
          </div>

          <div>
            <label htmlFor="settings-groq-key" className="field-label flex items-center gap-1.5">
              <Key weight="bold" className="w-3.5 h-3.5 text-warn" />
              Your Groq API key (optional)
            </label>
            <input
              id="settings-groq-key"
              type="password"
              autoComplete="off"
              value={settings.customGroqKey}
              onChange={(e) => setSettings({ ...settings, customGroqKey: e.target.value })}
              placeholder="gsk_..."
              className="input font-mono input-sm"
            />
            <p className="text-2xs text-ink-3 mt-1">
              Sent per request via <code className="font-mono">X-Groq-Api-Key</code> to use your own quota. Never synced anywhere.
            </p>
          </div>

          <div className="pt-3 border-t border-line">
            <span className="field-label flex items-center gap-1.5">
              <Shield weight="bold" className="w-3.5 h-3.5 text-ok" />
              Backend Basic Auth (if the server enables it)
            </span>
            <div className="grid grid-cols-2 gap-2">
              <div>
                <label htmlFor="settings-auth-user" className="sr-only">Basic Auth username</label>
                <input
                  id="settings-auth-user"
                  type="text"
                  autoComplete="off"
                  value={settings.appAuthUser}
                  onChange={(e) => setSettings({ ...settings, appAuthUser: e.target.value })}
                  placeholder="Username"
                  className="input input-sm"
                />
              </div>
              <div>
                <label htmlFor="settings-auth-pass" className="sr-only">Basic Auth password</label>
                <input
                  id="settings-auth-pass"
                  type="password"
                  autoComplete="new-password"
                  value={settings.appAuthPass}
                  onChange={(e) => setSettings({ ...settings, appAuthPass: e.target.value })}
                  placeholder="Password"
                  className="input input-sm"
                />
              </div>
            </div>
          </div>

          {/* Health check feedback */}
          <div aria-live="polite">
            {testResult && (
              <div
                className={`flex items-center gap-2 p-2.5 rounded-md2 border text-2xs ${
                  testResult.ok
                    ? "bg-ok-tint border-ok/30 text-ok"
                    : "bg-danger-tint border-danger/30 text-danger"
                }`}
              >
                <span className={`w-1.5 h-1.5 rounded-full shrink-0 ${testResult.ok ? "bg-ok" : "bg-danger"}`} />
                <span className="truncate">{testResult.message}</span>
              </div>
            )}
          </div>

          {/* Danger zone: local data */}
          <div className="pt-3 border-t border-line">
            <span className="field-label flex items-center gap-1.5">
              <WarningCircle weight="bold" className="w-3.5 h-3.5 text-danger" />
              Your data
            </span>
            <p className="text-2xs text-ink-3 mb-2 leading-relaxed">
              Your saved resume stays in this browser only. Erasing clears it, the captured job, and the onboarding flag.
              Connection settings are kept.
            </p>
            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={handleErase}
                className={eraseArmed ? "btn btn-sm btn-danger-primary" : "btn btn-sm btn-secondary text-danger"}
              >
                {eraseArmed ? "Click again to erase everything" : eraseDone ? "Erased" : "Forget my data"}
              </button>
              {eraseDone && (
                <span className="flex items-center gap-1 text-2xs text-ok">
                  <Check weight="bold" className="w-3 h-3" />
                  Local data cleared
                </span>
              )}
            </div>
          </div>

          {/* Delete-all: backend-side erase (history + runs ledger +
              checkpoints + artifact dirs). Requires typed confirmation. */}
          <div className="pt-3 border-t border-line">
            <span className="field-label flex items-center gap-1.5">
              <Trash weight="bold" className="w-3.5 h-3.5 text-danger" />
              Delete all server data
            </span>
            <p className="text-2xs text-ink-3 mb-2 leading-relaxed">
              Removes every application, run, checkpoint thread and artifact directory on the server.
              This deletes your data where the backend stores it; browser-side copies elsewhere are untouched.
            </p>
            <div className="flex flex-col sm:flex-row gap-2 items-stretch sm:items-center">
              <input
                type="text"
                value={deleteAllText}
                onChange={(e) => setDeleteAllText(e.target.value)}
                placeholder="Type DELETE ALL to confirm"
                className="input text-2xs sm:max-w-[260px]"
                aria-label="Type DELETE ALL to confirm"
              />
              <button
                type="button"
                onClick={handleDeleteAll}
                disabled={deleteAllBusy || deleteAllText.trim().toUpperCase() !== "DELETE ALL"}
                className="btn btn-danger-primary btn-sm"
              >
                {deleteAllBusy ? "Deleting..." : "Delete all"}
              </button>
            </div>
            {deleteAllDone && (
              <span className="flex items-center gap-1 text-2xs text-ok mt-1">
                <Check weight="bold" className="w-3 h-3" />
                All data deleted
              </span>
            )}
            {deleteAllError && (
              <p role="alert" className="flex items-start gap-1.5 mt-1.5 text-2xs text-danger leading-relaxed">
                <WarningCircle weight="bold" className="w-3.5 h-3.5 shrink-0 mt-px" />
                {deleteAllError}
              </p>
            )}
          </div>

          {/* Actions */}
          <div className="flex items-center justify-between pt-4 border-t border-line">
            <button type="button" onClick={handleTestConnection} disabled={testing} className="btn btn-secondary btn-sm">
              <ArrowClockwise weight="bold" className={`w-3.5 h-3.5 ${testing ? "animate-spin-slow" : ""}`} />
              <span>{testing ? "Testing..." : "Test connection"}</span>
            </button>

            <div className="flex items-center gap-2">
              <button type="button" onClick={onClose} className="btn btn-ghost btn-sm text-ink-3">
                Cancel
              </button>
              <button type="submit" className="btn btn-primary btn-sm">
                {savedSuccess ? (
                  <>
                    <Check weight="bold" className="w-3.5 h-3.5" />
                    <span>Saved</span>
                  </>
                ) : (
                  <span>Save settings</span>
                )}
              </button>
            </div>
          </div>
        </form>
      </div>
    </div>
  );
};
