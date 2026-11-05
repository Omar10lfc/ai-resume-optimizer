import React from "react";
import { WarningCircle, ArrowClockwise, Gear, X } from "@phosphor-icons/react";
import type { ErrorInfo } from "../lib/errors";

/**
 * Inline run-failure card. The counterpart of toasts: toasts carry transient
 * confirmations, ERROR STATEMENT stays visible until the user resolves or
 * dismisses it, with the next action one click away (retry / open settings
 * / start over), named in plain language by lib/errors.ts.
 */

interface ErrorStateProps {
  error: ErrorInfo;
  onRetry?: () => void;
  onOpenSettings?: () => void;
  onDismiss: () => void;
}

export const ErrorState: React.FC<ErrorStateProps> = ({ error, onRetry, onOpenSettings, onDismiss }) => {
  const openSettings = error.action?.includes("Settings") && onOpenSettings ? onOpenSettings : null;
  return (
    <div
      role="alert"
      className="mb-6 p-4 rounded-md2 bg-danger-tint border border-danger/35 animate-in"
    >
      <div className="flex items-start gap-2.5">
        <WarningCircle weight="bold" className="w-5 h-5 text-danger shrink-0 mt-0.5" />
        <div className="flex-1 min-w-0">
          <p className="text-xs font-semibold text-danger">{error.title}</p>
          <p className="text-2xs text-ink-2 leading-relaxed mt-1">{error.detail}</p>
          <div className="flex items-center gap-2 mt-2.5">
            {openSettings && (
              <button type="button" onClick={openSettings} className="btn btn-danger-primary btn-sm">
                <Gear weight="bold" className="w-3.5 h-3.5" />
                <span>Open settings</span>
              </button>
            )}
            {onRetry && (
              <button type="button" onClick={onRetry} className="btn btn-secondary btn-sm">
                <ArrowClockwise weight="bold" className="w-3.5 h-3.5" />
                <span>Try again</span>
              </button>
            )}
          </div>
        </div>
        <button
          type="button"
          onClick={onDismiss}
          aria-label="Dismiss error"
          className="p-1 rounded-sm2 text-ink-3 hover:text-ink transition-colors shrink-0"
        >
          <X weight="bold" className="w-4 h-4" />
        </button>
      </div>
    </div>
  );
};
