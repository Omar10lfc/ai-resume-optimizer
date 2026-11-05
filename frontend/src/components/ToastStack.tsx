import React, { useEffect } from "react";
import { X, CheckCircle, WarningCircle, Warning, Info } from "@phosphor-icons/react";
import type { Toast, ToastTone } from "../lib/useToasts";

/**
 * Toast stack: the single polite aria-live region for transient feedback.
 * Timers live in a ref map; each toast schedules exactly one dismissal, and
 * gone toasts release theirs on the next pass.
 */

interface ToastStackProps {
  toasts: Toast[];
  onDismiss: (id: number) => void;
}

const TONE_META: Record<ToastTone, { icon: React.ElementType; className: string }> = {
  info: { icon: Info, className: "text-info" },
  success: { icon: CheckCircle, className: "text-ok" },
  warn: { icon: Warning, className: "text-warn" },
  danger: { icon: WarningCircle, className: "text-danger" },
};

export const ToastStack: React.FC<ToastStackProps> = ({ toasts, onDismiss }) => {
  return (
    <div
      aria-live="polite"
      role="status"
      className="fixed bottom-4 left-1/2 -translate-x-1/2 z-[60] flex flex-col items-center gap-2 w-[calc(100%-2rem)] max-w-sm pointer-events-none"
    >
      {toasts.map((toast) => {
        const meta = TONE_META[toast.tone];
        const Icon = meta.icon;
        return (
          <ToastItem key={toast.id} toast={toast} icon={Icon} iconClass={meta.className} onDismiss={onDismiss} />
        );
      })}
    </div>
  );
};

const ToastItem: React.FC<{
  toast: Toast;
  icon: React.ElementType;
  iconClass: string;
  onDismiss: (id: number) => void;
}> = ({ toast, icon: Icon, iconClass, onDismiss }) => {
  // One effect per toast — the timer's lifetime equals the toast's mount.
  useEffect(() => {
    const t = setTimeout(() => onDismiss(toast.id), toast.ttlMs ?? 5000);
    return () => clearTimeout(t);
  }, [toast, onDismiss]);

  return (
    <div className="animate-in pointer-events-auto flex w-full items-start gap-2.5 px-3.5 py-2.5 rounded-md2 bg-raised border border-line-strong shadow-pop">
      <Icon weight="bold" className={`w-4.5 h-4.5 shrink-0 mt-0.5 ${iconClass}`} />
      <p className="flex-1 text-xs text-ink leading-relaxed">{toast.message}</p>
      <button
        type="button"
        onClick={() => onDismiss(toast.id)}
        aria-label="Dismiss notification"
        className="p-0.5 rounded-sm2 text-ink-3 hover:text-ink transition-colors"
      >
        <X weight="bold" className="w-3.5 h-3.5" />
      </button>
    </div>
  );
};
