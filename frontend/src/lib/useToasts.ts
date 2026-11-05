import { useCallback, useRef, useState } from "react";

/**
 * Toast queue state. Kept outside components/ so the component module only
 * exports JSX (react-refresh rule) and this hook stays independently testable.
 *
 * Toasts carry TRANSIENT confirmations only ("copied", "restored your
 * resume"). Errors use the always-visible ErrorState card — never toasts —
 * because auto-dismissing an error hides it.
 */

export type ToastTone = "info" | "success" | "warn" | "danger";

export interface Toast {
  id: number;
  tone: ToastTone;
  message: string;
  /** Auto-dismiss delay; errors must not ride this path. */
  ttlMs?: number;
}

export function useToasts() {
  const [toasts, setToasts] = useState<Toast[]>([]);
  const nextId = useRef(1);

  const push = useCallback((tone: ToastTone, message: string, ttlMs = 5000) => {
    const id = nextId.current++;
    setToasts((prev) => [...prev.slice(-3), { id, tone, message, ttlMs }]);
    return id;
  }, []);

  const dismiss = useCallback((id: number) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
  }, []);

  return { toasts, push, dismiss };
}
