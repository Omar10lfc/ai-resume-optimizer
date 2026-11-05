import React, { useState } from "react";
import { Sparkle, ArrowClockwise, Gear, Monitor, Sun, Moon } from "@phosphor-icons/react";
import { useTheme, type ThemePref } from "../lib/theme";

interface HeaderProps {
  healthOk: boolean | null;
  healthMessage?: string | null;
  onReset: () => void;
  onOpenSettings?: () => void;
  isProcessing: boolean;
}

const THEME_OPTIONS: Array<{ pref: ThemePref; icon: React.ElementType; label: string }> = [
  { pref: "system", icon: Monitor, label: "Follow system theme" },
  { pref: "light", icon: Sun, label: "Light theme" },
  { pref: "dark", icon: Moon, label: "Dark theme" },
];

const HEALTH_META = {
  ok: { label: "Provider online", dotClass: "bg-ok" },
  down: { label: "Provider unreachable", dotClass: "bg-danger" },
  checking: { label: "Checking provider", dotClass: "bg-warn animate-pulse" },
} as const;

export const Header: React.FC<HeaderProps> = ({
  healthOk,
  healthMessage,
  onReset,
  onOpenSettings,
  isProcessing,
}) => {
  const { pref, resolved, setPref } = useTheme();
  const [showHealthDetail, setShowHealthDetail] = useState(false);
  const meta =
    healthOk === true ? HEALTH_META.ok : healthOk === false ? HEALTH_META.down : HEALTH_META.checking;

  return (
    <header className="w-full border-b border-line bg-surface/80 backdrop-blur-md sticky top-0 z-40">
      <div className="w-full px-3 sm:px-6 h-12 flex items-center justify-between gap-2">
        {/* Brand + attribution: icon carries identity; wordmark collapses on narrow panels */}
        <div className="flex items-center gap-2.5 min-w-0">
          <div className="w-8 h-8 rounded-md2 bg-accent-tint border border-accent/30 flex items-center justify-center text-accent shrink-0">
            <Sparkle weight="bold" className="w-4 h-4" />
          </div>
          <div className="min-w-0">
            <span className="block text-[13px] font-semibold tracking-tight text-ink truncate">
              Resume Optimizer
            </span>
            <span className="hidden sm:inline text-2xs text-ink-3 font-mono tracking-wide">
              ATS · cover letter · interview prep
            </span>
          </div>
        </div>

        {/* Tools: health, theme, settings, reset */}
        <div className="flex items-center gap-1.5">
          {/* Health pill doubles as the settings affordance when down */}
          <div className="relative">
            <button
              type="button"
              onClick={() => setShowHealthDetail((s) => !s)}
              aria-expanded={showHealthDetail}
              aria-haspopup="dialog"
              aria-label={`${meta.label}. ${healthMessage ?? ""}`.trim()}
              className={`flex items-center gap-1.5 px-2 py-1 rounded-md2 border text-2xs font-mono transition-colors ${
                healthOk === false
                  ? "bg-danger-tint border-danger/30 text-danger hover:bg-danger-tint/70"
                  : "bg-inset border-line text-ink-2 hover:border-line-strong"
              }`}
            >
              <span className={`w-1.5 h-1.5 rounded-full ${meta.dotClass}`} aria-hidden="true" />
              <span>{healthOk === true ? "Online" : healthOk === false ? "Offline" : "Checking"}</span>
            </button>

            {showHealthDetail && (
              <div
                role="dialog"
                aria-label="Provider health details"
                className="absolute right-0 top-full mt-2 w-64 p-3 rounded-md2 bg-raised border border-line-strong text-xs text-ink-2 shadow-pop z-50 animate-in"
              >
                <div className="flex items-center justify-between pb-1.5 mb-1.5 border-b border-line">
                  <span className={`font-semibold text-2xs uppercase tracking-wide ${healthOk ? "text-ok" : "text-danger"}`}>
                    {healthOk === true ? "Provider active" : "Inference status"}
                  </span>
                  <span className="text-2xs text-ink-3 font-mono italic">/api/v1/health</span>
                </div>
                <p className="text-2xs text-ink-2 leading-relaxed break-words font-mono">
                  {healthMessage || "No response from server yet."}
                </p>
                {healthOk === false && (
                  <p className="text-2xs text-ink-3 mt-2 pt-1.5 border-t border-line">
                    ATS scoring and document exports still function without the AI provider.
                  </p>
                )}
                <button
                  type="button"
                  onClick={() => {
                    setShowHealthDetail(false);
                    onOpenSettings?.();
                  }}
                  className="mt-2 w-full flex items-center justify-center gap-1.5 px-2 py-1.5 rounded-sm2 bg-inset border border-line text-2xs text-accent hover:border-accent/50 transition-colors"
                >
                  <Gear weight="bold" className="w-3 h-3" />
                  Open settings
                </button>
              </div>
            )}
          </div>

          {/* Theme control: 3-state segmented (system / light / dark) */}
          <div
            role="radiogroup"
            aria-label="Color theme"
            className="flex items-center p-0.5 rounded-md2 border border-line bg-inset"
          >
            {THEME_OPTIONS.map(({ pref: p, icon: Icon, label }) => (
              <button
                key={p}
                type="button"
                role="radio"
                aria-checked={pref === p}
                aria-label={label}
                title={label}
                onClick={() => setPref(p)}
                className={`flex items-center justify-center w-7 h-7 rounded-sm2 transition-colors ${
                  pref === p
                    ? "bg-raised text-accent shadow-card"
                    : "text-ink-3 hover:text-ink hover:bg-raised-hover"
                }`}
              >
                <Icon weight="bold" className="w-3.5 h-3.5" />
              </button>
            ))}
          </div>

          {onOpenSettings && (
            <button
              type="button"
              onClick={onOpenSettings}
              title="Settings and connection"
              aria-label="Settings and connection"
              className="p-1.5 rounded-md2 text-ink-2 hover:text-ink hover:bg-raised-hover border border-transparent hover:border-line transition-all active:scale-95"
            >
              <Gear weight="bold" className="w-4 h-4" />
            </button>
          )}

          <button
            type="button"
            onClick={onReset}
            disabled={isProcessing}
            title={isProcessing ? "Can't reset during a run" : "Start over (clears results, keeps your resume)"}
            className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-md2 text-xs font-medium text-ink-2 hover:text-ink hover:bg-raised-hover border border-transparent hover:border-line transition-all disabled:opacity-40 disabled:cursor-not-allowed active:scale-95"
          >
            <ArrowClockwise weight="bold" className="w-3.5 h-3.5" />
            <span className="hidden sm:inline">Reset</span>
          </button>
        </div>
      </div>
      {/* current theme's resolved value exposed for e2e/testing hooks */}
      <span data-testid="resolved-theme" data-value={resolved} className="hidden" aria-hidden="true" />
    </header>
  );
};
