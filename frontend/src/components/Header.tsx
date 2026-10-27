import React from "react";
import { Sparkle, ArrowClockwise } from "@phosphor-icons/react";

interface HeaderProps {
  healthOk: boolean | null;
  healthMessage?: string | null;
  onReset: () => void;
  isProcessing: boolean;
}

export const Header: React.FC<HeaderProps> = ({ healthOk, healthMessage, onReset, isProcessing }) => {
  return (
    <header className="w-full border-b border-[#1e2436] bg-[#090a0f]/80 backdrop-blur-md sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 h-16 flex items-center justify-between">
        {/* Brand identity */}
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-[#06b6d4]/20 to-[#06b6d4]/5 border border-[#06b6d4]/30 flex items-center justify-center text-[#06b6d4]">
            <Sparkle weight="bold" className="w-5 h-5" />
          </div>
          <div>
            <span className="text-base font-semibold tracking-tight text-[#f8fafc]">
              Resume Optimizer
            </span>
            <span className="ml-2 text-xs font-mono px-2 py-0.5 rounded-full bg-[#10131c] text-[#06b6d4] border border-[#1e2436]">
              Agentic AI
            </span>
          </div>
        </div>

        {/* Right tools: Provider health + Reset button */}
        <div className="flex items-center gap-4">
          <div className="relative group">
            <div
              className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-[#10131c] border border-[#1e2436] text-xs font-mono text-[#94a3b8] cursor-help transition-colors hover:border-[#2a334c]"
              title={healthMessage || undefined}
            >
              <span
                className={`w-2 h-2 rounded-full ${
                  healthOk === true
                    ? "bg-emerald-400 shadow-[0_0_8px_rgba(52,211,153,0.5)]"
                    : healthOk === false
                    ? "bg-rose-400 shadow-[0_0_8px_rgba(251,113,133,0.3)]"
                    : "bg-amber-400 animate-pulse"
                }`}
              />
              <span>{healthOk === true ? "Groq Online" : healthOk === false ? "Degraded" : "Connecting"}</span>
            </div>

            {/* Hover details popover */}
            {healthMessage && (
              <div className="absolute right-0 top-full mt-2 hidden group-hover:block w-72 p-3 rounded-xl bg-[#151926] border border-[#2a334c] text-xs text-[#cbd5e1] shadow-2xl z-50 animate-fade-in pointer-events-none">
                <div className="flex items-center justify-between pb-1.5 mb-1.5 border-b border-[#1e2436]">
                  <span className="font-semibold text-rose-400 font-mono text-[11px] uppercase tracking-wider">
                    {healthOk === true ? "Provider Active" : "Inference Status"}
                  </span>
                  <span className="text-[10px] text-[#64748b] font-mono">/api/v1/health</span>
                </div>
                <p className="text-[11px] text-[#94a3b8] leading-relaxed break-words font-mono">
                  {healthMessage}
                </p>
                {healthOk === false && (
                  <p className="text-[10px] text-[#64748b] mt-2 pt-1.5 border-t border-[#1e2436]/60">
                    Fallback path active: Deterministic ATS scoring & document exports still function.
                  </p>
                )}
              </div>
            )}
          </div>

          <button
            onClick={onReset}
            disabled={isProcessing}
            title="Reset session state"
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium text-[#94a3b8] hover:text-[#f8fafc] hover:bg-[#151926] border border-transparent hover:border-[#1e2436] transition-all disabled:opacity-40 disabled:cursor-not-allowed active:scale-[0.98]"
          >
            <ArrowClockwise weight="bold" className={`w-3.5 h-3.5 ${isProcessing ? "animate-spin" : ""}`} />
            <span>Reset</span>
          </button>
        </div>
      </div>
    </header>
  );
};
