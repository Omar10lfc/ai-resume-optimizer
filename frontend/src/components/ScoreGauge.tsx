import React, { useMemo } from "react";
import { CheckCircle } from "@phosphor-icons/react";

/**
 * Score gauge: two facts, one view.
 *  - the ring is the COMPOSITE score (60% ATS + 40% quality by default, as
 *    weighted serverside) with the pass threshold (80) marked as a notch;
 *  - the two bars beneath split it into the parts the user can influence:
 *    deterministic ATS keyword coverage and LLM presentation quality.
 *
 * The quality component isn't sent by the server — it's derived from
 * composite, ATS% and weights client-side, clamped 0–100.
 */

interface ScoreGaugeProps {
  score: number;
  scoreClass: "good" | "warn" | "bad";
  atsPercentage: number;
  weights: { ats: number; llm: number };
}

const R = 34;
const C = 2 * Math.PI * R;

function polarToPoint(cx: number, cy: number, r: number, deg: number): { x: number; y: number } {
  const rad = ((deg - 90) * Math.PI) / 180;
  return { x: cx + r * Math.cos(rad), y: cy + r * Math.sin(rad) };
}

function qualityFrom(score: number, ats: number, weights: ScoreGaugeProps["weights"]): number {
  const wAts = weights.ats > 0 && weights.ats < 1 ? weights.ats : 0.6;
  const wLlm = weights.llm > 0 && weights.llm < 1 ? weights.llm : 0.4;
  const derived = (score - wAts * ats) / wLlm;
  if (!Number.isFinite(derived)) return 0;
  return Math.min(100, Math.max(0, Math.round(derived)));
}

const CLASS_COLORS: Record<ScoreGaugeProps["scoreClass"], { varName: string; label: string }> = {
  good: { varName: "var(--ok)", label: "Ready to apply" },
  warn: { varName: "var(--warn)", label: "Moderate match" },
  bad: { varName: "var(--danger)", label: "Needs review" },
};

export const ScoreGauge: React.FC<ScoreGaugeProps> = ({ score, scoreClass, atsPercentage, weights }) => {
  const quality = qualityFrom(score, atsPercentage, weights);
  const clamped = Math.min(100, Math.max(0, Math.round(score)));
  const ringColor = CLASS_COLORS[scoreClass].varName;
  // Dash arcs: fill for score, notch at the pass threshold (80).
  const arc = (C * clamped) / 100;

  const bars = useMemo(
    () => [
      { label: "ATS keywords", pct: Math.round(atsPercentage), color: "bg-accent", weightNote: `${Math.round(weights.ats * 100)}% weight` },
      { label: "Presentation quality", pct: quality, color: "bg-line-strong", weightNote: `${Math.round(weights.llm * 100)}% weight` },
    ],
    [atsPercentage, quality, weights.ats, weights.llm]
  );

  const threshold = polarToPoint(50, 50, R - 4, (80 / 100) * 360);

  return (
    <div
      role="img"
      aria-label={`Score ${clamped} of 100, ${CLASS_COLORS[scoreClass].label}. ATS keyword coverage ${Math.round(atsPercentage)} percent, presentation quality ${quality} percent.`}
      className="flex items-center gap-4"
    >
      <div className="relative shrink-0" style={{ width: 80, height: 80 }}>
        <svg viewBox="0 0 100 100" width={80} height={80}>
          <circle cx="50" cy="50" r={R} fill="none" stroke="var(--inset)" strokeWidth="9" />
          <circle
            cx="50"
            cy="50"
            r={R}
            fill="none"
            stroke={ringColor}
            strokeWidth="9"
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeDasharray={`${arc} ${C - arc}`}
            /* rotate dash start from 3 o'clock to 12 o'clock; matches polarToPoint */
            transform="rotate(-90 50 50)"
            className="transition-[stroke-dasharray] duration-700 ease-out motion-reduce:transition-none"
          />
        </svg>
        {/* threshold notch at 80 (the server's pass mark) */}
        <svg viewBox="0 0 100 100" width={80} height={80} className="absolute inset-0 pointer-events-none">
          <circle cx={threshold.x} cy={threshold.y} r="2.5" fill="var(--ink-3)" />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className="text-xl font-bold text-ink font-mono tabular-nums">{clamped}</span>
          <span className="text-2xs text-ink-3 font-mono">/ 100</span>
        </div>
      </div>

      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2 mb-2.5">
          <span
            className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-2xs font-semibold border"
            style={{
              color: CLASS_COLORS[scoreClass].varName,
              borderColor: `color-mix(in srgb, ${CLASS_COLORS[scoreClass].varName} 40%, transparent)`,
              background: `color-mix(in srgb, ${CLASS_COLORS[scoreClass].varName} 12%, transparent)`,
            }}
          >
            <CheckCircle weight="bold" className="w-3 h-3" />
            {CLASS_COLORS[scoreClass].label}
          </span>
        </div>
        {bars.map((bar) => (
          <div key={bar.label} className="mb-2 last:mb-0">
            <div className="flex items-baseline justify-between mb-1">
              <span className="text-2xs text-ink-2 font-medium">{bar.label}</span>
              <span className="text-2xs text-accent font-mono tabular-nums">{bar.pct}%</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="h-1.5 flex-1 rounded-full bg-inset overflow-hidden">
                <div
                  className={`h-full rounded-full ${bar.color} transition-[width] duration-700 ease-out motion-reduce:transition-none`}
                  style={{ width: `${bar.pct}%` }}
                />
              </div>
              <span className="text-2xs text-ink-3 font-mono shrink-0">{bar.weightNote}</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
