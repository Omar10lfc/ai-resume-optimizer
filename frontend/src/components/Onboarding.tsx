import React, { useEffect, useRef, useState } from "react";
import { Sparkle, MagnifyingGlass, ShieldCheck, ArrowRight, ArrowLeft, X } from "@phosphor-icons/react";
import { writeOnboarded } from "../lib/onboarding";

/**
 * First-run onboarding: three steps, skippable at any point, shown once.
 * The store/read/write helpers live in lib/onboarding.ts so this module
 * exports only the dialog component.
 */

const STEPS: Array<{
  icon: React.ElementType;
  title: string;
  body: string;
}> = [
  {
    icon: Sparkle,
    title: "One panel, three deliverables",
    body: "Open a job posting, add your resume, and get a tailored resume, cover letter, ATS report, and interview questions — all checked against skills the posting asks for.",
  },
  {
    icon: MagnifyingGlass,
    title: "You approve every claim",
    body: "Step one finds your skill gaps. You confirm what's true and add context before anything is rewritten — the rez never invents experience you didn't approve.",
  },
  {
    icon: ShieldCheck,
    title: "Your data stays with you",
    body: "Your resume is stored only in your browser for reuse, never synced anywhere. You can erase everything in Settings at any time.",
  },
];

interface OnboardingProps {
  open: boolean;
  onDone: () => void;
}

export const Onboarding: React.FC<OnboardingProps> = ({ open, onDone }) => {
  const [step, setStep] = useState(0);
  const primaryRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    if (!open) return;
    primaryRef.current?.focus();
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") handleFinish();
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open]);

  useEffect(() => {
    if (open) {
      setStep(0);
    }
  }, [open]);

  const finish = () => {
    void writeOnboarded();
    onDone();
  };

  const handleFinish = () => finish();

  if (!open) return null;
  const StepIcon = STEPS[step].icon;
  const isLast = step === STEPS.length - 1;

  return (
    <div
      className="fixed inset-0 z-[70] flex items-center justify-center p-4 bg-black/40 dark:bg-black/60"
      style={{ background: "color-mix(in srgb, var(--canvas) 75%, black)" }}
      role="dialog"
      aria-modal="true"
      aria-labelledby="onboarding-title"
    >
      <div className="panel w-full max-w-sm p-6 shadow-pop animate-in">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-1.5" aria-hidden="true">
            {STEPS.map((_, i) => (
              <span
                key={i}
                className={`h-1.5 rounded-full transition-all duration-200 ${i === step ? "w-6 bg-accent" : "w-1.5 bg-line-strong"}`}
              />
            ))}
          </div>
          <button
            type="button"
            onClick={finish}
            aria-label="Skip onboarding"
            className="p-1 rounded-sm2 text-ink-3 hover:text-ink transition-colors"
          >
            <X weight="bold" className="w-4 h-4" />
          </button>
        </div>

        <div className="w-10 h-10 rounded-md2 flex items-center justify-center mb-4 bg-accent-tint text-accent border border-line">
          <StepIcon weight="bold" className="w-5 h-5" />
        </div>
        <h2 id="onboarding-title" className="text-base font-semibold text-ink mb-1.5">
          {STEPS[step].title}
        </h2>
        <p className="text-xs text-ink-2 leading-relaxed mb-6">{STEPS[step].body}</p>

        <div className="flex items-center justify-between">
          <button type="button" onClick={finish} className="btn btn-ghost btn-sm text-ink-3">
            Skip tour
          </button>
          <div className="flex items-center gap-2">
            {step > 0 && (
              <button type="button" onClick={() => setStep((s) => s - 1)} className="btn btn-secondary">
                <ArrowLeft weight="bold" className="w-3.5 h-3.5" />
                <span>Back</span>
              </button>
            )}
            <button
              ref={primaryRef}
              type="button"
              onClick={() => (isLast ? finish() : setStep((s) => s + 1))}
              className="btn btn-primary"
            >
              <span>{isLast ? "Get started" : "Next"}</span>
              <ArrowRight weight="bold" className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
