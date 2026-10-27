import React from "react";
import { Sparkle, ShieldCheck, ArrowRight, Lightbulb } from "@phosphor-icons/react";

interface HumanReviewModalProps {
  missingSkills: string;
  userNotes: string;
  setUserNotes: (val: string) => void;
  onSubmit: () => void;
  isProcessing: boolean;
}

export const HumanReviewModal: React.FC<HumanReviewModalProps> = ({
  missingSkills,
  userNotes,
  setUserNotes,
  onSubmit,
  isProcessing,
}) => {
  return (
    <div className="w-full bg-[#10131c] border border-[#06b6d4]/40 shadow-[0_0_24px_rgba(6,182,212,0.08)] rounded-2xl p-6 mb-8">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-4 mb-5 border-b border-[#1e2436]">
        <div>
          <div className="flex items-center gap-2">
            <span className="w-2.5 h-2.5 rounded-full bg-emerald-400" />
            <span className="text-xs font-mono uppercase tracking-wider text-[#06b6d4]">
              Step 1 Complete : Checkpoint Reached
            </span>
          </div>
          <h3 className="text-lg font-semibold text-[#f8fafc] mt-1">
            Review Gap Analysis and Add Evidence
          </h3>
        </div>

        <div className="flex items-center gap-2 text-xs font-medium text-emerald-400 bg-emerald-950/40 border border-emerald-800/40 px-3 py-1.5 rounded-lg self-start">
          <ShieldCheck weight="bold" className="w-4 h-4 shrink-0" />
          <span>Truthfulness Guardrail Active</span>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Left: Missing skills detected by Scanner */}
        <div className="flex flex-col justify-between">
          <div>
            <h4 className="text-xs font-mono uppercase tracking-wider text-[#94a3b8] mb-2 flex items-center gap-1.5">
              <Lightbulb weight="bold" className="w-4 h-4 text-amber-400" />
              <span>Detected Skill Gaps from Job Description</span>
            </h4>
            <div className="p-4 rounded-xl bg-[#090a0f] border border-[#1e2436] text-xs font-mono text-[#cbd5e1] leading-relaxed whitespace-pre-wrap max-h-48 overflow-y-auto">
              {missingSkills || "No critical skill gaps identified."}
            </div>
          </div>

          <div className="p-3 rounded-xl bg-[#151926] border border-[#1e2436] text-xs text-[#94a3b8] mt-4 leading-relaxed">
            <strong className="text-[#f8fafc] font-medium">Why this matters:</strong> The AI Improver strictly forbids inventing fake metrics or unearned roles. Provide bullet context on the right for any tools or concepts you have touched.
          </div>
        </div>

        {/* Right: Notes editor for user */}
        <div>
          <label htmlFor="human-notes-textarea" className="block text-xs font-mono uppercase tracking-wider text-[#94a3b8] mb-2">
            Your Notes and Project Claims
          </label>
          <textarea
            id="human-notes-textarea"
            rows={7}
            value={userNotes}
            onChange={(e) => setUserNotes(e.target.value)}
            placeholder="Add relevant projects, courses, or hands-on context for the gaps above (e.g., 'Implemented CI/CD pipelines in my graduation project using GitHub Actions and Docker')..."
            disabled={isProcessing}
            className="w-full p-3.5 rounded-xl bg-[#090a0f] border border-[#1e2436] text-sm text-[#f8fafc] placeholder-[#64748b] focus:outline-none focus:border-[#06b6d4]/60 transition-colors resize-none font-mono"
          />

          <div className="pt-4 flex justify-end">
            <button
              type="button"
              onClick={onSubmit}
              disabled={isProcessing}
              className="w-full sm:w-auto px-6 py-2.5 rounded-xl bg-[#06b6d4] hover:bg-[#0891b2] text-[#090a0f] font-semibold text-sm transition-all flex items-center justify-center gap-2 shadow-[0_0_20px_rgba(6,182,212,0.25)] active:scale-[0.98] disabled:opacity-40"
            >
              {isProcessing ? (
                <>
                  <span className="w-4 h-4 border-2 border-[#090a0f] border-t-transparent rounded-full animate-spin" />
                  <span>Generating Tailored Package...</span>
                </>
              ) : (
                <>
                  <Sparkle weight="bold" className="w-4 h-4" />
                  <span>Step 2: Generate All Documents</span>
                  <ArrowRight weight="bold" className="w-4 h-4" />
                </>
              )}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
