import React from "react";
import { CheckCircle, CircleNotch, Circle } from "@phosphor-icons/react";
import type { PipelineStage } from "../types";

interface StepTimelineProps {
  stage: PipelineStage;
  statusMessage: string;
  /** Graph node ids in arrival order, from status events carrying a node id. */
  nodeLog: string[];
}

interface TimelineNode {
  id: string;
  label: string;
  stepNum: number;
}

// Full pipeline topology incl. the cover_letter / interview_prep fan-out.
// "checkpoint" is a pseudo-node: the graph INTERRUPTS before the improver and
// never announces it as a node update - the pipeline paused there during the
// step 1 review.
const NODES: TimelineNode[] = [
  { id: "loader", label: "Ingest Docs", stepNum: 1 },
  { id: "scanner", label: "Gap Analysis", stepNum: 1 },
  { id: "checkpoint", label: "Human Review", stepNum: 1 },
  { id: "improver", label: "Tailor Resume", stepNum: 2 },
  { id: "reviewer", label: "Quality Audit", stepNum: 2 },
  { id: "ats_check", label: "ATS Match", stepNum: 2 },
  { id: "cover_letter", label: "Cover Letter", stepNum: 2 },
  { id: "interview_prep", label: "Interview Prep", stepNum: 2 },
  { id: "pdf_exporter", label: "Export PDF", stepNum: 2 },
];

const ORDER = NODES.map((n) => n.id);
const isStep1Node = (id: string) =>
  id === "loader" || id === "scanner" || id === "checkpoint";

export const StepTimeline: React.FC<StepTimelineProps> = ({ stage, statusMessage, nodeLog }) => {
  // Progress is derived from the node arrival log (server truth), NOT from
  // prose-sniffing status messages. The LAST arrival is the running node;
  // anything already announced stays completed (the improver <-> reviewer ->
  // ats_check correction loop re-announces nodes, which arrival history
  // handles naturally).
  const current = nodeLog.length > 0 ? nodeLog[nodeLog.length - 1] : null;
  const currentIdx = current ? ORDER.indexOf(current) : -1;
  const seen = new Set(nodeLog);

  const getNodeState = (nodeId: string): "completed" | "running" | "pending" => {
    if (stage === "idle") return "pending";
    if (stage === "step2_done") return "completed";

    if (stage === "step1_running") {
      if (current === nodeId) return "running";
      if (seen.has(nodeId)) return "completed";
      return nodeId === "loader" ? "running" : "pending"; // before the first update arrives
    }

    if (stage === "step1_review") {
      // the graph is paused at the interrupt: the human IS the checkpoint
      if (nodeId === "checkpoint") return "running";
      if (nodeId === "loader" || nodeId === "scanner") return "completed";
      return "pending";
    }

    // step2_running
    if (nodeLog.length === 0) {
      if (nodeId === "improver") return "running"; // resume segment starts here
      return isStep1Node(nodeId) ? "completed" : "pending";
    }
    if (current === nodeId) return "running";
    if (seen.has(nodeId)) return "completed";

    if (isStep1Node(nodeId)) {
      // Plain resume: segment 1 is never re-announced -> it is history.
      // Fresh re-run: loader/scanner arrive again, so only step-1 nodes
      // behind the last arrival in canonical order count as done.
      if (seen.has("loader") || seen.has("scanner")) {
        return ORDER.indexOf(nodeId) < currentIdx ? "completed" : "pending";
      }
      return "completed";
    }
    // Fan-out pair runs in parallel: without a completion event for an
    // unseen node it stays honest-pending until its own arrival.
    return "pending";
  };

  return (
    <div className="w-full bg-[#10131c] border border-[#1e2436] rounded-2xl p-5 mb-8">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-4 mb-4 border-b border-[#1e2436]">
        <div>
          <h2 className="text-sm font-semibold text-[#f8fafc] tracking-tight">Agentic Pipeline Flow</h2>
          <p className="text-xs text-[#94a3b8] mt-0.5">Two-stage execution with LangGraph state checkpointing</p>
        </div>
        {statusMessage && (
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-[#151926] border border-[#1e2436] text-xs font-mono text-[#06b6d4]">
            <span className="w-2 h-2 rounded-full bg-[#06b6d4] animate-pulse" />
            <span className="truncate max-w-[280px]">{statusMessage}</span>
          </div>
        )}
      </div>

      {/* Node stages */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-9 gap-2">
        {NODES.map((node) => {
          const state = getNodeState(node.id);
          const isCompleted = state === "completed";
          const isRunning = state === "running";

          return (
            <div
              key={node.id}
              className={`p-3 rounded-xl border transition-all flex flex-col gap-1.5 ${
                isRunning
                  ? "bg-[#06b6d4]/10 border-[#06b6d4]/40 shadow-[0_0_12px_rgba(6,182,212,0.1)]"
                  : isCompleted
                  ? "bg-[#151926] border-[#1e2436] text-[#f8fafc]"
                  : "bg-[#090a0f] border-[#1e2436]/50 text-[#64748b]"
              }`}
            >
              <div className="flex items-center justify-between">
                <span className="text-[10px] font-mono uppercase tracking-wider text-[#94a3b8]">
                  Step {node.stepNum}
                </span>
                {isCompleted ? (
                  <CheckCircle weight="fill" className="w-4 h-4 text-emerald-400" />
                ) : isRunning ? (
                  <CircleNotch weight="bold" className="w-4 h-4 text-[#06b6d4] animate-spin" />
                ) : (
                  <Circle weight="regular" className="w-4 h-4 text-[#64748b]/40" />
                )}
              </div>
              <span
                className={`text-xs font-medium truncate ${
                  isRunning ? "text-[#06b6d4] font-semibold" : isCompleted ? "text-[#f8fafc]" : "text-[#64748b]"
                }`}
              >
                {node.label}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
};
