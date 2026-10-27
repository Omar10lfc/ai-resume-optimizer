import React from "react";
import { CheckCircle, CircleNotch, Circle } from "@phosphor-icons/react";
import type { PipelineStage } from "../types";

interface StepTimelineProps {
  stage: PipelineStage;
  statusMessage: string;
}

interface TimelineNode {
  id: string;
  label: string;
  stepNum: number;
}

const NODES: TimelineNode[] = [
  { id: "loader", label: "Ingest Docs", stepNum: 1 },
  { id: "scanner", label: "Gap Analysis", stepNum: 1 },
  { id: "checkpoint", label: "Human Review", stepNum: 1 },
  { id: "improver", label: "Tailor Resume", stepNum: 2 },
  { id: "reviewer", label: "Quality Audit", stepNum: 2 },
  { id: "ats_check", label: "ATS Match", stepNum: 2 },
  { id: "pdf_exporter", label: "Export PDF", stepNum: 2 },
];

export const StepTimeline: React.FC<StepTimelineProps> = ({ stage, statusMessage }) => {
  // Determine state of each node based on the pipeline stage and status message
  const getNodeState = (nodeId: string): "completed" | "running" | "pending" => {
    if (stage === "idle") return "pending";

    if (stage === "step1_running") {
      if (statusMessage.toLowerCase().includes("scanning") || statusMessage.toLowerCase().includes("gap")) {
        if (nodeId === "loader") return "completed";
        if (nodeId === "scanner") return "running";
        return "pending";
      }
      if (nodeId === "loader") return "running";
      return "pending";
    }

    if (stage === "step1_review") {
      if (nodeId === "loader" || nodeId === "scanner") return "completed";
      if (nodeId === "checkpoint") return "running";
      return "pending";
    }

    if (stage === "step2_running") {
      if (nodeId === "loader" || nodeId === "scanner" || nodeId === "checkpoint") return "completed";

      const msg = statusMessage.toLowerCase();
      if (msg.includes("exporting") || msg.includes("pdf") || msg.includes("latex")) {
        if (["improver", "reviewer", "ats_check"].includes(nodeId)) return "completed";
        if (nodeId === "pdf_exporter") return "running";
        return "pending";
      }
      if (msg.includes("ats") || msg.includes("keyword")) {
        if (["improver", "reviewer"].includes(nodeId)) return "completed";
        if (nodeId === "ats_check") return "running";
        return "pending";
      }
      if (msg.includes("reviewing") || msg.includes("judging") || msg.includes("quality")) {
        if (nodeId === "improver") return "completed";
        if (nodeId === "reviewer") return "running";
        return "pending";
      }
      if (nodeId === "improver") return "running";
      return "pending";
    }

    if (stage === "step2_done") {
      return "completed";
    }

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
      <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-7 gap-2">
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
