import React, { useEffect, useRef, useState } from "react";
import { CheckCircle, CircleNotch, Circle, WarningCircle, Clock } from "@phosphor-icons/react";
import type { PipelineStage } from "../types";

interface StepTimelineProps {
  stage: PipelineStage;
  statusMessage: string;
  /** Graph node ids in arrival order, from status events carrying a node id. */
  nodeLog: string[];
  /** Epoch ms of the most recent status event, or null (drives staleness hint). */
  lastEventAt: number | null;
  /** Node the run failed on (last active when the error arrived), for state marking. */
  failedNode: string | null;
}

interface TimelineNode {
  id: string;
  label: string;
  stepNum: 1 | 2;
  /** Honest coarse estimate shown while waiting; static text, not a countdown. */
  hint: string;
}

// Full pipeline topology incl. the cover_letter / interview_prep fan-out.
// "checkpoint" is a pseudo-node: the graph INTERRUPTS before the improver and
// never announces it as a node update - the pipeline paused there during the
// step 1 review.
const NODES: TimelineNode[] = [
  { id: "loader", label: "Ingest docs", stepNum: 1, hint: "a few seconds" },
  { id: "scanner", label: "Gap analysis", stepNum: 1, hint: "usually under 1 min" },
  { id: "checkpoint", label: "Your review", stepNum: 1, hint: "waits for you" },
  { id: "improver", label: "Tailor resume", stepNum: 2, hint: "usually 1–3 min" },
  { id: "polisher", label: "Polish with review notes", stepNum: 2, hint: "second pass" },
  { id: "reviewer", label: "Quality audit", stepNum: 2, hint: "usually under 1 min" },
  { id: "ats_check", label: "ATS match", stepNum: 2, hint: "seconds" },
  { id: "cover_letter", label: "Cover letter", stepNum: 2, hint: "usually under 1 min" },
  { id: "interview_prep", label: "Interview prep", stepNum: 2, hint: "usually under 1 min" },
  { id: "pdf_exporter", label: "Export PDF", stepNum: 2, hint: "seconds" },
];

const ORDER = NODES.map((n) => n.id);
const isStep1Node = (id: string) =>
  id === "loader" || id === "scanner" || id === "checkpoint";

const STALE_MS = 90_000; // nudge only, never an error: long docs legitimately wait

function formatClock(ms: number): string {
  const total = Math.max(0, Math.floor(ms / 1000));
  const m = Math.floor(total / 60);
  const s = total % 60;
  return `${m}:${String(s).padStart(2, "0")}`;
}

export const StepTimeline: React.FC<StepTimelineProps> = ({
  stage,
  statusMessage,
  nodeLog,
  lastEventAt,
  failedNode,
}) => {
  const [now, setNow] = useState(() => Date.now());
  const runStartRef = useRef<number | null>(null);
  const [runStart, setRunStart] = useState<number | null>(null);
  const [lastDuration, setLastDuration] = useState<number | null>(null);

  // Timer window: from run start to done/failure — resets when the stage changes.
  useEffect(() => {
    if (stage === "step1_running" || stage === "step2_running") {
      if (runStartRef.current === null) {
        runStartRef.current = Date.now();
        setRunStart(runStartRef.current);
      }
      const t = setInterval(() => setNow(Date.now()), 1000);
      return () => clearInterval(t);
    }
    // leaving the running phase: freeze the total so the done view can show it
    if (runStartRef.current !== null) {
      setLastDuration(Date.now() - runStartRef.current);
    }
    runStartRef.current = null;
    setRunStart(null);
  }, [stage]);

  const isRunning = stage === "step1_running" || stage === "step2_running";
  const elapsed = isRunning
    ? runStart !== null
      ? now - runStart
      : 0
    : null;

  const stale =
    stage === "step1_running" || stage === "step2_running"
      ? lastEventAt !== null && now - lastEventAt > STALE_MS
      : false;
  // `stale` is checked after the running-phase gate; when a run finishes the
  // banner disappears with it — never a stale nag on a done screen.

  // Progress is derived from the node arrival log (server truth), NOT from
  // prose-sniffing status messages. The LAST arrival is the running node;
  // anything already announced stays completed (the improver <-> reviewer ->
  // ats_check correction loop re-announces nodes, which arrival history
  // handles naturally).
  const current = nodeLog.length > 0 ? nodeLog[nodeLog.length - 1] : null;
  const currentIdx = current ? ORDER.indexOf(current) : -1;
  const seen = new Set(nodeLog);

  type NodeState = "completed" | "running" | "pending" | "failed";
  const getNodeState = (nodeId: string): NodeState => {
    if (stage === "idle") return "pending";
    if (stage === "step2_done") return "completed";

    if (stage === "step1_running") {
      if (failedNode === nodeId) return "failed";
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
    if (failedNode === nodeId) return "failed";
    if (nodeLog.length === 0) {
      if (nodeId === "improver") return failedNode ? "failed" : "running"; // resume segment starts here
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

  const nodeIcon = (state: NodeState, className: string) => {
    switch (state) {
      case "completed":
        return <CheckCircle weight="fill" className={`${className} text-ok`} />;
      case "running":
        return <CircleNotch weight="bold" className={`${className} text-accent animate-spin-slow`} />;
      case "failed":
        return <WarningCircle weight="fill" className={`${className} text-danger`} />;
      default:
        return <Circle weight="regular" className={`${className} text-ink-3/50`} />;
    }
  };

  return (
    <div className="panel p-4 mb-6">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2.5 pb-3 mb-3.5 border-b border-line">
        <div>
          <h2 className="text-xs font-semibold text-ink">Agent pipeline</h2>
          <p className="text-2xs text-ink-3 mt-0.5">
            {stage === "step2_done"
              ? "All nodes finished"
              : stage === "idle"
              ? "Runs after you start Step 1"
              : "Streaming updates as each node starts"}
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          {elapsed !== null && (
            <span className="flex items-center gap-1.5 px-2 py-1 rounded-md2 bg-inset border border-line text-2xs font-mono text-ink-2 tabular-nums">
              <Clock weight="bold" className="w-3 h-3 text-ink-3" />
              {formatClock(elapsed)}
            </span>
          )}
          {!isRunning && stage === "step2_done" && lastDuration !== null && (
            <span className="flex items-center gap-1.5 px-2 py-1 rounded-md2 bg-ok-tint border border-ok/25 text-2xs font-mono text-ok tabular-nums">
              <CheckCircle weight="bold" className="w-3 h-3" />
              Finished in {formatClock(lastDuration)}
            </span>
          )}
          {statusMessage && (stage === "step1_running" || stage === "step2_running") && (
            <span
              aria-live="polite"
              className="flex items-center gap-1.5 px-2 py-1 rounded-md2 bg-inset border border-line text-2xs font-mono text-accent max-w-full"
            >
              <span className="w-1.5 h-1.5 rounded-full bg-accent animate-pulse shrink-0" aria-hidden="true" />
              <span className="truncate">{statusMessage}</span>
            </span>
          )}
          {stale && (
            <span
              role="status"
              className="flex items-center gap-1.5 px-2 py-1 rounded-md2 bg-warn-tint border border-warn/30 text-2xs text-warn"
            >
              <CircleNotch weight="bold" className="w-3 h-3" />
              No updates for a while — long documents can take up to ~10 min total
            </span>
          )}
        </div>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
        {NODES.map((node) => {
          const state = getNodeState(node.id);
          const isRunningNow = state === "running";
          return (
            <div
              key={node.id}
              aria-current={isRunningNow ? "step" : undefined}
              className={`p-2.5 rounded-md2 border transition-colors flex flex-col gap-1 ${
                isRunningNow
                  ? "bg-accent-tint border-accent/40"
                  : state === "failed"
                  ? "bg-danger-tint border-danger/40"
                  : state === "completed"
                  ? "bg-raised border-line"
                  : "bg-inset border-line/50"
              }`}
            >
              <div className="flex items-center justify-between gap-1">
                <span className="text-2xs font-mono text-ink-3">Step {node.stepNum}</span>
                {nodeIcon(state, "w-3.5 h-3.5 shrink-0")}
              </div>
              <span
                className={`text-2xs font-medium truncate ${
                  isRunningNow ? "text-accent font-semibold" : state === "failed" ? "text-danger font-medium" : state === "completed" ? "text-ink" : "text-ink-3"
                }`}
              >
                {node.label}
              </span>
              <span className="text-2xs text-ink-3/80 leading-tight">{node.hint}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
};
