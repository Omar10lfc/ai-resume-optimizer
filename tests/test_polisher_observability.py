"""Tests for Issue 4: polisher ThreadPoolExecutor must propagate contextvars
(so LangSmith spans & TokenUsageTracker callbacks record tokens/children) and
POLISHER_REASONING_EFFORT defaults to 'low' to prevent 60s+ reasoning stalls.
"""
import contextvars
import importlib

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda

import agent.config as cfg
import agent.llms as llms_mod
from agent.nodes import polisher as polisher_mod
from agent.provider import TokenUsageTracker


_PARENT_SPAN_VAR: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "test_parent_span_var", default=None
)


class _RecordingChatModel(BaseChatModel):
    """Minimal BaseChatModel capturing parent ContextVars across worker threads."""

    seen_spans: list[str | None] = []

    @property
    def _llm_type(self) -> str:
        return "recording-fake"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.seen_spans.append(_PARENT_SPAN_VAR.get())
        prompt_text = str(messages[-1].content)
        if "## Experience" in prompt_text.splitlines():
            content = "## Experience\n**Engineer** | Acme | 2024 -- Present\n- Built faster ETL."
        else:
            content = "## Professional Summary\nSenior Python engineer building reliable pipelines."
        msg = AIMessage(
            content=content,
            usage_metadata={"input_tokens": 100, "output_tokens": 40, "total_tokens": 140},
        )
        return ChatResult(generations=[ChatGeneration(message=msg)])


def test_polisher_threadpool_propagates_contextvars_and_tokens(monkeypatch):
    """Multi-section polisher ThreadPoolExecutor must copy contextvars into
    each worker so LangSmith parent spans and TokenUsageTracker propagate."""
    fake = _RecordingChatModel()
    monkeypatch.setattr(llms_mod, "llm_creative", fake)

    tracker = TokenUsageTracker()
    state = {
        "resume_text": (
            "# Alex Chen\nalex@example.com\n\n"
            "## Professional Summary\nPython engineer.\n\n"
            "## Experience\n**Engineer** | Acme | 2024 -- Present\n- Built ETL.\n"
        ),
        "optimized_resume": (
            "# Alex Chen\nalex@example.com\n\n"
            "## Professional Summary\nPython engineer.\n\n"
            "## Experience\n**Engineer** | Acme | 2024 -- Present\n- Built ETL.\n"
        ),
        "job_text": "## Requirements\n- Python pipelines\n",
        "human_notes": "",
        "missing_skills": "",
        "feedback": "Sharpen summary and quantify ETL throughput.",
        "feedback_sections": ["Professional Summary", "Experience"],
        "iteration": 1,
    }

    token = _PARENT_SPAN_VAR.set("parent-trace-xyz")
    try:
        RunnableLambda(polisher_mod.polisher_node).invoke(
            state,
            config={
                "callbacks": [tracker],
                "metadata": {"langgraph_node": "polisher"},
            },
        )
    finally:
        _PARENT_SPAN_VAR.reset(token)

    assert len(fake.seen_spans) == 2
    assert fake.seen_spans == ["parent-trace-xyz", "parent-trace-xyz"]
    totals = tracker.totals()
    assert totals["llm_calls"] == 2
    assert totals["total_tokens"] == 280
    tasks = tracker.task_totals()
    assert tasks["polisher"]["llm_calls"] == 2
    assert tasks["polisher"]["total_tokens"] == 280


def test_polisher_reasoning_effort_defaults_to_low(monkeypatch):
    """Default POLISHER_REASONING_EFFORT must be 'low' when unset so polisher
    does not stall for 60s+ on unbounded reasoning tokens."""
    monkeypatch.delenv("POLISHER_REASONING_EFFORT", raising=False)
    reloaded = importlib.reload(cfg)
    try:
        assert reloaded.POLISHER_REASONING_EFFORT == "low"
    finally:
        importlib.reload(cfg)
