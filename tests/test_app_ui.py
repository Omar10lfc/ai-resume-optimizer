"""UI-layer tests: streaming generator arity, button state handling, and the
session checkpoint flow.

Guards against the Gradio failure mode where a yield tuple's length doesn't
match the declared outputs (silently breaks the UI) and where buttons stay
disabled after an error.
"""

import os
import sys
from pathlib import Path

import pytest

os.environ["LANGCHAIN_TRACING_V2"] = "false"  # keep test runs out of LangSmith
os.environ.setdefault("GROQ_API_KEY", "gsk_test_dummy_key_for_tests")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gradio as gr  # noqa: E402
import app  # noqa: E402


STEP1_OUTPUTS = 5    # btn1, btn2, session, notes_in, status_out
STEP2_OUTPUTS = 14   # btn1, btn2, session, 8 result components, pdf, docx, tex, status_out


class FakeState:
    def __init__(self):
        self.values = {}
        self.next = ()


class FakeAgentApp:
    """Simulates the interrupt/checkpoint flow of the interactive graph."""

    def __init__(self):
        self._states = {}
        self.fail = False

    def _state(self, config):
        tid = config["configurable"]["thread_id"]
        if tid not in self._states:
            self._states[tid] = FakeState()
        return self._states[tid]

    def stream(self, inputs, config=None, stream_mode=None, subgraphs=None):
        st = self._state(config or {})
        if self.fail:
            raise RuntimeError("provider down")
        if inputs is not None:  # first segment: loader → scanner → interrupt
            st.values.update(inputs)
            st.values["missing_skills"] = "- Kubernetes"
            st.values["request_id"] = "fake-rid"
            yield {"loader": {"job_text": "jd", "resume_text": "cv"}}
            yield {"scanner": {"missing_skills": "- Kubernetes", "iteration": 0}}
            st.next = ("improver",)
        else:  # resume: improver → ... → pdf_exporter → done
            st.values.update({
                "optimized_resume": "## Resume",
                "score": 82, "feedback": "Good", "review_failed": False,
                "ats_result": "## ATS", "ats_percentage": 90.0,
                "cover_letter": "Dear...", "interview_questions": "1. Q",
                "resume_pdf_path": "", "cover_letter_pdf_path": "",
                "iteration": 1, "output_dir": "unused",
            })
            yield {"improver": {"optimized_resume": "## Resume", "iteration": 1}}
            yield {"reviewer": {"score": 82, "feedback": "Good",
                                "llm_quality_score": 85, "review_failed": False}}
            yield {"ats_check": {"ats_result": "## ATS", "ats_percentage": 90.0,
                                 "score": 88}}
            yield {"cover_letter": {"cover_letter": "Dear..."}}
            yield {"interview_prep": {"interview_questions": "1. Q"}}
            yield {"pdf_exporter": {"resume_pdf_path": "",
                                    "cover_letter_pdf_path": "",
                                    "output_dir": "unused"}}
            st.next = ()

    def get_state(self, config):
        return self._state(config or {})

    def update_state(self, config, values):
        st = self._state(config or {})
        st.values.update(values)
        return st


@pytest.fixture
def fake_agent(monkeypatch):
    fake = FakeAgentApp()
    monkeypatch.setattr(app, "agent_app", fake)
    return fake


@pytest.fixture
def valid_inputs():
    return ("Python dev job posting", None, "Python resume text", "- Kubernetes", None)


def test_step1_streams_nodes_and_parks_at_interrupt(fake_agent, valid_inputs):
    yields = list(app.step1_analyze(*valid_inputs[:4]))
    assert len(yields) >= 2
    for y in yields:
        assert len(y) == STEP1_OUTPUTS, f"step1 yield has {len(y)} items, expected {STEP1_OUTPUTS}"
    final = yields[-1]
    assert final[0].get("interactive") is True and final[1].get("interactive") is True
    assert final[2]  # session thread id returned
    assert "Kubernetes" in final[3]
    # streaming statuses included per-node updates
    statuses = " ".join(y[4] for y in yields)
    assert "Scanning for skill gaps" in statuses


def test_step2_resumes_session_and_streams(fake_agent, valid_inputs):
    # Step 1 first so the session is parked at the interrupt
    s1 = list(app.step1_analyze(*valid_inputs[:4]))
    session = s1[-1][2]

    yields = list(app.step2_optimize(*valid_inputs[:4], session))
    assert len(yields) >= 2
    for y in yields:
        assert len(y) == STEP2_OUTPUTS, f"step2 yield has {len(y)} items, expected {STEP2_OUTPUTS}"
    final = yields[-1]
    assert final[0].get("interactive") is True and final[1].get("interactive") is True
    assert final[3] == "## Resume"
    assert "/100" in final[4]
    assert "composite" in final[13]
    statuses = " ".join(y[13] for y in yields)
    assert "Improving resume" in statuses and "Exporting PDFs" in statuses


def test_step2_without_step1_runs_full_pipeline(fake_agent, valid_inputs):
    """Skipping Step 1: the graph runs loader→scanner up to the interrupt,
    injects the notes, then resumes — all within Step 2."""
    yields = list(app.step2_optimize(*valid_inputs))
    final = yields[-1]
    assert len(final) == STEP2_OUTPUTS
    assert final[3] == "## Resume"
    # user notes injected at the checkpoint
    st = fake_agent._states[[k for k in fake_agent._states][0]]
    assert st.values.get("human_notes") == "- Kubernetes"


def test_step2_restarts_fresh_after_completed_run(fake_agent, valid_inputs):
    s1 = list(app.step1_analyze(*valid_inputs[:4]))
    session = s1[-1][2]
    list(app.step2_optimize(*valid_inputs[:4], session))  # completes the run
    yields = list(app.step2_optimize(*valid_inputs[:4], session))
    assert len(yields[-1]) == STEP2_OUTPUTS
    # a new thread id was issued for the fresh run
    assert yields[-1][2] != session


def test_step1_error_reenables_buttons(monkeypatch, valid_inputs):
    fake = FakeAgentApp()
    fake.fail = True
    monkeypatch.setattr(app, "agent_app", fake)
    gen = app.step1_analyze(*valid_inputs[:4])
    first = next(gen)
    assert len(first) == STEP1_OUTPUTS
    reenable = next(gen)  # error path yields re-enable before raising
    assert reenable[0].get("interactive") is True
    with pytest.raises(gr.Error):
        next(gen)


def test_step2_error_reenables_buttons(monkeypatch, valid_inputs):
    fake = FakeAgentApp()
    fake.fail = True
    monkeypatch.setattr(app, "agent_app", fake)
    gen = app.step2_optimize(*valid_inputs)
    first = next(gen)       # initial yield — disables buttons
    assert len(first) == STEP2_OUTPUTS
    reenable = next(gen)    # error path yields re-enable before raising
    assert reenable[0].get("interactive") is True
    with pytest.raises(gr.Error):
        next(gen)           # now the Error is raised


def test_score_emoji_aligned_with_gate_threshold():
    assert app._score_emoji(80) == "🟢"   # gate passes at 80 -> must be green
    assert app._score_emoji(60) == "🟡"
    assert app._score_emoji(59) == "🔴"
