"""Deterministic unevidenced-tool guard - reviewer half (run 4afa44cb)."""
import copy

import agent
from agent.tool_claims import (build_denied_tools, matching_tools,
                               remove_sentences_naming_tools,
                               sanitize_feedback)

JD = ("We're hiring an AI Automation Intern. You'll work with the FortyGuard "
      "Temperature MCP, build demos using Claude Code, Cursor, and Codex, and "
      "use n8n-style automations and REST APIs.")
RESUME = ("Alex Candidate\nAI/ML Engineer | alex@example.com\n\n"
          "## Experience\n- Built Python and FastAPI services\n"
          "- Deployed multi-agent workflows on AWS\n")
DECLINED = ["Experience with n8n",
            "Hands-on work using Claude Code, Cursor, or Codex",
            "Direct experience with MCP servers"]


def test_case_insensitive_word_boundary_variants():
    assert matching_tools("n8n-style, n8n like, n8n's automation") == {"n8n"}
    assert matching_tools("N8N workflows") == {"n8n"}
    assert matching_tools("temperature-MCP automation") == {"MCP"}
    assert matching_tools("model context protocol servers") == {"MCP"}
    # A tool name must not match inside a longer word.
    assert matching_tools("n8nish toolset") == set()


def test_multiword_names_are_units():
    found = matching_tools("Claude-Code, Cursor and Codex-based demos")
    assert found == {"Claude Code", "Cursor", "Codex"}
    assert matching_tools("langchain and langgraph agents") == {
        "LangChain", "LangGraph"}


def test_ambiguous_words_need_non_initial_capital_or_explicit_form():
    assert matching_tools("make a plan for the team") == set()
    assert matching_tools("database cursor implementation") == set()
    assert matching_tools("Make a plan.") == set()
    assert matching_tools("We integrated Make.com and a Cursor-based workflow") == {
        "Make", "Cursor"}


def test_evidenced_tool_is_never_denied():
    denied = build_denied_tools(
        "Experience with n8n and Claude Code.",
        declined_requests=DECLINED, job_text=JD)
    assert "n8n" not in denied
    assert "Claude Code" not in denied
    assert "MCP" in denied  # named by the posting, absent from the evidence


def test_denied_terms_come_from_declined_posting_and_vocabulary():
    denied = build_denied_tools(RESUME, declined_requests=DECLINED,
                                job_text=JD)
    assert {"n8n", "Claude Code", "Cursor", "Codex", "MCP"} <= denied
    assert "Python" not in denied  # not a tool in the maintained vocabulary


def test_sentence_removal_keeps_other_sentences_and_paragraphs():
    text = ("Use strong action verbs throughout.\n\n"
            "Link automation work to n8n-like platforms and Claude Code demos. "
            "Keep every true metric in the bullets.")
    cleaned, removals = remove_sentences_naming_tools(
        text, {"n8n", "Claude Code"}, stage="reviewer feedback")
    assert cleaned == ("Use strong action verbs throughout.\n\n"
                       "Keep every true metric in the bullets.")
    assert len(removals) == 1
    assert set(removals[0].terms) == {"Claude Code", "n8n"}
    assert "reviewer feedback" in removals[0].warning
    assert "n8n" in removals[0].warning


def test_sanitize_feedback_never_repeats_denied_term():
    cleaned, removals = sanitize_feedback(
        "Link your existing automation work to workflow orchestration "
        "and n8n-like platforms.", {"n8n"}, stage="reviewer feedback")
    assert "n8n" not in cleaned.lower()
    assert len(removals) == 1
    assert "n8n" in removals[0].warning  # logged, never fed back to the polisher


BAD_FEEDBACK = (
    "Strengthen the summary and project descriptions by explicitly linking "
    "your existing automation work to workflow orchestration and n8n-like "
    "platforms, using the same action-verb style as your quantified bullets.")


def test_reviewer_node_sanitizes_feedback_and_logs_warning(monkeypatch):
    class CapturingReviewer:
        def with_structured_output(self, schema):
            class Structured:
                def invoke(self, messages):
                    return agent.ReviewOutput(
                        score=85, feedback=BAD_FEEDBACK,
                        sections=["Professional Summary", "Projects"])
            return Structured()

    monkeypatch.setattr(agent.llms, "llm_strict", CapturingReviewer())
    from agent.nodes.reviewer import reviewer_node

    state = {
        "optimized_resume": "# Alex Candidate\n## Projects\n- Built agents\n",
        "resume_text": RESUME,
        "job_text": JD,
        "declined_requests": list(DECLINED),
        "human_notes": "",
    }
    snapshot = copy.deepcopy(state)
    out = reviewer_node(state)
    assert state == snapshot  # no mutation
    assert "n8n" not in out["feedback"].lower()
    assert "n8n-like" not in out["feedback"].lower()
    assert out["llm_quality_score"] == 85
    assert any("reviewer feedback" in w and "n8n" in w
               for w in out["guardrail_warnings"])
    assert any("n8n" in w for w in out["scrub_removals"])
    # Deterministic on identical inputs.
    assert reviewer_node(state)["feedback"] == out["feedback"]


def test_reviewer_keeps_feedback_when_tool_is_evidenced(monkeypatch):
    class CapturingReviewer:
        def with_structured_output(self, schema):
            class Structured:
                def invoke(self, messages):
                    return agent.ReviewOutput(score=80, feedback=BAD_FEEDBACK,
                                              sections=[])
            return Structured()

    monkeypatch.setattr(agent.llms, "llm_strict", CapturingReviewer())
    from agent.nodes.reviewer import reviewer_node
    state = {
        "optimized_resume": "# Alex Candidate\n## Projects\n- Built agents\n",
        "resume_text": RESUME + "\n- Used n8n for internal automations\n",
        "job_text": JD,
        "declined_requests": DECLINED,
        "human_notes": "",
    }
    out = reviewer_node(state)
    assert out["feedback"] == BAD_FEEDBACK
    assert out["guardrail_warnings"] == []
