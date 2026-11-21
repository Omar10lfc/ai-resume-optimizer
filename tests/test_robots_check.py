"""Offline tests for agent/robots_check.py.

Ported from the MIT ai-job-search repo's tests/test_robots_check.py
(adapted): no network anywhere - every case exercises the parser against
literal robots.txt bodies, matching the repo's CI policy of making no live
portal requests. The GHSA-xhqf-8q8x-4jwm backtracking regression is pinned.
"""
import time

import pytest

from agent import robots_check
from agent.robots_check import allowed, gate, has_undecodable_rule, is_robots_body

# Real body served by privatebank.barclays.com: blank lines sit between the
# User-agent line and its rules. Python's robotparser treats those as record
# separators and drops every rule, so /cs/ reads as allowed.
BARCLAYS = "User-agent: *\n\n\nAllow: /\n\nDisallow: /cs/\n\nSitemap: https://x/sitemap.xml\n"

JOBUP = "User-agent: *\nDisallow: /api/\n"


# ==========================================
# Path rules (allowed())
# ==========================================

class TestPathRules:
    def test_blank_lines_inside_record_do_not_end_it(self):
        assert not allowed(BARCLAYS, "*", "/cs/")

    def test_allowed_path_on_same_site_still_allowed(self):
        assert allowed(BARCLAYS, "*", "/careers/")

    def test_longest_match_wins_over_rule_order(self):
        body = "User-agent: *\nAllow: /\nDisallow: /cs/\n"
        assert not allowed(body, "*", "/cs/deep/page")

    def test_longest_match_can_unblock(self):
        body = "User-agent: *\nDisallow: /\nAllow: /jobs/\n"
        assert allowed(body, "*", "/jobs/x")
        assert not allowed(body, "*", "/other")

    def test_equal_specificity_tie_goes_to_disallow(self):
        assert not allowed("User-agent: *\nDisallow: /a\nAllow: /a\n", "*", "/a")

    def test_equal_specificity_tie_goes_to_disallow_when_allow_listed_first(self):
        assert not allowed("User-agent: *\nAllow: /a\nDisallow: /a\n", "*", "/a")

    def test_api_block_and_sibling_path(self):
        assert not allowed(JOBUP, "*", "/api/v1/public/search")
        assert allowed(JOBUP, "*", "/en/jobs/")

    def test_wildcard_and_end_anchor(self):
        body = "User-agent: *\nDisallow: /*.pdf$\n"
        assert not allowed(body, "*", "/files/cv.pdf")
        assert allowed(body, "*", "/files/cv.pdf.html")

    def test_dollar_sign_mid_pattern_is_literal_not_end_anchor(self):
        body = "User-agent: *\nDisallow: /tag/$var\n"
        assert not allowed(body, "*", "/tag/$var")
        assert not allowed(body, "*", "/tag/$var/sub")

    def test_repeated_wildcards_do_not_catastrophically_backtrack(self):
        # GHSA-xhqf-8q8x-4jwm: every '*' became a regex '.*', so a rule with
        # repeated wildcard segments that does not match took exponential time
        # in Python's backtracking engine (16 segments: minutes). The matcher
        # must run in O(len(pattern) * len(path)) whatever the rule says.
        pat = "/" + "*a" * 13 + "*b$"
        body = f"User-agent: *\nDisallow: {pat}\n"
        path = "/" + "a" * 28 + "X"
        start = time.perf_counter()
        res = allowed(body, "*", path)
        elapsed = time.perf_counter() - start
        assert res
        assert elapsed < 0.5, f"_match took {elapsed:.2f}s (catastrophic backtracking)"

    def test_consecutive_wildcards_match_identically_to_single_wildcard(self):
        body = "User-agent: *\nDisallow: /docs/*******.pdf$\n"
        assert not allowed(body, "*", "/docs/cv.pdf")
        assert not allowed(body, "*", "/docs/sub/folder/cv.pdf")
        assert allowed(body, "*", "/docs/sub/folder/cv.html")

    def test_wildcard_semantics_survive_the_matcher_change(self):
        body = (
            "User-agent: *\n"
            "Disallow: /a*b\n"
            "Disallow: /x/*/y$\n"
            "Disallow: /end*$\n"
        )
        assert not allowed(body, "*", "/ab")
        assert not allowed(body, "*", "/a/long/path/b/and-more")
        assert allowed(body, "*", "/a")
        assert not allowed(body, "*", "/x/anything/at/all/y")
        assert allowed(body, "*", "/x/anything/y/more")
        assert not allowed(body, "*", "/end")
        assert not allowed(body, "*", "/ending/here")
        assert allowed(body, "*", "/en")

    def test_percent_encoded_query_string_matches_rule_in_gate(self, monkeypatch):
        body = "User-agent: *\nDisallow: /search?q=machine%20learning\n"
        monkeypatch.setattr(robots_check, "_fetch", lambda url, ua: (body, 200))
        rc, msg = gate("https://example.com/search?q=machine%20learning")
        assert rc == 1
        assert "DISALLOWED" in msg

    def test_empty_disallow_means_allow_everything(self):
        assert allowed("User-agent: *\nDisallow:\n", "*", "/anything")

    def test_empty_or_ruleless_robots_allows(self):
        assert allowed("", "*", "/x")
        assert allowed("# just a comment\n", "*", "/x")

    def test_comments_are_stripped(self):
        assert not allowed("User-agent: *\nDisallow: /x  # nope\n", "*", "/x")


class TestAgentSelection:
    def test_named_claude_user_opt_out_is_honored(self):
        body = "User-agent: Claude-User\nDisallow: /\n\nUser-agent: *\nAllow: /\n"
        assert not allowed(body, "Claude-User", "/a")
        assert allowed(body, "*", "/a")

    def test_agent_match_is_case_insensitive(self):
        body = "User-agent: CLAUDE-USER\nDisallow: /x\n"
        assert not allowed(body, "claude-user", "/x")


# ==========================================
# is_robots_body / has_undecodable_rule
# ==========================================

class TestBodies:
    def test_empty_or_whitespace_body_is_a_valid_allow_all(self):
        assert is_robots_body("")
        assert is_robots_body("   \n")

    def test_comment_only_body_is_not_a_robots_body(self):
        assert not is_robots_body("# comment only\n")

    def test_html_error_page_is_not_a_robots_body(self):
        assert not is_robots_body("<!DOCTYPE html><html>404 not found</html>")

    def test_leading_bom_is_tolerated(self):
        assert is_robots_body("\ufeffUser-agent: *\nDisallow: /\n")

    def test_undecodable_rule_lines_are_rejected(self):
        assert has_undecodable_rule("User-agent: *\nDisallow: /\ufffd\n")
        assert not has_undecodable_rule("User-agent: *\nDisallow: /x  # \ufffd in comment\n")
        assert not has_undecodable_rule("User-agent: *\nDisallow: /x\n")


# ==========================================
# gate() network behavior (fake _fetch)
# ==========================================

class TestGate:
    def test_404_is_allowed(self, monkeypatch):
        monkeypatch.setattr(robots_check, "_fetch", lambda url, ua: ("", 404))
        rc, msg = gate("https://example.com/job")
        assert rc == 0
        assert "ALLOWED - no robots.txt published" in msg

    def test_disallow_star_blocks(self, monkeypatch):
        body = "User-agent: *\nDisallow: /\n"
        monkeypatch.setattr(robots_check, "_fetch", lambda url, ua: (body, 200))
        rc, msg = gate("https://example.com/job")
        assert rc == 1 and "DISALLOWED" in msg

    def test_fetch_failure_is_unconfirmed(self, monkeypatch):
        def _boom(url, ua):
            raise RuntimeError("connection reset")
        monkeypatch.setattr(robots_check, "_fetch", _boom)
        rc, msg = gate("https://example.com/job")
        assert rc == 1 and "UNCONFIRMED" in msg

    def test_non_robots_200_is_unconfirmed(self, monkeypatch):
        monkeypatch.setattr(robots_check, "_fetch",
                            lambda url, ua: ("<html>404 not found</html>", 200))
        rc, msg = gate("https://example.com/job")
        assert rc == 1 and "UNCONFIRMED" in msg

    def test_undecodable_rule_is_unconfirmed(self, monkeypatch):
        monkeypatch.setattr(robots_check, "_fetch",
                            lambda url, ua: ("User-agent: *\nDisallow: /\ufffd\n", 200))
        rc, msg = gate("https://example.com/job")
        assert rc == 1 and "UNCONFIRMED" in msg

    def test_unsupported_scheme_is_unconfirmed(self, monkeypatch):
        rc, msg = gate("ftp://example.com/job")
        assert rc == 1 and "UNCONFIRMED" in msg

    def test_permitted_path_is_allowed(self, monkeypatch):
        body = "User-agent: *\nAllow: /careers/\nDisallow: /\n"
        monkeypatch.setattr(robots_check, "_fetch", lambda url, ua: (body, 200))
        rc, msg = gate("https://example.com/careers/123")
        assert rc == 0 and "permits" in msg
