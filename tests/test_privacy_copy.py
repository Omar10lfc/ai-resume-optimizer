"""Item 3 (follow-up) gap test for commit 39a1fd7: the privacy copy is
factual - old false claims are gone, new truthful copy is present, and the
web privacy & terms page names the provider and states that names, emails
and phone numbers are sent to it unredacted."""

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FRONT = REPO / "frontend" / "src"


def test_privacy_copy_is_factual():
    onboarding = (FRONT / "components" / "Onboarding.tsx").read_text(encoding="utf-8")
    app = (FRONT / "App.tsx").read_text(encoding="utf-8")
    privacy_page = (FRONT / "PrivacyTermsPage.tsx").read_text(encoding="utf-8")
    # Old false claims are gone.
    for gone in ("Your data stays with you", "never synced anywhere",
                 "stored only in your browser",
                 "Resume text stays in this browser"):
        assert gone not in onboarding, gone
        assert gone not in app, gone
    # New factual copy is present.
    assert "Your data stays local" in onboarding
    assert "model provider" in onboarding
    assert "Generated text is reviewable AI output" in onboarding
    assert "Local app · model provider processes data" in app
    assert "Privacy & terms" in app
    # Web privacy & terms page: reachable, names the provider, states the
    # unredacted sending of names/emails/phone numbers, has limitations.
    assert "PrivacyTermsPage" in app
    assert "Groq" in privacy_page
    # Normalize the JSX whitespace for the (wrapped) assertion strings.
    normalized_page = " ".join(privacy_page.split())
    assert ("names, emails and phone numbers are sent to the provider "
            "unredacted") in normalized_page
    assert "Limitations" in normalized_page
