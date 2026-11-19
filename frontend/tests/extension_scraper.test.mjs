import { test } from "node:test";
import assert from "node:assert/strict";

import {
  SITE_RULES,
  extractGenericJob,
  scrapeCurrentPage,
} from "../../extension/src/sites.ts";

/** Minimal DOM element: only textContent + getAttribute are used by sites.ts. */
function el(textContent, attrs = {}) {
  return { textContent, getAttribute: (name) => attrs[name] ?? null };
}

/** Installs a selector-keyed fixture DOM. Selector priority is exercised by
 *  supplying several selectors from the same rule; querySelector returns the
 *  first one the rule asks for. */
function installDom({ url, title = "", selectors = {} }) {
  const elements = Object.fromEntries(
    Object.entries(selectors).map(([selector, value]) => [
      selector,
      typeof value === "string" ? el(value) : value,
    ]),
  );
  globalThis.window = { location: { href: url } };
  globalThis.document = {
    title,
    querySelector: (selector) => elements[selector] ?? null,
  };
}

const LONG_TEXT = ("Build Python services and REST APIs. ".repeat(12));

test("every site rule matches only its own board", () => {
  const cases = [
    ["LinkedIn", "https://www.linkedin.com/jobs/view/123"],
    ["Greenhouse", "https://boards.greenhouse.io/acme/jobs/123"],
    ["Lever", "https://jobs.lever.co/beta/123"],
    ["Indeed", "https://www.indeed.com/viewjob?jk=abc"],
    ["Workday", "https://acme.myworkdayjobs.com/en-US/careers/job/123"],
    ["Ashby", "https://jobs.ashbyhq.com/epsilon/123"],
  ];
  for (const [name, url] of cases) {
    const matches = SITE_RULES.filter((rule) => rule.matches(url)).map((r) => r.name);
    assert.deepEqual(matches, [name], `${url} should match only ${name}`);
  }
  assert.equal(
    SITE_RULES.some((r) => r.matches("https://www.linkedin.com/feed/")),
    false,
    "a LinkedIn feed URL is not a job posting",
  );
});

test("LinkedIn extracts title, company, location and strips upsell footer", () => {
  installDom({
    url: "https://www.linkedin.com/jobs/view/123",
    selectors: {
      ".job-details-jobs-unified-top-card__job-title": "AI Automation Intern",
      ".job-details-jobs-unified-top-card__company-name": "FortyGuard",
      ".job-details-jobs-unified-top-card__bullet": "Remote",
      ".jobs-description__content":
        "Build AI agents and workflows.\nJob search faster with Premium.\n" +
        "Access company insights like strategic priorities at Fortune 500 companies.",
    },
  });
  const job = scrapeCurrentPage();
  assert.equal(job.source, "LinkedIn");
  assert.equal(job.title, "AI Automation Intern");
  assert.equal(job.company, "FortyGuard");
  assert.equal(job.location, "Remote");
  assert.equal(job.description, "Build AI agents and workflows.");
  assert.equal(job.url, "https://www.linkedin.com/jobs/view/123");
});

test("LinkedIn honours selector priority (specific h1 before generic h1)", () => {
  installDom({
    url: "https://www.linkedin.com/jobs/view/123",
    selectors: {
      "h1.t-24": "Specific Title",
      h1: "Generic Title",
      ".jobs-description": "Design agents and demos for the team.",
    },
  });
  assert.equal(scrapeCurrentPage().title, "Specific Title");
});

test("Greenhouse, Lever, Indeed, Workday and Ashby extract their fields", () => {
  const cases = [
    {
      url: "https://boards.greenhouse.io/acme/jobs/123",
      title: "Senior Engineer",
      company: "Acme",
      source: "Greenhouse",
      selectors: {
        "h1.app-title": "Senior Engineer",
        ".company-name": "Acme",
        "#content": "Own Python services.",
      },
    },
    {
      url: "https://jobs.lever.co/beta/123",
      title: "Data Engineer",
      company: "Beta",
      source: "Lever",
      selectors: {
        ".posting-headline h2": "Data Engineer",
        ".main-header-logo img": el("", { alt: "Beta" }),
        ".posting-page": "Design ETL pipelines.",
      },
    },
    {
      url: "https://www.indeed.com/viewjob?jk=abc",
      title: "Backend Developer",
      company: "Gamma",
      source: "Indeed",
      selectors: {
        "h1.jobsearch-JobInfoHeader-title": "Backend Developer",
        '[data-testid="inlineHeader-companyName"]': "Gamma",
        "#jobDescriptionText": "Ship FastAPI services.",
      },
    },
    {
      url: "https://acme.myworkdayjobs.com/en-US/careers/job/123",
      title: "Platform Engineer",
      company: "Delta",
      source: "Workday",
      titleTag: "Delta - Platform Engineer - Remote",
      selectors: {
        '[data-automation-id="jobPostingHeader"]': "Platform Engineer",
        '[data-automation-id="jobPostingDescription"]': "Build Kubernetes platforms.",
      },
    },
    {
      url: "https://jobs.ashbyhq.com/epsilon/123",
      title: "ML Engineer",
      company: "Epsilon",
      source: "Ashby",
      titleTag: "ML Engineer - Epsilon",
      selectors: {
        h1: "ML Engineer",
        "div._description_": "Train models.",
      },
    },
  ];
  for (const c of cases) {
    installDom({ url: c.url, title: c.titleTag ?? "", selectors: c.selectors });
    const job = scrapeCurrentPage();
    assert.equal(job.source, c.source, c.url);
    assert.equal(job.title, c.title, c.url);
    assert.equal(job.company, c.company, c.url);
    assert.ok(job.description.length > 0, c.url);
    assert.equal(job.url, c.url);
  }
});

test("a site rule with no description falls through to the generic fallback", () => {
  installDom({
    url: "https://www.linkedin.com/jobs/view/123",
    title: "Fallback Co - Intern",
    selectors: { article: LONG_TEXT },
  });
  const job = scrapeCurrentPage();
  assert.equal(job.source, "Generic Web Scraper");
  assert.ok(job.description.length > 300);
});

test("generic fallback needs a meaningful text block", () => {
  installDom({ url: "https://example.com/jobs/1", selectors: { article: "Too short" } });
  assert.equal(extractGenericJob(), null);
});

test("generic fallback skips non-matching LinkedIn feed URLs", () => {
  installDom({
    url: "https://www.linkedin.com/feed/",
    title: "LinkedIn",
    selectors: { article: LONG_TEXT },
  });
  const job = scrapeCurrentPage();
  assert.equal(job.source, "Generic Web Scraper");
  assert.equal(job.url, "https://www.linkedin.com/feed/");
});

test("site rules strip promotional footers from scraped descriptions", () => {
  installDom({
    url: "https://boards.greenhouse.io/acme/jobs/123",
    selectors: {
      "#content": "Own Python services.\nJob search faster with Premium.\nMore noise.",
    },
  });
  const description = scrapeCurrentPage().description;
  assert.equal(description, "Own Python services.");
});
