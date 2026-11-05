/**
 * Site-specific selectors and DOM extraction rules for popular job boards.
 */

export interface ScrapedJob {
  title: string;
  company: string;
  location?: string;
  description: string;
  url: string;
  source: string;
}

export interface SiteRule {
  name: string;
  matches: (url: string) => boolean;
  extract: () => ScrapedJob | null;
}

function cleanText(text: string | null | undefined): string {
  if (!text) return "";
  return text
    .replace(/\r\n/g, "\n")
    .replace(/\t/g, " ")
    .replace(/[ \u00a0]+/g, " ")
    .replace(/\n\s*\n\s*\n+/g, "\n\n")
    .trim();
}

export const SITE_RULES: SiteRule[] = [
  // 1. LinkedIn
  {
    name: "LinkedIn",
    matches: (url) => url.includes("linkedin.com/jobs"),
    extract: () => {
      const titleElem =
        document.querySelector(".job-details-jobs-unified-top-card__job-title") ||
        document.querySelector("h1.t-24") ||
        document.querySelector(".jobs-unified-top-card__job-title") ||
        document.querySelector("h1");

      const companyElem =
        document.querySelector(".job-details-jobs-unified-top-card__company-name") ||
        document.querySelector(".jobs-unified-top-card__company-name") ||
        document.querySelector(".artdeco-entity-lockup__subtitle");

      const descElem =
        document.querySelector(".jobs-description__content") ||
        document.querySelector("#job-details") ||
        document.querySelector(".jobs-box__html-content") ||
        document.querySelector(".jobs-description");

      const locationElem =
        document.querySelector(".job-details-jobs-unified-top-card__bullet") ||
        document.querySelector(".jobs-unified-top-card__bullet");

      let description = cleanText(descElem?.textContent || "");
      if (!description) return null;

      // Strip LinkedIn upsell / promotional footers
      description = description
        .replace(/Job search faster with Premium[\s\S]*/i, "")
        .replace(/Access company insights like strategic priorities[\s\S]*/i, "")
        .trim();

      return {
        title: cleanText(titleElem?.textContent || "Target Role"),
        company: cleanText(companyElem?.textContent || "Hiring Company"),
        location: cleanText(locationElem?.textContent || ""),
        description,
        url: window.location.href,
        source: "LinkedIn",
      };
    },
  },

  // 2. Greenhouse
  {
    name: "Greenhouse",
    matches: (url) => url.includes("greenhouse.io"),
    extract: () => {
      const titleElem =
        document.querySelector("h1.app-title") ||
        document.querySelector(".job-name") ||
        document.querySelector("h1");

      const companyElem =
        document.querySelector(".company-name") ||
        document.querySelector(".sub-header");

      const descElem =
        document.querySelector("#content") ||
        document.querySelector(".body") ||
        document.querySelector("#app-body");

      let description = cleanText(descElem?.textContent || "");
      if (!description) return null;

      // Strip LinkedIn upsell / promotional footers
      description = description
        .replace(/Job search faster with Premium[\s\S]*/i, "")
        .replace(/Access company insights like strategic priorities[\s\S]*/i, "")
        .trim();

      return {
        title: cleanText(titleElem?.textContent || "Target Role"),
        company: cleanText(companyElem?.textContent || "Company"),
        description,
        url: window.location.href,
        source: "Greenhouse",
      };
    },
  },

  // 3. Lever
  {
    name: "Lever",
    matches: (url) => url.includes("jobs.lever.co"),
    extract: () => {
      const titleElem =
        document.querySelector(".posting-headline h2") ||
        document.querySelector("h2");

      const companyElem = document.querySelector(".main-header-logo img");
      const company = companyElem?.getAttribute("alt") || "Company";

      const descElem =
        document.querySelector(".posting-page") ||
        document.querySelector(".section-page") ||
        document.querySelector(".content");

      let description = cleanText(descElem?.textContent || "");
      if (!description) return null;

      // Strip LinkedIn upsell / promotional footers
      description = description
        .replace(/Job search faster with Premium[\s\S]*/i, "")
        .replace(/Access company insights like strategic priorities[\s\S]*/i, "")
        .trim();

      return {
        title: cleanText(titleElem?.textContent || "Target Role"),
        company: cleanText(company),
        description,
        url: window.location.href,
        source: "Lever",
      };
    },
  },

  // 4. Indeed
  {
    name: "Indeed",
    matches: (url) => url.includes("indeed.com"),
    extract: () => {
      const titleElem =
        document.querySelector("h1.jobsearch-JobInfoHeader-title") ||
        document.querySelector("h1");

      const companyElem =
        document.querySelector('[data-testid="inlineHeader-companyName"]') ||
        document.querySelector(".jobsearch-CompanyInfoContainer");

      const descElem =
        document.querySelector("#jobDescriptionText") ||
        document.querySelector(".jobsearch-jobDescriptionText");

      let description = cleanText(descElem?.textContent || "");
      if (!description) return null;

      // Strip LinkedIn upsell / promotional footers
      description = description
        .replace(/Job search faster with Premium[\s\S]*/i, "")
        .replace(/Access company insights like strategic priorities[\s\S]*/i, "")
        .trim();

      return {
        title: cleanText(titleElem?.textContent || "Target Role"),
        company: cleanText(companyElem?.textContent || "Company"),
        description,
        url: window.location.href,
        source: "Indeed",
      };
    },
  },

  // 5. Workday
  {
    name: "Workday",
    matches: (url) => url.includes("workday.com") || url.includes("myworkdayjobs.com"),
    extract: () => {
      const titleElem =
        document.querySelector('[data-automation-id="jobPostingHeader"]') ||
        document.querySelector("h1");

      const descElem =
        document.querySelector('[data-automation-id="jobPostingDescription"]') ||
        document.querySelector(".GWTCKEditor-Disabled");

      let description = cleanText(descElem?.textContent || "");
      if (!description) return null;

      // Strip LinkedIn upsell / promotional footers
      description = description
        .replace(/Job search faster with Premium[\s\S]*/i, "")
        .replace(/Access company insights like strategic priorities[\s\S]*/i, "")
        .trim();

      return {
        title: cleanText(titleElem?.textContent || "Target Role"),
        company: cleanText(document.title.split("-")[0] || "Company"),
        description,
        url: window.location.href,
        source: "Workday",
      };
    },
  },

  // 6. Ashby
  {
    name: "Ashby",
    matches: (url) => url.includes("ashbyhq.com"),
    extract: () => {
      const titleElem = document.querySelector("h1");
      const descElem =
        document.querySelector("div._description_") ||
        document.querySelector("div[class*='description']");

      let description = cleanText(descElem?.textContent || "");
      if (!description) return null;

      // Strip LinkedIn upsell / promotional footers
      description = description
        .replace(/Job search faster with Premium[\s\S]*/i, "")
        .replace(/Access company insights like strategic priorities[\s\S]*/i, "")
        .trim();

      return {
        title: cleanText(titleElem?.textContent || "Target Role"),
        company: cleanText(document.title.split("-")[1] || "Company"),
        description,
        url: window.location.href,
        source: "Ashby",
      };
    },
  },
];

/**
 * Fallback heuristic extractor when no specific site rule matches.
 * Looks for common container IDs/classes or the largest meaningful text block.
 */
export function extractGenericJob(): ScrapedJob | null {
  const candidates = [
    document.querySelector("article"),
    document.querySelector("[class*='job-description']"),
    document.querySelector("[id*='job-description']"),
    document.querySelector("[class*='description']"),
    document.querySelector("main"),
  ];

  for (const candidate of candidates) {
    if (candidate) {
      const text = cleanText(candidate.textContent || "");
      // Meaningful job description threshold (at least 300 characters)
      if (text.length > 300) {
        const titleElem = document.querySelector("h1") || document.querySelector("h2");
        return {
          title: cleanText(titleElem?.textContent || document.title),
          company: cleanText(document.title.split(/[-|–]/)[0] || "Hiring Company"),
          description: text,
          url: window.location.href,
          source: "Generic Web Scraper",
        };
      }
    }
  }

  return null;
}

/**
 * Main scraping entrypoint.
 */
export function scrapeCurrentPage(): ScrapedJob | null {
  const currentUrl = window.location.href;
  for (const rule of SITE_RULES) {
    if (rule.matches(currentUrl)) {
      const job = rule.extract();
      if (job) return job;
    }
  }
  return extractGenericJob();
}

