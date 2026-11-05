import { scrapeCurrentPage, type ScrapedJob } from "./sites";

const BUTTON_ID = "resume-optimizer-trigger-btn";

function injectTriggerButton(job: ScrapedJob) {
  if (document.getElementById(BUTTON_ID)) return;

  const btn = document.createElement("button");
  btn.id = BUTTON_ID;
  btn.type = "button";
  btn.innerHTML = `
    <svg width="18" height="18" viewBox="0 0 256 256" fill="currentColor" style="display:inline-block; vertical-align:middle; margin-right:6px;">
      <path d="M216,40H40A16,16,0,0,0,24,56V200a16,16,0,0,0,16,16H216a16,16,0,0,0,16-16V56A16,16,0,0,0,216,40ZM216,200H40V56H216V200Zm-40-80a8,8,0,0,1-8,8H88a8,8,0,0,1,0-16h80A8,8,0,0,1,176,120Zm0,32a8,8,0,0,1-8,8H88a8,8,0,0,1,0-16h80A8,8,0,0,1,176,152Zm-48-64a8,8,0,0,1-8,8H88a8,8,0,0,1,0-16h32A8,8,0,0,1,128,88Z"/>
    </svg>
    <span>Optimize Resume</span>
  `;

  // Inline CSS to ensure host styles don't distort it
  Object.assign(btn.style, {
    position: "fixed",
    bottom: "24px",
    right: "24px",
    zIndex: "2147483647",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    padding: "10px 18px",
    borderRadius: "9999px",
    backgroundColor: "#06b6d4",
    color: "#090a0f",
    fontFamily: "system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
    fontSize: "13px",
    fontWeight: "600",
    letterSpacing: "-0.01em",
    border: "1px solid rgba(255, 255, 255, 0.2)",
    boxShadow: "0 8px 24px -4px rgba(6, 182, 212, 0.4), 0 4px 12px rgba(0, 0, 0, 0.3)",
    cursor: "pointer",
    transition: "all 0.2s cubic-bezier(0.16, 1, 0.3, 1)",
    transform: "translateY(0)",
  });

  btn.addEventListener("mouseenter", () => {
    btn.style.transform = "translateY(-2px)";
    btn.style.boxShadow = "0 12px 28px -4px rgba(6, 182, 212, 0.5), 0 6px 16px rgba(0, 0, 0, 0.4)";
  });

  btn.addEventListener("mouseleave", () => {
    btn.style.transform = "translateY(0)";
    btn.style.boxShadow = "0 8px 24px -4px rgba(6, 182, 212, 0.4), 0 4px 12px rgba(0, 0, 0, 0.3)";
  });

  btn.addEventListener("click", () => {
    const freshJob = scrapeCurrentPage() || job;

    // 1. Store in chrome.storage so side panel can read it on mount
    if (typeof chrome !== "undefined" && chrome.storage && chrome.storage.local) {
      chrome.storage.local.set({ pendingScrapedJob: freshJob });
    }

    // 2. Broadcast to background service worker and open side panel
    if (typeof chrome !== "undefined" && chrome.runtime && chrome.runtime.sendMessage) {
      chrome.runtime.sendMessage({
        type: "JOB_EXTRACTED",
        job: freshJob,
      });
    }

    // Visual confirmation feedback
    const originalText = btn.innerHTML;
    btn.innerHTML = "✓ Opening Optimizer...";
    btn.style.backgroundColor = "#10b981";
    setTimeout(() => {
      btn.innerHTML = originalText;
      btn.style.backgroundColor = "#06b6d4";
    }, 2500);
  });

  document.body.appendChild(btn);
}

// Check for job posting after page load and mutation
function checkAndInject() {
  const job = scrapeCurrentPage();
  if (job && job.description.length > 200) {
    injectTriggerButton(job);
  }
}

// Initial check with delay for dynamic single-page applications (LinkedIn/Greenhouse)
setTimeout(checkAndInject, 1500);
setTimeout(checkAndInject, 3500);

// Observe DOM changes (e.g. user clicks another job on LinkedIn sidebar)
let lastUrl = window.location.href;
const observer = new MutationObserver(() => {
  if (window.location.href !== lastUrl) {
    lastUrl = window.location.href;
    setTimeout(checkAndInject, 1000);
  }
});

observer.observe(document.body, { childList: true, subtree: true });

// Listen for direct queries from the side panel
if (typeof chrome !== "undefined" && chrome.runtime && chrome.runtime.onMessage) {
  chrome.runtime.onMessage.addListener((request, _sender, sendResponse) => {
    if (request.type === "GET_ACTIVE_JOB") {
      const job = scrapeCurrentPage();
      sendResponse({ ok: !!job, job });
    }
  });
}

