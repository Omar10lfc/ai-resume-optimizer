/**
 * Background Service Worker (Manifest V3)
 * Manages the Side Panel lifecycle, cross-tab messaging, and storage synchronization.
 */

// Configure side panel to open when clicking the toolbar action icon
chrome.runtime.onInstalled.addListener(() => {
  if (chrome.sidePanel && chrome.sidePanel.setPanelBehavior) {
    chrome.sidePanel
      .setPanelBehavior({ openPanelOnActionClick: true })
      .catch((err) => console.warn("[Background] Failed to set side panel behavior:", err));
  }

  // Set default settings if not already present
  chrome.storage.local.get(["backendUrl"], (data) => {
    if (!data.backendUrl) {
      chrome.storage.local.set({
        backendUrl: "http://localhost:8000",
      });
    }
  });

  console.log("[Background] Job Application Optimizer extension installed.");
});

// Fallback: if setPanelBehavior failed to stick (or a future Chrome change),
// a plain action click must still open the panel. When openPanelOnActionClick
// is active, Chrome never dispatches onClicked, so this is dormant insurance.
chrome.action?.onClicked?.addListener((tab) => {
  if (!chrome.sidePanel?.open) return;
  const target = tab?.windowId != null
    ? { windowId: tab.windowId }
    : tab?.id != null ? { tabId: tab.id } : undefined;
  if (!target) return;
  chrome.sidePanel.open(target).catch((err) =>
    console.warn("[Background] Could not open side panel on action click:", err));
});

// Relay messages between content script and side panel
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type === "JOB_EXTRACTED" && message.job) {
    // 1. Cache the most recently scraped job
    chrome.storage.local.set({ pendingScrapedJob: message.job }, () => {
      // 2. Open side panel if tab ID is available
      if (sender.tab?.id && chrome.sidePanel && chrome.sidePanel.open) {
        chrome.sidePanel
          .open({ tabId: sender.tab.id })
          .catch((err) => console.warn("[Background] Could not open side panel programmatically:", err));
      }

      // 3. Notify any already-open side panel
      chrome.runtime.sendMessage({
        type: "JOB_DETECTED",
        job: message.job,
      }).catch(() => {
        // Ignored if side panel is not yet open
      });
    });

    sendResponse({ ok: true });
    return true;
  }
});

