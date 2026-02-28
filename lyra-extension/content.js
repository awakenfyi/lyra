/**
 * content.js — DOM Injector for Lyra Traffic Light
 *
 * Listens for coherence updates from the interceptor and renders
 * the traffic light indicator below each AI response in ChatGPT.
 *
 * Minimal, non-intrusive. A small colored tag that tells you
 * whether the model was confident or guessing.
 *
 * MIT License | awaken.fyi
 */

// Inject the interceptor into the main page context
// (content scripts run in an isolated world — the interceptor
// needs access to window.fetch in the page's world)
const script = document.createElement("script");
script.src = chrome.runtime.getURL("interceptor.js");
(document.head || document.documentElement).appendChild(script);

// Listen for coherence calculations from the interceptor
window.addEventListener("lyra_coherence_update", function (e) {
    const { status, confidence, mass, tokenCount } = e.detail;
    renderTrafficLight(status, confidence, mass, tokenCount);
});

function renderTrafficLight(status, confidence, mass, tokenCount) {
    // Find the latest AI response block in the DOM
    // ChatGPT uses various selectors — try multiple
    const selectors = [
        '[data-message-author-role="assistant"] .markdown',
        '.agent-turn .markdown',
        '.markdown',
    ];

    let responseBlocks = null;
    for (const selector of selectors) {
        responseBlocks = document.querySelectorAll(selector);
        if (responseBlocks.length > 0) break;
    }

    if (!responseBlocks || responseBlocks.length === 0) return;

    const latestResponse = responseBlocks[responseBlocks.length - 1];

    // Prevent duplicate injections
    if (latestResponse.querySelector(".lyra-indicator")) {
        // Update existing indicator instead
        const existing = latestResponse.querySelector(".lyra-indicator");
        applyStyle(existing, status, confidence, mass);
        return;
    }

    // Build the indicator element
    const indicator = document.createElement("div");
    indicator.className = "lyra-indicator";
    applyStyle(indicator, status, confidence, mass);
    latestResponse.appendChild(indicator);
}

function applyStyle(element, status, confidence, mass) {
    // Base styles
    element.style.padding = "6px 12px";
    element.style.marginTop = "12px";
    element.style.borderRadius = "6px";
    element.style.fontFamily = "'SF Mono', 'Fira Code', 'Consolas', monospace";
    element.style.fontSize = "11px";
    element.style.display = "inline-flex";
    element.style.alignItems = "center";
    element.style.gap = "6px";
    element.style.lineHeight = "1.4";
    element.style.userSelect = "none";
    element.style.transition = "opacity 0.3s ease";

    const confPct = Math.round(confidence * 100);
    const massPct = Math.round(mass * 100);

    if (status === "GREEN") {
        element.style.backgroundColor = "rgba(0, 200, 83, 0.08)";
        element.style.border = "1px solid rgba(0, 200, 83, 0.2)";
        element.style.color = "#00C853";
        element.textContent = `\u25CF Lyra: ${confPct}% confident \u00B7 committed to output`;
    } else if (status === "YELLOW") {
        element.style.backgroundColor = "rgba(255, 214, 0, 0.08)";
        element.style.border = "1px solid rgba(255, 214, 0, 0.2)";
        element.style.color = "#B8860B";
        element.textContent = `\u25CF Lyra: ${confPct}% confident \u00B7 verify specific claims`;
        if (mass < 0.30) {
            element.textContent += ` \u00B7 scattered (${massPct}% mass)`;
        }
    } else {
        element.style.backgroundColor = "rgba(255, 23, 68, 0.08)";
        element.style.border = "1px solid rgba(255, 23, 68, 0.2)";
        element.style.color = "#FF1744";
        element.textContent = `\u25CF Lyra: Safety Pause \u00B7 ${confPct}% confident \u00B7 high hallucination risk`;
        if (mass < 0.30) {
            element.textContent += ` \u00B7 scattered (${massPct}% mass)`;
        }
    }
}
