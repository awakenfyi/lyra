/**
 * interceptor.js — Fetch Override for ChatGPT
 *
 * Injected into the page context to intercept ChatGPT's SSE stream.
 * Extracts logprobs from the response as tokens arrive, runs the
 * coherence math, and dispatches a custom event with the result.
 *
 * This runs in the PAGE context (not the extension context) because
 * Manifest V3 content scripts can't read response bodies directly.
 *
 * MIT License | awaken.fyi
 */

const originalFetch = window.fetch;

window.fetch = async function (...args) {
    const response = await originalFetch.apply(this, args);

    // Only intercept ChatGPT conversation API calls
    const url = args[0] && typeof args[0] === "string" ? args[0] : "";
    if (!url.includes("/backend-api/conversation")) {
        return response;
    }

    // Clone the response so the original stream is unaffected
    const clonedResponse = response.clone();

    // Process asynchronously — don't block the UI
    clonedResponse.text().then(text => {
        try {
            const lines = text.split("\n");
            let sequenceMetrics = [];

            for (let line of lines) {
                if (!line.startsWith("data: ") || line === "data: [DONE]") continue;

                try {
                    const data = JSON.parse(line.substring(6));

                    // ChatGPT's response format nests logprobs in message.metadata
                    const logprobs = data?.message?.metadata?.logprobs;
                    if (!logprobs || !logprobs.content) continue;

                    for (let tokenData of logprobs.content) {
                        const topLp = tokenData.top_logprobs;
                        if (topLp && topLp.length > 0) {
                            // calculateCoherence is from lyra_math.js (loaded first)
                            const metric = calculateCoherence(topLp);
                            sequenceMetrics.push(metric);
                        }
                    }
                } catch (parseErr) {
                    // Incomplete SSE chunk — skip
                }
            }

            if (sequenceMetrics.length > 0) {
                // getTrafficLight is from lyra_math.js
                const status = getTrafficLight(sequenceMetrics);
                const avgConf = sequenceMetrics.reduce((s, m) => s + m.confidence, 0) / sequenceMetrics.length;
                const avgMass = sequenceMetrics.reduce((s, m) => s + m.mass, 0) / sequenceMetrics.length;

                window.dispatchEvent(new CustomEvent("lyra_coherence_update", {
                    detail: {
                        status: status,
                        confidence: avgConf,
                        mass: avgMass,
                        tokenCount: sequenceMetrics.length,
                    }
                }));
            }
        } catch (e) {
            // Stream processing error — fail silently
            console.debug("[Lyra] Stream processing error:", e.message);
        }
    });

    return response;
};
