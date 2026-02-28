/**
 * lyra_math.js — Coherence Proxy (JavaScript)
 *
 * Direct translation of bridge/coherence_proxy.py.
 * Calculates entropy, margin, mass, and confidence from logprobs.
 *
 * The mass term (Gem's fix) catches the blind spot where all top-K
 * tokens have tiny raw probabilities. After normalization they look
 * stable, but the model is actually scattered across the vocabulary.
 *
 * MIT License | awaken.fyi
 */

function calculateCoherence(topLogprobs, entropyCeiling = 4.0, massFloor = 0.30) {
    if (!topLogprobs || topLogprobs.length === 0) {
        return { entropy: 0.0, margin: 1.0, mass: 1.0, confidence: 1.0 };
    }

    // Convert logprobs to raw probabilities
    let rawProbs = topLogprobs.map(lp => Math.exp(lp.logprob));
    rawProbs.sort((a, b) => b - a);

    // --- Mass ---
    // Total probability captured by top-K tokens.
    // If top-5 combined < 30% of distribution, model is lost.
    let mass = rawProbs.reduce((sum, p) => sum + p, 0);
    mass = Math.min(mass, 1.0);

    // Normalize for entropy/margin
    let total = rawProbs.reduce((sum, p) => sum + p, 0);
    let normProbs = total > 0 ? rawProbs.map(p => p / total) : rawProbs;

    // --- Entropy ---
    let entropy = normProbs.reduce((sum, p) => {
        return p > 1e-10 ? sum - (p * Math.log2(p)) : sum;
    }, 0);
    let entropyNormalized = Math.min(entropy / entropyCeiling, 1.0);

    // --- Margin ---
    let margin = normProbs.length > 1 ? normProbs[0] - normProbs[1] : 1.0;

    // --- Mass Penalty ---
    let massPenalty = mass >= massFloor ? 1.0 : mass / massFloor;

    // --- Confidence ---
    // Matches Python: 0.6 * margin + 0.4 * (1 - entropy_norm) * mass_penalty
    let rawConfidence = 0.6 * margin + 0.4 * (1.0 - entropyNormalized);
    let confidence = Math.max(0.0, Math.min(1.0, rawConfidence * massPenalty));

    return { entropy, entropyNormalized, margin, mass, confidence };
}

function getTrafficLight(sequenceMetrics, greenThreshold = 0.70, yellowThreshold = 0.45, sustainedWindow = 5, minTokensForRed = 10) {
    if (!sequenceMetrics || sequenceMetrics.length === 0) return "GREEN";

    let confidences = sequenceMetrics.map(m => m.confidence);

    // Check for sustained critically low confidence
    let consecutiveLow = 0;
    let maxConsecutiveLow = 0;
    for (let conf of confidences) {
        if (conf < yellowThreshold) {
            consecutiveLow++;
            maxConsecutiveLow = Math.max(maxConsecutiveLow, consecutiveLow);
        } else {
            consecutiveLow = 0;
        }
    }

    // RED: sustained low after minimum tokens
    if (maxConsecutiveLow >= sustainedWindow && confidences.length >= minTokensForRed) {
        return "RED";
    }

    // YELLOW: any token or average dropped below green
    let avg = confidences.reduce((s, c) => s + c, 0) / confidences.length;
    let min = Math.min(...confidences);
    if (min < greenThreshold || avg < greenThreshold) {
        return "YELLOW";
    }

    return "GREEN";
}
