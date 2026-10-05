package com.i3u8.sleepdesk.audio

import kotlin.math.exp

/** Frozen whole-clip binary suggestion. A negative score is not a silence or speech verdict. */
internal class SpectralSnoreClassifier : EventSoundClassifier {
    override fun classify(candidate: CandidateAudio, eventStartMs: Long): SoundDecision {
        require(candidate.pcm.isNotEmpty() && candidate.clipPcm.isNotEmpty())
        if (Thread.currentThread().isInterrupted) throw InterruptedException("Spectral analysis cancelled")
        val margin = margin(SpectralFeatures.extract(candidate.clipPcm))
        val support = (1.0 / (1.0 + exp(-margin.coerceIn(-60.0, 60.0)))).toFloat()
        val snore = margin > SpectralParameters.THRESHOLD
        return SoundDecision(
            type = if (snore) NightEventType.SNORE else NightEventType.UNKNOWN,
            confidence = support,
            status = if (snore) ClassificationStatus.SUGGESTED else ClassificationStatus.UNCERTAIN,
            classScores = mapOf("Snoring" to support, "Non-snore" to 1f - support),
            suggestedTypes = if (snore) listOf(NightEventType.SNORE.name) else emptyList(),
            modelVersion = VERSION,
            reason = if (snore) "spectral_whole_clip_snore_suggestion;uncalibrated;non_medical"
                else "spectral_non_snore_unresolved;uncalibrated;non_medical",
            // Binary snore evidence cannot rule out speech anywhere in the recorded context.
            contextSpeechScore = null
        )
    }

    companion object {
        const val VERSION = "spectral-linear-48-v1"

        fun margin(features: Map<String, Double>): Double {
            var score = SpectralParameters.BIAS
            for (i in SpectralParameters.names.indices) {
                val value = features.getValue(SpectralParameters.names[i])
                require(value.isFinite()) { "Nonfinite spectral feature" }
                score += value * SpectralParameters.weights[i]
            }
            check(score.isFinite()) { "Nonfinite spectral score" }
            return score
        }
    }
}
