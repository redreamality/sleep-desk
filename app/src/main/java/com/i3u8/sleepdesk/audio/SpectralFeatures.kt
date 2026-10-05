package com.i3u8.sleepdesk.audio

import java.util.concurrent.CancellationException
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.ln1p
import kotlin.math.log10
import kotlin.math.max
import kotlin.math.sqrt

/** The env/spec/dyn subset of tools/audio_feature_study.py, at 16 kHz. */
internal object SpectralFeatures {
    private const val SAMPLE_RATE = 16000
    private const val MAX_SAMPLES = 240000
    private const val NFFT = 512
    private const val HOP = 160
    private const val FIRST_BIN = 3
    private const val LAST_BIN = 192
    private const val BINS = LAST_BIN - FIRST_BIN + 1
    private val window = DoubleArray(NFFT) { .5 - .5 * cos(2 * PI * it / NFFT) }
    private val edges = intArrayOf(80, 200, 400, 800, 1600, 3200, 6001)
    private val modulationKeys = listOf(
        "env.mod_0.5_3", "env.mod_3_8", "env.mod_8_20", "env.mod_20_50", "env.mod_50_90"
    )

    fun extract(pcm: ShortArray, sampleRate: Int = SAMPLE_RATE): Map<String, Double> {
        checkCancelled()
        require(sampleRate == SAMPLE_RATE) { "Spectral features require 16000 Hz mono PCM" }
        require(pcm.isNotEmpty() && pcm.size <= MAX_SAMPLES) { "Expected 1..240000 samples" }
        return extract(DoubleArray(pcm.size) {
            if (it % 4096 == 0) checkCancelled()
            pcm[it] / 32768.0
        })
    }

    fun extract(samples: DoubleArray): Map<String, Double> {
        checkCancelled()
        require(samples.isNotEmpty() && samples.size <= MAX_SAMPLES) { "Expected 1..240000 samples" }
        var sum = 0.0
        for (i in samples.indices) {
            if (i % 4096 == 0) checkCancelled()
            require(samples[i].isFinite()) { "Expected finite mono PCM" }
            sum += samples[i]
        }
        val mean = sum / samples.size
        require(mean.isFinite()) { "PCM mean overflow" }
        val y = DoubleArray(samples.size)
        var peak = 0.0
        for (i in samples.indices) {
            if (i % 4096 == 0) checkCancelled()
            y[i] = samples[i] - mean
            peak = max(peak, abs(y[i]))
        }
        require(peak.isFinite()) { "PCM range overflow" }
        if (peak > 1e-12) {
            for (i in y.indices) {
                if (i % 4096 == 0) checkCancelled()
                y[i] /= peak
            }
        }
        val count = 1 + (max(NFFT, y.size) - NFFT) / HOP
        val rms = DoubleArray(count)
        val norm = Array(count) { DoubleArray(BINS) }
        val real = DoubleArray(NFFT)
        val imag = DoubleArray(NFFT)
        for (frame in 0 until count) {
            checkCancelled()
            var squares = 0.0
            for (j in 0 until NFFT) {
                val index = frame * HOP + j
                val value = if (index < y.size) y[index] else 0.0
                squares += value * value
                real[j] = value * window[j]
                imag[j] = 0.0
            }
            rms[frame] = sqrt(squares / NFFT)
            SpectralDsp.fft(real, imag)
            var total = 0.0
            for (bin in FIRST_BIN..LAST_BIN) {
                val power = real[bin] * real[bin] + imag[bin] * imag[bin]
                norm[frame][bin - FIRST_BIN] = power
                total += power
            }
            total = max(total, 1e-18)
            for (bin in 0 until BINS) norm[frame][bin] /= total
        }
        val threshold = max(rms.maxOrNull()!! * .25, 1e-12)
        val active = BooleanArray(count) { rms[it] >= threshold }
        if (!active.any { it }) active.fill(true)
        val out = linkedMapOf<String, Double>()
        spectral(out, norm, active)
        envelope(out, y, rms, active)
        checkCancelled()
        val modulation = SpectralDsp.modulation(y)
        checkCancelled()
        for (key in modulationKeys) out["hand.$key"] = modulation.getValue(key)
        require(out.size == 48 && out.values.all { it.isFinite() }) { "Nonfinite acoustic features" }
        return out
    }

    private fun spectral(out: MutableMap<String, Double>, norm: Array<DoubleArray>, active: BooleanArray) {
        val count = norm.size
        val bands = Array(6) { DoubleArray(count) }
        val centroid = DoubleArray(count)
        val bandwidth = DoubleArray(count)
        val flatness = DoubleArray(count)
        val entropy = DoubleArray(count)
        val rolloff = DoubleArray(count)
        val lengths = DoubleArray(count)
        for (frame in norm.indices) {
            checkCancelled()
            val p = norm[frame]
            var cumulative = 0.0
            var logSum = 0.0
            var rolled = false
            // NumPy argmax returns zero if no bin reaches the threshold (silence).
            rolloff[frame] = FIRST_BIN * SAMPLE_RATE.toDouble() / NFFT
            for (bin in p.indices) {
                val hz = (bin + FIRST_BIN) * SAMPLE_RATE.toDouble() / NFFT
                for (band in 0..5) {
                    if (hz >= edges[band] && hz < edges[band + 1]) bands[band][frame] += p[bin]
                }
                centroid[frame] += p[bin] * hz
                val log = ln(max(p[bin], 1e-18))
                logSum += log
                entropy[frame] -= p[bin] * log
                lengths[frame] += p[bin] * p[bin]
                cumulative += p[bin]
                if (!rolled && cumulative >= .85) {
                    rolloff[frame] = hz
                    rolled = true
                }
            }
            for (bin in p.indices) {
                val delta = (bin + FIRST_BIN) * SAMPLE_RATE.toDouble() / NFFT - centroid[frame]
                bandwidth[frame] += p[bin] * delta * delta
            }
            bandwidth[frame] = sqrt(bandwidth[frame])
            flatness[frame] = exp(logSum / BINS) * BINS
            entropy[frame] /= ln(BINS.toDouble())
            lengths[frame] = sqrt(lengths[frame])
        }
        for (band in 0..5) {
            describe(out, "spec.band_${edges[band]}_${edges[band + 1]}", bands[band], active)
        }
        for ((name, values) in listOf(
            "centroid" to centroid, "bandwidth" to bandwidth, "flatness" to flatness,
            "entropy" to entropy, "rolloff" to rolloff
        )) describe(out, "spec.$name", values, active)
        val adjacent = BooleanArray(max(0, count - 1)) { active[it] && active[it + 1] }
        val flux = DoubleArray(adjacent.size)
        val step = DoubleArray(adjacent.size)
        for (i in adjacent.indices) {
            checkCancelled()
            for (bin in 0 until BINS) flux[i] += abs(norm[i + 1][bin] - norm[i][bin])
            flux[i] *= .5
            step[i] = abs(ln1p(centroid[i + 1]) - ln1p(centroid[i]))
        }
        describe(out, "dyn.flux", flux, adjacent)
        describe(out, "dyn.centroid_step", step, adjacent)
        for (lag in intArrayOf(1, 3, 10)) {
            val valid = BooleanArray(max(0, count - lag)) { active[it] && active[it + lag] }
            val similarity = DoubleArray(valid.size)
            for (i in valid.indices) {
                checkCancelled()
                var dot = 0.0
                for (bin in 0 until BINS) dot += norm[i][bin] * norm[i + lag][bin]
                similarity[i] = dot / max(lengths[i] * lengths[i + lag], 1e-18)
            }
            describe(out, "dyn.cosine_lag_$lag", similarity, valid)
        }
    }

    private fun envelope(
        out: MutableMap<String, Double>, y: DoubleArray, rms: DoubleArray, active: BooleanArray
    ) {
        val sorted = rms.sortedArray()
        val mean = rms.average()
        val variance = moment(rms, mean, 2)
        val span = max(sorted.last() - sorted.first(), 1e-12)
        fun put(key: String, value: Double) { out["hand.env.$key"] = value }
        put("cv", sqrt(variance) / max(mean, 1e-12))
        put("active_fraction", active.count { it }.toDouble() / active.size)
        var peak = 0.0
        var squares = 0.0
        for (i in y.indices) {
            if (i % 4096 == 0) checkCancelled()
            peak = max(peak, abs(y[i]))
            squares += y[i] * y[i]
        }
        put("crest_db", 20 * log10(max(peak, 1e-12) / max(sqrt(squares / y.size), 1e-12)))
        put("p90_p10_db", 20 * log10(max(quantile(sorted, .9), 1e-12) / max(quantile(sorted, .1), 1e-12)))
        put("median_peak_ratio", quantile(sorted, .5) / max(sorted.last(), 1e-12))
        put("kurtosis", if (sqrt(variance) > 1e-10) moment(rms, mean, 4) / (variance * variance) else 0.0)
        val waveMean = y.average()
        val waveVariance = moment(y, waveMean, 2)
        put("wave_kurtosis", if (sqrt(waveVariance) > 1e-10) {
            moment(y, waveMean, 4) / (waveVariance * waveVariance)
        } else 0.0)
        val rise = DoubleArray(rms.size - 1) { max((rms[it + 1] - rms[it]) / span, 0.0) }
        val fall = DoubleArray(rise.size) { max((rms[it] - rms[it + 1]) / span, 0.0) }
        put("rise_p95", quantile(rise.sortedArray(), .95))
        put("fall_p95", quantile(fall.sortedArray(), .95))
        val prominences = peakProminences(rms, .25 * span)
        put("peaks_per_second", prominences.size / max(y.size.toDouble() / SAMPLE_RATE, .01))
        put("prominence_mean", if (prominences.isEmpty()) 0.0 else prominences.average() / span)
    }

    private fun peakProminences(values: DoubleArray, minimum: Double): List<Double> {
        val peaks = mutableListOf<Int>()
        var i = 1
        while (i < values.lastIndex) {
            checkCancelled()
            if (values[i - 1] < values[i]) {
                var end = i
                while (end < values.lastIndex && values[end + 1] == values[i]) end++
                if (end < values.lastIndex && values[end + 1] < values[i]) peaks.add((i + end) / 2)
                i = end + 1
            } else i++
        }
        // SciPy applies distance by peak height BEFORE testing prominence.
        val keep = BooleanArray(peaks.size) { true }
        // Equal-height priority is deterministic here (rightmost first).
        // NumPy's default unstable argsort may order mixed-height ties differently.
        val order = peaks.indices.sortedWith(compareBy<Int> { values[peaks[it]] }.thenBy { it })
        for (index in order.asReversed()) {
            checkCancelled()
            if (!keep[index]) continue
            var neighbor = index - 1
            while (neighbor >= 0 && peaks[index] - peaks[neighbor] < 8) keep[neighbor--] = false
            neighbor = index + 1
            while (neighbor < peaks.size && peaks[neighbor] - peaks[index] < 8) keep[neighbor++] = false
        }
        val result = mutableListOf<Double>()
        for (index in peaks.indices) {
            checkCancelled()
            if (!keep[index]) continue
            val peak = peaks[index]
            val height = values[peak]
            var left = height
            var right = height
            i = peak
            while (i >= 0 && values[i] <= height) {
                left = minOf(left, values[i])
                i--
            }
            i = peak
            while (i < values.size && values[i] <= height) {
                right = minOf(right, values[i])
                i++
            }
            val prominence = height - max(left, right)
            if (prominence >= minimum) result.add(prominence)
        }
        return result
    }

    private fun moment(values: DoubleArray, mean: Double, order: Int): Double {
        var sum = 0.0
        for (i in values.indices) {
            if (i % 4096 == 0) checkCancelled()
            val delta = values[i] - mean
            val square = delta * delta
            sum += if (order == 2) square else square * square
        }
        return sum / values.size
    }

    private fun describe(
        out: MutableMap<String, Double>, name: String, values: DoubleArray, valid: BooleanArray
    ) {
        checkCancelled()
        val sorted = values.filterIndexed { index, value -> valid[index] && value.isFinite() }.sorted().toDoubleArray()
        out["hand.$name.median"] = quantile(sorted, .5)
        out["hand.$name.iqr"] = quantile(sorted, .75) - quantile(sorted, .25)
    }

    private fun quantile(sorted: DoubleArray, q: Double): Double {
        if (sorted.isEmpty()) return 0.0
        val position = (sorted.size - 1) * q
        val lower = position.toInt()
        val fraction = position - lower
        val a = sorted[lower]
        val b = sorted[minOf(lower + 1, sorted.lastIndex)]
        return if (fraction >= .5) b - (b - a) * (1 - fraction) else a + (b - a) * fraction
    }

    private fun checkCancelled() {
        if (Thread.currentThread().isInterrupted) throw CancellationException("Spectral extraction cancelled")
    }
}
