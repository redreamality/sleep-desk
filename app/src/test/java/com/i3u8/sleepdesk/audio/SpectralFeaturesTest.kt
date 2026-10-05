package com.i3u8.sleepdesk.audio

import java.util.concurrent.CancellationException
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.sin
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test

class SpectralFeaturesTest {
    @Test fun syntheticPythonReferenceMatchesAll48Features() {
        // Generated solely from this arithmetic waveform using audio_feature_study.py:
        // NumPy 2.2.6, SciPy 1.15.3, librosa 0.11.0. No recorded audio.
        val samples = DoubleArray(6400) { i ->
            ((i * 37) % 101 - 50) / 50.0 * if ((i / 800) % 3 == 0) 1.0 else .07
        }
        val expected = """
            dyn.centroid_step.iqr=0.004916384270984064
            dyn.centroid_step.median=0.0023393410324059616
            dyn.cosine_lag_1.iqr=0.04106861183393806
            dyn.cosine_lag_1.median=0.9845537425801913
            dyn.cosine_lag_10.iqr=0.040845439987341914
            dyn.cosine_lag_10.median=0.9684443332155466
            dyn.cosine_lag_3.iqr=0.0733472785415662
            dyn.cosine_lag_3.median=0.9841313073711546
            dyn.flux.iqr=0.27560117615470137
            dyn.flux.median=0.17137221346486983
            env.active_fraction=0.5675675675675675
            env.crest_db=8.90976642496803
            env.cv=0.8319046782589954
            env.fall_p95=0.5267753296225317
            env.kurtosis=1.2837152451769986
            env.median_peak_ratio=0.5592447919783149
            env.mod_0.5_3=0.002780927423714571
            env.mod_20_50=0.08654111119034659
            env.mod_3_8=0.6603319362373814
            env.mod_50_90=0.02743798192339801
            env.mod_8_20=0.22290804322515959
            env.p90_p10_db=23.10369848804382
            env.peaks_per_second=5.0
            env.prominence_mean=0.9996956384765482
            env.rise_p95=0.2705872670226848
            env.wave_kurtosis=4.72188179737034
            spec.band_1600_3200.iqr=0.005330609269996199
            spec.band_1600_3200.median=0.06080696376625297
            spec.band_200_400.iqr=0.000489898127200681
            spec.band_200_400.median=0.000695750156823534
            spec.band_3200_6001.iqr=0.00023689866171383578
            spec.band_3200_6001.median=0.8479185166102998
            spec.band_400_800.iqr=0.0003031092648157884
            spec.band_400_800.median=0.008017881757588748
            spec.band_800_1600.iqr=0.006769689301480711
            spec.band_800_1600.median=0.08240951350059536
            spec.band_80_200.iqr=5.808254160478629e-06
            spec.band_80_200.median=0.0009757354892913131
            spec.bandwidth.iqr=4.934193355841899
            spec.bandwidth.median=1460.795873484738
            spec.centroid.iqr=16.47741166988908
            spec.centroid.median=4956.804102392012
            spec.entropy.iqr=0.09863397786790079
            spec.entropy.median=0.4192281261043881
            spec.flatness.iqr=0.0704502221478831
            spec.flatness.median=0.026403742391189484
            spec.rolloff.iqr=0.0
            spec.rolloff.median=5875.0
        """.trimIndent().lines().associate {
            val (key, value) = it.split('=')
            "hand.$key" to value.toDouble()
        }
        assertFeatures(expected, SpectralFeatures.extract(samples), 1e-9)
    }

    @Test fun silenceAndDcHavePythonFallbacksIncludingRolloff() {
        for (size in intArrayOf(1, 200, 512, 673, 16000)) {
            for (constant in doubleArrayOf(0.0, .5)) {
                val features = SpectralFeatures.extract(DoubleArray(size) { constant })
                assertEquals(48, features.size)
                for ((key, value) in features) {
                    val expected = when (key) {
                        "hand.env.active_fraction" -> 1.0
                        "hand.spec.rolloff.median" -> 93.75
                        "hand.spec.flatness.median" -> 190e-18
                        else -> 0.0
                    }
                    assertEquals(key, expected, value, 1e-25)
                }
            }
        }
    }

    @Test fun shortNonconstantInputPadsRmsButNotWaveformAndHasNoDynamics() {
        val features = SpectralFeatures.extract(doubleArrayOf(-1.0, 1.0))
        assertEquals(0.0, features.getValue("hand.env.crest_db"), 0.0)
        assertEquals(1.0, features.getValue("hand.env.wave_kurtosis"), 0.0)
        assertEquals(1.0, features.getValue("hand.env.median_peak_ratio"), 0.0)
        assertEquals(0.0, features.getValue("hand.env.kurtosis"), 0.0)
        assertTrue(features.filterKeys { it.startsWith("hand.dyn.") }.values.all { it == 0.0 })
        assertTrue(features.values.all { it.isFinite() })
    }

    @Test fun pcmConversionGainOffsetAndCallerArrayArePreserved() {
        val pcm = ShortArray(4097) { ((it * 37) % 101 * 200 - 10000).toShort() }
        val doubles = DoubleArray(pcm.size) { pcm[it] / 32768.0 }
        val original = doubles.copyOf()
        val features = SpectralFeatures.extract(doubles)
        assertFeatures(features, SpectralFeatures.extract(pcm), 1e-12)
        assertFeatures(features, SpectralFeatures.extract(DoubleArray(doubles.size) { doubles[it] * .125 + .25 }), 1e-8)
        assertTrue(original.contentEquals(doubles))
    }

    @Test fun lagValidityUsesOriginalEndpointsWithoutCompressingQuietFrames() {
        val samples = DoubleArray(512 + 160 * 12)
        for (i in 0 until 160) {
            samples[i] = sin(2 * PI * 1000 * i / 16000)
            samples[1920 + i] = samples[i]
        }
        val features = SpectralFeatures.extract(samples)
        // Active frames are 0,9,10,11,12. Lag 3 includes (9,12), not
        // pairs from a compressed list of the five active spectra.
        assertEquals(5.0 / 13, features.getValue("hand.env.active_fraction"), 1e-12)
        assertEquals(.25687033526608516, features.getValue("hand.dyn.flux.median"), 1e-10)
        assertEquals(.22247006650692536, features.getValue("hand.dyn.flux.iqr"), 1e-10)
        assertEquals(.00983452097951698, features.getValue("hand.dyn.centroid_step.median"), 1e-10)
        assertEquals(.02687132375662138, features.getValue("hand.dyn.centroid_step.iqr"), 1e-10)
        assertEquals(.952072469786892, features.getValue("hand.dyn.cosine_lag_1.median"), 1e-10)
        assertEquals(.20066752169636604, features.getValue("hand.dyn.cosine_lag_1.iqr"), 1e-10)
        assertEquals(.6386928304154603, features.getValue("hand.dyn.cosine_lag_3.median"), 1e-10)
        assertEquals(.9921014854416871, features.getValue("hand.dyn.cosine_lag_10.median"), 1e-10)
        assertEquals(0.0, features.getValue("hand.dyn.cosine_lag_3.iqr"), 0.0)
        assertEquals(0.0, features.getValue("hand.dyn.cosine_lag_10.iqr"), 0.0)
    }

    @Test fun quantilesUseLinearInterpolationAndDescribeDropsNonfiniteValues() {
        val method = SpectralFeatures::class.java.getDeclaredMethod(
            "describe", MutableMap::class.java, String::class.java,
            DoubleArray::class.java, BooleanArray::class.java
        )
        method.isAccessible = true
        val out = mutableMapOf<String, Double>()
        method.invoke(SpectralFeatures, out, "test",
            doubleArrayOf(30.0, Double.NaN, 0.0, 10000.0, 10.0, 20.0, Double.POSITIVE_INFINITY),
            booleanArrayOf(true, true, true, false, true, true, true))
        assertEquals(15.0, out.getValue("hand.test.median"), 0.0)
        assertEquals(15.0, out.getValue("hand.test.iqr"), 0.0)
        method.invoke(SpectralFeatures, out, "empty", doubleArrayOf(Double.NaN), booleanArrayOf(true))
        assertEquals(0.0, out.getValue("hand.empty.median"), 0.0)
        assertEquals(0.0, out.getValue("hand.empty.iqr"), 0.0)
    }

    @Test fun scipyPlateausDistanceAndProminenceOrder() {
        assertEquals(listOf(2.0), prominences(doubleArrayOf(0.0, 2.0, 2.0, 0.0), .5))
        assertEquals(emptyList<Double>(), prominences(doubleArrayOf(2.0, 2.0, 0.0), .5))
        val separated = DoubleArray(20)
        separated[2] = 2.0
        separated[10] = 3.0
        assertEquals(listOf(2.0, 3.0), prominences(separated, 1.0))
        separated[9] = 4.0
        assertEquals(listOf(4.0), prominences(separated, 1.0))
        val equal = DoubleArray(12)
        equal[2] = 2.0
        equal[8] = 2.0
        assertEquals(listOf(2.0), prominences(equal, .5))
        // Taller, low-prominence peak eliminates the shorter peak first,
        // then itself fails prominence. Filtering prominence first is wrong.
        assertEquals(emptyList<Double>(), prominences(
            doubleArrayOf(0.0, 3.0, 0.0, 3.5, 4.0, 3.5, 5.0), 1.0
        ))
        assertEquals(listOf(2.0, 2.0), prominences(
            doubleArrayOf(0.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 0.0), 1.0
        ))
    }

    @Test fun invalidInputsAreRejectedBeforeDspAllocation() {
        assertThrows(IllegalArgumentException::class.java) { SpectralFeatures.extract(doubleArrayOf()) }
        assertThrows(IllegalArgumentException::class.java) { SpectralFeatures.extract(shortArrayOf()) }
        for (value in doubleArrayOf(Double.NaN, Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY)) {
            assertThrows(IllegalArgumentException::class.java) { SpectralFeatures.extract(doubleArrayOf(value)) }
        }
        assertThrows(IllegalArgumentException::class.java) { SpectralFeatures.extract(DoubleArray(240001)) }
        assertThrows(IllegalArgumentException::class.java) { SpectralFeatures.extract(ShortArray(240001)) }
        assertThrows(IllegalArgumentException::class.java) { SpectralFeatures.extract(shortArrayOf(1), 8000) }
        assertEquals(48, SpectralFeatures.extract(DoubleArray(240000)).size)
    }

    @Test fun interruptionIsPreservedAndExtractionCanResumeAfterClearingIt() {
        assertFalse(Thread.currentThread().isInterrupted)
        try {
            Thread.currentThread().interrupt()
            assertThrows(CancellationException::class.java) { SpectralFeatures.extract(DoubleArray(512)) }
            assertThrows(CancellationException::class.java) { SpectralFeatures.extract(ShortArray(512)) }
            assertTrue(Thread.currentThread().isInterrupted)
        } finally {
            Thread.interrupted()
        }
        assertEquals(48, SpectralFeatures.extract(DoubleArray(512)).size)
    }

    private fun prominences(values: DoubleArray, minimum: Double): List<*> {
        val method = SpectralFeatures::class.java.getDeclaredMethod(
            "peakProminences", DoubleArray::class.java, Double::class.javaPrimitiveType
        )
        method.isAccessible = true
        return method.invoke(SpectralFeatures, values, minimum) as List<*>
    }

    private fun assertFeatures(expected: Map<String, Double>, actual: Map<String, Double>, tolerance: Double) {
        assertEquals(expected.keys, actual.keys)
        for ((key, value) in expected) {
            assertEquals(key, value, actual.getValue(key), tolerance * max(1.0, abs(value)))
        }
    }
}
