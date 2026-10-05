package com.i3u8.sleepdesk.audio

import java.util.concurrent.CancellationException
import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.sin
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test

class SpectralDspTest {
    private val keys = listOf(
        "env.mod_0.5_3", "env.mod_3_8", "env.mod_8_20", "env.mod_20_50", "env.mod_50_90",
    )

    @Test
    fun forwardAndInverseMatchIndependentComplexDft() {
        for (n in listOf(1, 2, 3, 4, 5, 7, 8, 15, 16, 17, 31, 64, 97, 127, 256)) {
            val inputReal = DoubleArray(n) { sin(it * 0.71) + 0.2 * cos(it * 0.31) }
            val inputImag = DoubleArray(n) { cos(it * 0.43) - 0.1 * sin(it * 0.13) }
            for (inverse in listOf(false, true)) {
                val expectedReal = DoubleArray(n)
                val expectedImag = DoubleArray(n)
                for (k in 0 until n) {
                    for (j in 0 until n) {
                        val angle = (if (inverse) 2.0 else -2.0) * PI * j * k / n
                        expectedReal[k] += inputReal[j] * cos(angle) - inputImag[j] * sin(angle)
                        expectedImag[k] += inputReal[j] * sin(angle) + inputImag[j] * cos(angle)
                    }
                    if (inverse) {
                        expectedReal[k] /= n.toDouble()
                        expectedImag[k] /= n.toDouble()
                    }
                }
                val real = inputReal.copyOf()
                val imag = inputImag.copyOf()
                SpectralDsp.fft(real, imag, inverse)
                assertArrayEquals("real n=$n inverse=$inverse", expectedReal, real, 2e-10)
                assertArrayEquals("imag n=$n inverse=$inverse", expectedImag, imag, 2e-10)
            }
        }
    }

    @Test
    fun roundTripIncludesLargePrimeAndMaximumLength() {
        for (n in listOf(511, 512, 1009, 16001, 262139, 262144)) {
            val originalReal = DoubleArray(n) { sin(it * 0.83) }
            val originalImag = DoubleArray(n) { cos(it * 0.37) }
            val real = originalReal.copyOf()
            val imag = originalImag.copyOf()
            SpectralDsp.fft(real, imag)
            SpectralDsp.fft(real, imag, inverse = true)
            assertArrayEquals("real n=$n", originalReal, real, 2e-11)
            assertArrayEquals("imag n=$n", originalImag, imag, 2e-11)
        }
    }

    @Test
    fun dcImpulseAndEvenNyquistHaveExpectedBins() {
        for (n in listOf(7, 8, 31, 32)) {
            val dc = DoubleArray(n) { 1.0 }
            val imag = DoubleArray(n)
            SpectralDsp.fft(dc, imag)
            assertEquals(n.toDouble(), dc[0], 1e-12)
            assertTrue(dc.drop(1).all { kotlin.math.abs(it) < 1e-12 })
            assertArrayEquals(DoubleArray(n), imag, 1e-12)
            val impulse = DoubleArray(n).also { it[0] = 1.0 }
            imag.fill(0.0)
            SpectralDsp.fft(impulse, imag)
            assertArrayEquals(DoubleArray(n) { 1.0 }, impulse, 1e-12)
            assertArrayEquals(DoubleArray(n), imag, 1e-12)
        }
        val nyquist = DoubleArray(32) { if (it % 2 == 0) 1.0 else -1.0 }
        val imag = DoubleArray(32)
        SpectralDsp.fft(nyquist, imag)
        assertArrayEquals(DoubleArray(32).also { it[16] = 32.0 }, nyquist, 1e-12)
        assertArrayEquals(DoubleArray(32), imag, 1e-12)
    }

    @Test
    fun silenceAndShortSignalsKeepFiniteFiveKeyShapeWithoutMutatingInput() {
        for (n in listOf(1, 2, 79, 80, 81, 159, 160, 161, 241, 512, 1009, 16000)) {
            val silence = SpectralDsp.modulation(DoubleArray(n))
            assertEquals(keys, silence.keys.toList())
            assertTrue(silence.values.all { it == 0.0 })
            val y = synthetic(n)
            val before = y.copyOf()
            val actual = SpectralDsp.modulation(y)
            assertArrayEquals(before, y, 0.0)
            assertEquals(keys, actual.keys.toList())
            assertTrue(actual.values.all { it.isFinite() && it >= 0.0 && it <= 1.0 })
            assertEquals(if (n <= 160) 0.0 else 1.0, actual.values.sum(), 1e-12)
        }
    }

    // Golden values generated ONLY from synthetic PCM with NumPy 2.2.6 / SciPy 1.15.3:
    // e = resample_poly(abs(hilbert(sosfilt(butter(3,(80,2500),fs=16000,
    //     btype="bandpass",output="sos"), y))), 1, 80)
    // p = abs(np.fft.rfft((e-e.mean()) * np.hanning(len(e))))**2
    // f = np.fft.rfftfreq(len(e), 1/200); total = max(p[(f>=.5)&(f<90)].sum(),1e-18)
    // Each fixture stores p[(f>=lo)&(f<hi)].sum()/total for the five bands.
    @Test
    fun modulationMatchesScipyForOddEvenPrimeAndFifteenSecondInputs() {
        val fixtures = listOf(
            1 to doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0),
            81 to doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0),
            161 to doubleArrayOf(0.0, 0.0, 0.0, 0.0, 1.0),
            241 to doubleArrayOf(0.0, 0.0, 0.0, 0.0, 1.0),
            511 to doubleArrayOf(0.0, 0.0, 0.0, 0.4178801766933155, 0.5821198233066844),
            512 to doubleArrayOf(0.0, 0.0, 0.0, 0.42063026285298244, 0.5793697371470177),
            1009 to doubleArrayOf(0.0, 0.0, 0.09258535770520096, 0.4605076333359517, 0.44690700895884716),
            16000 to doubleArrayOf(0.4263544579109207, 0.31153670130909805,
                0.14473055801369503, 0.08124161536515849, 0.03613666740112785),
            16001 to doubleArrayOf(0.5128163845249534, 0.22563125255109726,
                0.1444233305623062, 0.08106910305571358, 0.036059929305929696),
            32001 to doubleArrayOf(0.5099096709175173, 0.22696689346938853,
                0.145283531298461, 0.08156088167966499, 0.036279022634968475),
            240000 to doubleArrayOf(0.5099060574495731, 0.22697013728029863,
                0.145285215581504, 0.08156001011453655, 0.03627857957408786),
        )
        for ((n, expected) in fixtures) assertModulation(expected, synthetic(n))
    }

    @Test
    fun boundaryImpulsesMatchScipyZeroPaddingAndFilterAlignment() {
        assertModulation(doubleArrayOf(0.0, 0.0, 0.9738739797688075,
            0.018275312650199667, 0.007850707580992856), DoubleArray(1009).also { it[0] = 1.0 })
        assertModulation(doubleArrayOf(0.0, 0.0, 0.9423064332104973,
            0.0166651352351345, 0.041028431554368056), DoubleArray(1009).also { it[1008] = 1.0 })
        assertModulation(doubleArrayOf(0.9995807052985235, 0.0003850600546160551,
            3.119632789259718e-5, 1.2800935259925673e-6, 1.7582254414726262e-6),
            DoubleArray(16000).also { it[15999] = 1.0 })
    }

    @Test
    fun tinyGainPreservesDenominatorFloorInsteadOfRenormalizing() {
        val normal = SpectralDsp.modulation(synthetic(16001))
        val quiet = SpectralDsp.modulation(synthetic(16001).map { it * 1e-12 }.toDoubleArray())
        assertTrue(quiet.values.sum() > 0.0)
        assertTrue(quiet.values.sum() < 0.01)
        for (key in keys) {
            assertEquals(normal.getValue(key), quiet.getValue(key) / quiet.values.sum(), 2e-10)
        }
    }

    @Test
    fun rejectsInvalidInputsAndExcessiveLengths() {
        assertThrows(IllegalArgumentException::class.java) { SpectralDsp.fft(DoubleArray(0), DoubleArray(0)) }
        assertThrows(IllegalArgumentException::class.java) { SpectralDsp.fft(DoubleArray(2), DoubleArray(3)) }
        val alias = DoubleArray(8)
        assertThrows(IllegalArgumentException::class.java) { SpectralDsp.fft(alias, alias) }
        assertThrows(IllegalArgumentException::class.java) {
            SpectralDsp.fft(DoubleArray(262145), DoubleArray(262145))
        }
        for (y in listOf(DoubleArray(0), DoubleArray(262145), doubleArrayOf(Double.NaN),
            doubleArrayOf(Double.POSITIVE_INFINITY), doubleArrayOf(Double.NEGATIVE_INFINITY))) {
            assertThrows(IllegalArgumentException::class.java) { SpectralDsp.modulation(y) }
        }
        assertThrows(IllegalArgumentException::class.java) {
            SpectralDsp.fft(doubleArrayOf(Double.NaN), doubleArrayOf(0.0))
        }
    }

    @Test
    fun interruptionCancelsBothEntryPointsAndPreservesFlag() {
        try {
            Thread.currentThread().interrupt()
            assertThrows(CancellationException::class.java) {
                SpectralDsp.fft(DoubleArray(17), DoubleArray(17))
            }
            assertTrue(Thread.currentThread().isInterrupted)
            assertThrows(CancellationException::class.java) {
                SpectralDsp.modulation(DoubleArray(16000))
            }
            assertTrue(Thread.currentThread().isInterrupted)
        } finally {
            Thread.interrupted()
        }
    }

    private fun assertModulation(expected: DoubleArray, y: DoubleArray) {
        val actual = SpectralDsp.modulation(y)
        assertEquals(keys, actual.keys.toList())
        assertArrayEquals("n=${y.size}", expected, keys.map { actual.getValue(it) }.toDoubleArray(), 2e-10)
    }

    private fun synthetic(n: Int) = DoubleArray(n) { i ->
        val t = i / 16000.0
        (0.6 + 0.15 * cos(2 * PI * 2 * t) + 0.1 * sin(2 * PI * 6 * t) +
            0.08 * cos(2 * PI * 13 * t) + 0.06 * sin(2 * PI * 37 * t) +
            0.04 * cos(2 * PI * 71 * t)) * sin(2 * PI * 437 * t) +
            0.03 * cos(2 * PI * 1703 * t)
    }
}
