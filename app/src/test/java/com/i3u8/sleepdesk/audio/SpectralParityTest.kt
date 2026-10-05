package com.i3u8.sleepdesk.audio

import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import org.junit.Assert.*
import org.junit.Assume.assumeTrue
import org.junit.Test
import kotlin.math.abs
import kotlin.math.max

class SpectralParityTest {
    @Test fun syntheticFeaturesAndDecisionsMatchFrozenPythonReference() {
        val text = javaClass.getResourceAsStream("/spectral-fixtures.tsv")!!
            .bufferedReader().use { it.readText() }
        val rows = text.trim().lines().map { it.split('\t') }
        val names = rows.first().drop(4)
        assertEquals(SpectralParameters.names.toList(), names)
        for (row in rows.drop(1)) {
            val pcm = synthetic(row[1].toInt(), row[2].toInt())
            val f = SpectralFeatures.extract(pcm)
            names.forEachIndexed { i, name ->
                val expected = row[i + 4].toDouble()
                assertEquals("${row[0]} / $name", expected, f.getValue(name),
                    max(2e-7, abs(expected) * 2e-7))
            }
            val expected = row[3].toDouble()
            assertEquals("${row[0]} margin", expected, SpectralSnoreClassifier.margin(f), 2e-6)
            val audio = CandidateAudio(0, pcm.size.toLong(), -60f, pcm, pcm, 0, pcm.size.toLong())
            val decision = SpectralSnoreClassifier().classify(audio, 0)
            assertEquals(expected > 0, decision.type == NightEventType.SNORE)
            assertEquals(SpectralSnoreClassifier.VERSION, decision.modelVersion)
            assertNull(decision.contextSpeechScore)
        }
    }

    @Test fun optionalPrivatePcmParity() {
        val root = System.getenv("SLEEP_DESK_SPECTRAL_PARITY")
        assumeTrue(root != null)
        val rows = File(root!!, "expected.tsv").readLines().map { it.split('\t') }
        val names = rows.first().drop(2)
        for (row in rows.drop(1)) {
            require(row[0].matches(Regex("[A-Za-z0-9_-]+")))
            val input = ByteBuffer.wrap(File(root, "${row[0]}.pcm").readBytes())
                .order(ByteOrder.LITTLE_ENDIAN).asShortBuffer()
            val pcm = ShortArray(input.remaining())
            input.get(pcm)
            val f = SpectralFeatures.extract(pcm)
            names.forEachIndexed { i, name ->
                val expected = row[i + 2].toDouble()
                assertEquals("${row[0]} / $name", expected, f.getValue(name),
                    max(2e-6, abs(expected) * 2e-6))
            }
            val expected = row[1].toDouble()
            val score = SpectralSnoreClassifier.margin(f)
            assertEquals("${row[0]} margin", expected, score, 2e-5)
            assertEquals("${row[0]} binary verdict", expected > 0, score > 0)
        }
        println("Private native feature parity: ${rows.size - 1} clips")
    }

    companion object {
        fun synthetic(size: Int, kind: Int): ShortArray {
            var state = 17L
            return ShortArray(size) { i ->
                state = (state * 1664525 + 1013904223) and 0xffffffffL
                val noise = ((state ushr 16).toInt() and 65535) - 32768
                val triangle = (abs(i % 160 - 80) - 40) * 200
                when (kind) {
                    0 -> 0
                    1 -> 111
                    2 -> triangle
                    3 -> noise / 8
                    4 -> if (i % 48000 in 16000 until 32000) triangle / 2 + noise / 32 else noise / 512
                    else -> noise / 128 + if (i % 8000 < 64) 15000 / (i % 8000 + 1) else 0
                }.toShort()
            }
        }
    }
}
