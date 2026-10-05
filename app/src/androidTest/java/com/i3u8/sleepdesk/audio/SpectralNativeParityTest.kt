package com.i3u8.sleepdesk.audio

import android.os.SystemClock
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import java.io.File
import org.junit.Assert.*
import org.junit.Test
import org.junit.runner.RunWith
import kotlin.math.abs
import kotlin.math.max

@RunWith(AndroidJUnit4::class)
class SpectralNativeParityTest {
    @Test fun productionAppHasNoLargeModelAndNativeFeaturesMatchSyntheticReference() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val applicationModels = instrumentation.targetContext.assets.list("models").orEmpty()
        assertFalse("Large model leaked into application", applicationModels.any { it.endsWith(".tflite") })
        val rows = instrumentation.context.assets.open("spectral-fixtures.tsv").bufferedReader()
            .use { it.readLines() }.map { it.split('\t') }
        val names = rows.first().drop(4)
        val timings = mutableListOf("case\tms")
        for (row in rows.drop(1)) {
            val size = row[1].toInt()
            val kind = row[2].toInt()
            var state = 17L
            val pcm = ShortArray(size) { i ->
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
            val start = SystemClock.elapsedRealtime()
            val values = SpectralFeatures.extract(pcm)
            timings.add("${row[0]}\t${SystemClock.elapsedRealtime() - start}")
            for ((i, name) in names.withIndex()) {
                val expected = row[i + 4].toDouble()
                assertEquals("${row[0]} / $name", expected, values.getValue(name),
                    max(2e-7, abs(expected) * 2e-7))
            }
            assertEquals("${row[0]} margin", row[3].toDouble(), SpectralSnoreClassifier.margin(values), 2e-6)
        }
        File(instrumentation.targetContext.filesDir, "spectral-native-timings.tsv")
            .writeText(timings.joinToString("\n"))
    }
}
