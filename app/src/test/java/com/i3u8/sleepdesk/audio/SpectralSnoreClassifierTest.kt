package com.i3u8.sleepdesk.audio

import org.junit.Assert.*
import org.junit.Test

class SpectralSnoreClassifierTest {
    private fun candidate(pcm: ShortArray) =
        CandidateAudio(0, pcm.size.toLong(), -60f, pcm, pcm, 0, pcm.size.toLong())

    @Test fun silenceIsUnresolvedNotAFabricatedSoundClass() {
        val result = SpectralSnoreClassifier().classify(candidate(ShortArray(16000)), 0)
        assertEquals(NightEventType.UNKNOWN, result.type)
        assertEquals(ClassificationStatus.UNCERTAIN, result.status)
        assertTrue(result.suggestedTypes.isEmpty())
        assertNull(result.contextSpeechScore)
        assertTrue(result.reason!!.contains("non_medical"))
        assertEquals(SpectralSnoreClassifier.VERSION, result.modelVersion)
    }

    @Test fun wholePlaybackContextMatchesTheTrainingScope() {
        val full = SpectralParityTest.synthetic(48000, 4)
        val body = full.copyOfRange(2000, 2127)
        val audio = CandidateAudio(2000, 2127, -60f, body, full, 0, full.size.toLong())
        val result = SpectralSnoreClassifier().classify(audio, 0)
        val direct = SpectralSnoreClassifier().classify(candidate(full), 0)
        assertEquals(direct.type, result.type)
        assertEquals(direct.confidence, result.confidence, 0f)
        assertNull(result.contextSpeechScore)
    }

    @Test fun binarySnoreResultCannotAuthorizeRecordingWithSpeechSavingDisabled() {
        var writes = 0
        val observed = mutableListOf<NightEvent>()
        val classifier = object : EventSoundClassifier {
            override fun classify(candidate: CandidateAudio, eventStartMs: Long) = SoundDecision(
                NightEventType.SNORE, .9f, ClassificationStatus.SUGGESTED,
                mapOf("Snoring" to .9f), listOf("SNORE"), SpectralSnoreClassifier.VERSION,
                "spectral_whole_clip_snore_suggestion;non_medical"
            )
        }
        val processor = SoundEventProcessor(
            AudioAlgoConfig(saveSpeechClips = false),
            writeClip = { _, _ -> writes++; "must-not-save.m4a" },
            publish = observed::add, classifierFactory = { classifier }
        )
        processor.process(NightEvent("event", NightEventType.UNKNOWN, 0, 1000, 0f),
            candidate(ShortArray(16000)))
        assertEquals(0, writes)
        assertEquals(NightEventType.SNORE, observed.last().type)
        assertEquals(ClipStatus.DISABLED, observed.last().clipStatus)
        assertNull(observed.last().clipRelativePath)
        processor.close()
    }

    @Test fun frozenDecisionCanBePersistedWithoutAnyLegacySoundModel() {
        val observed = mutableListOf<NightEvent>()
        val processor = SoundEventProcessor(AudioAlgoConfig(),
            modelFactory = { error("Legacy model must not load") },
            writeClip = { _, _ -> "synthetic.m4a" }, publish = observed::add,
            classifierFactory = { SpectralSnoreClassifier() })
        processor.process(NightEvent("event", NightEventType.UNKNOWN, 0, 1000, 0f),
            candidate(ShortArray(16000)))
        assertEquals(SpectralSnoreClassifier.VERSION, observed.last().modelVersion)
        assertEquals(ClipStatus.SAVED, observed.last().clipStatus)
        assertEquals(2, observed.size)
        processor.close()
    }
}
