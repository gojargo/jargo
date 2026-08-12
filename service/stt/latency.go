package stt

import "time"

// P99 time-to-final-segment latencies, one per transcription service.
//
// TTFS measures the wait between the user finishing an utterance and its
// transcript being available. Turn stop strategies read it off the metadata
// frame a service publishes at pipeline start and size their safety net by it:
// too low truncates the tail of an utterance, too high spends the difference as
// silence on every turn the provider does not close explicitly.
//
// Every value here was measured with a VAD stop window of DefaultStopSecs, the
// recommended default. Under a different stop window, or in a deployment whose
// region, network or self-hosted instance makes its own latency, re-measure and
// pass the result as the TTFSP99 field of the service's config.
//
// A turn-based service — one whose server defines the turn boundary — has no
// meaningful TTFS: there is no separate wait between the speech ending and the
// transcript to measure. Those report SupportsTTFS false rather than a value.
const (
	// DefaultTTFSP99 is the conservative fallback for a service with no measured
	// value of its own.
	DefaultTTFSP99 = time.Second

	AssemblyAITTFSP99         = 420 * time.Millisecond
	AWSTranscribeTTFSP99      = 1900 * time.Millisecond
	AzureTTFSP99              = 1800 * time.Millisecond
	CartesiaTTFSP99           = 810 * time.Millisecond
	DeepgramTTFSP99           = 350 * time.Millisecond
	DeepgramSageMakerTTFSP99  = 350 * time.Millisecond
	ElevenLabsTTFSP99         = 2010 * time.Millisecond
	ElevenLabsRealtimeTTFSP99 = 410 * time.Millisecond
	FalTTFSP99                = 2070 * time.Millisecond
	GladiaTTFSP99             = 1490 * time.Millisecond
	GoogleTTFSP99             = 1570 * time.Millisecond
	GradiumTTFSP99            = 620 * time.Millisecond
	GroqTTFSP99               = 1540 * time.Millisecond
	MistralTTFSP99            = 1890 * time.Millisecond
	OpenAITTFSP99             = 2010 * time.Millisecond
	OpenAIRealtimeTTFSP99     = 1660 * time.Millisecond
	SarvamTTFSP99             = 1170 * time.Millisecond
	SmallestTTFSP99           = 1590 * time.Millisecond
	SonioxTTFSP99             = 350 * time.Millisecond
	SpeechmaticsTTFSP99       = 740 * time.Millisecond
	XAITTFSP99                = 2140 * time.Millisecond
	TogetherTTFSP99           = time.Second

	// These services run locally, where the hardware decides the latency. Measure
	// yours and pass it in rather than trusting the fallback.
	NvidiaTTFSP99  = DefaultTTFSP99
	WhisperTTFSP99 = DefaultTTFSP99
)
