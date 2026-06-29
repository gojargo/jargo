package run

import (
	"os"

	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/provider/anthropic"
	"github.com/gojargo/jargo/provider/deepgram"
	"github.com/gojargo/jargo/provider/elevenlabs"
)

// The default stack each example falls back to for the categories its own
// provider does not cover: Deepgram (STT), Anthropic (LLM), ElevenLabs (TTS).
// A provider's example overrides only the category (or categories) it serves.

// Service returns a Builder that, on each call, validates cfg and constructs the
// service with ctor. Use it to wire a provider in one line, e.g.
//
//	TTS: run.Service(cartesia.Config{APIKey: os.Getenv("CARTESIA_API_KEY")}, cartesia.NewTTS),
func Service[C interface{ Validate() error }, S processor.Processor](cfg C, ctor func(C) S) Builder {
	return func() (processor.Processor, error) {
		return Build(cfg, ctor)
	}
}

// DefaultSTT builds the default STT service (Deepgram), reading DEEPGRAM_API_KEY.
func DefaultSTT() (processor.Processor, error) {
	return Build(deepgram.Config{APIKey: os.Getenv("DEEPGRAM_API_KEY"), SampleRate: SampleRate}, deepgram.NewSTT)
}

// DefaultLLM builds the default LLM service (Anthropic), reading ANTHROPIC_API_KEY.
func DefaultLLM() (processor.Processor, error) {
	return Build(anthropic.Config{APIKey: os.Getenv("ANTHROPIC_API_KEY")}, anthropic.NewLLM)
}

// DefaultTTS builds the default TTS service (ElevenLabs), reading ELEVENLABS_API_KEY.
func DefaultTTS() (processor.Processor, error) {
	return Build(elevenlabs.Config{APIKey: os.Getenv("ELEVENLABS_API_KEY")}, elevenlabs.NewTTS)
}
