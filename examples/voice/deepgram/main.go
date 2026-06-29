// Command voice-deepgram is the voice bot with Deepgram for both speech-to-text
// and text-to-speech (Anthropic handles the LLM). Run it, open the URL it logs,
// click start and allow the mic.
//
//	DEEPGRAM_API_KEY=… ANTHROPIC_API_KEY=… go run ./examples/voice/deepgram
package main

import (
	"log"
	"os"

	"github.com/gojargo/jargo/examples/voice/run"
	"github.com/gojargo/jargo/provider/deepgram"
)

func main() {
	if err := run.Voice(run.Builders{
		STT: run.Service(deepgram.Config{
			APIKey:     os.Getenv("DEEPGRAM_API_KEY"),
			SampleRate: run.SampleRate,
		}, deepgram.NewSTT),
		LLM: run.DefaultLLM,
		TTS: run.Service(deepgram.TTSConfig{APIKey: os.Getenv("DEEPGRAM_API_KEY")}, deepgram.NewTTS),
	}); err != nil {
		log.Fatal(err)
	}
}
