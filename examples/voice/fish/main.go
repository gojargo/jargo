// Command voice-fish is the voice bot with Fish Audio text-to-speech (Deepgram
// STT, Anthropic LLM). Set FISH_REFERENCE_ID to pick a voice. Run it, open the
// URL it logs, click start and allow the mic:
//
//	DEEPGRAM_API_KEY=… ANTHROPIC_API_KEY=… FISH_API_KEY=… FISH_REFERENCE_ID=… go run ./examples/voice/fish
package main

import (
	"log"
	"os"

	"github.com/gojargo/jargo/examples/voice/run"
	"github.com/gojargo/jargo/provider/fish"
)

func main() {
	if err := run.Voice(run.Builders{
		STT: run.DefaultSTT,
		LLM: run.DefaultLLM,
		TTS: run.Service(fish.Config{
			APIKey:      os.Getenv("FISH_API_KEY"),
			ReferenceID: os.Getenv("FISH_REFERENCE_ID"),
		}, fish.NewTTS),
	}); err != nil {
		log.Fatal(err)
	}
}
