// Command voice-rime is the voice bot with Rime text-to-speech (Deepgram STT,
// Anthropic LLM). Run it, open the URL it logs, click start and allow the mic:
//
//	DEEPGRAM_API_KEY=… ANTHROPIC_API_KEY=… RIME_API_KEY=… go run ./examples/voice/rime
package main

import (
	"log"
	"os"

	"github.com/gojargo/jargo/examples/voice/run"
	"github.com/gojargo/jargo/provider/rime"
)

func main() {
	if err := run.Voice(run.Builders{
		STT: run.DefaultSTT,
		LLM: run.DefaultLLM,
		TTS: run.Service(rime.Config{APIKey: os.Getenv("RIME_API_KEY")}, rime.NewTTS),
	}); err != nil {
		log.Fatal(err)
	}
}
