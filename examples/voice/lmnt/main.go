// Command voice-lmnt is the voice bot with LMNT text-to-speech (Deepgram STT,
// Anthropic LLM). Run it, open the URL it logs, click start and allow the mic:
//
//	DEEPGRAM_API_KEY=… ANTHROPIC_API_KEY=… LMNT_API_KEY=… go run ./examples/voice/lmnt
package main

import (
	"log"
	"os"

	"github.com/gojargo/jargo/examples/voice/run"
	"github.com/gojargo/jargo/provider/lmnt"
)

func main() {
	if err := run.Voice(run.Builders{
		STT: run.DefaultSTT,
		LLM: run.DefaultLLM,
		TTS: run.Service(lmnt.Config{APIKey: os.Getenv("LMNT_API_KEY")}, lmnt.NewTTS),
	}); err != nil {
		log.Fatal(err)
	}
}
