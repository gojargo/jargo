// Command voice-cartesia is the voice bot with Cartesia text-to-speech
// (Deepgram STT, Anthropic LLM). Run it, open the URL it logs, click start and
// allow the mic:
//
//	DEEPGRAM_API_KEY=… ANTHROPIC_API_KEY=… CARTESIA_API_KEY=… go run ./examples/voice/cartesia
package main

import (
	"log"
	"os"

	"github.com/gojargo/jargo/examples/voice/run"
	"github.com/gojargo/jargo/provider/cartesia"
)

func main() {
	if err := run.Voice(run.Builders{
		STT: run.DefaultSTT,
		LLM: run.DefaultLLM,
		TTS: run.Service(cartesia.Config{APIKey: os.Getenv("CARTESIA_API_KEY")}, cartesia.NewTTS),
	}); err != nil {
		log.Fatal(err)
	}
}
