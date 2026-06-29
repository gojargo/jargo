// Command voice-anthropic is the voice bot with an Anthropic LLM (Deepgram STT,
// ElevenLabs TTS). Run it, open the URL it logs, click start and allow the mic:
//
//	DEEPGRAM_API_KEY=… ANTHROPIC_API_KEY=… ELEVENLABS_API_KEY=… go run ./examples/voice/anthropic
package main

import (
	"log"
	"os"

	"github.com/gojargo/jargo/examples/voice/run"
	"github.com/gojargo/jargo/provider/anthropic"
)

func main() {
	if err := run.Voice(run.Builders{
		STT: run.DefaultSTT,
		LLM: run.Service(anthropic.Config{APIKey: os.Getenv("ANTHROPIC_API_KEY")}, anthropic.NewLLM),
		TTS: run.DefaultTTS,
	}); err != nil {
		log.Fatal(err)
	}
}
