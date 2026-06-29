// Command voice-elevenlabs is the voice bot with ElevenLabs text-to-speech
// (Deepgram STT, Anthropic LLM). Run it, open the URL it logs, click start and
// allow the mic:
//
//	DEEPGRAM_API_KEY=… ANTHROPIC_API_KEY=… ELEVENLABS_API_KEY=… go run ./examples/voice/elevenlabs
package main

import (
	"log"
	"os"

	"github.com/gojargo/jargo/examples/voice/run"
	"github.com/gojargo/jargo/provider/elevenlabs"
)

func main() {
	if err := run.Voice(run.Builders{
		STT: run.DefaultSTT,
		LLM: run.DefaultLLM,
		TTS: run.Service(elevenlabs.Config{APIKey: os.Getenv("ELEVENLABS_API_KEY")}, elevenlabs.NewTTS),
	}); err != nil {
		log.Fatal(err)
	}
}
