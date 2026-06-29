// Command voice-gladia is the voice bot with Gladia speech-to-text (Anthropic
// LLM, ElevenLabs TTS). Run it, open the URL it logs, click start and allow the
// mic:
//
//	GLADIA_API_KEY=… ANTHROPIC_API_KEY=… ELEVENLABS_API_KEY=… go run ./examples/voice/gladia
package main

import (
	"log"
	"os"

	"github.com/gojargo/jargo/examples/voice/run"
	"github.com/gojargo/jargo/provider/gladia"
)

func main() {
	if err := run.Voice(run.Builders{
		STT: run.Service(gladia.Config{
			APIKey:     os.Getenv("GLADIA_API_KEY"),
			SampleRate: run.SampleRate,
		}, gladia.NewSTT),
		LLM: run.DefaultLLM,
		TTS: run.DefaultTTS,
	}); err != nil {
		log.Fatal(err)
	}
}
