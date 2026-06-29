// Command voice-soniox is the voice bot with Soniox speech-to-text (Anthropic
// LLM, ElevenLabs TTS). Run it, open the URL it logs, click start and allow the
// mic:
//
//	SONIOX_API_KEY=… ANTHROPIC_API_KEY=… ELEVENLABS_API_KEY=… go run ./examples/voice/soniox
package main

import (
	"log"
	"os"

	"github.com/gojargo/jargo/examples/voice/run"
	"github.com/gojargo/jargo/provider/soniox"
)

func main() {
	if err := run.Voice(run.Builders{
		STT: run.Service(soniox.Config{
			APIKey:     os.Getenv("SONIOX_API_KEY"),
			SampleRate: run.SampleRate,
		}, soniox.NewSTT),
		LLM: run.DefaultLLM,
		TTS: run.DefaultTTS,
	}); err != nil {
		log.Fatal(err)
	}
}
