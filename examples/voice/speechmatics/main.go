// Command voice-speechmatics is the voice bot with Speechmatics speech-to-text
// (Anthropic LLM, ElevenLabs TTS). Run it, open the URL it logs, click start
// and allow the mic:
//
//	SPEECHMATICS_API_KEY=… ANTHROPIC_API_KEY=… ELEVENLABS_API_KEY=… go run ./examples/voice/speechmatics
package main

import (
	"log"
	"os"

	"github.com/gojargo/jargo/examples/voice/run"
	"github.com/gojargo/jargo/provider/speechmatics"
)

func main() {
	if err := run.Voice(run.Builders{
		STT: run.Service(speechmatics.Config{
			APIKey:     os.Getenv("SPEECHMATICS_API_KEY"),
			SampleRate: run.SampleRate,
		}, speechmatics.NewSTT),
		LLM: run.DefaultLLM,
		TTS: run.DefaultTTS,
	}); err != nil {
		log.Fatal(err)
	}
}
