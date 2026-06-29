// Command voice-google is the voice bot with a Google Gemini LLM (Deepgram STT,
// ElevenLabs TTS). Run it, open the URL it logs, click start and allow the mic:
//
//	DEEPGRAM_API_KEY=… GEMINI_API_KEY=… ELEVENLABS_API_KEY=… go run ./examples/voice/google
package main

import (
	"log"
	"os"

	"github.com/gojargo/jargo/examples/voice/run"
	"github.com/gojargo/jargo/provider/google"
)

func main() {
	if err := run.Voice(run.Builders{
		STT: run.DefaultSTT,
		LLM: run.Service(google.Config{APIKey: os.Getenv("GEMINI_API_KEY")}, google.NewLLM),
		TTS: run.DefaultTTS,
	}); err != nil {
		log.Fatal(err)
	}
}
