// Command voice-together is the voice bot with a Together AI LLM, served through
// jargo's OpenAI-compatible client (Deepgram STT, ElevenLabs TTS). Run it, open
// the URL it logs, click start and allow the mic:
//
//	DEEPGRAM_API_KEY=… TOGETHER_API_KEY=… ELEVENLABS_API_KEY=… go run ./examples/voice/together
package main

import (
	"log"
	"os"

	"github.com/gojargo/jargo/examples/voice/run"
	"github.com/gojargo/jargo/provider/openai"
	"github.com/gojargo/jargo/provider/together"
)

func main() {
	if err := run.Voice(run.Builders{
		STT: run.DefaultSTT,
		LLM: run.Service(openai.LLMConfig{APIKey: os.Getenv("TOGETHER_API_KEY")}, together.NewLLM),
		TTS: run.DefaultTTS,
	}); err != nil {
		log.Fatal(err)
	}
}
