// Command voice-nebius is the voice bot with a Nebius AI Studio LLM, served
// through jargo's OpenAI-compatible client (Deepgram STT, ElevenLabs TTS). Run
// it, open the URL it logs, click start and allow the mic:
//
//	DEEPGRAM_API_KEY=… NEBIUS_API_KEY=… ELEVENLABS_API_KEY=… go run ./examples/voice/nebius
package main

import (
	"log"
	"os"

	"github.com/gojargo/jargo/examples/voice/run"
	"github.com/gojargo/jargo/provider/nebius"
	"github.com/gojargo/jargo/provider/openai"
)

func main() {
	if err := run.Voice(run.Builders{
		STT: run.DefaultSTT,
		LLM: run.Service(openai.LLMConfig{APIKey: os.Getenv("NEBIUS_API_KEY")}, nebius.NewLLM),
		TTS: run.DefaultTTS,
	}); err != nil {
		log.Fatal(err)
	}
}
