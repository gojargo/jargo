// Command voice-deepseek is the voice bot with a DeepSeek LLM, served through
// jargo's OpenAI-compatible client (Deepgram STT, ElevenLabs TTS). Run it, open
// the URL it logs, click start and allow the mic:
//
//	DEEPGRAM_API_KEY=… DEEPSEEK_API_KEY=… ELEVENLABS_API_KEY=… go run ./examples/voice/deepseek
package main

import (
	"log"
	"os"

	"github.com/gojargo/jargo/examples/voice/run"
	"github.com/gojargo/jargo/provider/deepseek"
	"github.com/gojargo/jargo/provider/openai"
)

func main() {
	if err := run.Voice(run.Builders{
		STT: run.DefaultSTT,
		LLM: run.Service(openai.LLMConfig{APIKey: os.Getenv("DEEPSEEK_API_KEY")}, deepseek.NewLLM),
		TTS: run.DefaultTTS,
	}); err != nil {
		log.Fatal(err)
	}
}
