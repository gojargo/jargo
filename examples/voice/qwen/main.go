// Command voice-qwen is the voice bot with an Alibaba Qwen LLM (DashScope),
// served through jargo's OpenAI-compatible client (Deepgram STT, ElevenLabs
// TTS). Run it, open the URL it logs, click start and allow the mic:
//
//	DEEPGRAM_API_KEY=… DASHSCOPE_API_KEY=… ELEVENLABS_API_KEY=… go run ./examples/voice/qwen
package main

import (
	"log"
	"os"

	"github.com/gojargo/jargo/examples/voice/run"
	"github.com/gojargo/jargo/provider/openai"
	"github.com/gojargo/jargo/provider/qwen"
)

func main() {
	if err := run.Voice(run.Builders{
		STT: run.DefaultSTT,
		LLM: run.Service(openai.LLMConfig{APIKey: os.Getenv("DASHSCOPE_API_KEY")}, qwen.NewLLM),
		TTS: run.DefaultTTS,
	}); err != nil {
		log.Fatal(err)
	}
}
