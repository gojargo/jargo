// Command voice-groq is the voice bot with Groq for both speech-to-text
// (Whisper) and the LLM, served through jargo's OpenAI-compatible clients
// (ElevenLabs TTS). Run it, open the URL it logs, click start and allow the mic:
//
//	GROQ_API_KEY=… ELEVENLABS_API_KEY=… go run ./examples/voice/groq
//
// Groq STT is segmented, so it relies on turn-taking to delimit turns — keep the
// ONNX runtime available (JARGO_ONNXRUNTIME_LIB) for the best experience.
package main

import (
	"log"
	"os"

	"github.com/gojargo/jargo/examples/voice/run"
	"github.com/gojargo/jargo/provider/groq"
	"github.com/gojargo/jargo/provider/openai"
)

func main() {
	key := os.Getenv("GROQ_API_KEY")
	if err := run.Voice(run.Builders{
		STT: run.Service(openai.STTConfig{APIKey: key, SampleRate: run.SampleRate}, groq.NewSTT),
		LLM: run.Service(openai.LLMConfig{APIKey: key}, groq.NewLLM),
		TTS: run.DefaultTTS,
	}); err != nil {
		log.Fatal(err)
	}
}
