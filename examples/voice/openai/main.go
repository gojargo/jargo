// Command voice-openai is the all-OpenAI voice bot: OpenAI for speech-to-text,
// the LLM, and text-to-speech. Run it, open the URL it logs, click start and
// allow the mic:
//
//	OPENAI_API_KEY=… go run ./examples/voice/openai
//
// OpenAI STT is segmented (it transcribes a finished utterance rather than
// streaming partials), so it relies on turn-taking to delimit turns — keep the
// ONNX runtime available (JARGO_ONNXRUNTIME_LIB) for the best experience.
package main

import (
	"log"
	"os"

	"github.com/gojargo/jargo/examples/voice/run"
	"github.com/gojargo/jargo/provider/openai"
)

func main() {
	key := os.Getenv("OPENAI_API_KEY")
	if err := run.Voice(run.Builders{
		STT: run.Service(openai.STTConfig{APIKey: key, SampleRate: run.SampleRate}, openai.NewSTT),
		LLM: run.Service(openai.LLMConfig{APIKey: key}, openai.NewLLM),
		TTS: run.Service(openai.TTSConfig{APIKey: key}, openai.NewTTS),
	}); err != nil {
		log.Fatal(err)
	}
}
