// Command voice-ollama is the voice bot with a local Ollama LLM, served through
// jargo's OpenAI-compatible client — no LLM API key needed (Deepgram STT,
// ElevenLabs TTS). Start Ollama first (e.g. `ollama run llama3.2`), then run it,
// open the URL it logs, click start and allow the mic:
//
//	DEEPGRAM_API_KEY=… ELEVENLABS_API_KEY=… go run ./examples/voice/ollama
package main

import (
	"log"

	"github.com/gojargo/jargo/examples/voice/run"
	"github.com/gojargo/jargo/provider/ollama"
	"github.com/gojargo/jargo/provider/openai"
)

func main() {
	if err := run.Voice(run.Builders{
		STT: run.DefaultSTT,
		LLM: run.Service(openai.LLMConfig{}, ollama.NewLLM), // local; no API key required
		TTS: run.DefaultTTS,
	}); err != nil {
		log.Fatal(err)
	}
}
