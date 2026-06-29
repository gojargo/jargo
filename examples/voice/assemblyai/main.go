// Command voice-assemblyai is the voice bot with AssemblyAI speech-to-text
// (Anthropic LLM, ElevenLabs TTS). Run it, open the URL it logs, click start
// and allow the mic:
//
//	ASSEMBLYAI_API_KEY=… ANTHROPIC_API_KEY=… ELEVENLABS_API_KEY=… go run ./examples/voice/assemblyai
package main

import (
	"log"
	"os"

	"github.com/gojargo/jargo/examples/voice/run"
	"github.com/gojargo/jargo/provider/assemblyai"
)

func main() {
	if err := run.Voice(run.Builders{
		STT: run.Service(assemblyai.Config{
			APIKey:     os.Getenv("ASSEMBLYAI_API_KEY"),
			SampleRate: run.SampleRate,
		}, assemblyai.NewSTT),
		LLM: run.DefaultLLM,
		TTS: run.DefaultTTS,
	}); err != nil {
		log.Fatal(err)
	}
}
