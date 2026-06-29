// Command voice-hume is the voice bot with Hume Octave text-to-speech (Deepgram
// STT, Anthropic LLM). Set HUME_VOICE_ID to pick a voice. Run it, open the URL
// it logs, click start and allow the mic:
//
//	DEEPGRAM_API_KEY=… ANTHROPIC_API_KEY=… HUME_API_KEY=… HUME_VOICE_ID=… go run ./examples/voice/hume
package main

import (
	"log"
	"os"

	"github.com/gojargo/jargo/examples/voice/run"
	"github.com/gojargo/jargo/provider/hume"
)

func main() {
	if err := run.Voice(run.Builders{
		STT: run.DefaultSTT,
		LLM: run.DefaultLLM,
		TTS: run.Service(hume.Config{
			APIKey:  os.Getenv("HUME_API_KEY"),
			VoiceID: os.Getenv("HUME_VOICE_ID"),
		}, hume.NewTTS),
	}); err != nil {
		log.Fatal(err)
	}
}
