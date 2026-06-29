// Command voice-minimax is the voice bot with MiniMax text-to-speech (Deepgram
// STT, Anthropic LLM). MINIMAX_GROUP_ID is required; set MINIMAX_VOICE_ID to
// pick a voice. Run it, open the URL it logs, click start and allow the mic:
//
//	DEEPGRAM_API_KEY=… ANTHROPIC_API_KEY=… MINIMAX_API_KEY=… MINIMAX_GROUP_ID=… go run ./examples/voice/minimax
package main

import (
	"log"
	"os"

	"github.com/gojargo/jargo/examples/voice/run"
	"github.com/gojargo/jargo/provider/minimax"
)

func main() {
	if err := run.Voice(run.Builders{
		STT: run.DefaultSTT,
		LLM: run.DefaultLLM,
		TTS: run.Service(minimax.Config{
			APIKey:  os.Getenv("MINIMAX_API_KEY"),
			GroupID: os.Getenv("MINIMAX_GROUP_ID"),
			VoiceID: os.Getenv("MINIMAX_VOICE_ID"),
		}, minimax.NewTTS),
	}); err != nil {
		log.Fatal(err)
	}
}
