package kokoro

import (
	"github.com/gojargo/jargo/provider/openai/chat"
	"github.com/gojargo/jargo/service/tts"
)

// NewTTS builds a Kokoro TTS service. Kokoro-FastAPI returns 24 kHz PCM, which
// matches the OpenAI "pcm" response format this wrapper requests.
func NewTTS(cfg Config) *tts.Base {
	return chat.NewCompatTTS("KokoroTTS", cfg.BaseURL, defaultModel, defaultVoice, chat.TTSConfig{
		APIKey:  cfg.APIKey,
		BaseURL: cfg.BaseURL,
		Model:   cfg.Model,
		Voice:   cfg.Voice,
		Speed:   cfg.Speed,
	})
}
