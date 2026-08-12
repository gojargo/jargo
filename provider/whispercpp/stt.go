package whispercpp

import (
	"cmp"

	"github.com/gojargo/jargo/provider/openai/chat"
	"github.com/gojargo/jargo/service/stt"
)

// NewSTT builds a whisper.cpp transcription service.
func NewSTT(cfg Config) *stt.SegmentService {
	return chat.NewCompatSTT("WhisperCppSTT", cfg.BaseURL, defaultModel, chat.STTConfig{
		APIKey:     cfg.APIKey,
		BaseURL:    cfg.BaseURL,
		Model:      cfg.Model,
		Language:   cfg.Language,
		SampleRate: cfg.SampleRate,
		// Whisper runs on whatever hardware hosts it rather than against a hosted
		// endpoint, so it carries the fallback rather than the OpenAI measurement
		// the compatible service would otherwise report.
		TTFSP99: cmp.Or(cfg.TTFSP99, stt.WhisperTTFSP99),
	})
}
