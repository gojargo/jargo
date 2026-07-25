package whispercpp

import (
	"github.com/gojargo/jargo/provider/openai"
	"github.com/gojargo/jargo/service/stt"
)

// NewSTT builds a whisper.cpp transcription service.
func NewSTT(cfg Config) *stt.SegmentService {
	return openai.NewCompatSTT("WhisperCppSTT", cfg.BaseURL, defaultModel, openai.STTConfig{
		APIKey:     cfg.APIKey,
		BaseURL:    cfg.BaseURL,
		Model:      cfg.Model,
		Language:   cfg.Language,
		SampleRate: cfg.SampleRate,
	})
}
