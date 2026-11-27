package whispercpp

import (
	"github.com/gojargo/jargo/provider/openai/chat"
	"github.com/gojargo/jargo/service/stt"
)

// STTService is the whisper.cpp speech-to-text service NewSTT builds.
type STTService struct {
	*stt.SegmentService
}

// NewSTT builds a whisper.cpp transcription service.
func NewSTT(cfg Config) *STTService {
	s := &STTService{}
	s.SegmentService = buildSTT(cfg, stt.WithSelf(s))
	return s
}

// buildSTT builds the service NewSTT returns, registering the
// service opts names as the processor the pipeline runs.
func buildSTT(cfg Config, opts ...stt.Option) *stt.SegmentService {
	return chat.NewCompatSTT("WhisperCppSTT", cfg.BaseURL, defaultModel, chat.STTConfig{
		APIKey:     cfg.APIKey,
		BaseURL:    cfg.BaseURL,
		Model:      cfg.Model,
		Language:   cfg.Language,
		SampleRate: cfg.SampleRate,
		// Whisper runs on whatever hardware hosts it rather than against a hosted
		// endpoint, so it carries the fallback rather than the OpenAI measurement
		// the compatible service would otherwise report.
		TTFSP99: stt.TTFSP99Or(cfg.TTFSP99, stt.WhisperTTFSP99),
	}, opts...)
}
