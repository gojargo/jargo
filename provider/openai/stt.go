package openai

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"os"
	"strconv"

	"github.com/gojargo/jargo/language"
	"github.com/gojargo/jargo/service/stt"
)

const defaultSTTModel = "gpt-4o-transcribe"

// STTConfig configures an OpenAI (or OpenAI-compatible) transcription service.
type STTConfig struct {
	// APIKey is the API key; empty uses the provider's env var.
	APIKey string
	// BaseURL overrides the API base.
	BaseURL string
	// Model is the transcription model; empty uses the provider default.
	Model string
	// Language of the audio, sent as an ISO code; the zero value omits it
	// (auto-detect). Mapped to the base code.
	Language language.Language
	// Prompt steers the model's style or continues a previous segment; empty
	// omits it.
	Prompt string
	// Temperature is the sampling temperature (0.0 to 1.0); nil omits it.
	Temperature *float64
	// SampleRate is the input audio sample rate; 0 uses the transport's rate.
	SampleRate int
}

// NewSTT builds an OpenAI transcription service. It is segmented: a turn
// detector upstream delimits each utterance, which is transcribed in one request.
func NewSTT(cfg STTConfig) *stt.SegmentService {
	return NewCompatSTT("OpenAISTT", defaultLLMBaseURL, "OPENAI_API_KEY", defaultSTTModel, cfg)
}

// NewCompatSTT builds a transcription service for any endpoint that implements
// OpenAI's /audio/transcriptions API (e.g. Groq).
func NewCompatSTT(name, baseURL, envVar, defaultModel string, cfg STTConfig) *stt.SegmentService {
	if cfg.APIKey == "" {
		cfg.APIKey = os.Getenv(envVar)
	}
	if cfg.BaseURL == "" {
		cfg.BaseURL = baseURL
	}
	if cfg.Model == "" {
		cfg.Model = defaultModel
	}
	t := &transcriber{cfg: cfg, http: &http.Client{}}
	return stt.NewSegment(name, t, cfg.SampleRate)
}

type transcriber struct {
	cfg  STTConfig
	http *http.Client
}

// writeFields writes the transcription form fields, omitting optional ones that
// are unset.
func writeFields(w *multipart.Writer, cfg *STTConfig) error {
	if err := w.WriteField("model", cfg.Model); err != nil {
		return err
	}
	if err := w.WriteField("response_format", "json"); err != nil {
		return err
	}
	if cfg.Language != "" {
		if err := w.WriteField("language", cfg.Language.BaseCode()); err != nil {
			return err
		}
	}
	if cfg.Prompt != "" {
		if err := w.WriteField("prompt", cfg.Prompt); err != nil {
			return err
		}
	}
	if cfg.Temperature != nil {
		if err := w.WriteField("temperature", strconv.FormatFloat(*cfg.Temperature, 'g', -1, 64)); err != nil {
			return err
		}
	}
	return nil
}

// Transcribe uploads the segment as a WAV file and returns the transcript.
func (t *transcriber) Transcribe(ctx context.Context, audio []byte, sampleRate int) (string, error) {
	var body bytes.Buffer
	w := multipart.NewWriter(&body)
	part, err := w.CreateFormFile("file", "audio.wav")
	if err != nil {
		return "", err
	}
	if _, err = part.Write(stt.WAV(audio, sampleRate, 1)); err != nil {
		return "", err
	}
	if err = writeFields(w, &t.cfg); err != nil {
		return "", err
	}
	if err = w.Close(); err != nil {
		return "", err
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, t.cfg.BaseURL+"/audio/transcriptions", &body)
	if err != nil {
		return "", err
	}
	req.Header.Set("Authorization", "Bearer "+t.cfg.APIKey)
	req.Header.Set("Content-Type", w.FormDataContentType())

	resp, err := t.http.Do(req)
	if err != nil {
		return "", err
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusOK {
		msg, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return "", fmt.Errorf("%w %d: %s", errStatus, resp.StatusCode, msg)
	}
	var out struct {
		Text string `json:"text"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return "", err
	}
	return out.Text, nil
}
