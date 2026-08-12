package groq

import (
	"bytes"
	"cmp"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"strconv"
	"time"

	"github.com/gojargo/jargo/internal/validate"
	"github.com/gojargo/jargo/language"
	"github.com/gojargo/jargo/service/stt"
)

// STTConfig configures the Groq Whisper transcription service. It targets Groq's
// OpenAI-compatible /audio/transcriptions endpoint.
type STTConfig struct {
	// APIKey is the Groq API key. Required.
	APIKey string `validate:"required"`
	// BaseURL overrides the API base.
	BaseURL string
	// Model is the transcription model; empty uses the provider default.
	Model string
	// Language of the audio, sent as an ISO code; empty defaults to English.
	// Mapped to the base code.
	Language language.Language
	// Prompt steers the model's style or continues a previous segment; empty
	// omits it.
	Prompt string
	// Temperature is the sampling temperature (0.0 to 1.0); nil omits it.
	Temperature *float64
	// SampleRate is the input audio sample rate; 0 uses the transport's rate.
	SampleRate int

	// TTFSP99 overrides the measured transcript latency the turn strategies
	// size their wait by; 0 uses stt.GroqTTFSP99.
	TTFSP99 time.Duration
}

// Validate reports whether the configuration is usable.
func (c STTConfig) Validate() error { return validate.Struct(c) }

// NewSTT builds a Groq Whisper transcription service. It is segmented: a turn
// detector upstream delimits each utterance, which is transcribed in one request.
func NewSTT(cfg STTConfig) *stt.SegmentService {
	if cfg.BaseURL == "" {
		cfg.BaseURL = baseURL
	}
	if cfg.Model == "" {
		cfg.Model = defaultSTTModel
	}
	if cfg.Language == "" {
		cfg.Language = language.English
	}
	return stt.NewSegment("GroqSTT", &sttTranscriber{cfg: cfg, http: &http.Client{}}, cfg.SampleRate)
}

type sttTranscriber struct {
	cfg  STTConfig
	http *http.Client
}

// Metadata reports the model in use and the transcript latency the turn
// strategies size their wait by.
func (t *sttTranscriber) Metadata() stt.Metadata {
	return stt.Metadata{TTFSP99: cmp.Or(t.cfg.TTFSP99, stt.GroqTTFSP99), Model: t.cfg.Model}
}

// writeFields writes the transcription form fields, omitting optional ones that
// are unset.
func (t *sttTranscriber) writeFields(w *multipart.Writer) error {
	if err := w.WriteField("model", t.cfg.Model); err != nil {
		return err
	}
	if err := w.WriteField("response_format", "json"); err != nil {
		return err
	}
	if lang := t.cfg.Language.BaseCode(); lang != "" {
		if err := w.WriteField("language", lang); err != nil {
			return err
		}
	}
	if t.cfg.Prompt != "" {
		if err := w.WriteField("prompt", t.cfg.Prompt); err != nil {
			return err
		}
	}
	if t.cfg.Temperature != nil {
		if err := w.WriteField("temperature", strconv.FormatFloat(*t.cfg.Temperature, 'g', -1, 64)); err != nil {
			return err
		}
	}
	return nil
}

// Transcribe uploads the segment as a WAV file and returns the transcript.
func (t *sttTranscriber) Transcribe(ctx context.Context, audio []byte, sampleRate int) (string, error) {
	var body bytes.Buffer
	w := multipart.NewWriter(&body)
	part, err := w.CreateFormFile("file", "audio.wav")
	if err != nil {
		return "", err
	}
	if _, err = part.Write(stt.WAV(audio, sampleRate, 1)); err != nil {
		return "", err
	}
	if err = t.writeFields(w); err != nil {
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
