package gemini

import (
	"bytes"
	"cmp"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/gojargo/jargo/internal/validate"
	"github.com/gojargo/jargo/language"
	"github.com/gojargo/jargo/service/stt"
)

const sttEndpoint = "https://speech.googleapis.com/v1/speech:recognize"

// STTConfig configures the Google Cloud Speech-to-Text batch service. Only batch
// recognition is supported; streaming recognition requires gRPC and is out of
// scope.
type STTConfig struct {
	// APIKey is the Google API key. Required.
	APIKey string `validate:"required"`
	// Language of the audio, mapped to a Google language code; the zero value
	// uses US English.
	Language language.Language
	// SampleRate is the input audio sample rate; 0 uses the transport's rate.
	SampleRate int

	// TTFSP99 overrides the measured transcript latency the turn strategies
	// size their wait by; 0 uses stt.GoogleTTFSP99.
	TTFSP99 time.Duration
}

// Validate reports whether the configuration is usable.
func (c STTConfig) Validate() error { return validate.Struct(c) }

// NewSTT builds a Google batch Speech-to-Text service. It is segmented: a turn
// detector upstream delimits each utterance, which is transcribed in one
// request. Streaming recognition needs gRPC and is not supported.
func NewSTT(cfg STTConfig) *stt.SegmentService {
	t := &sttTranscriber{cfg: cfg, http: &http.Client{}}
	return stt.NewSegment("GoogleSTT", t, cfg.SampleRate)
}

type sttTranscriber struct {
	cfg  STTConfig
	http *http.Client
}

// Metadata reports the transcript latency the turn strategies size their
// wait by.
func (t *sttTranscriber) Metadata() stt.Metadata {
	return stt.Metadata{TTFSP99: cmp.Or(t.cfg.TTFSP99, stt.GoogleTTFSP99)}
}

// sttResponse is the subset of a recognize response the transcriber reads.
type sttResponse struct {
	Results []sttResult `json:"results"`
}

type sttResult struct {
	Alternatives []sttAlternative `json:"alternatives"`
}

type sttAlternative struct {
	Transcript string `json:"transcript"`
}

// requestBody builds the recognize request body for the buffered segment.
func (t *sttTranscriber) requestBody(audio []byte, sampleRate int) ([]byte, error) {
	return json.Marshal(map[string]any{
		"config": map[string]any{
			"encoding":        "LINEAR16",
			"sampleRateHertz": sampleRate,
			"languageCode":    languageToGoogleSTT(t.cfg.Language),
		},
		"audio": map[string]any{
			"content": base64.StdEncoding.EncodeToString(audio),
		},
	})
}

// Transcribe sends the segment to the batch recognize endpoint and returns the
// joined transcript.
func (t *sttTranscriber) Transcribe(ctx context.Context, audio []byte, sampleRate int) (string, error) {
	body, err := t.requestBody(audio, sampleRate)
	if err != nil {
		return "", err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, sttEndpoint+"?key="+t.cfg.APIKey,
		bytes.NewReader(body))
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", "application/json")
	return t.recognize(req)
}

// recognize issues req and joins the returned alternatives into one transcript.
func (t *sttTranscriber) recognize(req *http.Request) (string, error) {
	resp, err := t.http.Do(req)
	if err != nil {
		return "", err
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusOK {
		msg, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return "", fmt.Errorf("%w %d: %s", errStatus, resp.StatusCode, msg)
	}
	var out sttResponse
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return "", err
	}
	return joinTranscripts(out.Results), nil
}

// joinTranscripts concatenates the first alternative of each result.
func joinTranscripts(results []sttResult) string {
	parts := make([]string, 0, len(results))
	for _, r := range results {
		if len(r.Alternatives) > 0 && r.Alternatives[0].Transcript != "" {
			parts = append(parts, r.Alternatives[0].Transcript)
		}
	}
	return strings.TrimSpace(strings.Join(parts, " "))
}

// languageToGoogleSTT maps a language to a Google STT language code, defaulting
// to the full code when the base language is not modeled.
func languageToGoogleSTT(l language.Language) string {
	switch l.BaseCode() {
	case "en":
		return defaultLangCode
	case "fr":
		return "fr-FR"
	case "es":
		return "es-ES"
	case "de":
		return "de-DE"
	case "it":
		return "it-IT"
	case "nl":
		return "nl-NL"
	case "pt":
		return "pt-PT"
	case "pl":
		return "pl-PL"
	case "ru":
		return "ru-RU"
	case "ja":
		return "ja-JP"
	case "ko":
		return "ko-KR"
	case "zh":
		return "cmn-Hans-CN"
	default:
		if code := l.Code(); code != "" {
			return code
		}
		return defaultLangCode
	}
}
