package elevenlabs

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
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

// errSTTStatus is returned when the transcription API responds with a non-200
// status.
//
//nolint:gochecknoglobals // sentinel error
var errSTTStatus = errors.New("elevenlabs: unexpected stt status")

const (
	// defaultSTTBaseURL is the ElevenLabs HTTP API base.
	defaultSTTBaseURL = "https://api.elevenlabs.io"
	// defaultSTTModel is the ElevenLabs scribe transcription model.
	defaultSTTModel = "scribe_v2"
	// sttTTFSP99 is the time-to-final-segment P99 latency reported to downstream.
	sttTTFSP99 = 2010 * time.Millisecond
)

// STTConfig configures the ElevenLabs batch STT service. It transcribes one
// utterance per request; a turn detector upstream delimits each segment.
type STTConfig struct {
	// APIKey is the ElevenLabs API key, sent as the xi-api-key header. Required.
	APIKey string `validate:"required"`
	// BaseURL overrides the HTTP API base; empty uses the hosted API.
	BaseURL string
	// Model is the transcription model; empty uses the scribe default.
	Model string
	// Language of the audio; the zero value omits it (auto-detect). Mapped to
	// ElevenLabs' ISO-639-3 code.
	Language language.Language
	// TagAudioEvents includes audio events like (laughter) in the transcript;
	// nil leaves it unset.
	TagAudioEvents *bool
	// Keyterms biases transcription towards the given terms; empty applies none.
	Keyterms []string
	// SampleRate is the input audio sample rate; 0 uses the transport's rate.
	SampleRate int
}

// Validate reports whether the configuration is usable.
func (c STTConfig) Validate() error { return validate.Struct(c) }

// NewSTT builds an ElevenLabs batch STT service. It is segmented: a turn
// detector upstream delimits each utterance, which is transcribed in one request.
func NewSTT(cfg STTConfig) *stt.SegmentService {
	if cfg.BaseURL == "" {
		cfg.BaseURL = defaultSTTBaseURL
	}
	if cfg.Model == "" {
		cfg.Model = defaultSTTModel
	}
	t := &sttTranscriber{cfg: cfg, http: &http.Client{}}
	return stt.NewSegment("ElevenLabsSTT", t, cfg.SampleRate)
}

type sttTranscriber struct {
	cfg  STTConfig
	http *http.Client
}

// Metadata reports ElevenLabs' time-to-final-segment latency to downstream
// processors.
func (t *sttTranscriber) Metadata() stt.Metadata {
	return stt.Metadata{TTFSP99: sttTTFSP99, Model: t.cfg.Model}
}

// elevenlabsSTTLanguage maps a Language to ElevenLabs' ISO-639-3 STT language
// code, returned only for languages it supports; otherwise "" (auto-detect). A
// map keyed by the base code avoids the enum-exhaustiveness lint of a switch.
func elevenlabsSTTLanguage(l language.Language) string {
	codes := map[string]string{
		"ar": "ara", "bg": "bul", "cs": "ces", "da": "dan", "de": "deu",
		"el": "ell", "en": "eng", "es": "spa", "fi": "fin", langFil: langFil,
		"fr": "fra", "he": "heb", "hi": "hin", "hr": "hrv", "hu": "hun",
		"id": "ind", "it": "ita", "ja": "jpn", "ko": "kor", "ms": "msa",
		"nl": "nld", "no": "nor", "pl": "pol", "pt": "por", "ro": "ron",
		"ru": "rus", "sk": "slk", "sv": "swe", "ta": "tam", "th": "tha",
		"tr": "tur", "uk": "ukr", "vi": "vie", "zh": "zho",
	}
	return codes[l.BaseCode()]
}

// writeFields writes the transcription form fields, omitting optional ones that
// are unset.
func (t *sttTranscriber) writeFields(w *multipart.Writer) error {
	if err := w.WriteField("model_id", t.cfg.Model); err != nil {
		return err
	}
	if code := elevenlabsSTTLanguage(t.cfg.Language); code != "" {
		if err := w.WriteField("language_code", code); err != nil {
			return err
		}
	}
	if t.cfg.TagAudioEvents != nil {
		if err := w.WriteField("tag_audio_events", strconv.FormatBool(*t.cfg.TagAudioEvents)); err != nil {
			return err
		}
	}
	for _, kt := range t.cfg.Keyterms {
		if err := w.WriteField("keyterms", kt); err != nil {
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

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, t.cfg.BaseURL+"/v1/speech-to-text", &body)
	if err != nil {
		return "", err
	}
	req.Header.Set("xi-api-key", t.cfg.APIKey)
	req.Header.Set("Content-Type", w.FormDataContentType())

	resp, err := t.http.Do(req)
	if err != nil {
		return "", err
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusOK {
		msg, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return "", fmt.Errorf("%w %d: %s", errSTTStatus, resp.StatusCode, msg)
	}
	var out struct {
		Text string `json:"text"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return "", err
	}
	return out.Text, nil
}
