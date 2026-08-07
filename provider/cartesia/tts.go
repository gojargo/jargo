package cartesia

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/http"
	"regexp"
	"strings"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/language"
	"github.com/gojargo/jargo/service/tts"
	"github.com/gojargo/jargo/service/wsutil"
	uctx "github.com/gojargo/jargo/utils/context"
)

// GenerationConfig guides Cartesia generation; it applies to sonic-3 and
// sonic-3.5 models. Fields left at their zero value are omitted.
type GenerationConfig struct {
	// Volume multiplies the generated speech volume (0.5 to 2.0).
	Volume *float64 `json:"volume,omitempty"`
	// Speed multiplies the speaking rate (0.6 to 1.5).
	Speed *float64 `json:"speed,omitempty"`
	// Emotion guides the emotional tone (e.g. "neutral", "excited", "sad").
	Emotion string `json:"emotion,omitempty"`
}

// NewTTS builds a Cartesia TTS service.
func NewTTS(cfg Config) *tts.Base {
	if cfg.URL == "" {
		cfg.URL = defaultURL
	}
	if cfg.Version == "" {
		cfg.Version = defaultVersion
	}
	if cfg.Model == "" {
		cfg.Model = defaultModel
	}
	if cfg.VoiceID == "" {
		cfg.VoiceID = defaultVoiceID
	}
	if cfg.SampleRate == 0 {
		cfg.SampleRate = defaultSampleRate
	}
	if cfg.Encoding == "" {
		cfg.Encoding = defaultEncoding
	}
	if cfg.Container == "" {
		cfg.Container = defaultContainer
	}
	s := &synthesizer{cfg: cfg}
	if cfg.WordTimestamps {
		// Only the timestamp-aware type implements tts.WordTimestamps, so the base
		// takes the word-aligned path solely when the caller opts in.
		return tts.New("CartesiaTTS", &timedSynthesizer{synthesizer: s})
	}
	return tts.New("CartesiaTTS", s)
}

type synthesizer struct {
	cfg Config
}

// Metadata reports the Cartesia model and voice synthesis is billed against.
func (s *synthesizer) Metadata() tts.Metadata {
	return tts.Metadata{Model: s.cfg.Model, VoiceID: s.cfg.VoiceID}
}

// timedSynthesizer adds word-timestamp streaming on top of synthesizer. It
// implements tts.WordTimestamps.
type timedSynthesizer struct {
	*synthesizer
}

// SampleRate reports the requested PCM output rate.
func (s *synthesizer) SampleRate() int { return s.cfg.SampleRate }

// spacelessLanguage reports whether the configured language is written without
// spaces between words. Chinese and Japanese are, and this provider reports
// their characters separately but grouped into one message, which is what makes
// the message rather than the character the unit worth reporting.
func (s *synthesizer) spacelessLanguage() bool {
	switch s.cfg.Language.BaseCode() {
	case "zh", "ja":
		return true
	default:
		return false
	}
}

// cartesiaLanguage maps a Language to Cartesia's language code: Cartesia wants
// the base code, returned only for languages it supports; otherwise "" (Cartesia
// defaults to English).
func cartesiaLanguage(l language.Language) string {
	switch base := l.BaseCode(); base {
	case "cs", "da", "de", "el", "en", "es", "fi", "fr", "gu", "he", "hi",
		"hr", "hu", "id", "it", "ja", "ka", "kn", "ko", "ml", "mr", "ms",
		"nl", "no", "pa", "pl", "pt", "ro", "ru", "sk", "sv", "ta", "te",
		"th", "tl", "tr", "uk", "vi", "zh":
		return base
	default:
		return ""
	}
}

// wsMessage is the subset of a Cartesia WebSocket message we read.
type wsMessage struct {
	Type           string         `json:"type"`
	Data           string         `json:"data"`
	Message        string         `json:"message"`
	WordTimestamps *wsWordTimings `json:"word_timestamps"`
}

// wsWordTimings is the payload of a Cartesia "timestamps" message: parallel
// arrays of spoken words and their start times, in seconds from the start of the
// synthesis.
type wsWordTimings struct {
	Words []string  `json:"words"`
	Start []float64 `json:"start"`
}

// Synthesize opens a session, sends the transcript, and streams audio chunks.
func (s *synthesizer) RunTTS(ctx context.Context, text, _ string, yield func(f frames.Frame) error) error {
	emit := tts.PCMYielder(yield, s.SampleRate())
	conn, err := s.dial(ctx)
	if err != nil {
		return err
	}
	defer func() { _ = conn.Close(websocket.StatusNormalClosure, "") }()

	if err := s.request(ctx, conn, text, false); err != nil {
		return err
	}
	return s.receive(ctx, conn, emit, nil)
}

// SynthesizeTimed streams audio and reports per-word timing, implementing
// tts.WordTimestamps. It requests timestamps and forwards each Cartesia
// "timestamps" message (after merging any punctuation-only tokens into the
// preceding word) to word.
func (s *timedSynthesizer) RunTTSTimed(
	ctx context.Context,
	text, _ string,
	yield func(f frames.Frame) error,
	word func(words []uctx.WordTiming, opts tts.WordTimingOptions) error,
) error {
	emit := tts.PCMYielder(yield, s.SampleRate())
	conn, err := s.dial(ctx)
	if err != nil {
		return err
	}
	defer func() { _ = conn.Close(websocket.StatusNormalClosure, "") }()

	if err := s.request(ctx, conn, text, true); err != nil {
		return err
	}
	return s.receive(ctx, conn, emit, word)
}

func (s *synthesizer) dial(ctx context.Context) (*wsutil.Conn, error) {
	header := http.Header{}
	header.Set("X-API-Key", s.cfg.APIKey)
	header.Set("Cartesia-Version", s.cfg.Version)
	return wsutil.Dial(ctx, s.cfg.URL, header, readLimit)
}

func (s *synthesizer) request(ctx context.Context, conn *wsutil.Conn, text string, timestamps bool) error {
	msg := map[string]any{
		"model_id":   s.cfg.Model,
		"transcript": text,
		"voice":      map[string]any{"mode": "id", "id": s.cfg.VoiceID},
		"output_format": map[string]any{
			"container":   s.cfg.Container,
			"encoding":    s.cfg.Encoding,
			"sample_rate": s.cfg.SampleRate,
		},
		"context_id": "jargo",
		"continue":   false,
	}
	if timestamps {
		msg["add_timestamps"] = true
		msg["use_normalized_timestamps"] = false
	}
	if lang := cartesiaLanguage(s.cfg.Language); lang != "" {
		msg["language"] = lang
	}
	if s.cfg.GenerationConfig != nil {
		msg["generation_config"] = s.cfg.GenerationConfig
	}
	if s.cfg.PronunciationDictID != "" {
		msg["pronunciation_dict_id"] = s.cfg.PronunciationDictID
	}
	payload, err := json.Marshal(msg)
	if err != nil {
		return err
	}
	return conn.Write(ctx, websocket.MessageText, payload)
}

// receive reads audio chunks and, when word is non-nil, word-timestamp messages
// until the transcript is done.
func (s *synthesizer) receive(
	ctx context.Context,
	conn *wsutil.Conn,
	emit func(pcm []byte) error,
	word func(words []uctx.WordTiming, opts tts.WordTimingOptions) error,
) error {
	for {
		_, data, err := conn.Read(ctx)
		if err != nil {
			return err
		}
		var m wsMessage
		if err := json.Unmarshal(data, &m); err != nil {
			continue
		}
		switch m.Type {
		case "chunk":
			pcm, err := base64.StdEncoding.DecodeString(m.Data)
			if err != nil {
				return err
			}
			if err := emit(pcm); err != nil {
				return err
			}
		case "timestamps":
			if word != nil {
				if err := emitWordTimings(m.WordTimestamps, s.spacelessLanguage(), word); err != nil {
					return err
				}
			}
		case "done":
			return nil
		case "error":
			return fmt.Errorf("%w: %s", errProtocol, m.Message)
		}
	}
}

// cartesiaTagPattern matches the markup this provider accepts in the text it is
// given and reports back in its timings.
//
//nolint:gochecknoglobals // compiled once, read-only
var cartesiaTagPattern = regexp.MustCompile(`(?i)</?(?:spell|emotion|break|volume|speed)\b[^>]*>`)

// stripCartesiaTags removes that markup from one reported token, collapsing the
// whitespace it leaves behind. A token that was nothing but markup comes back
// empty and is dropped by the caller.
func stripCartesiaTags(text string) string {
	text = cartesiaTagPattern.ReplaceAllString(text, " ")
	return strings.TrimSpace(strings.Join(strings.Fields(text), " "))
}

// emitWordTimings normalizes one batch of reported timings and forwards it.
//
// The markup this provider was given comes back in the tokens it reports, and
// has to come off before anything downstream tries to match them against the
// text. A language written without spaces between words has its characters
// reported separately but grouped into one message, so the message is joined
// into a single token at the offset of its first character: that is the unit a
// reader of the language recognizes, where the characters alone are not.
func emitWordTimings(
	wt *wsWordTimings,
	spaceless bool,
	word func(words []uctx.WordTiming, opts tts.WordTimingOptions) error,
) error {
	if wt == nil {
		return nil
	}
	opts := tts.WordTimingOptions{IncludesInterFrameSpaces: spaceless}

	if spaceless {
		var combined strings.Builder
		for _, w := range wt.Words {
			combined.WriteString(stripCartesiaTags(w))
		}
		if combined.Len() == 0 || len(wt.Start) == 0 {
			return nil
		}
		return word([]uctx.WordTiming{{Word: combined.String(), Offset: wt.Start[0]}}, opts)
	}

	batch := make([]uctx.WordTiming, 0, len(wt.Words))
	for i, w := range wt.Words {
		cleaned := stripCartesiaTags(w)
		if cleaned == "" {
			continue
		}
		var start float64
		if i < len(wt.Start) {
			start = wt.Start[i]
		}
		batch = append(batch, uctx.WordTiming{Word: cleaned, Offset: start})
	}
	if len(batch) == 0 {
		return nil
	}
	return word(batch, opts)
}
