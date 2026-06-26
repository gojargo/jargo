// Package deepgram provides Deepgram's streaming speech-to-text service (over
// the live transcription WebSocket) and its Aura text-to-speech service.
//
// The STT service pushes InterimTranscriptionFrames and finalized
// TranscriptionFrames downstream; a finalized transcript with Deepgram's
// speech_final marks the end of the user's turn.
package deepgram

import (
	"context"
	"encoding/json"
	"net/http"
	"net/url"
	"os"
	"strconv"
	"sync"
	"time"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/language"
	"github.com/gojargo/jargo/service/stt"
)

const (
	listenURL       = "wss://api.deepgram.com/v1/listen"
	keepAlivePeriod = 8 * time.Second
	defaultSTTModel = "nova-3"
	defaultEncoding = "linear16"
	defaultChannels = 1
	// defaultEndpointingMs and defaultUtteranceEndMs preserve jargo's tuned
	// turn detection; Deepgram's own defaults are 10ms and unset respectively.
	defaultEndpointingMs  = 300
	defaultUtteranceEndMs = 1000
)

// Config configures the STT service. Fields left at their zero value fall back
// to jargo's defaults (which preserve its tuned turn detection); several of
// those defaults differ from Deepgram's own and are noted per field. Optional
// fields modeled as pointers or slices are omitted from the request when unset.
type Config struct {
	// APIKey is the Deepgram API key; empty uses the DEEPGRAM_API_KEY env var.
	APIKey string
	// ListenURL overrides the live-transcription WebSocket endpoint; empty uses
	// Deepgram's hosted endpoint.
	ListenURL string
	// Model is the Deepgram model; empty uses "nova-3".
	Model string
	// Language is the transcription language; empty uses English (US).
	Language language.Language
	// SampleRate is the input audio sample rate; 0 uses the transport's rate.
	SampleRate int
	// Encoding is the audio encoding; empty uses "linear16".
	Encoding string
	// Channels is the channel count; 0 uses 1.
	Channels int

	// InterimResults emits partial transcripts; nil defaults to true.
	InterimResults *bool
	// SmartFormat applies Deepgram smart formatting; nil defaults to true
	// (Deepgram's own default is false).
	SmartFormat *bool
	// Punctuate adds punctuation; nil defaults to true.
	Punctuate *bool
	// VADEvents requests speech-start/utterance events; nil defaults to true.
	VADEvents *bool
	// Endpointing is the silence in ms before a result is finalized; nil
	// defaults to 300 (Deepgram's own default is 10).
	Endpointing *int
	// UtteranceEndMs is the silence in ms before an utterance-end event; nil
	// defaults to 1000 (Deepgram leaves it unset).
	UtteranceEndMs *int

	// The remaining options are omitted from the request unless set, matching
	// Deepgram's defaults.

	// Numerals converts spoken numbers to numerals.
	Numerals *bool
	// ProfanityFilter filters profanity from transcripts.
	ProfanityFilter *bool
	// Diarize labels speakers in the transcript.
	Diarize *bool
	// DetectEntities enables named-entity detection.
	DetectEntities *bool
	// Dictation converts spoken commands (e.g. "comma") to punctuation.
	Dictation *bool
	// Multichannel transcribes each channel independently.
	Multichannel *bool
	// MipOptOut opts out of Deepgram's model-improvement program.
	MipOptOut *bool
	// Version pins the model version (e.g. "latest").
	Version string
	// Keywords boosts recognition of the given terms (legacy models); each entry
	// may carry an intensifier, e.g. "snuffleupagus:5".
	Keywords []string
	// Keyterm boosts recognition of the given terms (nova-3).
	Keyterm []string
	// Redact removes the given categories of sensitive data.
	Redact []string
	// Replace applies find/replace rules to the transcript.
	Replace []string
	// Search flags occurrences of the given terms.
	Search []string
	// Tag attaches billing tags to the request.
	Tag []string
	// ExtraQuery sets arbitrary additional Deepgram query parameters not modeled
	// above; values override any param of the same name set from other fields.
	ExtraQuery map[string]string
}

// NewSTT builds a Deepgram streaming STT service.
func NewSTT(cfg Config) *stt.StreamService {
	if cfg.APIKey == "" {
		cfg.APIKey = os.Getenv("DEEPGRAM_API_KEY")
	}
	if cfg.Model == "" {
		cfg.Model = defaultSTTModel
	}
	if cfg.Language == "" {
		cfg.Language = language.EnglishUS
	}
	if cfg.Encoding == "" {
		cfg.Encoding = defaultEncoding
	}
	if cfg.Channels == 0 {
		cfg.Channels = defaultChannels
	}
	if cfg.ListenURL == "" {
		cfg.ListenURL = listenURL
	}
	return stt.NewStream("DeepgramSTT", &connector{cfg: cfg}, cfg.SampleRate)
}

// deepgramLanguage maps a Language to Deepgram's code. Deepgram uses BCP-47
// codes directly, so the canonical code passes through unchanged.
func deepgramLanguage(l language.Language) string {
	return l.Code()
}

// query builds the live-transcription query string for the given sample rate.
func (cfg *Config) query(sampleRate int) url.Values {
	q := url.Values{}
	q.Set("model", cfg.Model)
	q.Set("language", deepgramLanguage(cfg.Language))
	q.Set("encoding", cfg.Encoding)
	q.Set("sample_rate", strconv.Itoa(sampleRate))
	q.Set("channels", strconv.Itoa(cfg.Channels))

	setBoolTrue(q, "interim_results", cfg.InterimResults)
	setBoolTrue(q, "smart_format", cfg.SmartFormat)
	setBoolTrue(q, "punctuate", cfg.Punctuate)
	setBoolTrue(q, "vad_events", cfg.VADEvents)
	setIntDefault(q, "endpointing", cfg.Endpointing, defaultEndpointingMs)
	setIntDefault(q, "utterance_end_ms", cfg.UtteranceEndMs, defaultUtteranceEndMs)

	setBoolOpt(q, "numerals", cfg.Numerals)
	setBoolOpt(q, "profanity_filter", cfg.ProfanityFilter)
	setBoolOpt(q, "diarize", cfg.Diarize)
	setBoolOpt(q, "detect_entities", cfg.DetectEntities)
	setBoolOpt(q, "dictation", cfg.Dictation)
	setBoolOpt(q, "multichannel", cfg.Multichannel)
	setBoolOpt(q, "mip_opt_out", cfg.MipOptOut)
	setStrOpt(q, "version", cfg.Version)

	addAll(q, "keywords", cfg.Keywords)
	addAll(q, "keyterm", cfg.Keyterm)
	addAll(q, "redact", cfg.Redact)
	addAll(q, "replace", cfg.Replace)
	addAll(q, "search", cfg.Search)
	addAll(q, "tag", cfg.Tag)

	for k, v := range cfg.ExtraQuery {
		q.Set(k, v)
	}
	return q
}

// setBoolTrue sets key to v, defaulting to true when v is nil.
func setBoolTrue(q url.Values, key string, v *bool) {
	val := true
	if v != nil {
		val = *v
	}
	q.Set(key, strconv.FormatBool(val))
}

// setBoolOpt sets key only when v is non-nil.
func setBoolOpt(q url.Values, key string, v *bool) {
	if v != nil {
		q.Set(key, strconv.FormatBool(*v))
	}
}

// setIntDefault sets key to v, or to def when v is nil.
func setIntDefault(q url.Values, key string, v *int, def int) {
	val := def
	if v != nil {
		val = *v
	}
	q.Set(key, strconv.Itoa(val))
}

// setStrOpt sets key only when v is non-empty.
func setStrOpt(q url.Values, key, v string) {
	if v != "" {
		q.Set(key, v)
	}
}

// addAll appends each value under key, preserving repeated query parameters.
func addAll(q url.Values, key string, vals []string) {
	for _, v := range vals {
		q.Add(key, v)
	}
}

type connector struct {
	cfg Config
}

// Connect dials the live transcription WebSocket for the given sample rate.
func (c *connector) Connect(ctx context.Context, sampleRate int) (stt.Stream, error) {
	q := c.cfg.query(sampleRate)

	header := http.Header{}
	header.Set("Authorization", "Token "+c.cfg.APIKey)

	conn, resp, err := websocket.Dial(ctx, c.cfg.ListenURL+"?"+q.Encode(), &websocket.DialOptions{HTTPHeader: header})
	if resp != nil && resp.Body != nil {
		_ = resp.Body.Close()
	}
	if err != nil {
		return nil, err
	}
	s := &stream{conn: conn, ctx: ctx}
	s.wg.Go(s.keepAlive)
	return s, nil
}

type stream struct {
	conn    *websocket.Conn
	ctx     context.Context
	writeMu sync.Mutex
	wg      sync.WaitGroup
}

// dgMessage is the subset of Deepgram's live transcription result we use.
type dgMessage struct {
	Type    string `json:"type"`
	Channel struct {
		Alternatives []struct {
			Transcript string `json:"transcript"`
		} `json:"alternatives"`
	} `json:"channel"`
	IsFinal     bool `json:"is_final"`
	SpeechFinal bool `json:"speech_final"`
}

// Send writes a chunk of PCM audio as a binary frame.
func (s *stream) Send(audio []byte) error {
	s.writeMu.Lock()
	defer s.writeMu.Unlock()
	return s.conn.Write(s.ctx, websocket.MessageBinary, audio)
}

// Recv reads the next result. A finalized result carries Deepgram's speech_final
// as the end-of-turn signal.
func (s *stream) Recv() ([]stt.Result, error) {
	for {
		_, data, err := s.conn.Read(s.ctx)
		if err != nil {
			return nil, err
		}
		var m dgMessage
		if err := json.Unmarshal(data, &m); err != nil {
			continue
		}
		if m.Type != "Results" || len(m.Channel.Alternatives) == 0 {
			continue
		}
		text := m.Channel.Alternatives[0].Transcript
		if text == "" {
			continue
		}
		return []stt.Result{{Text: text, Final: m.IsFinal, EndOfTurn: m.SpeechFinal}}, nil
	}
}

// keepAlive sends a periodic KeepAlive so Deepgram does not close an idle
// connection during silence.
func (s *stream) keepAlive() {
	ticker := time.NewTicker(keepAlivePeriod)
	defer ticker.Stop()
	for {
		select {
		case <-s.ctx.Done():
			return
		case <-ticker.C:
			s.writeMu.Lock()
			_ = s.conn.Write(s.ctx, websocket.MessageText, []byte(`{"type":"KeepAlive"}`))
			s.writeMu.Unlock()
		}
	}
}

// Close asks Deepgram to flush and then closes the socket.
func (s *stream) Close() error {
	s.writeMu.Lock()
	_ = s.conn.Write(context.Background(), websocket.MessageText, []byte(`{"type":"CloseStream"}`))
	s.writeMu.Unlock()
	s.wg.Wait()
	return s.conn.Close(websocket.StatusNormalClosure, "")
}
