package elevenlabs

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"strconv"
	"sync"
	"sync/atomic"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/internal/validate"
	"github.com/gojargo/jargo/language"
	"github.com/gojargo/jargo/service/tts"
	"github.com/gojargo/jargo/service/wsutil"
	uctx "github.com/gojargo/jargo/utils/context"
)

// realtimeTTSPath is the multi-stream-input endpoint. Multi-stream is what
// carries a context id on every message, which is how audio is attributed to the
// synthesis that asked for it.
const realtimeTTSPath = "/v1/text-to-speech/%s/multi-stream-input"

// defaultRealtimeTTSHost is the ElevenLabs WebSocket host.
const defaultRealtimeTTSHost = "wss://api.elevenlabs.io"

// Message keys on the multi-stream-input protocol.
const (
	keyText      = "text"
	keyContextID = "context_id"
)

// errRealtimeTTS wraps an error the server reported on the stream.
//
//nolint:gochecknoglobals // sentinel error
var errRealtimeTTS = errors.New("elevenlabs realtime tts")

// RealtimeTTSConfig configures the streaming WebSocket TTS service.
type RealtimeTTSConfig struct {
	// APIKey is the ElevenLabs API key, sent as the xi-api-key header. Required.
	APIKey string `validate:"required"`
	// Host overrides the WebSocket host; empty uses the hosted API.
	Host string
	// VoiceID is the ElevenLabs voice; empty uses a default public voice.
	VoiceID string
	// Model is the ElevenLabs model; empty uses the low-latency flash model.
	Model string
	// SampleRate is the PCM rate requested and emitted downstream. Empty uses
	// 48 kHz. Must be a rate ElevenLabs supports.
	SampleRate int
	// Language for multilingual models; the zero value leaves it unset.
	Language language.Language
	// VoiceSettings overrides the voice's default settings when non-nil.
	VoiceSettings *VoiceSettings
	// AutoMode lets ElevenLabs decide when it has enough text to start
	// generating, rather than waiting to be told. nil leaves it on, which is
	// what suits a sentence at a time.
	AutoMode *bool
	// ApplyTextNormalization controls spoken-form normalization ("auto", "on",
	// "off"); empty leaves it unset.
	ApplyTextNormalization string
	// EnableLogging toggles server-side logging; nil leaves it unset.
	EnableLogging *bool
	// PronunciationDictionaryLocators applies the given dictionaries.
	PronunciationDictionaryLocators []PronunciationDictionaryLocator
	// WordTimestamps reports per-word timing alongside the audio, which lets the
	// assistant context record what was actually spoken before an interruption.
	// ElevenLabs times every character, so the words are assembled from those.
	WordTimestamps bool
}

// Validate reports whether the configuration is usable.
func (c RealtimeTTSConfig) Validate() error { return validate.Struct(c) }

func (c RealtimeTTSConfig) withDefaults() RealtimeTTSConfig {
	if c.Host == "" {
		c.Host = defaultRealtimeTTSHost
	}
	if c.VoiceID == "" {
		c.VoiceID = defaultVoiceID
	}
	if c.Model == "" {
		c.Model = defaultModel
	}
	if c.SampleRate == 0 {
		c.SampleRate = defaultSampleRate
	}
	if c.AutoMode == nil {
		// Without auto mode the server holds text until it is explicitly flushed,
		// and closing a context discards whatever was never flushed — the synthesis
		// then ends with a final marker and no audio at all. Sentence-at-a-time
		// synthesis has nothing to gain from that buffering, so it is on unless the
		// caller turns it off.
		c.AutoMode = new(true)
	}
	return c
}

// NewRealtimeTTS builds an ElevenLabs TTS service over the streaming WebSocket
// API. Unlike NewTTS, which issues one HTTP request per sentence, this holds a
// single connection open for the session, so a sentence pays no connection setup
// before the first audio comes back.
func NewRealtimeTTS(cfg RealtimeTTSConfig) *tts.Base {
	s := &realtimeSynthesizer{cfg: cfg.withDefaults()}
	if cfg.WordTimestamps {
		// Only the timestamp-aware type implements tts.WordTimestamps, so the
		// base takes the word-aligned path solely when timings were requested.
		return tts.New("ElevenLabsRealtimeTTS", &timedRealtimeSynthesizer{realtimeSynthesizer: s})
	}
	return tts.New("ElevenLabsRealtimeTTS", s)
}

type realtimeSynthesizer struct {
	cfg RealtimeTTSConfig

	// mu guards the lazily dialed connection, which is reused across syntheses
	// so a sentence does not pay for a fresh handshake. It is held for the whole
	// of a synthesis: the base synthesizes one sentence at a time, and the
	// stream is read inline rather than by a background reader.
	mu   sync.Mutex
	conn *websocket.Conn

	// context ids only have to be unique within the connection.
	nextContext atomic.Uint64
}

// timedRealtimeSynthesizer adds word-timestamp streaming on top of
// realtimeSynthesizer. It implements tts.WordTimestamps.
type timedRealtimeSynthesizer struct {
	*realtimeSynthesizer
}

// SampleRate reports the requested PCM output rate.
func (s *realtimeSynthesizer) SampleRate() int { return s.cfg.SampleRate }

// Metadata reports the ElevenLabs model and voice synthesis is billed against.
func (s *realtimeSynthesizer) Metadata() tts.Metadata {
	return tts.Metadata{Model: s.cfg.Model, VoiceID: s.cfg.VoiceID}
}

// endpoint builds the multi-stream-input URL. The voice, model and output format
// are fixed for the connection; only the text varies per message.
func (s *realtimeSynthesizer) endpoint() string {
	q := url.Values{}
	q.Set("model_id", s.cfg.Model)
	q.Set("output_format", outputFormat(s.cfg.SampleRate))
	if s.cfg.AutoMode != nil {
		q.Set("auto_mode", strconv.FormatBool(*s.cfg.AutoMode))
	}
	if s.cfg.ApplyTextNormalization != "" {
		q.Set("apply_text_normalization", s.cfg.ApplyTextNormalization)
	}
	if s.cfg.EnableLogging != nil {
		q.Set("enable_logging", strconv.FormatBool(*s.cfg.EnableLogging))
	}
	if code := elevenlabsLanguage(s.cfg.Language); code != "" {
		q.Set("language_code", code)
	}
	path := fmt.Sprintf(realtimeTTSPath, s.cfg.VoiceID)
	return s.cfg.Host + path + "?" + q.Encode()
}

// stream returns the shared connection, dialing it on first use and redialing
// when a previous synthesis left it broken. Callers hold mu.
func (s *realtimeSynthesizer) stream(ctx context.Context) (*websocket.Conn, error) {
	if s.conn != nil {
		return s.conn, nil
	}
	header := http.Header{}
	header.Set("xi-api-key", s.cfg.APIKey)
	conn, err := wsutil.Dial(ctx, s.endpoint(), header, wsutil.DefaultReadLimit)
	if err != nil {
		return nil, err
	}
	s.conn = conn
	return conn, nil
}

// drop closes the connection so the next synthesis dials a fresh one. Used when
// a stream has failed and its state can no longer be trusted.
func (s *realtimeSynthesizer) drop() {
	if s.conn == nil {
		return
	}
	_ = s.conn.Close(websocket.StatusInternalError, "")
	s.conn = nil
}

// Close releases the shared connection, implementing tts.Closer.
func (s *realtimeSynthesizer) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.conn == nil {
		return nil
	}
	conn := s.conn
	s.conn = nil
	// Ask the server to close so it does not sit on a half-open stream.
	_ = writeJSON(context.Background(), conn, map[string]any{"close_socket": true})
	return conn.Close(websocket.StatusNormalClosure, "")
}

// Synthesize sends the sentence on the shared stream and emits the audio.
func (s *realtimeSynthesizer) Synthesize(ctx context.Context, text string, emit func(pcm []byte) error) error {
	return s.run(ctx, text, emit, nil)
}

// SynthesizeTimed streams audio and reports per-word timing, implementing
// tts.WordTimestamps.
func (s *timedRealtimeSynthesizer) SynthesizeTimed(
	ctx context.Context,
	text string,
	emit func(pcm []byte) error,
	word func(text string, offset float64) error,
) error {
	return s.run(ctx, text, emit, word)
}

// run drives one synthesis over the shared stream.
func (s *realtimeSynthesizer) run(
	ctx context.Context,
	text string,
	emit func(pcm []byte) error,
	word func(text string, offset float64) error,
) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	conn, err := s.stream(ctx)
	if err != nil {
		return err
	}
	contextID := strconv.FormatUint(s.nextContext.Add(1), 10)

	if err := s.send(ctx, conn, contextID, text); err != nil {
		// A failed write leaves the server's view of the stream unknown.
		s.drop()
		return err
	}
	if err := s.receive(ctx, conn, contextID, emit, word); err != nil {
		s.drop()
		return err
	}
	return nil
}

// send opens a context, hands it the sentence, and closes it. Closing is what
// makes the final marker arrive immediately after the last audio byte rather
// than after the server waits to see whether more text is coming.
func (s *realtimeSynthesizer) send(
	ctx context.Context,
	conn *websocket.Conn,
	contextID, text string,
) error {
	// The opening message carries the voice parameters for the context. A single
	// space rather than the sentence: it initializes without committing text.
	open := map[string]any{keyText: " ", keyContextID: contextID}
	if s.cfg.VoiceSettings != nil {
		open["voice_settings"] = s.cfg.VoiceSettings
	}
	if len(s.cfg.PronunciationDictionaryLocators) > 0 {
		open["pronunciation_dictionary_locators"] = s.cfg.PronunciationDictionaryLocators
	}
	messages := []map[string]any{
		open,
		{keyText: text, keyContextID: contextID},
		{keyContextID: contextID, "close_context": true},
	}
	for _, msg := range messages {
		if err := writeJSON(ctx, conn, msg); err != nil {
			return err
		}
	}
	return nil
}

// writeJSON sends one JSON message as a text frame.
func writeJSON(ctx context.Context, conn *websocket.Conn, msg map[string]any) error {
	payload, err := json.Marshal(msg)
	if err != nil {
		return err
	}
	return conn.Write(ctx, websocket.MessageText, payload)
}

// realtimeTTSMessage is one server message. Audio arrives base64-encoded, and
// alignment times every character rather than every word.
type realtimeTTSMessage struct {
	ContextID string `json:"contextId"` //nolint:tagliatelle // ElevenLabs wire keys are camelCase
	Audio     string `json:"audio"`
	IsFinal   bool   `json:"isFinal"` //nolint:tagliatelle // ElevenLabs wire keys are camelCase
	Alignment *struct {
		Chars            []string  `json:"chars"`
		CharStartTimesMs []float64 `json:"charStartTimesMs"` //nolint:tagliatelle // ElevenLabs wire keys are camelCase
	} `json:"alignment"`
	NormalizedAlignment *struct {
		Chars            []string  `json:"chars"`
		CharStartTimesMs []float64 `json:"charStartTimesMs"` //nolint:tagliatelle // ElevenLabs wire keys are camelCase
	} `json:"normalizedAlignment"` //nolint:tagliatelle // ElevenLabs wire keys are camelCase
	Error string `json:"error"`
}

// receive reads until the context is final, emitting its audio and words.
//
// Messages for any other context are skipped: an interruption abandons a
// synthesis mid-flight, and audio the server had already generated for it
// arrives afterwards. Attributing by context id keeps that audio out of the next
// sentence, which is what the ids are for.
func (s *realtimeSynthesizer) receive(
	ctx context.Context,
	conn *websocket.Conn,
	contextID string,
	emit func(pcm []byte) error,
	word func(text string, offset float64) error,
) error {
	var acc uctx.CharAccumulator
	for {
		msg, err := readRealtimeTTSMessage(ctx, conn)
		if err != nil {
			return err
		}
		if msg.ContextID != contextID {
			continue
		}
		if err := s.handleMessage(msg, &acc, emit, word); err != nil {
			return err
		}
		if msg.IsFinal {
			if word == nil {
				return nil
			}
			// The closing word ends on the utterance, not on a space.
			if wt, ok := acc.Flush(); ok {
				return word(wt.Word, wt.Offset)
			}
			return nil
		}
	}
}

// readRealtimeTTSMessage reads and decodes one server message.
func readRealtimeTTSMessage(ctx context.Context, conn *websocket.Conn) (*realtimeTTSMessage, error) {
	_, data, err := conn.Read(ctx)
	if err != nil {
		return nil, err
	}
	var msg realtimeTTSMessage
	if err := json.Unmarshal(data, &msg); err != nil {
		return nil, fmt.Errorf("%w: decode message: %w", errRealtimeTTS, err)
	}
	if msg.Error != "" {
		return nil, fmt.Errorf("%w: %s", errRealtimeTTS, msg.Error)
	}
	return &msg, nil
}

// handleMessage emits a message's audio and reports the words it completes.
func (s *realtimeSynthesizer) handleMessage(
	msg *realtimeTTSMessage,
	acc *uctx.CharAccumulator,
	emit func(pcm []byte) error,
	word func(text string, offset float64) error,
) error {
	if msg.Audio != "" {
		pcm, err := base64.StdEncoding.DecodeString(msg.Audio)
		if err != nil {
			return fmt.Errorf("%w: decode audio: %w", errRealtimeTTS, err)
		}
		if err := emit(pcm); err != nil {
			return err
		}
	}
	if word == nil {
		return nil
	}
	return s.reportWords(acc, msg, word)
}

// reportWords folds a message's character timings into whole words and reports
// each one that completed.
func (s *realtimeSynthesizer) reportWords(
	acc *uctx.CharAccumulator,
	msg *realtimeTTSMessage,
	word func(text string, offset float64) error,
) error {
	alignment := msg.Alignment
	if s.cfg.PronunciationDictionaryLocators != nil && msg.NormalizedAlignment != nil {
		// A pronunciation dictionary rewrites what is spoken, so the normalized
		// form is the one the timings belong to.
		alignment = msg.NormalizedAlignment
	}
	if alignment == nil {
		return nil
	}
	starts := make([]float64, len(alignment.CharStartTimesMs))
	for i, ms := range alignment.CharStartTimesMs {
		starts[i] = ms / 1000
	}
	words, err := acc.Add(alignment.Chars, starts)
	if err != nil {
		return err
	}
	for _, wt := range uctx.MergePunctTokens(words) {
		if err := word(wt.Word, wt.Offset); err != nil {
			return err
		}
	}
	return nil
}
