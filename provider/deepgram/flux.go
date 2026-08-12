package deepgram

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"net/url"
	"strconv"
	"sync"
	"time"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/internal/query"
	"github.com/gojargo/jargo/internal/validate"
	"github.com/gojargo/jargo/language"
	"github.com/gojargo/jargo/service/stt"
	"github.com/gojargo/jargo/service/wsutil"
)

const (
	// fluxListenURL is the streaming turn-aware transcription WebSocket.
	fluxListenURL = "wss://api.deepgram.com/v2/listen"
	// defaultFluxModel is the English conversational model.
	defaultFluxModel = "flux-general-en"
	// fluxMultilingualModel is the only model that honors language hints.
	fluxMultilingualModel = "flux-general-multi"

	// defaultWatchdogMinTimeout is the minimum idle period before silence is
	// injected to keep an active turn from dangling.
	defaultWatchdogMinTimeout = 500 * time.Millisecond
	// watchdogTick is how often the idle watchdog wakes.
	watchdogTick = 100 * time.Millisecond
	// watchdogSilenceSeconds is the duration of silence injected on a stall.
	watchdogSilenceSeconds = 0.5
	// pcmSampleWidth is the byte width of one linear16 PCM sample.
	pcmSampleWidth = 2
)

// Flux STT WebSocket message types (the top-level "type" field). The shared
// "Error" type (fluxMsgError) lives in deepgram.go.
const (
	fluxMsgConnected = "Connected"
	fluxMsgTurnInfo  = "TurnInfo"
)

// Flux TurnInfo event types (the "event" field on a TurnInfo message).
const (
	fluxEventStartOfTurn     = "StartOfTurn"
	fluxEventTurnResumed     = "TurnResumed"
	fluxEventEndOfTurn       = "EndOfTurn"
	fluxEventEagerEndOfTurn  = "EagerEndOfTurn"
	fluxEventUpdate          = "Update"
	fluxDefaultLanguageForEn = "en"
)

// FluxConfig configures the Flux streaming turn-aware STT service. Fields left
// at their zero value fall back to the service defaults; optional tuning fields
// modeled as pointers or slices are omitted from the request when unset.
type FluxConfig struct {
	// APIKey is the Deepgram API key. Required.
	APIKey string `validate:"required"`
	// ListenURL overrides the transcription WebSocket endpoint; empty uses the
	// hosted endpoint.
	ListenURL string
	// Model is the Flux model; empty uses "flux-general-en". Set
	// "flux-general-multi" to enable language hints.
	Model string
	// SampleRate is the input audio sample rate; 0 uses the transport's rate.
	SampleRate int

	// EagerEOTThreshold enables eager end-of-turn predictions (interim
	// transcripts before a turn is confirmed). Off when nil.
	EagerEOTThreshold *float64
	// EOTThreshold is the end-of-turn confidence required to finish a turn;
	// nil uses the service default.
	EOTThreshold *float64
	// EOTTimeoutMs is the time in ms after speech to finish a turn regardless of
	// end-of-turn confidence; nil uses the service default.
	EOTTimeoutMs *int
	// MinConfidence drops a finalized turn whose average word confidence does
	// not exceed it; nil accepts every finalized turn.
	MinConfidence *float64
	// MipOptOut opts out of Deepgram's model-improvement program.
	MipOptOut *bool
	// Numerals writes spoken numbers as digits ("twenty three" becomes "23");
	// nil leaves the service default. It is fixed when the session opens: Flux
	// does not take a change to it mid-stream.
	Numerals *bool
	// Keyterm boosts recognition of the given terms.
	Keyterm []string
	// Tag attaches billing tags to the request.
	Tag []string
	// LanguageHints biases detection toward the given languages. Only honored by
	// the multilingual model; ignored for other models.
	LanguageHints []language.Language
	// WatchdogMinTimeout is the minimum idle period before silence is injected
	// to keep an active turn from dangling; 0 uses 500ms.
	WatchdogMinTimeout time.Duration
	// ExtraQuery sets arbitrary additional query parameters; values override any
	// param of the same name set from other fields.
	ExtraQuery map[string]string
}

// Validate reports whether the configuration is usable.
func (cfg FluxConfig) Validate() error { return validate.Struct(cfg) }

// NewFluxSTT builds a Deepgram Flux streaming turn-aware STT service. Flux
// detects turn boundaries server-side, so the service recommends external user
// turns downstream.
func NewFluxSTT(cfg FluxConfig) *stt.StreamService {
	if cfg.Model == "" {
		cfg.Model = defaultFluxModel
	}
	if cfg.ListenURL == "" {
		cfg.ListenURL = fluxListenURL
	}
	if cfg.WatchdogMinTimeout == 0 {
		cfg.WatchdogMinTimeout = defaultWatchdogMinTimeout
	}
	return stt.NewStream("DeepgramFluxSTT", &fluxConnector{cfg: cfg}, cfg.SampleRate)
}

// fluxQuery builds the transcription query string for the given sample rate.
func fluxQuery(cfg FluxConfig, sampleRate int) url.Values {
	q := url.Values{}
	q.Set("model", cfg.Model)
	q.Set("sample_rate", strconv.Itoa(sampleRate))
	q.Set("encoding", fluxEncoding)

	if cfg.EagerEOTThreshold != nil {
		q.Set("eager_eot_threshold", strconv.FormatFloat(*cfg.EagerEOTThreshold, 'f', -1, 64))
	}
	if cfg.EOTThreshold != nil {
		q.Set("eot_threshold", strconv.FormatFloat(*cfg.EOTThreshold, 'f', -1, 64))
	}
	if cfg.EOTTimeoutMs != nil {
		q.Set("eot_timeout_ms", strconv.Itoa(*cfg.EOTTimeoutMs))
	}
	query.SetBoolOpt(q, "numerals", cfg.Numerals)
	query.SetBoolOpt(q, "mip_opt_out", cfg.MipOptOut)

	query.AddAll(q, "keyterm", cfg.Keyterm)
	query.AddAll(q, "tag", cfg.Tag)

	// Language hints are only meaningful on the multilingual model.
	if cfg.Model == fluxMultilingualModel {
		for _, l := range cfg.LanguageHints {
			if code := l.BaseCode(); code != "" {
				q.Add("language_hint", code)
			}
		}
	}

	for k, v := range cfg.ExtraQuery {
		q.Set(k, v)
	}
	return q
}

type fluxConnector struct {
	cfg FluxConfig
}

// Metadata recommends external user turns: Flux emits its own turn boundaries.
func (c *fluxConnector) Metadata() stt.Metadata {
	noTTFS := false
	return stt.Metadata{
		RecommendedUserTurns: frames.UserTurnExternal,
		SupportsTTFS:         &noTTFS,
		Model:                c.cfg.Model,
	}
}

// Connect dials the Flux transcription WebSocket for the given sample rate.
func (c *fluxConnector) Connect(ctx context.Context, sampleRate int) (stt.Stream, error) {
	q := fluxQuery(c.cfg, sampleRate)

	header := http.Header{}
	header.Set("Authorization", "Token "+c.cfg.APIKey)

	conn, err := wsutil.Dial(ctx, c.cfg.ListenURL+"?"+q.Encode(), header, 0)
	if err != nil {
		return nil, fmt.Errorf("deepgram flux: dial: %w", err)
	}
	s := &fluxStream{
		conn:        conn,
		ctx:         ctx,
		sampleRate:  sampleRate,
		model:       c.cfg.Model,
		minConf:     c.cfg.MinConfidence,
		watchdogMin: c.cfg.WatchdogMinTimeout,
		lastAudio:   time.Now(),
	}
	s.wg.Go(s.watchdog)
	return s, nil
}

type fluxStream struct {
	conn        *wsutil.Conn
	ctx         context.Context //nolint:containedctx // mirrors the STT stream lifetime
	sampleRate  int
	model       string
	minConf     *float64
	watchdogMin time.Duration

	writeMu sync.Mutex
	wg      sync.WaitGroup

	stateMu   sync.Mutex
	speaking  bool
	lastAudio time.Time
	lastChunk time.Duration
}

// fluxWord is one recognized word; confidence is absent on some events.
type fluxWord struct {
	Confidence *float64 `json:"confidence"`
}

// fluxMessage is the subset of a Flux message we consume.
type fluxMessage struct {
	Type       string     `json:"type"`
	Event      string     `json:"event"`
	Transcript string     `json:"transcript"`
	Error      string     `json:"error"`
	Languages  []string   `json:"languages"`
	Words      []fluxWord `json:"words"`
}

// Send writes a chunk of PCM audio as a binary frame and records the send time
// so the watchdog can detect a stalled turn.
func (s *fluxStream) Send(audio []byte) error {
	s.stateMu.Lock()
	s.lastAudio = time.Now()
	if s.sampleRate > 0 {
		secs := float64(len(audio)) / float64(s.sampleRate*pcmSampleWidth)
		s.lastChunk = time.Duration(secs * float64(time.Second))
	}
	s.stateMu.Unlock()

	s.writeMu.Lock()
	defer s.writeMu.Unlock()
	if err := s.conn.Write(s.ctx, websocket.MessageBinary, audio); err != nil {
		return fmt.Errorf("deepgram flux: send: %w", err)
	}
	return nil
}

// Recv reads the next transcription result. TurnInfo Update/EagerEndOfTurn/
// StartOfTurn events become interim results; EndOfTurn becomes a finalized,
// end-of-turn result.
func (s *fluxStream) Recv() ([]stt.Result, error) {
	for {
		_, data, err := s.conn.Read(s.ctx)
		if err != nil {
			return nil, fmt.Errorf("deepgram flux: recv: %w", err)
		}
		var m fluxMessage
		if err := json.Unmarshal(data, &m); err != nil {
			// Skipped rather than fatal, unlike the listen session: Flux reads
			// its own stream and keeps the turn state, so one message that does
			// not parse is not worth the turn in progress.
			slog.Error("decoding a deepgram flux message failed", "err", err)
			continue
		}
		switch m.Type {
		case fluxMsgError:
			return nil, fmt.Errorf("%w: %s", errFluxServer, m.Error)
		case fluxMsgTurnInfo:
			s.trackTurn(m.Event)
			if res := fluxResults(m, s.model, s.minConf); len(res) > 0 {
				return res, nil
			}
		default:
			// Connected/ConfigureSuccess/other control messages carry no text.
		}
	}
}

// trackTurn records whether the user is currently speaking so the watchdog only
// injects silence during an active turn.
func (s *fluxStream) trackTurn(event string) {
	switch event {
	case fluxEventStartOfTurn:
		s.stateMu.Lock()
		s.speaking = true
		s.stateMu.Unlock()
	case fluxEventEndOfTurn:
		s.stateMu.Lock()
		s.speaking = false
		s.stateMu.Unlock()
	}
}

// fluxResults maps a Flux message to zero or one transcription result.
func fluxResults(m fluxMessage, model string, minConf *float64) []stt.Result {
	if m.Type != fluxMsgTurnInfo {
		return nil
	}
	lang := primaryLanguage(m, model)
	switch m.Event {
	case fluxEventUpdate, fluxEventEagerEndOfTurn, fluxEventStartOfTurn:
		if m.Transcript == "" {
			return nil
		}
		return []stt.Result{{Text: m.Transcript, Final: false, Language: lang}}
	case fluxEventEndOfTurn:
		if !confidenceOK(m.Words, minConf) {
			return nil
		}
		return []stt.Result{{Text: m.Transcript, Final: true, EndOfTurn: true, Language: lang}}
	case fluxEventTurnResumed:
		return nil
	default:
		return nil
	}
}

// primaryLanguage reports the detected language of a turn, falling back to
// English on the English-only model where the field is absent.
func primaryLanguage(m fluxMessage, model string) string {
	if len(m.Languages) > 0 {
		return m.Languages[0]
	}
	if model == defaultFluxModel {
		return fluxDefaultLanguageForEn
	}
	return ""
}

// confidenceOK reports whether a finalized turn clears the minimum-confidence
// threshold. With no threshold every turn passes; with a threshold set a turn
// missing confidence data is dropped.
func confidenceOK(words []fluxWord, minConf *float64) bool {
	if minConf == nil || *minConf <= 0 {
		return true
	}
	avg, ok := averageConfidence(words)
	if !ok {
		return false
	}
	return avg > *minConf
}

// averageConfidence averages the confidences of the words that carry one.
func averageConfidence(words []fluxWord) (float64, bool) {
	sum := 0.0
	n := 0
	for _, w := range words {
		if w.Confidence != nil {
			sum += *w.Confidence
			n++
		}
	}
	if n == 0 {
		return 0, false
	}
	return sum / float64(n), true
}

// watchdog injects a short block of silence when audio stops flowing mid-turn,
// so an active turn does not dangle without an end-of-turn event.
func (s *fluxStream) watchdog() {
	ticker := time.NewTicker(watchdogTick)
	defer ticker.Stop()
	for {
		select {
		case <-s.ctx.Done():
			return
		case <-ticker.C:
			s.stateMu.Lock()
			speaking := s.speaking
			last := s.lastAudio
			threshold := max(s.watchdogMin, s.lastChunk*2)
			s.stateMu.Unlock()

			if speaking && !last.IsZero() && time.Since(last) > threshold {
				if err := s.sendSilence(watchdogSilenceSeconds); err != nil {
					return
				}
			}
		}
	}
}

// sendSilence writes durationSecs of PCM silence at the session's sample rate.
func (s *fluxStream) sendSilence(durationSecs float64) error {
	samples := int(float64(s.sampleRate) * durationSecs)
	if samples <= 0 {
		return nil
	}
	silence := make([]byte, samples*pcmSampleWidth)
	if err := s.Send(silence); err != nil {
		return err
	}
	return nil
}

// Close asks Flux to finish the stream and then closes the socket.
func (s *fluxStream) Close() error {
	s.writeMu.Lock()
	_ = s.conn.Write(context.Background(), websocket.MessageText, []byte(`{"type":"CloseStream"}`))
	s.writeMu.Unlock()
	s.wg.Wait()
	if err := s.conn.Close(websocket.StatusNormalClosure, ""); err != nil {
		return fmt.Errorf("deepgram flux: close: %w", err)
	}
	return nil
}
