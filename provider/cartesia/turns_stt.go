package cartesia

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/internal/validate"
	"github.com/gojargo/jargo/processor/turns"
	"github.com/gojargo/jargo/service/settings"
	"github.com/gojargo/jargo/service/stt"
	"github.com/gojargo/jargo/service/wsutil"
)

const (
	defaultTurnsURL   = "wss://api.cartesia.ai/stt/turns/websocket"
	defaultTurnsModel = "ink-2"
	// turnsDrainWindow is how long the server is given to flush the turns it was
	// still holding once it has been told the session is closing.
	turnsDrainWindow = 2 * time.Second
)

// Cartesia turn-detection message types.
const (
	// turnsConnected acknowledges the session.
	turnsConnected = "connected"
	// turnsStart marks the beginning of a user turn.
	turnsStart = "turn.start"
	// turnsUpdate carries the turn's transcript so far. Transcripts are
	// cumulative, so each update supersedes the previous one.
	turnsUpdate = "turn.update"
	// turnsEnd marks the end of a user turn and carries its final transcript.
	turnsEnd = "turn.end"
	// turnsEagerEnd is a predicted end of turn, which turnsResume may retract.
	turnsEagerEnd = "turn.eager_end"
	// turnsResume retracts an eager end: the user carried on speaking.
	turnsResume = "turn.resume"
	// turnsError reports a server-side failure.
	turnsError = "error"

	// turnsClose asks the server to finish the session, so it flushes whatever
	// turn it was still holding before the socket goes.
	turnsClose = "close"
)

// TurnsSTTConfig configures the Cartesia turn-detecting STT service.
type TurnsSTTConfig struct {
	// ShouldInterrupt barges in when the server signals the start of a new turn;
	// nil enables it. It is passed along to the user turn strategies this
	// service recommends, which own the interruption; strategies the application
	// configures itself override the recommendation and this setting with it.
	ShouldInterrupt *bool

	// APIKey is the Cartesia API key, sent as the X-API-Key header. Required.
	APIKey string `validate:"required"`
	// URL overrides the turn-detection WebSocket endpoint; empty uses the hosted
	// endpoint.
	URL string
	// Version sets the Cartesia-Version header; empty uses a pinned default.
	Version string
	// Headers are sent on the WebSocket handshake, on top of the ones the
	// service sets, for a deployment that authorizes or routes its own way.
	Headers map[string]string
	// Model is the transcription model; empty uses a default.
	Model string
	// SampleRate is the input audio sample rate; 0 uses the transport's rate.
	SampleRate int
	// Keyterm biases transcription towards the given terms or phrases. Cartesia
	// binds them to a connection, so changing them while the pipeline runs
	// reopens the session.
	Keyterm []string
	// WatchdogMinTimeout is the shortest gap in the audio the server is left to
	// sit through mid-turn before silence is submitted. The server decides the
	// end of a turn from the audio it is fed, so a turn it has opened never ends
	// while the audio is stopped. 0 uses 500ms.
	WatchdogMinTimeout time.Duration

	// OnTurnStart is called when the server reports the start of a turn, with
	// the transcript it carries (usually empty). It runs on the read goroutine,
	// so it must not block.
	OnTurnStart func(transcript string)
	// OnTurnUpdate is called with the turn's transcript so far. Transcripts are
	// cumulative, so each one supersedes the last. Same rules as OnTurnStart.
	OnTurnUpdate func(transcript string)
	// OnTurnEagerEnd is called when the server predicts the turn has ended,
	// which OnTurnResume may retract. Same rules as OnTurnStart.
	OnTurnEagerEnd func(transcript string)
	// OnTurnResume is called when the user carried on speaking after an eager
	// end, retracting it. Same rules as OnTurnStart.
	OnTurnResume func()
	// OnTurnEnd is called with the final transcript for a completed turn. Same
	// rules as OnTurnStart.
	OnTurnEnd func(transcript string)
}

// Validate reports whether the configuration is usable.
func (c TurnsSTTConfig) Validate() error { return validate.Struct(c) }

// NewTurnsSTT builds a Cartesia turn-detecting STT service. Unlike NewSTT, the
// server decides where a turn begins and ends and reports those boundaries, so
// the pipeline does not need its own end-of-turn detection. The service
// recommends external turn strategies accordingly.
func NewTurnsSTT(cfg TurnsSTTConfig) *stt.StreamService {
	if cfg.URL == "" {
		cfg.URL = defaultTurnsURL
	}
	if cfg.Version == "" {
		cfg.Version = defaultVersion
	}
	if cfg.Model == "" {
		cfg.Model = defaultTurnsModel
	}
	return stt.NewStream("CartesiaTurnsSTT", newTurnsConnector(cfg), cfg.SampleRate)
}

type turnsConnector struct {
	cfg TurnsSTTConfig
	// live is what a caller may ask to change. Nothing here takes effect on a
	// session already running, but an update still lands in it so the service
	// can say so.
	live *TurnsSTTSettings
}

// TurnsSTTSettings is the part of the turn-detecting configuration that can
// change while the pipeline runs. Only the keyterms take effect, and only by
// opening a new session, since Cartesia binds them to a connection.
type TurnsSTTSettings struct {
	settings.STT

	// Keyterm biases transcription towards the given terms or phrases, sent as
	// repeated keyterm query parameters.
	Keyterm settings.Opt[[]string] `settings:"keyterm"`
}

// newTurnsConnector builds the connector with its settings seeded, which is the
// only way it should be built: the settings store is not optional.
func newTurnsConnector(cfg TurnsSTTConfig) *turnsConnector {
	return &turnsConnector{cfg: cfg, live: newTurnsSettings(cfg)}
}

// newTurnsSettings is the starting state, taken from what the service was built
// with.
func newTurnsSettings(cfg TurnsSTTConfig) *TurnsSTTSettings {
	s := &TurnsSTTSettings{}
	s.Model = settings.Set(cfg.Model)
	if len(cfg.Keyterm) > 0 {
		s.Keyterm = settings.Set(cfg.Keyterm)
	}
	return s
}

// Settings is the configuration a caller may change while the pipeline runs.
func (c *turnsConnector) Settings() any { return c.live }

// UpdateSettings reopens the session when the keyterms changed, and says so
// when anything else did. Cartesia binds keyterms to a connection, so a new list
// reaches it only by opening another; the rest of what this service is told at
// the session cannot be changed on one already running, and a caller who asked
// should hear that rather than assume it took effect.
func (c *turnsConnector) UpdateSettings(_ context.Context, changed settings.Changed) (bool, error) {
	if rest := changed.Except("keyterm"); len(rest) > 0 {
		slog.Warn("runtime settings update is not supported by this service",
			"service", "CartesiaTurnsSTT", "fields", strings.Join(rest, ", "))
	}
	return changed.Has("keyterm"), nil
}

// Metadata tells downstream processors the service detects turns itself, so the
// user aggregator adopts external turn strategies rather than running its own.
func (c *turnsConnector) Metadata() stt.Metadata {
	noTTFS := false
	return stt.Metadata{
		UserTurnStrategies: turns.ExternalStrategies(turns.ExternalStrategiesConfig{
			EnableInterruptions: c.cfg.ShouldInterrupt,
		}),
		SupportsTTFS: &noTTFS,
		Model:        c.live.Model.Or(c.cfg.Model),
	}
}

// TurnWatchdog keeps a turn from stalling. The server decides where a turn ends
// from the audio it is fed, so a turn it has opened never ends while the audio
// is stopped; silence stands in for the audio that stopped.
func (c *turnsConnector) TurnWatchdog() stt.TurnWatchdogOptions {
	return stt.TurnWatchdogOptions{MinTimeout: c.cfg.WatchdogMinTimeout}
}

// Connect opens a turn-detection session and waits for the server to
// acknowledge it.
func (c *turnsConnector) Connect(ctx context.Context, sampleRate int) (stt.Stream, error) {
	q := url.Values{}
	q.Set("model", c.live.Model.Or(c.cfg.Model))
	q.Set("encoding", defaultSTTEncoding)
	q.Set("sample_rate", strconv.Itoa(sampleRate))
	// The turn-detecting service honors keyterms on every model it serves, so
	// they go out without the model check the transcription service makes.
	for _, term := range prepareKeyterms(c.live.Keyterm.Or(nil)) {
		q.Add("keyterm", term)
	}

	header := http.Header{}
	header.Set("X-API-Key", c.cfg.APIKey)
	header.Set("Cartesia-Version", c.cfg.Version)
	// Applied last, so a deployment can route or authorize the handshake its own
	// way, overriding what the service sets.
	for k, v := range c.cfg.Headers {
		header.Set(k, v)
	}

	conn, err := wsutil.Dial(ctx, c.cfg.URL+"?"+encodeQuery(q), header, readLimit)
	if err != nil {
		return nil, err
	}
	s := &turnsStream{conn: conn, ctx: ctx, cfg: c.cfg}
	if err := s.awaitConnected(); err != nil {
		_ = conn.Close(websocket.StatusInternalError, "no session acknowledgement")
		return nil, err
	}
	return s, nil
}

type turnsStream struct {
	conn    *wsutil.Conn
	ctx     context.Context
	cfg     TurnsSTTConfig
	writeMu sync.Mutex
}

// awaitConnected reads until the server acknowledges the session, so audio is
// not sent to one it has not opened yet. Nothing else is sent before the
// acknowledgement, so nothing is consumed by waiting for it.
func (s *turnsStream) awaitConnected() error {
	for {
		_, data, err := s.conn.Read(s.ctx)
		if err != nil {
			return err
		}
		var m turnsMessage
		if json.Unmarshal(data, &m) != nil {
			continue
		}
		switch m.Type {
		case turnsConnected:
			slog.Debug("connected to the Cartesia turn-detection endpoint",
				"request_id", m.RequestID)
			return nil
		case turnsError:
			return fmt.Errorf("%w: %s", errSTTProtocol, m.Message)
		}
	}
}

// turnsMessage is the subset of a Cartesia turn message we read.
type turnsMessage struct {
	Type string `json:"type"`
	// Transcript is the turn's text so far. It is cumulative, not a delta.
	Transcript string `json:"transcript"`
	Message    string `json:"message"`
	RequestID  string `json:"request_id"`
}

// Send writes a chunk of PCM as a binary frame.
func (s *turnsStream) Send(audio []byte) error {
	s.writeMu.Lock()
	defer s.writeMu.Unlock()
	return s.conn.Write(s.ctx, websocket.MessageBinary, audio)
}

// Recv reads the next turn event. An update carries the running transcript as an
// interim result; the turn's end finalizes it. An eager end is a prediction the
// server may retract with a resume, so it is not treated as the turn ending.
//
// Ink-2 runs its own turn detection, so the boundaries it reports are carried on
// the results as proposals for the pipeline's strategies to resolve. A turn that
// captured only silence ends with no transcript, which the watchdog's injected
// silence can force; the boundary is still reported, and only the empty text is
// left out so nothing downstream aggregates an empty user message.
func (s *turnsStream) Recv() ([]stt.Result, error) {
	for {
		_, data, err := s.conn.Read(s.ctx)
		if err != nil {
			return nil, err
		}
		var m turnsMessage
		if json.Unmarshal(data, &m) != nil {
			continue
		}
		switch m.Type {
		case turnsUpdate:
			call(s.cfg.OnTurnUpdate, m.Transcript)
			if m.Transcript == "" {
				continue
			}
			return []stt.Result{{Text: m.Transcript, Final: false}}, nil
		case turnsStart:
			call(s.cfg.OnTurnStart, m.Transcript)
			return []stt.Result{{Speech: stt.SpeechStarted}}, nil
		case turnsEnd:
			call(s.cfg.OnTurnEnd, m.Transcript)
			if m.Transcript == "" {
				return []stt.Result{{Speech: stt.SpeechStopped}}, nil
			}
			return []stt.Result{{
				Text: m.Transcript, Final: true, EndOfTurn: true, Speech: stt.SpeechStopped,
			}}, nil
		case turnsEagerEnd:
			// A prediction the server may retract, so it opens no boundary here.
			// It is reported for an application that wants to act on it early.
			call(s.cfg.OnTurnEagerEnd, m.Transcript)
			continue
		case turnsResume:
			// The eager end above, retracted.
			if s.cfg.OnTurnResume != nil {
				s.cfg.OnTurnResume()
			}
			continue
		case turnsError:
			return nil, fmt.Errorf("%w: %s", errSTTProtocol, m.Message)
		case turnsConnected:
			// Session bookkeeping, with no transcript to emit and no boundary to
			// report. Already consumed at connect time, so this is a repeat.
			continue
		}
	}
}

// Drain tells the server the session is finishing, so it flushes the turn it was
// still holding rather than losing it with the socket. The window is what the
// results already on their way are given to arrive.
func (s *turnsStream) Drain() (time.Duration, error) {
	b, err := json.Marshal(map[string]any{"type": turnsClose})
	if err != nil {
		return 0, err
	}
	s.writeMu.Lock()
	defer s.writeMu.Unlock()
	// Its own context: the session's is canceled as the pipeline stops, and the
	// request is what the drain is waiting on.
	if err := s.conn.Write(context.Background(), websocket.MessageText, b); err != nil {
		return 0, err
	}
	return turnsDrainWindow, nil
}

// Close tears the session down. Cartesia ends a turn session by closing the
// socket; there is no finalize command, and what it was still holding was asked
// for by Drain.
func (s *turnsStream) Close() error {
	return s.conn.Close(websocket.StatusNormalClosure, "")
}

// call runs an optional turn listener.
func call(fn func(string), transcript string) {
	if fn != nil {
		fn(transcript)
	}
}
