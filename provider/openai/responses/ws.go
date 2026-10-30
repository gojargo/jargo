package responses

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"log/slog"
	"net/http"
	"strconv"
	"sync"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/adapter"
	"github.com/gojargo/jargo/adapter/openai"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/service/llm"
	"github.com/gojargo/jargo/service/wsutil"
)

// maxTurnAttempts bounds how many times a turn is retried after a recoverable
// connection failure. Two is enough: the retry reconnects and sends the full
// context, which cannot hit the same fault again.
const maxTurnAttempts = 2

// laneCount is how many lanes the service's responses go on. A connection runs
// at most this many responses at once, abandoned ones included.
const laneCount = 16

// endsResponse reports whether an event ends a response on its lane: it
// completed, failed or was cut short, or its request got an error.
func endsResponse(eventType string) bool {
	switch eventType {
	case evtCompleted, evtFailed, evtIncomplete, evtError:
		return true
	default:
		return false
	}
}

// errPreviousResponseGone reports that the server no longer holds the response
// the turn tried to continue from. Its cache is connection-local, so a reconnect
// loses it; the turn retries with the full context.
//
//nolint:gochecknoglobals // sentinel error
var errPreviousResponseGone = errors.New("openai responses: previous response not found")

// errConnectionExpired reports that the connection hit the server's lifetime
// limit. The turn retries on a fresh one.
//
//nolint:gochecknoglobals // sentinel error
var errConnectionExpired = errors.New("openai responses: connection expired")

// errConnectionClosed reports that the connection ended mid-turn.
//
//nolint:gochecknoglobals // sentinel error
var errConnectionClosed = errors.New("openai responses: connection closed")

// wsEvent is one item from the read loop: an event, or the failure that ended
// the connection.
type wsEvent struct {
	ev  event
	err error
}

// session is one live connection and the read loop draining it.
type session struct {
	conn   *wsutil.Conn
	ctx    context.Context
	cancel context.CancelFunc
	events chan wsEvent
}

// Service is a Responses LLM processor over a persistent WebSocket.
type Service struct {
	*llm.Base
	cfg Config
	// http answers a one-shot inference. It is a plain request, so it does not
	// go near the connection this service holds open for its turns.
	http *http.Client

	mu   sync.Mutex
	sess *session
	// cont is what the server is known to hold, so the next turn can send only
	// what is new. It is reset whenever the connection is replaced.
	cont continuation
	// currentLane is the lane of the response in flight, from its
	// response.create until its last event, and draining holds the lanes still
	// draining an abandoned response (see chooseNextLane).
	currentLane string
	draining    map[string]bool
	wg          sync.WaitGroup
}

// wsCreate is the client message that starts a turn. The request travels
// pre-encoded so the configured extra fields are merged into it, not around it.
type wsCreate struct {
	Type string `json:"type"`
	// StreamID is the lane the response goes on, which each of its events
	// names.
	StreamID string          `json:"stream_id"`
	Response json.RawMessage `json:"response"`
}

// continuation records what the previous turn sent and what came back, which is
// what lets the next turn send only the new items.
type continuation struct {
	// responseID is the response the next turn continues from; empty disables
	// the optimization.
	responseID string
	// inputLen is how many input items the previous turn sent.
	inputLen int
	// inputHash fingerprints those items, so a conversation edited behind the
	// service's back is detected rather than silently continued from.
	inputHash string
	// outputLen is how many items the server's own response added, which the
	// next turn must skip because the server already has them.
	outputLen int
	// lane is the lane the response ran on, which the next response avoids.
	lane string
}

// NewLLM builds a Responses LLM service over a persistent WebSocket. It sends
// only the new items each turn when the conversation has grown from where the
// previous one left off, and the full context otherwise.
func NewLLM(cfg Config) *Service {
	if cfg.WSURL == "" {
		cfg.WSURL = defaultWSURL
	}
	if cfg.BaseURL == "" {
		cfg.BaseURL = defaultBaseURL
	}
	if cfg.Model == "" {
		cfg.Model = defaultModel
	}
	s := &Service{cfg: cfg, http: &http.Client{}}
	s.Base = llm.New("OpenAIResponsesLLM", s)
	s.Base.SetModel(cfg.Model)
	warnIfReasoningUnsupported("OpenAIResponsesLLM", cfg)
	return s
}

// Generate streams a completion, emitting each text delta.
func (s *Service) Generate(ctx context.Context, convo *frames.LLMContext, emit llm.Emit) error {
	return s.run(ctx, convo, textSink{emit: emit}, false)
}

// GenerateWithTools streams a completion, reporting text and any tool calls the
// model requests. It implements llm.ToolGenerator.
func (s *Service) GenerateWithTools(ctx context.Context, convo *frames.LLMContext, sink llm.Sink) error {
	return s.run(ctx, convo, sink, true)
}

// RunInference answers the conversation once, off to the side of the pipeline:
// no streaming, no frames, just the text. It implements llm.Inferencer.
func (s *Service) RunInference(
	ctx context.Context, convo *frames.LLMContext, opts llm.InferenceOptions,
) (string, error) {
	opts.ResponseSchema = s.CheckResponseSchema(ctx, opts.ResponseSchema)
	return runInference(ctx, s.cfg, s.http, convo, opts, func(req request) ([]byte, error) {
		return s.cfg.encode(req, s)
	})
}

// SupportsResponseSchema implements llm.ResponseSchemaSupporter.
func (s *Service) SupportsResponseSchema() bool { return true }

// ModelSupportsResponseSchema implements llm.ResponseSchemaSupporter. OpenAI
// models before gpt-4o-mini and gpt-4o-2024-08-06 cannot enforce a schema.
func (s *Service) ModelSupportsResponseSchema(model string) bool {
	return openai.ModelSupportsResponseSchema(model)
}

// Cleanup closes the connection and tears the processor down.
func (s *Service) Cleanup(ctx context.Context) error {
	s.disconnect()
	return s.Base.Cleanup(ctx)
}

// run drives one turn, retrying once when the connection turns out to be stale.
// Both recoverable faults are properties of the connection rather than the
// request, so the retry drops the continuation and resends the full context.
func (s *Service) run(ctx context.Context, convo *frames.LLMContext, sink llm.Sink, withTools bool) error {
	lane := s.chooseNextLane()
	var err error
	for attempt := range maxTurnAttempts {
		err = s.turn(ctx, convo, sink, withTools, lane)
		if !isRecoverable(err) {
			return err
		}
		slog.Debug("openai responses turn failed recoverably, retrying",
			"attempt", attempt+1, "err", err)
		s.disconnect()
	}
	return err
}

// isRecoverable reports whether a failed turn is worth retrying on a fresh
// connection. A canceled context is an interruption, not a fault.
func isRecoverable(err error) bool {
	if err == nil || errors.Is(err, context.Canceled) {
		return false
	}
	return errors.Is(err, errPreviousResponseGone) || errors.Is(err, errConnectionExpired)
}

// turn runs one exchange on a lane: it builds the request, sends it, and reads
// events until the response ends.
func (s *Service) turn(
	ctx context.Context, convo *frames.LLMContext, sink llm.Sink, withTools bool, lane string,
) error {
	sess, err := s.connect(ctx)
	if err != nil {
		return err
	}

	req, err := s.cfg.newRequest(convo, adapter.Options{SystemInstruction: s.SystemInstruction()}, withTools)
	if err != nil {
		return err
	}
	s.cfg.applyThinkingDefault(&req, s)
	full := req.Input
	sent := s.applyContinuation(&req)

	inner, err := encodeBody(req, s.cfg.Extra)
	if err != nil {
		return err
	}
	body, err := json.Marshal(wsCreate{Type: "response.create", StreamID: lane, Response: inner})
	if err != nil {
		return err
	}
	// The write uses the session's context, not the turn's: canceling a write
	// closes the socket, and an interruption has to leave it usable.
	s.StartTTFBMetrics()
	// Set before sending: an interruption during the send may still let the
	// response go out, and its lane must then count as draining.
	s.mu.Lock()
	s.currentLane = lane
	s.mu.Unlock()
	if werr := sess.conn.Write(sess.ctx, websocket.MessageText, body); werr != nil {
		return werr
	}

	state, err := s.readTurn(ctx, sess, sink, lane)
	if err != nil {
		if ctx.Err() != nil {
			// An interruption. The response drains from its lane, and the next
			// one goes on another (see chooseNextLane).
			s.abandonResponse()
			return ctx.Err()
		}
		return err
	}
	s.recordContinuation(state, full, sent, lane)
	s.reportUsage(ctx, state)
	return nil
}

// readTurn consumes the events of the response on lane until it ends. It
// watches the turn's context rather than passing it to the socket, so an
// interruption leaves the connection open for the next turn.
func (s *Service) readTurn(ctx context.Context, sess *session, sink llm.Sink, lane string) (*streamState, error) {
	state := newStreamState(sink)
	for {
		select {
		case <-ctx.Done():
			return state, ctx.Err()
		case m, ok := <-sess.events:
			if !ok {
				return state, errConnectionClosed
			}
			if m.err != nil {
				return state, m.err
			}
			if !s.followLane(m.ev, lane) {
				continue
			}
			if carriesModelOutput(m.ev) {
				s.StopTTFBMetrics()
			}
			if cerr := classifyStreamError(m.ev); cerr != nil {
				return state, cerr
			}
			done, err := state.handle(m.ev)
			if err != nil {
				return state, err
			}
			if done {
				return state, nil
			}
		}
	}
}

// chooseNextLane chooses the lane for the next response.
//
// Every response goes on a named lane of the connection (the API's stream_id),
// so each of its events says which lane it came from. That lets an interrupted
// response be abandoned rather than waited for: WebSocket mode cannot stop a
// response (it ignores response.cancel), so an abandoned response drains from
// its lane while the next one runs on another, and its events are discarded as
// they arrive. The lane is free again once the response ends.
//
// The lane chosen is not draining, since a lane runs its responses one at a
// time, and is not the previous response's lane. Each lane caches only its
// latest response, so an abandoned response on the previous response's lane
// would evict the previous response when it ended, and the next request could
// not continue from it. Continuing from another lane's response (a fork) costs
// no more than continuing on the same lane.
//
// If every lane is taken, the connection is closed, which ends the abandoned
// responses; the next send opens a new one.
func (s *Service) chooseNextLane() string {
	s.mu.Lock()
	for i := range laneCount {
		lane := laneName(i)
		if lane != s.cont.lane && !s.draining[lane] {
			s.mu.Unlock()
			return lane
		}
	}
	s.mu.Unlock()
	slog.Debug("openai responses: every lane is draining or holds the previous response, reconnecting")
	s.disconnect()
	return laneName(0)
}

// laneName is the stream_id of the ith lane.
func laneName(i int) string { return "lane-" + strconv.Itoa(i) }

// abandonResponse leaves the response in flight to drain from its lane.
func (s *Service) abandonResponse() {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.currentLane == "" {
		return
	}
	slog.Debug("openai responses: abandoning the response", "lane", s.currentLane)
	if s.draining == nil {
		s.draining = map[string]bool{}
	}
	s.draining[s.currentLane] = true
	s.currentLane = ""
}

// followLane reports whether an event belongs to the response on lane, and
// notes when it ends a response.
//
// An event on another lane is a draining lane's, from an abandoned response;
// the lane is free again once that response ends. Events with no stream_id
// concern the connection, and so the response in flight.
func (s *Service) followLane(ev event, lane string) bool {
	ends := endsResponse(ev.Type)
	s.mu.Lock()
	defer s.mu.Unlock()
	if ev.StreamID != "" && ev.StreamID != lane {
		if ends {
			delete(s.draining, ev.StreamID)
		}
		return false
	}
	if ends {
		s.currentLane = ""
	}
	return true
}

// classifyStreamError recognizes the two failures that are worth retrying on a
// fresh connection rather than reporting to the caller.
func classifyStreamError(ev event) error {
	if ev.Type != evtError {
		return nil
	}
	switch ev.Code {
	case "previous_response_not_found":
		return errPreviousResponseGone
	case "connection_expired", "session_expired":
		return errConnectionExpired
	}
	return nil
}

// applyContinuation trims the request's input to what the server does not
// already hold, and returns how many items were sent. When the conversation has
// not grown from where the previous turn left off, the full input goes.
func (s *Service) applyContinuation(req *request) int {
	s.mu.Lock()
	cont := s.cont
	s.mu.Unlock()

	full := req.Input
	if cont.responseID == "" || cont.inputHash == "" {
		return len(full)
	}
	// The server holds the previous input plus the items its own response added.
	known := cont.inputLen + cont.outputLen
	if len(full) <= known {
		// The conversation did not grow, or was rewritten shorter. Send it whole.
		return len(full)
	}
	if hashItems(full[:cont.inputLen]) != cont.inputHash {
		// The history the service previously sent has changed underneath it, so
		// continuing from the server's copy would replay the wrong conversation.
		return len(full)
	}
	req.Input = full[known:]
	req.PreviousResponseID = cont.responseID
	return len(full)
}

// recordContinuation stores what the server now holds, so the next turn can send
// only what is new. full is the whole input the conversation rendered to, sent
// is how many of those items the server has seen, and lane is the lane the
// response ran on.
func (s *Service) recordContinuation(state *streamState, full []inputItem, sent int, lane string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if state.responseID == "" {
		// Without a response to continue from the next turn sends everything.
		s.cont = continuation{}
		return
	}
	s.cont = continuation{
		responseID: state.responseID,
		inputLen:   sent,
		inputHash:  hashItems(full[:sent]),
		// The response's own output items land in the context before the next
		// turn, and the server already has them.
		outputLen: state.outputItems(),
		lane:      lane,
	}
}

// hashItems fingerprints a prefix of the input, so a conversation rewritten
// behind the service's back is detected instead of silently continued from.
func hashItems(items []inputItem) string {
	b, err := json.Marshal(items)
	if err != nil {
		return ""
	}
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:])
}

// connect returns the live session, dialing one if there is none.
func (s *Service) connect(ctx context.Context) (*session, error) {
	s.mu.Lock()
	sess := s.sess
	s.mu.Unlock()
	if sess != nil {
		return sess, nil
	}

	header := http.Header{}
	header.Set("Authorization", "Bearer "+s.cfg.APIKey)
	header.Set("OpenAI-Beta", "responses=v1")

	conn, err := wsutil.Dial(ctx, s.cfg.WSURL, header, readLimit)
	if err != nil {
		return nil, err
	}

	// The session outlives any one turn, so its context is rooted in the
	// background rather than in the turn that happened to open it.
	sessCtx, cancel := context.WithCancel(context.Background())
	sess = &session{
		conn:   conn,
		ctx:    sessCtx,
		cancel: cancel,
		events: make(chan wsEvent, 32),
	}

	s.mu.Lock()
	s.sess = sess
	// A new connection holds nothing, so the next turn starts from scratch.
	s.cont = continuation{}
	s.mu.Unlock()

	s.wg.Add(1)
	go s.readLoop(sess)
	return sess, nil
}

// readLoop drains the socket into the session's event channel until the
// connection ends or the session is torn down.
func (s *Service) readLoop(sess *session) {
	defer s.wg.Done()
	defer close(sess.events)
	for {
		_, data, err := sess.conn.Read(sess.ctx)
		if err != nil {
			select {
			case sess.events <- wsEvent{err: err}:
			case <-sess.ctx.Done():
			}
			return
		}
		ev, ok := decodeEvent(data)
		if !ok {
			continue // skip a malformed event rather than ending the session
		}
		select {
		case sess.events <- wsEvent{ev: ev}:
		case <-sess.ctx.Done():
			return
		}
	}
}

// disconnect closes the connection, stops the read loop, and forgets what the
// server held and which lanes were in use. It is safe to call more than once.
func (s *Service) disconnect() {
	s.mu.Lock()
	sess := s.sess
	s.sess = nil
	s.cont = continuation{}
	s.currentLane = ""
	s.draining = nil
	s.mu.Unlock()
	if sess == nil {
		return
	}
	sess.cancel()
	_ = sess.conn.Close(websocket.StatusNormalClosure, "")
	s.wg.Wait()
}

// reportUsage forwards the turn's token accounting, when usage metrics are on.
func (s *Service) reportUsage(ctx context.Context, state *streamState) {
	if state.usage == nil || !s.UsageMetricsEnabled() {
		return
	}
	_ = s.PushTokenUsage(ctx, state.usage.tokenUsage())
}
