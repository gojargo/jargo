package responses

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"log/slog"
	"net/http"
	"sync"
	"time"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/adapter"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/service/llm"
	"github.com/gojargo/jargo/service/wsutil"
)

// maxTurnAttempts bounds how many times a turn is retried after a recoverable
// connection failure. Two is enough: the retry reconnects and sends the full
// context, which cannot hit the same fault again.
const maxTurnAttempts = 2

// drainTimeout bounds how long an interrupted turn is given to finish draining
// before the connection is discarded instead.
const drainTimeout = 5 * time.Second

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
	wg   sync.WaitGroup
}

// wsCreate is the client message that starts a turn. The request travels
// pre-encoded so the configured extra fields are merged into it, not around it.
type wsCreate struct {
	Type     string          `json:"type"`
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
	return runInference(ctx, s.cfg, s.http, convo, opts)
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
	var err error
	for attempt := range maxTurnAttempts {
		err = s.turn(ctx, convo, sink, withTools)
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

// turn runs one exchange: it builds the request, sends it, and reads events
// until the response ends.
func (s *Service) turn(ctx context.Context, convo *frames.LLMContext, sink llm.Sink, withTools bool) error {
	sess, err := s.connect(ctx)
	if err != nil {
		return err
	}

	req, err := s.cfg.newRequest(convo, adapter.Options{SystemInstruction: s.SystemInstruction()}, withTools)
	if err != nil {
		return err
	}
	full := req.Input
	sent := s.applyContinuation(&req)

	inner, err := encodeBody(req, s.cfg.Extra)
	if err != nil {
		return err
	}
	body, err := json.Marshal(wsCreate{Type: "response.create", Response: inner})
	if err != nil {
		return err
	}
	// The write uses the session's context, not the turn's: canceling a write
	// closes the socket, and an interruption has to leave it usable.
	s.StartTTFBMetrics()
	if werr := sess.conn.Write(sess.ctx, websocket.MessageText, body); werr != nil {
		return werr
	}

	state, err := s.readTurn(ctx, sess, sink)
	if err != nil {
		if ctx.Err() != nil {
			// An interruption. The response is still being generated, and its
			// remaining events would otherwise be read as the next turn's.
			s.abandonTurn(sess, state.responseID)
			return ctx.Err()
		}
		return err
	}
	s.recordContinuation(state, full, sent)
	s.reportUsage(ctx, state)
	return nil
}

// readTurn consumes events until the response ends. It watches the turn's
// context rather than passing it to the socket, so an interruption leaves the
// connection open for the cleanup that follows.
func (s *Service) readTurn(ctx context.Context, sess *session, sink llm.Sink) (*streamState, error) {
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

// abandonTurn leaves the connection usable after an interruption. It asks the
// server to stop generating, then reads and discards the response's remaining
// events: the API gives no way to tell which response a delta belongs to, so
// events left unread would be attributed to the next turn.
//
// The continuation is dropped either way. The server holds however much of the
// response it produced before stopping, while the context holds only what was
// actually spoken, so the two have diverged and the next turn must send
// everything. If the connection cannot be cleaned in time it is discarded, and
// the next turn dials a fresh one.
func (s *Service) abandonTurn(sess *session, responseID string) {
	canceled := s.sendCancel(sess, responseID)
	deadline := time.After(drainTimeout)
	for {
		select {
		case <-deadline:
			slog.Debug("openai responses timed out draining an interrupted turn")
			s.disconnect()
			return
		case m, ok := <-sess.events:
			if !ok || m.err != nil {
				s.disconnect()
				return
			}
			// Interrupted before the response announced itself: cancel it now
			// that its id is known.
			if m.ev.Type == evtCreated && !canceled && m.ev.Response != nil {
				canceled = s.sendCancel(sess, m.ev.Response.ID)
				continue
			}
			switch m.ev.Type {
			case evtCompleted, evtFailed, evtIncomplete:
				s.resetContinuation()
				return
			}
		}
	}
}

// sendCancel asks the server to stop generating a response, reporting whether
// the request went out.
func (s *Service) sendCancel(sess *session, responseID string) bool {
	if responseID == "" {
		return false
	}
	body, err := json.Marshal(map[string]any{"type": "response.cancel", "response_id": responseID})
	if err != nil {
		return false
	}
	return sess.conn.Write(sess.ctx, websocket.MessageText, body) == nil
}

// resetContinuation forgets what the server held, so the next turn sends the
// whole conversation.
func (s *Service) resetContinuation() {
	s.mu.Lock()
	s.cont = continuation{}
	s.mu.Unlock()
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
// only what is new. full is the whole input the conversation rendered to and
// sent is how many of those items the server has seen.
func (s *Service) recordContinuation(state *streamState, full []inputItem, sent int) {
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
// server held. It is safe to call more than once.
func (s *Service) disconnect() {
	s.mu.Lock()
	sess := s.sess
	s.sess = nil
	s.cont = continuation{}
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
