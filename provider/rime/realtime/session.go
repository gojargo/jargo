package realtime

import (
	"context"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"errors"
	"io"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/provider/rime/internal/rimepcm"
	"github.com/gojargo/jargo/service/wsutil"
	"github.com/google/uuid"
)

const (
	// sessionLimit bounds the responses a session retains, the protected ones
	// and the ordinary ones each, and the operations it has pending.
	sessionLimit = 128
	// eventQueueSize bounds the events waiting to be consumed.
	eventQueueSize = 256
	// maxAudioChunk is the most audio one send takes.
	maxAudioChunk = 192000
	// wireAudioBytes is the audio one wire message carries: 40 ms.
	wireAudioBytes = 1280
	// modelPrism is the only model a session runs.
	modelPrism = "prism"
)

// notReady are the refusals that mean Prism is not ready for another reply
// yet, which is waited out.
//
//nolint:gochecknoglobals // a fixed set
var notReady = []string{"typed_turn_busy", "instructions_busy", "proactive_unavailable"}

// signal is a condition that can be set, cleared and waited for. It is guarded
// by the lock of the session it belongs to.
type signal struct {
	set bool
	ch  chan struct{}
}

func newSignal() *signal { return &signal{ch: make(chan struct{})} }

func (s *signal) fire() {
	if !s.set {
		s.set = true
		close(s.ch)
	}
}

func (s *signal) reset() {
	if s.set {
		s.set = false
		s.ch = make(chan struct{})
	}
}

// lock is a mutex whose acquisition can be given up on.
type lock chan struct{}

func newLock() lock { return make(lock, 1) }

func (l lock) acquire(ctx context.Context) error {
	select {
	case l <- struct{}{}:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

func (l lock) release() { <-l }

// locked reports whether the lock is held.
func (l lock) locked() bool { return len(l) > 0 }

// responseState is what the session keeps of one reply.
type responseState struct {
	ref     ResponseRef
	done    bool
	doneCh  chan struct{}
	changed *signal
	// calls holds the digest of each call's recorded result, nil until it has
	// one.
	calls      map[string]*string
	outputs    map[OutputRef]bool
	token      string
	continued  bool
	abandoned  bool
	holds      int
	cancelLock lock
}

func newResponseState(ref ResponseRef) *responseState {
	return &responseState{
		ref:        ref,
		doneCh:     make(chan struct{}),
		changed:    newSignal(),
		calls:      map[string]*string{},
		outputs:    map[OutputRef]bool{},
		cancelLock: newLock(),
	}
}

func (r *responseState) finish() {
	if !r.done {
		r.done = true
		close(r.doneCh)
	}
}

// protected reports whether the response is still needed: it is generating, it
// is held, or one of its tools has no result yet.
func (r *responseState) protected() bool {
	if !r.done || r.holds > 0 {
		return true
	}
	for _, result := range r.calls {
		if result == nil {
			return true
		}
	}
	return false
}

// pendingRequest is a request waiting for its acknowledgment.
type pendingRequest struct {
	req    *request
	target string
	// itemID is the item a typed turn created, which a later refusal names.
	itemID            string
	interruptOnSpeech *bool
	toolOutputDigest  *string
	requiresItemID    bool
	expectedItemID    string

	settled bool
	ack     any
	err     error
	done    chan struct{}
}

func (p *pendingRequest) settle(ack any, err error) {
	if !p.settled {
		p.settled, p.ack, p.err = true, ack, err
		close(p.done)
	}
}

// responseHold is a local lease on a response. Releasing it is idempotent and
// sends nothing.
type responseHold struct {
	session *session
	state   *responseState
}

func (h *responseHold) release() {
	s := h.session
	if s == nil {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	h.session = nil
	if h.state.holds > 0 {
		h.state.holds--
		_ = s.pruneResponses()
	}
}

// session is one Prism conversation. Its events are consumed while its
// operations run.
//
// It retains 128 ordinary responses and up to 128 protected ones, without
// retaining audio. A response that is generating, has a tool without a result,
// or is held is protected; other older references expire. A consumer too slow
// for the events closes the session rather than dropping audio or holding up
// control messages.
type session struct {
	conn     *wsutil.Conn
	timeouts Timeouts

	// The locks an operation holds across its request.
	admission    lock
	writeLock    lock
	audioLock    lock
	clearLock    lock
	truncateLock lock
	toolLock     lock

	readerDone chan struct{}
	closeOnce  sync.Once
	background sync.WaitGroup

	// mu guards everything below.
	mu          sync.Mutex
	info        *SessionInfo
	created     *sessionView
	createdErr  error
	createdCh   chan struct{}
	createdDone bool
	queue       []*Event
	queueWake   chan struct{}
	failure     error
	closed      bool
	ready       *signal
	readyEpoch  int
	pending     map[string]*pendingRequest
	// pendingOrder is the order the requests were made in.
	pendingOrder []string
	responses    map[string]*responseState
	// responseOrder is the order the responses started in.
	responseOrder []string
	// completed are the responses that expired, so a late event of one is told
	// from an event of a response that never existed.
	completed             []string
	policies              *interruptionPolicies
	responseInterruptions map[string]bool
	latest                string
	tasks                 int
	inputConverter        *rimepcm.InputConverter
}

// newSession starts reading conn.
func newSession(conn *wsutil.Conn, timeouts Timeouts) *session {
	s := &session{
		conn:                  conn,
		timeouts:              timeouts,
		admission:             newLock(),
		writeLock:             newLock(),
		audioLock:             newLock(),
		clearLock:             newLock(),
		truncateLock:          newLock(),
		toolLock:              newLock(),
		readerDone:            make(chan struct{}),
		createdCh:             make(chan struct{}),
		queueWake:             make(chan struct{}, 1),
		ready:                 newSignal(),
		pending:               map[string]*pendingRequest{},
		responses:             map[string]*responseState{},
		policies:              newInterruptionPolicies(),
		responseInterruptions: map[string]bool{},
	}
	go s.read()
	return s
}

// sessionInfo is the settings Prism last acknowledged.
func (s *session) sessionInfo() (SessionInfo, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.info == nil {
		return SessionInfo{}, inputError("The session has not initialized")
	}
	return *s.info, nil
}

// next returns the next event. It returns io.EOF once the session was closed
// and its events consumed, and the failure of a session that failed.
func (s *session) next(ctx context.Context) (*Event, error) {
	for {
		s.mu.Lock()
		if s.failure != nil {
			err := s.failure
			s.mu.Unlock()
			return nil, err
		}
		if len(s.queue) > 0 {
			event := s.queue[0]
			s.queue = s.queue[1:]
			s.mu.Unlock()
			if event == nil {
				return nil, io.EOF
			}
			return event, nil
		}
		closed := s.closed
		s.mu.Unlock()
		if closed {
			return nil, io.EOF
		}
		select {
		case <-s.queueWake:
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}
}

func (s *session) wakeConsumer() {
	select {
	case s.queueWake <- struct{}{}:
	default:
	}
}

// initialize waits for the session to be created and configures it.
func (s *session) initialize(ctx context.Context, settings map[string]any) error {
	timer := time.NewTimer(s.timeouts.Connect)
	defer timer.Stop()
	select {
	case <-s.createdCh:
	case <-timer.C:
		return newError(KindTimeout, "No session.created event arrived")
	case <-ctx.Done():
		return ctx.Err()
	}
	s.mu.Lock()
	err := s.createdErr
	s.mu.Unlock()
	if err != nil {
		return err
	}
	_, err = s.request(ctx, reqUpdate, map[string]any{keySession: settings}, "", nil)
	return err
}

// check reports why the session cannot be used, when it cannot.
func (s *session) check() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.checkLocked()
}

func (s *session) checkLocked() error {
	if s.failure != nil {
		return s.failure
	}
	if s.closed {
		return inputError("The realtime session is closed")
	}
	return nil
}

// responseLocked is the state of the response a reference names.
func (s *session) responseLocked(sessionID, responseID, callID string) (*responseState, error) {
	if err := s.checkLocked(); err != nil {
		return nil, err
	}
	if s.info == nil {
		return nil, inputError("The session has not initialized")
	}
	if sessionID != s.info.SessionID {
		return nil, inputError("Reference belongs to another session")
	}
	response, ok := s.responses[responseID]
	if !ok {
		return nil, inputError("Unknown or expired response reference")
	}
	if callID != "" {
		if _, ok := response.calls[callID]; !ok {
			return nil, inputError("Unknown tool call reference")
		}
	}
	return response, nil
}

func (s *session) protectedCount() int {
	count := 0
	for _, state := range s.responses {
		if state.protected() {
			count++
		}
	}
	return count
}

// pruneResponses drops the oldest responses nothing needs once more than the
// limit of them are kept.
func (s *session) pruneResponses() error {
	protected := s.protectedCount()
	if protected > sessionLimit {
		return resourceLimitError("Too many protected realtime responses")
	}
	ordinary := len(s.responses) - protected
	if ordinary <= sessionLimit {
		return nil
	}
	kept := s.responseOrder[:0:0]
	for _, key := range s.responseOrder {
		state := s.responses[key]
		if ordinary <= sessionLimit || state.protected() {
			kept = append(kept, key)
			continue
		}
		delete(s.responses, key)
		delete(s.responseInterruptions, key)
		s.completed = append(s.completed, key)
		if len(s.completed) > sessionLimit {
			s.completed = s.completed[1:]
		}
		ordinary--
	}
	s.responseOrder = kept
	return nil
}

// holdResponse protects a response until the hold is released or the session
// closes. Holds on one response share a count. Acquiring one at the protected
// limit fails, without evicting another owner or sending anything.
func (s *session) holdResponse(ref ResponseRef) (*responseHold, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	state, err := s.responseLocked(ref.SessionID, ref.ResponseID, "")
	if err != nil {
		return nil, err
	}
	if !state.protected() && s.protectedCount() >= sessionLimit {
		return nil, resourceLimitError("Too many protected realtime responses")
	}
	state.holds++
	return &responseHold{session: s, state: state}, nil
}

// receivedCloseCode is the close code the peer sent, or 0 when it sent none.
func receivedCloseCode(err error) int {
	code := int(websocket.CloseStatus(err))
	if code < 0 || code == int(websocket.StatusNoStatusRcvd) || code == int(websocket.StatusAbnormalClosure) {
		return 0
	}
	return code
}

// send writes one event, bounded by the request timeout. A write that fails or
// times out has an unknown outcome, so it ends the session.
func (s *session) send(event map[string]any, abandoned, submitted *atomic.Bool) error {
	if err := s.check(); err != nil {
		return err
	}
	ctx, cancel := context.WithTimeout(context.Background(), s.timeouts.Request)
	defer cancel()
	if err := s.writeLock.acquire(ctx); err != nil {
		return s.writeTimedOut()
	}
	defer s.writeLock.release()
	if err := s.check(); err != nil {
		return err
	}
	if abandoned != nil && abandoned.Load() {
		// Given up on before it was sent.
		return context.Canceled
	}
	encoded, err := json.Marshal(event)
	if err != nil {
		return inputError("Realtime events must contain finite JSON values")
	}
	if submitted != nil {
		submitted.Store(true)
	}
	if err := s.conn.Write(ctx, websocket.MessageText, encoded); err != nil {
		if ctx.Err() != nil || errors.Is(err, context.DeadlineExceeded) || errors.Is(err, wsutil.ErrSendTimeout) {
			return s.writeTimedOut()
		}
		failure := streamError("Realtime write failed; outcome unknown")
		failure.CloseCode = receivedCloseCode(err)
		s.fail(failure)
		return failure
	}
	return nil
}

func (s *session) writeTimedOut() error {
	failure := newError(KindTimeout, "Realtime write timed out; outcome unknown")
	s.fail(failure)
	return failure
}

// requestOutcome is how a request ended.
type requestOutcome struct {
	ack any
	err error
}

// request sends a request and waits for its acknowledgment.
//
// The request runs to a known outcome even when ctx ends first: it may already
// have been accepted, so it keeps matching the answer, and a reply it started
// that nobody is waiting for any more is canceled. Nothing is replayed.
func (s *session) request(
	ctx context.Context, req *request, body map[string]any, target string, continuation *responseState,
) (any, error) {
	s.mu.Lock()
	if err := s.checkLocked(); err != nil {
		s.mu.Unlock()
		return nil, err
	}
	if s.tasks >= sessionLimit {
		s.mu.Unlock()
		return nil, resourceLimitError("Too many pending realtime operations")
	}
	// The turn is reserved only once the local checks pass, and before the
	// lock is let go, so two callers cannot reserve the same turn.
	if req.kind == kindResponseNew {
		if continuation != nil {
			continuation.continued = true
		} else {
			s.ready.reset()
		}
	}
	epoch := s.readyEpoch
	s.tasks++
	s.mu.Unlock()

	abandoned := &atomic.Bool{}
	done := make(chan requestOutcome, 1)
	go func() {
		ack, err := s.runRequest(req, body, target, abandoned, epoch, continuation)
		s.mu.Lock()
		s.tasks--
		s.mu.Unlock()
		done <- requestOutcome{ack: ack, err: err}
	}()

	select {
	case outcome := <-done:
		if outcome.err != nil {
			return nil, outcome.err
		}
		if err := req.check(outcome.ack); err != nil {
			return nil, err
		}
		return outcome.ack, nil
	case <-ctx.Done():
		abandoned.Store(true)
		if req.kind != kindResponseNew {
			// The locks of an operation are held until its request has a known
			// outcome.
			<-done
			return nil, ctx.Err()
		}
		// The request may have been accepted already. A reply it started that
		// its caller never received is canceled, without replaying anything.
		select {
		case outcome := <-done:
			if started, ok := outcome.ack.(ResponseStarted); ok && outcome.err == nil {
				s.abandonResponse(started.Response)
			}
		default:
		}
		return nil, ctx.Err()
	}
}

// abandonResponse cancels a reply nobody is waiting for any more, and says
// so.
func (s *session) abandonResponse(ref ResponseRef) {
	s.mu.Lock()
	state, ok := s.responses[ref.ResponseID]
	if !ok || state.abandoned || s.closed {
		s.mu.Unlock()
		return
	}
	state.abandoned = true
	_ = s.publishLocked(ResponseAbandoned{Response: ref}, "", "", nil)
	s.mu.Unlock()
	s.spawn(func() { _ = s.cancel(context.Background(), ref) })
}

// spawn runs work the session owns in the background.
func (s *session) spawn(work func()) {
	s.mu.Lock()
	s.tasks++
	s.mu.Unlock()
	s.background.Go(func() {
		defer func() {
			s.mu.Lock()
			s.tasks--
			s.mu.Unlock()
		}()
		work()
	})
}

// newPending registers a request that is about to be sent.
func (s *session) newPending(req *request, body map[string]any, target string) (string, *pendingRequest) {
	eventID := "evt_" + strings.ReplaceAll(uuid.NewString(), "-", "")
	p := &pendingRequest{req: req, target: target, done: make(chan struct{})}
	if req.kind == kindSessionUpdate {
		if settings, ok := body[keySession].(map[string]any); ok {
			if turn, ok := settings[keyTurn].(map[string]any); ok {
				if interrupt, ok := turn["interrupt_response"].(bool); ok {
					p.interruptOnSpeech = &interrupt
				}
			}
		}
	}
	item, _ := body[keyItem].(map[string]any)
	if req.history {
		p.requiresItemID = true
		p.expectedItemID, _ = item[keyID].(string)
	}
	if item[keyType] == itemFunctionOut {
		output, _ := item[keyOutput].(string)
		digest := digestOf(output)
		p.toolOutputDigest = &digest
	}
	s.mu.Lock()
	s.pending[eventID] = p
	s.pendingOrder = append(s.pendingOrder, eventID)
	s.mu.Unlock()
	return eventID, p
}

func digestOf(output string) string {
	sum := sha256.Sum256([]byte(output))
	return hex.EncodeToString(sum[:])
}

// runRequest sends one request and waits for the event that answers it.
func (s *session) runRequest(
	req *request, body map[string]any, target string, abandoned *atomic.Bool, epoch int,
	continuation *responseState,
) (any, error) {
	eventID, p := s.newPending(req, body, target)
	submitted := &atomic.Bool{}
	defer s.finishRequest(req, eventID, submitted, epoch, continuation)

	timer := time.NewTimer(s.timeouts.Request)
	defer timer.Stop()
	message := map[string]any{keyType: req.kind, keyEventID: eventID}
	for key, value := range body {
		message[key] = value
	}
	if err := s.send(message, abandoned, submitted); err != nil {
		return nil, err
	}
	select {
	case <-p.done:
	case <-timer.C:
		failure := newError(KindTimeout, req.kind+": no acknowledgment; outcome unknown")
		failure.RequestID = eventID
		s.fail(failure)
		return nil, failure
	}
	s.mu.Lock()
	ack, err := p.ack, p.err
	s.mu.Unlock()
	if err != nil {
		s.refused(req, err, epoch, continuation)
		return nil, err
	}
	if started, ok := ack.(ResponseStarted); ok && abandoned.Load() {
		s.abandonResponse(started.Response)
	}
	return ack, nil
}

// refused gives back the turn a refused reply request had reserved, when the
// refusal took none.
func (s *session) refused(req *request, err error, epoch int, continuation *responseState) {
	refusal, ok := errors.AsType[*FaultError](err)
	if !ok || req.kind != kindResponseNew || refusal.Fault.Scope != scopeEvent {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	code := refusal.Fault.Code
	if continuation == nil && (code == "invalid_typed_input" || code == "invalid_instruction") &&
		epoch == s.readyEpoch {
		// These validation refusals consume no turn. Prism stays ready, so it
		// sends no other ready event.
		s.ready.fire()
	}
	if continuation != nil && code == "tool_continuation_not_ready" {
		// A refusal with no effect releases the request, abandoned or not.
		continuation.continued = false
	}
}

// finishRequest forgets a request. A reservation is released only when
// nothing was sent: a send that started has an unknown outcome, and a newer
// readiness the server reported is not overridden.
func (s *session) finishRequest(
	req *request, eventID string, submitted *atomic.Bool, epoch int, continuation *responseState,
) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if req.kind == kindResponseNew && !submitted.Load() {
		if continuation != nil {
			continuation.continued = false
		} else if epoch == s.readyEpoch {
			s.ready.fire()
		}
	}
	delete(s.pending, eventID)
	s.pendingOrder = slices.DeleteFunc(s.pendingOrder, func(id string) bool { return id == eventID })
}

func nonblank(value string) error {
	if strings.TrimSpace(value) == "" {
		return inputError("Text must be nonblank")
	}
	return nil
}

// create asks for a reply once Prism is ready for one, and returns when the
// reply is created.
func (s *session) create(ctx context.Context, body map[string]any, req *request) (ResponseRef, error) {
	deadline, cancel := context.WithTimeout(ctx, s.timeouts.Ready)
	defer cancel()
	if err := s.admission.acquire(deadline); err != nil {
		if ctx.Err() != nil {
			return ResponseRef{}, ctx.Err()
		}
		return ResponseRef{}, &AdmissionTimeoutError{Message: "Prism admission lock exceeded ready_s"}
	}
	defer s.admission.release()
	for {
		s.mu.Lock()
		if err := s.checkLocked(); err != nil {
			s.mu.Unlock()
			return ResponseRef{}, err
		}
		ready := s.ready.ch
		s.mu.Unlock()
		select {
		case <-ready:
		case <-deadline.Done():
			if ctx.Err() != nil {
				return ResponseRef{}, ctx.Err()
			}
			return ResponseRef{}, &AdmissionTimeoutError{
				Message: "Prism did not admit the turn before ready_s expired",
			}
		}
		if err := s.check(); err != nil {
			return ResponseRef{}, err
		}
		// Once submitted, the request timeout governs the acknowledgment. An
		// admission timeout must never hide an unknown wire outcome.
		ack, err := s.request(ctx, req, map[string]any{keyResponse: body}, "", nil)
		if err == nil {
			started, _ := ack.(ResponseStarted)
			return started.Response, nil
		}
		refusal, ok := errors.AsType[*FaultError](err)
		if !ok || refusal.Fault.Scope != scopeEvent || !slices.Contains(notReady, refusal.Fault.Code) {
			return ResponseRef{}, err
		}
	}
}

// sessionPatch is a change to a session's settings. A nil field keeps a
// setting; empty instructions or tools clear it, and a tool list given replaces
// all the tools. Replies already started keep their settings.
type sessionPatch struct {
	instructions      *string
	tools             *[]toolDefinition
	voice             *string
	interruptOnSpeech *bool
	speed             *float64
	turnDetection     *TurnDetectionUpdate
	truncation        *Truncation
}

func (p sessionPatch) empty() bool {
	return p.instructions == nil && p.tools == nil && p.voice == nil && p.interruptOnSpeech == nil &&
		p.speed == nil && p.turnDetection == nil && p.truncation == nil
}

// settings is the wire form of the patch.
func (p sessionPatch) settings() (map[string]any, error) {
	settings := map[string]any{}
	if p.truncation != nil {
		value, err := truncationSettings(*p.truncation)
		if err != nil {
			return nil, err
		}
		settings["truncation"] = value
	}
	if p.instructions != nil {
		settings["instructions"] = *p.instructions
	}
	if p.voice != nil {
		settings["voice"] = *p.voice
	}
	if p.tools != nil {
		tools, err := toolSettings(*p.tools)
		if err != nil {
			return nil, err
		}
		settings[keyTools] = tools
	}
	if p.speed != nil {
		speed, err := speedSetting(*p.speed)
		if err != nil {
			return nil, err
		}
		settings["speed"] = speed
	}
	turn := map[string]any{}
	if p.turnDetection != nil {
		rules, err := interruptionSettings(*p.turnDetection)
		if err != nil {
			return nil, err
		}
		turn = rules
		settings[keyTurn] = turn
	}
	if p.interruptOnSpeech != nil {
		turn["interrupt_response"] = *p.interruptOnSpeech
		settings[keyTurn] = turn
	}
	return settings, nil
}

// update applies the settings given and waits for the server to acknowledge
// them. Giving up after the request was sent does not undo an accepted update.
func (s *session) update(ctx context.Context, patch sessionPatch) (SessionInfo, error) {
	if err := s.check(); err != nil {
		return SessionInfo{}, err
	}
	settings, err := patch.settings()
	if err != nil {
		return SessionInfo{}, err
	}
	ack, err := s.request(ctx, reqUpdate, map[string]any{keySession: settings}, "", nil)
	if err != nil {
		return SessionInfo{}, err
	}
	updated, _ := ack.(sessionUpdated)
	return *updated.info, nil
}

// sendText starts a typed user turn, and returns when its reply is created.
func (s *session) sendText(ctx context.Context, text string) (ResponseRef, error) {
	if err := nonblank(text); err != nil {
		return ResponseRef{}, err
	}
	metadata, err := requestMetadata(nil, map[string]string{keyCause: "user_text"})
	if err != nil {
		return ResponseRef{}, err
	}
	return s.create(ctx, map[string]any{"prism_input_text": text, keyMetadata: metadata}, reqText)
}

// requestReply asks Prism to reply without the user speaking, following
// instruction when one is given and reporting the result of call when one is
// named.
func (s *session) requestReply(ctx context.Context, instruction *string, call *ToolCallRef) (ResponseRef, error) {
	metadata, err := requestMetadata(nil, map[string]string{keyCause: causeProactive})
	if err != nil {
		return ResponseRef{}, err
	}
	body := map[string]any{keyMetadata: metadata}
	if instruction != nil {
		if err := nonblank(*instruction); err != nil {
			return ResponseRef{}, err
		}
		body["prism_instruction"] = *instruction
	}
	if call != nil {
		s.mu.Lock()
		response, err := s.responseLocked(call.SessionID, call.ResponseID, call.CallID)
		recorded := err == nil && response.calls[call.CallID] != nil
		s.mu.Unlock()
		if err != nil {
			return ResponseRef{}, err
		}
		if !recorded {
			return ResponseRef{}, inputError("Record the tool result before requesting its report")
		}
		body["prism_call_id"] = call.CallID
	}
	return s.create(ctx, body, reqCreate)
}

// history reads the items the next reply would see, in the server's order.
func (s *session) history(ctx context.Context) ([]historyItem, error) {
	ack, err := s.request(ctx, reqGetHistory, map[string]any{}, "", nil)
	if err != nil {
		return nil, err
	}
	listed, _ := ack.(historyListed)
	return listed.items, nil
}

// deleteItem hides an item from future replies. A call and its result are
// hidden together.
func (s *session) deleteItem(ctx context.Context, itemID string) error {
	if _, err := historyID(itemID); err != nil {
		return inputError("item_id must be a nonempty string")
	}
	_, err := s.request(ctx, reqDeleteItem, map[string]any{keyItemID: itemID}, itemID, nil)
	return err
}

// createItem adds an item to the end of the history without starting a reply,
// and returns the identity the server gave it. A tool result follows its call.
func (s *session) createItem(ctx context.Context, item historyItem) (string, error) {
	value, err := itemSettings(item)
	if err != nil {
		return "", err
	}
	body := map[string]any{keyItem: value}
	var createdID string
	if item.kind == itemFunctionOut {
		ack, err := s.request(ctx, reqHistoryResult, body, item.callID, nil)
		if err != nil {
			return "", err
		}
		created, _ := ack.(toolResultCreated)
		createdID = created.itemID
	} else {
		ack, err := s.request(ctx, reqHistoryItem, body, "", nil)
		if err != nil {
			return "", err
		}
		created, _ := ack.(itemCreated)
		createdID = created.itemID
	}
	if createdID == "" || (item.id != "" && item.id != createdID) {
		failure := streamError("History acknowledgment names an invalid item")
		s.fail(failure)
		return "", failure
	}
	return createdID, nil
}

// addMessage adds a user or assistant message to the end of the history.
func (s *session) addMessage(ctx context.Context, role, text string) (string, error) {
	if role != "user" && role != "assistant" {
		return "", inputError("History role must be user or assistant")
	}
	if err := nonblank(text); err != nil {
		return "", err
	}
	return s.createItem(ctx, historyItem{kind: itemMessage, role: role, text: text})
}

// submitToolResult records the result of a tool call. It runs to a known
// outcome whatever becomes of its caller, and a result recorded once cannot
// change.
func (s *session) submitToolResult(call ToolCallRef, output string) error {
	s.mu.Lock()
	_, err := s.responseLocked(call.SessionID, call.ResponseID, call.CallID)
	if err == nil && s.tasks >= sessionLimit {
		err = resourceLimitError("Too many pending realtime operations")
	}
	s.mu.Unlock()
	if err != nil {
		return err
	}
	ctx := context.Background()
	_ = s.toolLock.acquire(ctx)
	defer s.toolLock.release()

	digest := digestOf(output)
	s.mu.Lock()
	response, err := s.responseLocked(call.SessionID, call.ResponseID, call.CallID)
	var recorded *string
	if err == nil {
		recorded = response.calls[call.CallID]
	}
	s.mu.Unlock()
	if err != nil {
		return err
	}
	if !utf8Valid(output) {
		return inputError("Tool output must be a string")
	}
	if recorded != nil {
		if *recorded != digest {
			return inputError("A recorded tool result cannot change")
		}
		return nil
	}
	body := map[string]any{keyItem: map[string]any{
		keyType: itemFunctionOut, keyCallID: call.CallID, keyOutput: output,
	}}
	if _, err := s.request(ctx, reqToolResult, body, call.CallID, nil); err != nil {
		return err
	}
	s.mu.Lock()
	response.calls[call.CallID] = &digest
	response.changed.fire()
	s.mu.Unlock()
	return nil
}

func utf8Valid(value string) bool { return strings.ToValidUTF8(value, "") == value }

// continueReply asks Prism to continue a reply whose tools have all returned,
// once the reply has ended.
func (s *session) continueReply(ctx context.Context, parent ResponseRef) (ResponseRef, error) {
	metadata, err := requestMetadata(nil, map[string]string{
		keyCause:                   "tool_continuation",
		"prism_parent_response_id": parent.ResponseID,
	})
	if err != nil {
		return ResponseRef{}, err
	}
	deadline, cancel := context.WithTimeout(ctx, s.timeouts.Ready)
	defer cancel()
	var response *responseState
	for {
		s.mu.Lock()
		response, err = s.responseLocked(parent.SessionID, parent.ResponseID, "")
		ready := false
		if err == nil {
			ready, err = s.continuableLocked(parent, response)
		}
		if err != nil || ready {
			s.mu.Unlock()
			break
		}
		response.changed.reset()
		changed := response.changed.ch
		s.mu.Unlock()
		select {
		case <-changed:
		case <-deadline.Done():
			if ctx.Err() != nil {
				return ResponseRef{}, ctx.Err()
			}
			return ResponseRef{}, &AdmissionTimeoutError{
				Message: "Tool results or parent completion did not arrive before ready_s expired",
			}
		}
	}
	if err != nil {
		return ResponseRef{}, err
	}
	body := map[string]any{keyResponse: map[string]any{keyMetadata: metadata}}
	ack, err := s.request(ctx, reqCreate, body, "", response)
	if err != nil {
		return ResponseRef{}, err
	}
	started, _ := ack.(ResponseStarted)
	return started.Response, nil
}

// continuableLocked reports whether a tool round can be continued now: its
// reply has ended and every one of its tools has returned. It fails for a
// round that never can be.
func (s *session) continuableLocked(parent ResponseRef, response *responseState) (bool, error) {
	if parent.ResponseID != s.latest || response.continued {
		return false, inputError("The tool round is superseded or already continued")
	}
	if !response.done {
		return false, nil
	}
	if len(response.calls) == 0 {
		return false, inputError("The response has no tool calls")
	}
	for _, result := range response.calls {
		if result == nil {
			return false, nil
		}
	}
	return true, nil
}

// cancel asks a reply to stop if it is still running, and returns once it has
// ended. A reply can finish before the cancellation takes effect, so returning
// does not mean it was canceled: its end says what became of it.
func (s *session) cancel(ctx context.Context, response ResponseRef) error {
	s.mu.Lock()
	state, err := s.responseLocked(response.SessionID, response.ResponseID, "")
	s.mu.Unlock()
	if err != nil {
		return err
	}
	if err := state.cancelLock.acquire(ctx); err != nil {
		return err
	}
	defer state.cancelLock.release()
	s.mu.Lock()
	done := state.done
	s.mu.Unlock()
	if done {
		return nil
	}
	_, err = s.request(ctx, reqCancel, map[string]any{keyResponseID: response.ResponseID}, response.ResponseID, nil)
	return err
}

// setResponseMetadata replaces the labels later replies carry, and returns
// once Prism confirms the change. A reply already started keeps its own.
func (s *session) setResponseMetadata(ctx context.Context, labels map[string]string) error {
	body := map[string]any{keySession: map[string]any{keyLabels: responseMetadata(labels)}}
	_, err := s.request(ctx, reqUpdate, body, "", nil)
	return err
}

// clearAudio drops the audio sent that Prism has not committed to a turn.
func (s *session) clearAudio(ctx context.Context) error {
	if err := s.audioLock.acquire(ctx); err != nil {
		return err
	}
	defer s.audioLock.release()
	if err := s.clearLock.acquire(ctx); err != nil {
		return err
	}
	defer s.clearLock.release()
	_, err := s.request(ctx, reqClear, map[string]any{}, "", nil)
	return err
}

// sendAudio sends one bounded chunk, waiting on the transport when it is slow.
// A change of format starts the conversion afresh.
func (s *session) sendAudio(chunk audioChunk) error {
	if err := chunk.validate(); err != nil {
		return err
	}
	_ = s.audioLock.acquire(context.Background())
	defer s.audioLock.release()
	s.mu.Lock()
	err := s.checkLocked()
	previous := s.inputConverter
	s.mu.Unlock()
	if err != nil {
		return err
	}
	if len(chunk.data) > maxAudioChunk {
		return newError(KindAudioFormat, "Send audio in chunks of at most 192000 bytes")
	}
	converter := rimepcm.NewInputConverter(chunk.format)
	if previous != nil && previous.Format() == chunk.format {
		converter = previous.Clone()
	}
	s.mu.Lock()
	s.inputConverter = converter
	s.mu.Unlock()
	data := converter.Process(chunk.data)
	// 40 ms to a wire message, so no unbounded queue of encoded audio builds
	// up.
	for offset := 0; offset < len(data); offset += wireAudioBytes {
		part := data[offset:min(offset+wireAudioBytes, len(data))]
		if err := s.send(map[string]any{
			keyType: "input_audio_buffer.append",
			"audio": base64.StdEncoding.EncodeToString(part),
		}, nil, nil); err != nil {
			return err
		}
	}
	return nil
}

// reportInterrupted tells Prism how much of a message was heard before it was
// cut off.
func (s *session) reportInterrupted(ctx context.Context, report playbackInterrupted) error {
	if err := report.validate(); err != nil {
		return err
	}
	s.mu.Lock()
	state, err := s.responseLocked(report.output.Response.SessionID, report.output.Response.ResponseID, "")
	if err == nil && !state.outputs[report.output] {
		err = inputError("Unknown output reference")
	}
	s.mu.Unlock()
	if err != nil {
		return err
	}
	if err := s.truncateLock.acquire(ctx); err != nil {
		return err
	}
	defer s.truncateLock.release()
	_, err = s.request(ctx, reqTruncate, map[string]any{
		keyItemID:       report.output.ItemID,
		keyContentIndex: report.output.ContentIndex,
		keyAudioEndMS:   report.audioEndMS,
	}, report.output.ItemID, nil)
	return err
}

// reportFinished tells Prism how much of a reply was played, once the reply
// has ended.
func (s *session) reportFinished(ctx context.Context, report playbackFinished) error {
	if err := report.validate(); err != nil {
		return err
	}
	s.mu.Lock()
	state, err := s.responseLocked(report.response.SessionID, report.response.ResponseID, "")
	s.mu.Unlock()
	if err != nil {
		return err
	}
	timer := time.NewTimer(s.timeouts.Ready)
	defer timer.Stop()
	select {
	case <-state.doneCh:
	case <-timer.C:
		return newError(KindTimeout, "Response did not finish before playback report deadline")
	case <-ctx.Done():
		return ctx.Err()
	}
	s.mu.Lock()
	err = s.checkLocked()
	token := state.token
	s.mu.Unlock()
	if err != nil || token == "" {
		return err
	}
	event := map[string]any{
		keyType:           "prism.playback.drained",
		keyResponseID:     report.response.ResponseID,
		"drain_token":     token,
		"played_ms":       nil,
		"audible_tail_ms": nil,
	}
	if report.playedMS != nil {
		event["played_ms"] = *report.playedMS
	}
	if report.audibleTailMS != nil {
		event["audible_tail_ms"] = *report.audibleTailMS
	}
	return s.send(event, nil, nil)
}

// settleLocked answers the request waiting under key, if one is.
func (s *session) settleLocked(key string, ack any) error {
	p, ok := s.pending[key]
	if !ok {
		return nil
	}
	if err := p.req.check(ack); err != nil {
		return err
	}
	p.settle(ack, nil)
	return nil
}

func (s *session) rejectLocked(key string, err error) {
	if p, ok := s.pending[key]; ok {
		p.settle(nil, err)
	}
}

// fail ends the session with a failure. Everything waiting on it is woken to
// see the failure.
func (s *session) fail(failure error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.failLocked(failure)
}

func (s *session) failLocked(failure error) {
	if s.closed {
		return
	}
	s.failure = failure
	s.closed = true
	s.ready.fire()
	for _, state := range s.responses {
		state.finish()
		state.changed.fire()
	}
	for _, p := range s.pending {
		p.settle(nil, failure)
	}
	if !s.createdDone {
		s.createdDone, s.createdErr = true, failure
		close(s.createdCh)
	}
	if len(s.queue) >= eventQueueSize {
		s.queue = s.queue[1:]
	}
	s.queue = append(s.queue, nil)
	s.wakeConsumer()
	s.background.Go(func() { _ = s.conn.CloseNow() })
}

// read takes the server's events until the connection ends.
func (s *session) read() {
	defer close(s.readerDone)
	defer func() { _ = s.conn.CloseNow() }()
	for {
		kind, data, err := s.conn.Read(context.Background())
		if err != nil {
			s.connectionEnded(err)
			return
		}
		if err := s.receive(kind == websocket.MessageBinary, data); err != nil {
			s.fail(err)
			return
		}
	}
}

// connectionEnded fails the session whose connection ended under it.
func (s *session) connectionEnded(err error) {
	failure := streamError("Invalid event or lost realtime connection")
	status := websocket.CloseStatus(err)
	if status == websocket.StatusNormalClosure || status == websocket.StatusGoingAway {
		failure = streamError("Realtime connection closed; reconnect is not automatic")
	}
	failure.CloseCode = receivedCloseCode(err)
	s.fail(failure)
}

// receive decodes one message and acts on it.
func (s *session) receive(binary bool, data []byte) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	var sessionID *string
	if s.created != nil {
		sessionID = &s.created.id
	}
	event, err := decode(binary, data, sessionID)
	if err != nil || event == nil {
		return err
	}
	return s.dispatchLocked(event)
}

// checkEchoLocked refuses an acknowledgment that does not match the request it
// echoes, before it changes any session state.
func (s *session) checkEchoLocked(payload any, echo string) error {
	kind, ok := ackKindOf(payload)
	if !ok || kind == ackAudioCleared || kind == ackItemTruncated || kind == ackResponseEnded {
		return nil
	}
	p, ok := s.pending[echo]
	if !ok {
		return nil
	}
	created, isCreated := payload.(itemCreated)
	result, isResult := payload.(toolResultCreated)
	if !isCreated || !p.req.allowsItemCreated {
		if err := p.req.check(payload); err != nil {
			return err
		}
	}
	if p.requiresItemID {
		itemID := created.itemID
		if isResult {
			itemID = result.itemID
		}
		if (!isCreated && !isResult) || itemID == "" || (p.expectedItemID != "" && p.expectedItemID != itemID) {
			return streamError("History acknowledgment names an invalid item")
		}
	}
	if isResult && p.target != result.callID {
		return streamError("Tool result acknowledgment names another call")
	}
	return nil
}

// dispatchLocked acts on one event: it settles the request the event answers,
// keeps the session's view of its responses, and hands the event on to the
// consumer.
func (s *session) dispatchLocked(event *serverEvent) error {
	echo := event.requestID
	switch payload := event.payload.(type) {
	case reviewRequest:
		return streamError("Speech review arrived without a configured callback")
	case sessionIdle:
		s.failLocked(&IdleTimeoutError{IdleTimeout: payload.idleTimeout})
		return nil
	}
	if err := s.checkEchoLocked(event.payload, echo); err != nil {
		return err
	}
	publish, err := s.applyLocked(event)
	if err != nil || publish == nil {
		return err
	}
	return s.publishLocked(publish, event.eventID, echo, event.serverTimeMS)
}

// applyLocked takes an event into the session's state, and returns what the
// consumer is to be told, if anything.
func (s *session) applyLocked(event *serverEvent) (EventPayload, error) {
	echo := event.requestID
	switch payload := event.payload.(type) {
	case sessionCreated:
		if !s.createdDone {
			view := payload.session
			s.created, s.createdDone = &view, true
			close(s.createdCh)
		}
		return nil, nil //nolint:nilnil // nothing to hand on
	case sessionUpdated:
		return nil, s.sessionUpdatedLocked(payload, echo)
	case itemDeleted:
		if p, ok := s.pending[echo]; ok && p.target != payload.itemID {
			return nil, streamError("Deletion acknowledgment names another item")
		}
		return nil, s.settleLocked(echo, payload)
	case historyListed:
		return nil, s.settleLocked(echo, payload)
	case toolResultCreated:
		return nil, s.toolResultCreatedLocked(payload, echo)
	case itemCreated:
		if p, ok := s.pending[echo]; ok && p.req.allowsItemCreated {
			p.itemID = payload.itemID
			return nil, nil //nolint:nilnil // nothing to hand on
		}
		return nil, s.settleLocked(echo, payload)
	case audioCleared:
		return nil, s.audioClearedLocked(payload)
	case ItemTruncated:
		return payload, s.itemTruncatedLocked(payload)
	case responseEnded:
		return s.responseEndedLocked(payload)
	default:
		return s.applyConversationLocked(event)
	}
}

// applyConversationLocked takes the events of the conversation itself: the
// user's input and the replies.
func (s *session) applyConversationLocked(event *serverEvent) (EventPayload, error) {
	switch payload := event.payload.(type) {
	case InputReady:
		s.readyEpoch++
		s.ready.fire()
		return payload, nil
	case SpeechStarted:
		s.readyEpoch++
		s.ready.reset()
		for _, state := range s.responses {
			state.changed.fire()
		}
		s.latest = ""
		return payload, nil
	case ResponseStarted:
		return payload, s.responseStartedLocked(payload, event.requestID)
	case ToolCall:
		return s.toolCallLocked(payload)
	case MessageStarted:
		state, ok := s.responses[payload.Output.Response.ResponseID]
		if !ok {
			return nil, streamError("Invalid event or lost realtime connection")
		}
		if len(state.outputs) >= sessionLimit {
			return nil, resourceLimitError("Too many messages in one response")
		}
		state.outputs[payload.Output] = true
		return payload, nil
	case TextDelta:
		return payload, s.knownResponseLocked(payload.Output)
	case TextDone:
		return payload, s.knownResponseLocked(payload.Output)
	case AudioDelta:
		return payload, s.knownResponseLocked(payload.Output)
	case AudioDone:
		return payload, s.knownResponseLocked(payload.Output)
	case FaultEvent:
		return s.faultLocked(payload)
	default:
		published, _ := event.payload.(EventPayload)
		return published, nil
	}
}

func (s *session) knownResponseLocked(output OutputRef) error {
	if _, ok := s.responses[output.Response.ResponseID]; !ok {
		return streamError("Event refers to an unknown response")
	}
	return nil
}

func (s *session) sessionUpdatedLocked(payload sessionUpdated, echo string) error {
	p, ok := s.pending[echo]
	if !ok {
		return nil
	}
	view := payload.session
	fallback := true
	if s.info != nil {
		fallback = s.info.InterruptOnSpeech
	}
	if p.interruptOnSpeech != nil {
		fallback = *p.interruptOnSpeech
	}
	info := &SessionInfo{
		SessionID:               s.created.id,
		Model:                   modelPrism,
		Speed:                   view.speed,
		Review:                  view.review,
		Voice:                   view.voice,
		InterruptOnSpeech:       fallback,
		ToolResultTimeout:       view.toolResultTimeout,
		ToolContinuationTimeout: view.toolContinuationTimeout,
		TurnDetection:           view.turnDetection,
		SettingsRevision:        view.settingsRevision,
		Truncation:              view.truncation,
		RequestID:               view.requestID,
		ResponseMetadata:        view.responseMetadata,
	}
	switch {
	case view.version != nil:
		info.Version = *view.version
	case s.info != nil:
		info.Version = s.info.Version
	case s.created.version != nil:
		info.Version = *s.created.version
	}
	if view.interruptOnSpeech != nil {
		info.InterruptOnSpeech = *view.interruptOnSpeech
	}
	s.info = info
	s.policies.record(view.settingsRevision, info.InterruptOnSpeech)
	payload.info = info
	return s.settleLocked(echo, payload)
}

func (s *session) toolResultCreatedLocked(payload toolResultCreated, echo string) error {
	if p, ok := s.pending[echo]; ok && p.toolOutputDigest != nil {
		for _, response := range s.responses {
			if result, ok := response.calls[payload.callID]; ok && result == nil {
				response.calls[payload.callID] = p.toolOutputDigest
				response.changed.fire()
			}
		}
		if err := s.pruneResponses(); err != nil {
			return err
		}
	}
	return s.settleLocked(echo, payload)
}

func (s *session) audioClearedLocked(payload audioCleared) error {
	for _, key := range s.pendingOrder {
		if s.pending[key].req.kind == kindAudioClear {
			s.inputConverter = nil
			return s.settleLocked(key, payload)
		}
	}
	return nil
}

func (s *session) itemTruncatedLocked(payload ItemTruncated) error {
	for _, key := range s.pendingOrder {
		p := s.pending[key]
		if p.req.kind == kindItemTruncate && p.target == payload.ItemID && payload.ContentIndex == 0 {
			if err := s.settleLocked(key, payload); err != nil {
				return err
			}
		}
	}
	return nil
}

func (s *session) responseStartedLocked(payload ResponseStarted, echo string) error {
	ref := payload.Response
	if _, ok := s.responses[ref.ResponseID]; ok || slices.Contains(s.completed, ref.ResponseID) {
		return streamError("Duplicate response.created")
	}
	if s.protectedCount() >= sessionLimit {
		return resourceLimitError("Too many protected realtime responses")
	}
	interrupt, err := s.policies.resolve(payload.SettingsRevision)
	if err != nil {
		return err
	}
	s.responseInterruptions[ref.ResponseID] = interrupt
	s.responses[ref.ResponseID] = newResponseState(ref)
	s.responseOrder = append(s.responseOrder, ref.ResponseID)
	for _, state := range s.responses {
		state.changed.fire()
	}
	s.latest = ref.ResponseID
	s.readyEpoch++
	if err := s.pruneResponses(); err != nil {
		return err
	}
	s.ready.reset()
	return s.settleLocked(echo, payload)
}

func (s *session) responseEndedLocked(payload responseEnded) (EventPayload, error) {
	state, ok := s.responses[payload.Response.ResponseID]
	if !ok {
		if slices.Contains(s.completed, payload.Response.ResponseID) {
			return nil, nil //nolint:nilnil // a late end of an expired response
		}
		return nil, streamError("Event refers to an unknown response")
	}
	if state.done {
		return nil, nil //nolint:nilnil // a repeated end
	}
	state.token = payload.drainToken
	state.finish()
	state.changed.fire()
	if err := s.pruneResponses(); err != nil {
		return nil, err
	}
	for _, key := range s.pendingOrder {
		p := s.pending[key]
		if p.req.kind == kindResponseStop && p.target == state.ref.ResponseID {
			if err := s.settleLocked(key, payload); err != nil {
				return nil, err
			}
		}
	}
	ended := payload.ResponseEnded
	ended.Response = state.ref
	return ended, nil
}

func (s *session) toolCallLocked(payload ToolCall) (EventPayload, error) {
	state, ok := s.responses[payload.Call.ResponseID]
	if !ok {
		return nil, streamError("Invalid event or lost realtime connection")
	}
	callID := payload.Call.CallID
	if _, ok := state.calls[callID]; ok {
		return nil, nil //nolint:nilnil // a repeated call
	}
	for _, other := range s.responses {
		if _, ok := other.calls[callID]; ok {
			return nil, streamError("Tool call ID reused by another response")
		}
	}
	if len(state.calls) >= sessionLimit {
		return nil, resourceLimitError("Too many tool calls in one response")
	}
	if !state.protected() && s.protectedCount() >= sessionLimit {
		return nil, resourceLimitError("Too many protected realtime responses")
	}
	state.calls[callID] = nil
	if err := s.pruneResponses(); err != nil {
		return nil, err
	}
	return payload, nil
}

func (s *session) faultLocked(payload FaultEvent) (EventPayload, error) {
	fault := payload.Error
	failure := &FaultError{Fault: fault}
	if slices.Contains(notReady, fault.Code) {
		s.ready.reset()
	}
	s.rejectLocked(fault.RequestID, failure)
	if fault.Code == "typed_turn_superseded" && fault.ItemID != "" {
		for _, p := range s.pending {
			if p.itemID == fault.ItemID {
				p.settle(nil, failure)
			}
		}
	}
	if fault.Scope == scopeSession {
		s.failLocked(failure)
		return nil, nil //nolint:nilnil // the failure ends the events
	}
	return payload, nil
}

// publishLocked hands an event to the consumer. A consumer that has fallen
// behind by the whole queue ends the session.
func (s *session) publishLocked(payload EventPayload, eventID, requestID string, serverTimeMS *int) error {
	if len(s.queue) >= eventQueueSize {
		failure := resourceLimitError("Realtime event consumer is too slow")
		s.failLocked(failure)
		return failure
	}
	sessionID := ""
	if s.created != nil {
		sessionID = s.created.id
	}
	s.queue = append(s.queue, &Event{
		SessionID:    sessionID,
		EventID:      eventID,
		RequestID:    requestID,
		Payload:      payload,
		ServerTimeMS: serverTimeMS,
	})
	s.wakeConsumer()
	return nil
}

// close ends the session. Its events end normally, and everything it holds is
// let go.
func (s *session) close() {
	s.closeOnce.Do(func() {
		s.mu.Lock()
		if !s.closed {
			s.failLocked(streamError("Realtime session closed"))
			// An explicit close ends the events normally.
			s.failure = nil
		}
		s.mu.Unlock()
		_ = s.conn.Close(websocket.StatusNormalClosure, "")
		s.mu.Lock()
		for _, state := range s.responses {
			state.holds = 0
		}
		s.responses = map[string]*responseState{}
		s.responseOrder, s.completed = nil, nil
		s.responseInterruptions = map[string]bool{}
		s.mu.Unlock()
		<-s.readerDone
		s.background.Wait()
	})
}
