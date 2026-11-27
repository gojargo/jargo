package realtime

import (
	"context"
	"errors"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/google/uuid"
)

// retentionLimit bounds the responses, receipts and tool policies the service
// keeps.
const retentionLimit = 128

// output is one generated message and the estimate of how far it has played.
type output struct {
	ref       OutputRef
	contextID string
	text      string
	audioMS   float64
	// startedAt is when the message started playing; zero until it has.
	startedAt time.Time
	closed    bool
}

func newOutput(ref OutputRef) *output {
	return &output{ref: ref, contextID: strings.ReplaceAll(uuid.NewString(), "-", "")}
}

// response is the state of a reply, kept until everything that depends on it
// has let go of it.
type response struct {
	ref     ResponseRef
	outputs map[string]*output
	// outputOrder is the order the messages started in.
	outputOrder   []string
	calls         map[string]bool
	inputIDs      []string
	ended         bool
	stopped       bool
	audioResponse bool
	continued     bool
	// continuing says a continuation was asked for and has not started yet.
	continuing    bool
	toolsDisabled bool
	hold          *responseHold
	operations    int
}

// tool is one call the application runs, open until its final result is
// submitted.
type tool struct {
	event     ToolCall
	submitted bool
	final     bool
	// superseded says a later reply, or Prism's deadline for the result, closed
	// the call's round.
	superseded bool
	// report says the result of a closed round is to be reported: Prism mentions
	// it only when asked.
	report bool
	// progress says progress went into the history beside the open call.
	progress bool
}

// conversation owns the references to the replies, apart from how long their
// output is kept. It is guarded by the lock of the service it belongs to, the
// settings lock excepted.
type conversation struct {
	responses map[string]*response
	// order is the order the replies started in.
	order []string
	calls map[string]*tool
	// settingsLock serializes the changes to the session's settings.
	settingsLock        lock
	suppressedRevisions map[int]bool
	policyFloor         int
	suppressingTools    bool
}

func newConversation() *conversation {
	return &conversation{
		responses:           map[string]*response{},
		calls:               map[string]*tool{},
		settingsLock:        newLock(),
		suppressedRevisions: map[int]bool{},
	}
}

// start keeps a reply while its generation, its tools or a request need it.
func (c *conversation) start(sess *session, event ResponseStarted) (*response, error) {
	c.prune()
	protected := 0
	for _, value := range c.responses {
		if value.hold != nil {
			protected++
		}
	}
	if protected >= retentionLimit {
		return nil, resourceLimitError("Prism protected response limit exceeded")
	}
	if event.SettingsRevision != nil && *event.SettingsRevision < c.policyFloor {
		return nil, resourceLimitError("Prism response tool policy expired")
	}
	hold, err := sess.holdResponse(event.Response)
	if err != nil {
		return nil, err
	}
	disabled := c.suppressingTools
	if event.SettingsRevision != nil {
		disabled = c.suppressedRevisions[*event.SettingsRevision]
	}
	value := &response{
		ref:           event.Response,
		outputs:       map[string]*output{},
		calls:         map[string]bool{},
		inputIDs:      event.InputItemIDs,
		audioResponse: true,
		hold:          hold,
		toolsDisabled: disabled,
	}
	c.responses[event.Response.ResponseID] = value
	c.order = append(c.order, event.Response.ResponseID)
	return value, nil
}

// release lets go of the session's reference once all the work that depends on
// the reply has settled.
func (c *conversation) release(value *response) {
	if value.ended && len(value.calls) == 0 && value.operations == 0 && value.hold != nil {
		value.hold.release()
		value.hold = nil
	}
}

// prune keeps at most 128 ordinary replies, without evicting one that is still
// owned.
func (c *conversation) prune() {
	var ordinary []string
	for _, key := range c.order {
		if c.responses[key].hold == nil {
			ordinary = append(ordinary, key)
		}
	}
	if len(ordinary) < retentionLimit {
		return
	}
	for _, key := range ordinary[:len(ordinary)-(retentionLimit-1)] {
		delete(c.responses, key)
		c.order = slices.DeleteFunc(c.order, func(id string) bool { return id == key })
		for callID, call := range c.calls {
			if call.event.Call.ResponseID == key {
				delete(c.calls, callID)
			}
		}
	}
}

// suppress remembers the tool policy the replies at this revision selected.
func (c *conversation) suppress(info SessionInfo) {
	if info.SettingsRevision == nil {
		return
	}
	c.suppressedRevisions[*info.SettingsRevision] = true
	if len(c.suppressedRevisions) <= retentionLimit {
		return
	}
	oldest := *info.SettingsRevision
	for revision := range c.suppressedRevisions {
		oldest = min(oldest, revision)
	}
	delete(c.suppressedRevisions, oldest)
	c.policyFloor = oldest + 1
}

// close lets go of every reference, when the conversation closes.
func (c *conversation) close() {
	for _, value := range c.responses {
		if value.hold != nil {
			value.hold.release()
			value.hold = nil
		}
	}
}

// ledger appends context messages to one conversation's history. Each append
// is validated whole before anything is written, and the writes are serialized
// with the submission of tool results through its lock.
type ledger struct {
	session *session
	lock    lock

	// mu guards writeErr, which the event loop reads without the lock.
	mu sync.Mutex
	// writeErr is the error the last write failed with, recorded before the
	// lock is let go (see failedWith).
	writeErr error
}

func newLedger(sess *session) *ledger { return &ledger{session: sess, lock: newLock()} }

// failedWith reports whether err is the error a write to the history failed
// with: the session ended because a history write got no answer, so what the
// history now holds is uncertain.
func (l *ledger) failedWith(err error) bool {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.writeErr != nil && errors.Is(err, l.writeErr)
}

// append writes items to the end of the history, in order.
func (l *ledger) append(ctx context.Context, items func() ([]historyItem, error)) error {
	if err := l.lock.acquire(ctx); err != nil {
		return err
	}
	defer l.lock.release()
	writes := 0
	list, err := items()
	if err == nil {
		for _, item := range list {
			if _, err = l.session.createItem(ctx, item); err != nil {
				break
			}
			writes++
		}
	}
	if err == nil {
		return nil
	}
	// Recorded while the lock is still held, so the event loop, seeing the
	// session end, finds either the lock held or this error: there is no moment
	// at which the write has failed and neither says so.
	l.mu.Lock()
	l.writeErr = err
	l.mu.Unlock()
	status := historyUncertain
	if knownOutcome(err) {
		status = historyRejected
		if writes > 0 {
			status = historyPartial
		}
	}
	return &HistoryUpdateError{Status: status, Err: err}
}

// knownOutcome reports whether a failed write is known to have written
// nothing: it was refused before it was sent, or Prism refused it.
func knownOutcome(err error) bool {
	if _, ok := errors.AsType[*SettingsError](err); ok {
		return true
	}
	if failure, ok := errors.AsType[*Error](err); ok {
		return failure.Kind == KindInput
	}
	if refusal, ok := errors.AsType[*FaultError](err); ok {
		return refusal.Fault.Scope == scopeEvent
	}
	return false
}

// discardReceipt is a reply whose receipt is still to be sent.
type discardReceipt struct {
	ref   ResponseRef
	hold  *responseHold
	ended bool
}

// discardPlayback reports the replies of a conversation that discards all its
// audio as played for no time at all, in the order they were created.
type discardPlayback struct {
	session *session
	// pending and order are guarded by the lock of the service.
	pending map[string]*discardReceipt
	order   []string
	lock    lock
}

func newDiscardPlayback(sess *session) *discardPlayback {
	return &discardPlayback{session: sess, pending: map[string]*discardReceipt{}, lock: newLock()}
}

// start keeps a reply until its receipt can follow those of all the earlier
// output.
func (d *discardPlayback) start(ref ResponseRef) error {
	if len(d.pending) >= retentionLimit {
		return &failureError{message: "Prism pending discard receipt limit exceeded"}
	}
	hold, err := d.session.holdResponse(ref)
	if err != nil {
		return err
	}
	d.pending[ref.ResponseID] = &discardReceipt{ref: ref, hold: hold}
	d.order = append(d.order, ref.ResponseID)
	return nil
}

// end records that a reply completed, before another reply can be admitted.
func (d *discardPlayback) end(ref ResponseRef) {
	if entry, ok := d.pending[ref.ResponseID]; ok {
		entry.ended = true
	}
}

// close lets go of the references still pending, without claiming their
// playback completed.
func (d *discardPlayback) close() {
	for _, entry := range d.pending {
		entry.hold.release()
	}
	d.pending, d.order = map[string]*discardReceipt{}, nil
}
