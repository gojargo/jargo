package llmworker

import (
	"context"
	"errors"
	"fmt"
	"io"
	"maps"
	"sync"
	"time"

	"github.com/gojargo/jargo/pipeline/jobcontext"
)

// errNotOpen is reported by Next on a session that was never opened.
var errNotOpen = errors.New("the backend session is not open")

// detachReason is what the attach job is called off for when a frontend lets go
// of its backend.
const detachReason = "frontend detached"

// BackendSession is a frontend's side of the attached contract: one backend
// worker, attached for the session's life.
//
// Opening the session sends the attach job; closing it cancels the job, which
// stops the backend's work. In between, Next yields what the backend produces,
// and Send puts messages to it. The stream ends when the backend goes away, with
// an error if it failed.
//
//	session, err := llmworker.OpenBackendSession(ctx, worker, "backend", 30*time.Second)
//	if err != nil { … }
//	defer session.Close(ctx)
//	status, err := session.Send(ctx, request)
//	for {
//	    event, err := session.Next(ctx)
//	    if err != nil { break }
//	    …
//	}
type BackendSession struct {
	requester   jobcontext.Requester
	backendName string
	timeout     time.Duration

	mu    sync.Mutex
	group *jobcontext.Group
	// first is the update Open took off the stream while waiting for the
	// attachment to be taken up, which Next hands over before reading on.
	first        *jobcontext.GroupEvent
	capabilities map[string]bool
	detached     bool
}

// NewBackendSession builds a session with the named backend worker, not yet
// attached: Open attaches it. Send works on a session that is not attached, and
// is refused by a backend with no frontend attached.
//
// requester is the worker attaching (for a pipeline processor, the pipeline
// worker it runs in). timeout bounds how long a message may take to be
// acknowledged, including the wait for the backend to become ready; zero waits
// as long as it takes. The attachment itself has no timeout.
func NewBackendSession(requester jobcontext.Requester, backendName string, timeout time.Duration) *BackendSession {
	return &BackendSession{requester: requester, backendName: backendName, timeout: timeout}
}

// OpenBackendSession builds a session with the named backend worker and opens
// it. See NewBackendSession and Open.
func OpenBackendSession(
	ctx context.Context, requester jobcontext.Requester, backendName string, timeout time.Duration,
) (*BackendSession, error) {
	s := NewBackendSession(requester, backendName, timeout)
	if err := s.Open(ctx); err != nil {
		return nil, err
	}
	return s, nil
}

// Open attaches the session: it sends the attach job, whose stream Next reads.
//
// It returns once the backend has taken the attachment up, with the first
// update of its stream or by refusing it, so a message sent next finds it
// attached: every job is handled on a goroutine of its own, and one sent right
// behind the attachment could otherwise reach the backend first. The wait for
// that is bounded by the session's timeout, and a refusal is reported by Next.
func (s *BackendSession) Open(ctx context.Context) error {
	group, err := s.requester.CreateGroupAndRequestJob(ctx, []string{s.backendName}, jobcontext.Request{
		Name: AttachJobName,
	})
	if err != nil {
		return err
	}
	waitCtx := ctx
	if s.timeout > 0 {
		var cancel context.CancelFunc
		waitCtx, cancel = context.WithTimeout(ctx, s.timeout)
		defer cancel()
	}
	first, ok := group.NextEvent(waitCtx)

	s.mu.Lock()
	s.group = group
	if ok {
		s.first = &first
	}
	s.mu.Unlock()
	return nil
}

// attachment is the attach job's group, or nil before Open.
func (s *BackendSession) attachment() *jobcontext.Group {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.group
}

// BackendName is the name of the backend worker.
func (s *BackendSession) BackendName() string { return s.backendName }

// Capabilities is what the backend said it can do, once its stream has opened.
func (s *BackendSession) Capabilities() map[string]bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := make(map[string]bool, len(s.capabilities))
	maps.Copy(out, s.capabilities)
	return out
}

// Detached reports whether the stream ended because this side let go of the
// backend (the session was closed, or the attaching worker stopped), as opposed
// to the backend going away.
func (s *BackendSession) Detached() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.detached
}

// Close detaches: it cancels the attach job, which stops the backend's work.
// The cancellation reaches the backend even when ctx has already ended, since a
// backend left working for a frontend that is gone is worse than one stopped
// late.
func (s *BackendSession) Close(ctx context.Context) {
	group := s.attachment()
	if group != nil && s.requester.HasGroup(group.JobID) {
		s.requester.CancelGroup(context.WithoutCancel(ctx), group.JobID, detachReason)
	}
}

// Next returns the next thing the backend produced, waiting for it.
//
// It returns io.EOF once the stream has ended without a failure: the session
// was closed (Detached then reports true) or the backend went away. It returns
// an error wrapping jobcontext.ErrJob, naming the backend's reason where it gave
// one, when the backend refused the attachment or failed, and ctx's error when
// ctx ends first. Updates of a type the session does not know are skipped.
func (s *BackendSession) Next(ctx context.Context) (BackendEvent, error) {
	group := s.attachment()
	if group == nil {
		return nil, errNotOpen
	}
	for {
		e, ok := s.nextEvent(ctx, group)
		if !ok {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
			return nil, s.ended(ctx, group)
		}
		if e.Type != jobcontext.EventUpdate || len(e.Data) == 0 {
			continue
		}
		if payloadString(e.Data, payloadType) == AttachedUpdateType {
			s.mu.Lock()
			s.capabilities = capabilitiesOf(e.Data["capabilities"])
			s.mu.Unlock()
			continue
		}
		if event := eventFromPayload(e.Data); event != nil {
			return event, nil
		}
	}
}

// nextEvent is the next update of the stream, starting with the one Open kept.
func (s *BackendSession) nextEvent(ctx context.Context, group *jobcontext.Group) (jobcontext.GroupEvent, bool) {
	s.mu.Lock()
	first := s.first
	s.first = nil
	s.mu.Unlock()
	if first != nil {
		return *first, true
	}
	return group.NextEvent(ctx)
}

// ended says how the stream ended, once the group has finished.
func (s *BackendSession) ended(ctx context.Context, group *jobcontext.Group) error {
	err := group.Wait(ctx)
	if err == nil {
		return io.EOF
	}
	response, answered := group.Response(s.backendName)
	if !answered {
		// Called off from this side: the session was closed, or the requester's
		// worker stopped.
		s.mu.Lock()
		s.detached = true
		s.mu.Unlock()
		return io.EOF
	}
	return withReason(fmt.Errorf("%w: %w", jobcontext.ErrJob, err), response)
}

// Send puts a message to the backend, the text appended to its conversation,
// and returns what the backend was doing when it arrived: "idle" or "working".
// It reports an error wrapping jobcontext.ErrJob, naming the backend's reason,
// when the backend refused the message or did not answer in time.
func (s *BackendSession) Send(ctx context.Context, request string) (string, error) {
	response, err := s.ask(ctx, jobcontext.Request{
		Name:    MessageJobName,
		Payload: map[string]any{"request": request},
		Timeout: s.timeout,
	})
	if err != nil {
		return "", err
	}
	status := payloadString(response, "backend")
	if status == "" {
		status = "idle"
	}
	return status, nil
}

// ask runs one short job on the backend and returns its response.
func (s *BackendSession) ask(ctx context.Context, req jobcontext.Request) (map[string]any, error) {
	job, err := jobcontext.RunJob(ctx, s.requester, s.backendName, req, nil)
	if err != nil {
		var response map[string]any
		if job != nil {
			response = job.Response()
		}
		if !errors.Is(err, jobcontext.ErrJob) {
			err = fmt.Errorf("%w: %w", jobcontext.ErrJob, err)
		}
		return nil, withReason(err, response)
	}
	return job.Response(), nil
}

// eventFromPayload turns an attach stream update into the event it carries, if
// it carries one. An update with no type is an output, as the first backends
// sent them.
func eventFromPayload(payload map[string]any) BackendEvent {
	updateType := payloadString(payload, payloadType)
	if updateType == "" {
		updateType = OutputUpdateType
	}
	switch updateType {
	case OutputUpdateType:
		return BackendOutputFromPayload(payload)
	case ToolCallUpdateType:
		return BackendToolCallFromPayload(payload)
	case IdleUpdateType:
		return BackendIdle{}
	case ErrorUpdateType:
		return BackendError{Error: payloadString(payload, payloadError)}
	default:
		return nil
	}
}

// capabilitiesOf reads the capabilities an attached update carries, as a
// backend in this process sends them or as they arrive from another.
func capabilitiesOf(v any) map[string]bool {
	out := map[string]bool{}
	switch caps := v.(type) {
	case map[string]bool:
		maps.Copy(out, caps)
	case map[string]any:
		for k, on := range caps {
			out[k] = truthy(on)
		}
	}
	return out
}

// withReason is the job error with the backend's own reason for it, when its
// response gave one.
func withReason(err error, response map[string]any) error {
	if reason := payloadString(response, payloadError); reason != "" {
		return fmt.Errorf("%w: %s", err, reason)
	}
	return err
}
