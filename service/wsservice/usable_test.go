package wsservice_test

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/service/wsservice"
	"github.com/gojargo/jargo/service/wsutil"
	errs "github.com/gojargo/jargo/utils/errors"
)

// usableService is a processor holding a websocket whose dials succeed or fail
// on demand, reporting errors through its own PushErrorFrame, as a service does.
type usableService struct {
	*processor.Base
	connectErr error

	mu        sync.Mutex
	attempts  int
	connected bool
	sent      int
	reported  []*frames.ErrorFrame
	permanent []bool
}

func newUsableService(connectErr error) *usableService {
	s := &usableService{connectErr: connectErr}
	s.Base = processor.New("UsableService", s)
	return s
}

func (s *usableService) ConnectWebsocket(context.Context) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.attempts++
	if s.connectErr != nil {
		return s.connectErr
	}
	s.connected = true
	return nil
}

func (s *usableService) DisconnectWebsocket(context.Context) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.connected = false
	return nil
}

// ReceiveMessages returns at once: the peer closed the connection.
func (s *usableService) ReceiveMessages(context.Context) error { return nil }

func (s *usableService) Connected() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.connected
}

func (s *usableService) report(ctx context.Context, ef *frames.ErrorFrame, forceTreatAsPermanent bool) {
	s.mu.Lock()
	s.reported = append(s.reported, ef)
	s.permanent = append(s.permanent, forceTreatAsPermanent)
	s.mu.Unlock()
	s.PushErrorFrame(ctx, ef, forceTreatAsPermanent)
}

func fastBase(h wsservice.Handler, reconnectOnError bool, maxRetries int) *wsservice.Base {
	return wsservice.New(h, wsservice.Config{
		ReconnectOnError:        &reconnectOnError,
		MaxRetries:              maxRetries,
		ReconnectBackoffMinWait: time.Millisecond,
		ReconnectBackoffMaxWait: time.Millisecond,
	})
}

var errRefused = errors.New("refused")

// TestReportedErrorsCarryTheException checks each reported attempt carries the
// error behind it, so the refusal is categorized from it.
func TestReportedErrorsCarryTheException(t *testing.T) {
	refusal := &wsutil.HandshakeError{StatusCode: 401, Err: errRefused}
	s := newUsableService(refusal)
	fastBase(s, true, 3).TryReconnect(t.Context(), s.report)
	if len(s.reported) == 0 {
		t.Fatal("nothing reported")
	}
	if !errors.Is(s.reported[0].Err, refusal) {
		t.Errorf("reported error = %v, want the handshake refusal", s.reported[0].Err)
	}
	if s.reported[0].Category != errs.Authentication {
		t.Errorf("category = %q, want authentication", s.reported[0].Category)
	}
}

// TestAFailedAttemptAloneLeavesTheServiceUsable checks a transient refusal is
// reported as a server error, which does not cost the service its usability;
// only running out of attempts does.
func TestAFailedAttemptAloneLeavesTheServiceUsable(t *testing.T) {
	s := newUsableService(&wsutil.HandshakeError{StatusCode: 503, Err: errRefused})
	fastBase(s, true, 2).TryReconnect(t.Context(), s.report)
	if len(s.reported) != 3 {
		t.Fatalf("reported %d errors, want one per attempt and the give-up", len(s.reported))
	}
	if s.reported[0].Category != errs.Server {
		t.Errorf("category = %q, want server", s.reported[0].Category)
	}
	if s.permanent[0] {
		t.Error("a single failed attempt was reported as permanent")
	}
	if s.Usable() {
		t.Error("the service is usable after running out of attempts")
	}
}

// TestADroppedConnectionLeavesTheReceiveLoopAlone checks that with reconnecting
// off, a drop is reported once, not as permanent, and nothing is redialed.
func TestADroppedConnectionLeavesTheReceiveLoopAlone(t *testing.T) {
	s := newUsableService(nil)
	fastBase(s, false, 3).ReceiveTaskHandler(t.Context(), s.report)
	if !s.Usable() {
		t.Error("the service is unusable after a drop with reconnecting off")
	}
	if s.attempts != 0 {
		t.Errorf("dials = %d, want none", s.attempts)
	}
	if len(s.permanent) != 1 || s.permanent[0] {
		t.Errorf("reports = %v, want one that is not permanent", s.permanent)
	}
}

// TestADroppedConnectionStillReconnectsOnDemand checks that with reconnecting
// off, a send after a drop still reconnects and goes out.
func TestADroppedConnectionStillReconnectsOnDemand(t *testing.T) {
	s := newUsableService(nil)
	b := fastBase(s, false, 3)
	b.ReceiveTaskHandler(t.Context(), s.report)
	//nolint:staticcheck // the deprecated send is what reconnects on demand
	err := b.SendWithRetry(t.Context(), func() error {
		s.mu.Lock()
		s.sent++
		s.mu.Unlock()
		return nil
	}, s.report)
	if err != nil {
		t.Fatalf("send: %v", err)
	}
	if s.attempts != 1 || s.sent != 1 {
		t.Errorf("dials = %d, sent = %d, want 1 and 1", s.attempts, s.sent)
	}
}
