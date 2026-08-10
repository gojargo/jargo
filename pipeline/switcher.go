package pipeline

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"sync"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
)

// errNoServices is returned when a switcher is built without a service.
//
//nolint:gochecknoglobals // sentinel error
var errNoServices = errors.New("pipeline: service switcher needs at least one service")

// inactiveUpdateRing is how many settings updates handed to inactive services
// are remembered so they can be consumed on the way out. An update crosses its
// service long before the ring wraps.
const inactiveUpdateRing = 64

// SwitcherStrategy decides which of a switcher's services is active and when
// that changes. Build one with a StrategyFunc.
//
// Two are supplied: NewManualStrategy switches only when asked, and
// NewFailoverStrategy additionally moves on when the active service reports an
// error. Implement the interface for anything else.
type SwitcherStrategy interface {
	// Services are the services the strategy chooses between.
	Services() []processor.Processor
	// ActiveService is the service currently in use.
	ActiveService() processor.Processor
	// HandleFrame acts on a frame that steers the switcher, returning the
	// service it switched to, or nil if it did not switch. A frame it does not
	// act on travels on, so another switcher can have its turn.
	HandleFrame(f frames.Frame, dir processor.Direction) processor.Processor
	// HandleError reacts to a non-fatal error from the active service,
	// returning the service it switched to, or nil if it did not switch.
	HandleError(ef *frames.ErrorFrame) processor.Processor
	// OnServiceSwitched registers fn to be called whenever the active service
	// changes.
	OnServiceSwitched(fn func(processor.Processor))
}

// StrategyFunc builds a strategy over a set of services. A switcher is given the
// constructor rather than a built strategy, so the strategy is always paired
// with the services the switcher actually manages.
type StrategyFunc func(services []processor.Processor) SwitcherStrategy

// ServiceSwitcher routes the pipeline through one of several interchangeable
// services at a time. Every service is started and kept warm, but only the
// active one receives data; the rest are gated off.
//
// It is a ParallelPipeline: each service becomes a branch wrapped in a pair of
// filters that pass lifecycle frames (so every service stays ready) but gate
// everything else on whether that service is active. What leaves the switcher is
// decided on the way out, which is where the copies the gating produces are
// dropped so the rest of the pipeline sees one service, not several.
//
// A settings update is the exception to the gating. One addressed to a member
// service reaches it whether or not it is in use, and one marked
// ReachInactiveServices reaches every member, so whichever service becomes
// active later is already configured.
type ServiceSwitcher struct {
	*ParallelPipeline
	strategy SwitcherStrategy

	mu sync.Mutex
	// inactiveUpdates are the ids of the settings updates handed to services
	// that were not active, so they can be consumed again on their way out.
	inactiveUpdates []uint64
}

// NewServiceSwitcher builds a switcher over services, the first of which starts
// active, choosing between them with the strategy newStrategy builds. A nil
// newStrategy switches only when asked.
func NewServiceSwitcher(
	services []processor.Processor, newStrategy StrategyFunc,
) (*ServiceSwitcher, error) {
	return newServiceSwitcherAs(nil, "ServiceSwitcher", services, newStrategy)
}

// newServiceSwitcherAs builds a switcher on behalf of self, the processor
// embedding it, under the given name. A nil self means the switcher is the
// outermost processor and stands for itself.
func newServiceSwitcherAs(
	self processor.Processor,
	name string,
	services []processor.Processor,
	newStrategy StrategyFunc,
) (*ServiceSwitcher, error) {
	if len(services) == 0 {
		return nil, errNoServices
	}
	if newStrategy == nil {
		newStrategy = NewManualStrategy
	}

	s := &ServiceSwitcher{strategy: newStrategy(services)}
	if self == nil {
		self = s
	}

	// The filters decide system frames too, so a branch that is gated off stops
	// hearing the conversation rather than following it in the background. The
	// lifecycle frames still reach it, so it starts and shuts down with the rest.
	down, up := processor.Downstream, processor.Upstream
	branches := make([][]processor.Processor, len(services))
	for i, svc := range services {
		branches[i] = []processor.Processor{
			processor.NewFunctionFilter(fmt.Sprintf("Switch::In%d", i), &down, s.gate(svc),
				processor.WithFilterSystemFrames()),
			svc,
			processor.NewFunctionFilter(fmt.Sprintf("Switch::Out%d", i), &up, s.gate(svc),
				processor.WithFilterSystemFrames()),
		}
	}

	// The parallel pipeline is built on the switcher's behalf, so the frames its
	// branches produce leave through the switcher's own PushFrame.
	pp, err := newParallelAs(self, name, branches...)
	if err != nil {
		return nil, err
	}
	s.ParallelPipeline = pp
	return s, nil
}

// Strategy is the strategy choosing between the switcher's services.
func (s *ServiceSwitcher) Strategy() SwitcherStrategy { return s.strategy }

// Services are the services the switcher manages.
func (s *ServiceSwitcher) Services() []processor.Processor { return s.strategy.Services() }

// ActiveService is the service currently in use.
func (s *ServiceSwitcher) ActiveService() processor.Processor { return s.strategy.ActiveService() }

// SwitchTo makes svc the active service, reporting false if svc is not one of
// the switcher's services.
func (s *ServiceSwitcher) SwitchTo(svc processor.Processor) bool {
	return s.switchTo(svc) != nil
}

// OnSwitch registers fn to be called whenever the active service changes.
func (s *ServiceSwitcher) OnSwitch(fn func(processor.Processor)) {
	s.strategy.OnServiceSwitched(fn)
}

// switchTo asks the strategy to activate svc and, when it does, asks the new
// service to broadcast its metadata, so what the rest of the pipeline knows
// describes the service now in use.
func (s *ServiceSwitcher) switchTo(svc processor.Processor) processor.Processor {
	switched := s.strategy.HandleFrame(frames.NewManuallySwitchServiceFrame(svc), processor.Downstream)
	if switched != nil {
		s.requestMetadata(switched)
	}
	return switched
}

// requestMetadata asks svc to broadcast its metadata again. It is queued
// straight onto the service, past the filters, since the switch has only just
// happened.
func (s *ServiceSwitcher) requestMetadata(svc processor.Processor) {
	req := frames.NewServiceSwitcherRequestMetadataFrame(frames.ServiceTarget(svc))
	_ = svc.QueueFrame(context.Background(), req, processor.Downstream)
}

// ProcessFrame steers the switcher on the frames that ask it to, and hands every
// settings update to the member services the gating would otherwise keep it from.
func (s *ServiceSwitcher) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if _, isSwitcher := f.(frames.SwitcherFrame); isSwitcher {
		if switched := s.strategy.HandleFrame(f, dir); switched != nil {
			s.requestMetadata(switched)
			// The request was ours to answer, so it stops here.
			return nil
		}
		// Not one of ours. Pass it on for the switcher that manages it.
	}

	if err := s.ParallelPipeline.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}

	if u, ok := f.(frames.SettingsUpdate); ok {
		s.updateInactiveServices(ctx, u, dir)
	}
	return nil
}

// PushFrame decides what leaves the switcher.
//
// The branches are copies of one another as far as the rest of the pipeline is
// concerned, so what escapes has to be the active service's alone: its metadata,
// its settings update, its errors. It is also where a non-fatal error from the
// active service reaches the strategy, which is what drives failover.
func (s *ServiceSwitcher) PushFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	switch fr := f.(type) {
	case *frames.ServiceSwitcherRequestMetadataFrame:
		// The service it was aimed at has answered it, so it goes no further.
		if fr.Service == frames.ServiceTarget(s.ActiveService()) {
			return nil
		}
	case frames.ServiceMetadata:
		// Only the active service describes the switcher. Every service
		// broadcasts at startup, since the lifecycle frames reach them all.
		if fr.Service() != s.ActiveService().Name() {
			return nil
		}
	case frames.SettingsUpdate:
		// The copies handed to the inactive services have been delivered; the
		// one the rest of the pipeline sees is the active service's.
		if s.consumeInactiveUpdate(fr.ID()) {
			return nil
		}
	case frames.ErrorReport:
		// Let the strategy react to the active service failing, while the error
		// still travels on so the application hears about it too.
		ef := fr.ErrorInfo()
		if !ef.Fatal && ef.Source != nil && ef.Source.Name() == s.ActiveService().Name() {
			if switched := s.strategy.HandleError(ef); switched != nil {
				s.requestMetadata(switched)
			}
		}
	}
	return s.ParallelPipeline.Base.PushFrame(ctx, f, dir)
}

// updateInactiveServices hands a settings update to the member services the
// gating keeps it from, when it is one they should have.
//
// The active service receives the update through its branch like any other
// frame. Each inactive one is handed a copy of its own, so the copies can be
// told apart from the original and dropped on the way out.
func (s *ServiceSwitcher) updateInactiveServices(
	ctx context.Context, u frames.SettingsUpdate, dir processor.Direction,
) {
	for _, svc := range s.inactiveUpdateTargets(u) {
		copied := u.Copy()
		s.rememberInactiveUpdate(copied.ID())
		_ = svc.QueueFrame(ctx, copied, dir)
	}
}

// inactiveUpdateTargets are the inactive member services that should apply an
// update, which may be none.
func (s *ServiceSwitcher) inactiveUpdateTargets(u frames.SettingsUpdate) []processor.Processor {
	active := s.ActiveService()
	var inactive []processor.Processor
	for _, svc := range s.Services() {
		if svc != active {
			inactive = append(inactive, svc)
		}
	}

	update := u.ServiceUpdate()
	if update.Service != nil {
		// An addressed update goes to its service alone, active or not. One
		// addressed elsewhere is left for the switcher that manages it.
		for _, svc := range inactive {
			if update.Service == frames.ServiceTarget(svc) {
				return []processor.Processor{svc}
			}
		}
		return nil
	}
	// Any other update crosses to the inactive services only if it opts in,
	// since a setting is usually specific to one provider.
	if update.ReachInactiveServices {
		return inactive
	}
	return nil
}

func (s *ServiceSwitcher) rememberInactiveUpdate(id uint64) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if len(s.inactiveUpdates) == inactiveUpdateRing {
		s.inactiveUpdates = s.inactiveUpdates[1:]
	}
	s.inactiveUpdates = append(s.inactiveUpdates, id)
}

// consumeInactiveUpdate reports whether id is one of the copies handed to an
// inactive service, removing it when it is.
func (s *ServiceSwitcher) consumeInactiveUpdate(id uint64) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	for i, seen := range s.inactiveUpdates {
		if seen == id {
			s.inactiveUpdates = append(s.inactiveUpdates[:i], s.inactiveUpdates[i+1:]...)
			return true
		}
	}
	return false
}

// gate returns the filter predicate for svc: frames pass only while svc is the
// active service. The lifecycle frames are not decided here; the filter passes
// those whatever this says, so every service starts and shuts down with the
// pipeline and stays ready to take over.
//
// A settings update meant for a service that is not in use is not let through
// here. It is handed to that service directly instead, as a copy of its own, so
// only one of them travels on through the pipeline.
func (s *ServiceSwitcher) gate(svc processor.Processor) processor.FilterFunc {
	return func(frames.Frame) bool { return s.ActiveService() == svc }
}

// manualStrategy switches only when asked.
type manualStrategy struct {
	mu       sync.Mutex
	services []processor.Processor
	active   processor.Processor
	onSwitch func(processor.Processor)
}

// NewManualStrategy builds a strategy that changes the active service only on an
// explicit request. The first service starts active.
func NewManualStrategy(services []processor.Processor) SwitcherStrategy {
	return &manualStrategy{services: services, active: services[0]}
}

func (m *manualStrategy) Services() []processor.Processor { return m.services }

func (m *manualStrategy) ActiveService() processor.Processor {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.active
}

func (m *manualStrategy) OnServiceSwitched(fn func(processor.Processor)) {
	m.mu.Lock()
	m.onSwitch = fn
	m.mu.Unlock()
}

// HandleFrame activates the service a manual switch request names.
func (m *manualStrategy) HandleFrame(f frames.Frame, _ processor.Direction) processor.Processor {
	sf, ok := f.(*frames.ManuallySwitchServiceFrame)
	if !ok {
		return nil
	}
	return m.setActiveIfAvailable(sf.Service)
}

// HandleError does nothing: switching is manual.
func (m *manualStrategy) HandleError(*frames.ErrorFrame) processor.Processor { return nil }

// setActiveIfAvailable activates target if it is one of the strategy's services.
// A target it does not manage is left alone, since the request was meant for
// another switcher.
func (m *manualStrategy) setActiveIfAvailable(target frames.ServiceTarget) processor.Processor {
	m.mu.Lock()
	var found processor.Processor
	for _, svc := range m.services {
		if frames.ServiceTarget(svc) == target {
			found = svc
			break
		}
	}
	if found == nil {
		m.mu.Unlock()
		return nil
	}
	m.active = found
	cb := m.onSwitch
	m.mu.Unlock()

	if cb != nil {
		cb(found)
	}
	return found
}

// failoverStrategy switches when asked, and moves on when the active service
// reports an error.
type failoverStrategy struct {
	*manualStrategy
}

// NewFailoverStrategy builds a strategy that switches to the next service when
// the active one reports a non-fatal error, and can still be switched by hand.
// The failed service stays in the list and can be switched back to.
func NewFailoverStrategy(services []processor.Processor) SwitcherStrategy {
	return &failoverStrategy{manualStrategy: &manualStrategy{services: services, active: services[0]}}
}

// HandleError moves to the next service in the list.
func (f *failoverStrategy) HandleError(ef *frames.ErrorFrame) processor.Processor {
	f.mu.Lock()
	if len(f.services) <= 1 {
		f.mu.Unlock()
		slog.Error("no other service to switch to", "err", ef.Error)
		return nil
	}
	idx := 0
	for i, svc := range f.services {
		if svc == f.active {
			idx = i
			break
		}
	}
	next := f.services[(idx+1)%len(f.services)]
	f.mu.Unlock()

	slog.Warn("service reported an error, switching", "service", ef.Source.Name(), "err", ef.Error)
	return f.setActiveIfAvailable(frames.ServiceTarget(next))
}
