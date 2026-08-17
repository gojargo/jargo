package pipeline

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"sync"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
	errs "github.com/gojargo/jargo/utils/errors"
	"github.com/gojargo/jargo/utils/events"
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
	// UsableServices are the services that can still be given work, in order.
	UsableServices() []processor.Processor
	// ActiveService is the service currently in use.
	ActiveService() processor.Processor
	// HandleFrame acts on a frame that steers the switcher, returning the
	// service it switched to, or nil if it did not switch. A frame it does not
	// act on travels on, so another switcher can have its turn.
	HandleFrame(f frames.Frame, dir processor.Direction) processor.Processor
	// HandleError reacts to a non-fatal error from the active service,
	// returning the service it switched to, or nil if it did not switch.
	//
	// It is called for every non-fatal error the active service reports,
	// whether or not the service can carry on from it, so the strategy decides
	// which errors are worth switching away from. ErrorFrame.Source.Usable()
	// tells a service that is finished from one having a bad moment.
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
// A switcher is only as dead as its last service, so its own usability is a
// reading of theirs rather than a state of its own.
type ServiceSwitcher struct {
	*ParallelPipeline
	strategy SwitcherStrategy

	mu sync.Mutex
	// inactiveUpdates are the ids of the settings updates handed to services
	// that were not active, so they can be consumed again on their way out.
	inactiveUpdates []uint64
	// announcedUsable is the last reading announced, kept because a service
	// changing its own usability does not always change the switcher's.
	announcedUsable bool
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

	// What the switcher reports is read from the services, so it hears about
	// each of them changing and announces its own reading when that moves.
	s.announcedUsable = s.Usable()
	for _, svc := range services {
		events.On(svc.Events(), processor.EventUsableChanged,
			func(ctx context.Context, _ bool) { s.announceUsable(ctx) })
	}
	return s, nil
}

// Usable reports whether any of the switched services can still be given work.
//
// A switcher is only as dead as its last service: it can keep doing its job by
// moving work to a different one, so it reports itself unusable only once none
// of them can do theirs. This is a reading of the services rather than a state
// of its own, so bringing a service back with SetUsable brings the switcher back
// too, and calling that on the switcher itself does nothing. The switcher raises
// EventUsableChanged for itself as the reading moves, so watching it is enough
// to hear about the services it holds.
func (s *ServiceSwitcher) Usable() bool {
	for _, svc := range s.Services() {
		if svc.Usable() {
			return true
		}
	}
	return false
}

// SetUsable ignores an attempt to set this switcher's usability directly.
//
// A switcher has no usability of its own to set: what it reports is a reading of
// its services. Bring an individual service back with its own SetUsable, which
// brings the switcher back with it.
func (s *ServiceSwitcher) SetUsable(ctx context.Context, usable bool) {
	// Not calling through to the base: it would set a flag this switcher never
	// reads, and announce a change that did not happen.
	slog.DebugContext(ctx, "ignoring set_usable; a switcher reports whether any of its services can be given work",
		"switcher", s.Name(), "usable", usable)
}

// announceUsable raises EventUsableChanged when what this switcher reports has
// moved.
func (s *ServiceSwitcher) announceUsable(ctx context.Context) {
	usable := s.Usable()
	s.mu.Lock()
	if usable == s.announcedUsable {
		s.mu.Unlock()
		return
	}
	s.announcedUsable = usable
	s.mu.Unlock()

	if usable {
		slog.DebugContext(ctx, "switcher usable", "switcher", s.Name())
	} else {
		slog.DebugContext(ctx, "switcher no longer usable", "switcher", s.Name())
	}
	s.Events().Call(ctx, processor.EventUsableChanged, s, usable)
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
		if handled := s.handleServiceError(ctx, fr.ErrorInfo()); handled {
			return nil
		}
	}
	return s.ParallelPipeline.Base.PushFrame(ctx, f, dir)
}

// handleServiceError answers for an error one of the switched services reported,
// returning whether it has been dealt with here and goes no further.
//
// The rest of the pipeline deals with the switcher and not with what it holds,
// so an error from a service the switcher is not using stops here outright: it
// is not being given work, and nothing about it bears on whether the switcher
// can do its job. Every error from the active service goes to the strategy,
// which decides whether it warrants a switch; a successful switch absorbs it,
// the switcher having gone on doing its job. An error that leaves the active
// service unable to do its job with no switch to be made is reported against the
// switcher, so the pipeline judges it by what the switcher has left rather than
// by the one service that failed. Every other error travels upstream as usual.
func (s *ServiceSwitcher) handleServiceError(ctx context.Context, ef *frames.ErrorFrame) bool {
	if ef.Fatal {
		return false
	}
	failed, ok := ef.Source.(processor.Processor)
	if !ok {
		return false
	}
	active := s.ActiveService()
	if failed != active {
		// Errors from anywhere else are just passing through, and travel on.
		// Watch a service's own EventUsableChanged to hear about the ones held
		// in reserve.
		return s.isMember(failed)
	}

	if switched := s.strategy.HandleError(ef); switched != nil {
		s.requestMetadata(switched)
		return true
	}
	// No switch was made. If the service can no longer work, the switcher has
	// nowhere left to send the work, so it reports the failure as its own.
	if !failed.Usable() {
		s.reportServiceFailure(ctx, failed, ef)
		return true
	}
	return false
}

// isMember reports whether p is one of the services this switcher manages.
func (s *ServiceSwitcher) isMember(p processor.Processor) bool {
	for _, svc := range s.Services() {
		if svc == p {
			return true
		}
	}
	return false
}

// reportServiceFailure reports a service failure the switcher could not switch
// away from.
//
// It re-attributes the error to the switcher, which is the processor the rest of
// the pipeline deals with. Whether losing this service matters depends on what
// the switcher has left, not on the service that just failed, so Usable on the
// reported error answers for the switcher as a whole.
func (s *ServiceSwitcher) reportServiceFailure(
	ctx context.Context, failed processor.Processor, ef *frames.ErrorFrame,
) {
	s.PushError(ctx,
		fmt.Sprintf("%s can no longer do its job: %s", failed.Name(), ef.Error),
		ef.Err, false,
		// A permanent category would cost the switcher its own usability,
		// writing it off along with the one service that failed.
		processor.WithErrorCategory(errs.Unknown))
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

// UsableServices are the services that can still be given work, in order.
func (m *manualStrategy) UsableServices() []processor.Processor {
	var usable []processor.Processor
	for _, svc := range m.services {
		if svc.Usable() {
			usable = append(usable, svc)
		}
	}
	return usable
}

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
// another switcher. A service that can no longer do its job is refused: making
// it active would only route work to something that cannot handle it. Bring it
// back with its own SetUsable first, once whatever stopped it working has been
// dealt with.
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
	if !found.Usable() {
		m.mu.Unlock()
		slog.Warn("not switching to a service that can no longer do its job",
			"service", found.Name())
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

// HandleError moves to the next service that can still do its job.
//
// Only an error leaving the service unable to do its job is worth a failover;
// anything it can carry on from is ignored, so a provider hiccup does not cost
// one. When one does, it switches to the next service in the list that can still
// do its own, wrapping around from the end. The failed service stays in the list
// and can be switched back to once it has been brought back with SetUsable.
func (f *failoverStrategy) HandleError(ef *frames.ErrorFrame) processor.Processor {
	failed, ok := ef.Source.(processor.Processor)
	if !ok {
		failed = f.ActiveService()
	}
	// The service is still working, so there is nothing to fail over from.
	if failed.Usable() {
		return nil
	}

	slog.Warn("service reported an error, switching", "service", failed.Name(), "err", ef.Error)

	// Walk the list from the one after the active service, so failover follows
	// the order the services were given in.
	f.mu.Lock()
	idx := 0
	for i, svc := range f.services {
		if svc == f.active {
			idx = i
			break
		}
	}
	services := f.services
	f.mu.Unlock()

	for offset := 1; offset < len(services); offset++ {
		candidate := services[(idx+offset)%len(services)]
		if candidate.Usable() {
			return f.setActiveIfAvailable(frames.ServiceTarget(candidate))
		}
	}
	slog.Error("no other service available to switch to", "err", ef.Error)
	return nil
}
