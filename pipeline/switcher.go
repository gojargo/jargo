package pipeline

import (
	"context"
	"errors"
	"fmt"
	"sync"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
)

// errNoServices is returned by NewServiceSwitcher when no service is supplied.
//
//nolint:gochecknoglobals // sentinel error
var errNoServices = errors.New("pipeline: service switcher needs at least one service")

// SwitcherStrategy selects how a ServiceSwitcher changes its active service.
type SwitcherStrategy int

const (
	// SwitchManual changes the active service only on an explicit request.
	SwitchManual SwitcherStrategy = iota
	// SwitchFailover additionally moves to the next service when the active one
	// reports a non-fatal error.
	SwitchFailover
)

// SwitchServiceFrame requests that a ServiceSwitcher make Service active. Queue
// it downstream into the pipeline, or call ServiceSwitcher.SwitchTo directly.
type SwitchServiceFrame struct {
	frames.BaseControlFrame
	// Service is the service to activate; it must belong to the switcher.
	Service processor.Processor
}

// NewSwitchServiceFrame builds a SwitchServiceFrame targeting svc.
func NewSwitchServiceFrame(svc processor.Processor) *SwitchServiceFrame {
	return &SwitchServiceFrame{
		BaseControlFrame: frames.NewBaseControlFrame("SwitchServiceFrame"),
		Service:          svc,
	}
}

// ServiceSwitcher routes the pipeline through one of several interchangeable
// services at a time. Every service is started and kept warm, but only the
// active one receives data; the rest are gated off. Switching is manual (via
// SwitchTo or a SwitchServiceFrame) and, under SwitchFailover, automatic when
// the active service reports a non-fatal error.
//
// It is built on a ParallelPipeline: each service becomes a branch wrapped in a
// pair of filters that pass lifecycle and system frames (so every service stays
// ready) but gate data frames on whether the service is active. A control
// processor in front consumes switch requests and watches for the errors that
// drive failover.
type ServiceSwitcher struct {
	*Pipeline
	state *switcherState
}

// NewServiceSwitcher builds a switcher over services, the first of which starts
// active, using the given switching strategy.
func NewServiceSwitcher(services []processor.Processor, strategy SwitcherStrategy) (*ServiceSwitcher, error) {
	if len(services) == 0 {
		return nil, errNoServices
	}
	state := &switcherState{services: services, active: services[0], mode: strategy}

	branches := make([][]processor.Processor, len(services))
	for i, svc := range services {
		branches[i] = []processor.Processor{
			processor.NewFunctionFilter(fmt.Sprintf("Switch::In%d", i), processor.Downstream, gate(state, svc)),
			svc,
			processor.NewFunctionFilter(fmt.Sprintf("Switch::Out%d", i), processor.Upstream, gate(state, svc)),
		}
	}
	pp, err := NewParallel(branches...)
	if err != nil {
		return nil, err
	}
	return &ServiceSwitcher{Pipeline: New(newSwitchControl(state), pp), state: state}, nil
}

// SwitchTo makes svc the active service, returning false if svc is not one of
// the switcher's services.
func (s *ServiceSwitcher) SwitchTo(svc processor.Processor) bool { return s.state.switchTo(svc) }

// ActiveService returns the currently active service.
func (s *ServiceSwitcher) ActiveService() processor.Processor { return s.state.activeService() }

// OnSwitch registers fn to be called whenever the active service changes.
func (s *ServiceSwitcher) OnSwitch(fn func(processor.Processor)) { s.state.setOnSwitch(fn) }

// gate returns the filter predicate for svc: lifecycle and system frames always
// pass (so the service stays started and ready), and other frames pass only
// while svc is the active service.
func gate(state *switcherState, svc processor.Processor) processor.FilterFunc {
	return func(f frames.Frame) bool {
		return alwaysPass(f) || state.isActive(svc)
	}
}

// alwaysPass reports whether a frame must reach every branch regardless of which
// service is active: every system frame, and the EndFrame (a control frame the
// parallel pipeline synchronizes across all branches).
func alwaysPass(f frames.Frame) bool {
	if _, ok := f.(frames.SystemFrame); ok {
		return true
	}
	_, ok := f.(*frames.EndFrame)
	return ok
}

// switcherState holds the shared, concurrency-safe switching state.
type switcherState struct {
	mu       sync.Mutex
	services []processor.Processor
	active   processor.Processor
	mode     SwitcherStrategy
	onSwitch func(processor.Processor)
}

func (s *switcherState) isActive(svc processor.Processor) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.active == svc
}

func (s *switcherState) activeService() processor.Processor {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.active
}

func (s *switcherState) setOnSwitch(fn func(processor.Processor)) {
	s.mu.Lock()
	s.onSwitch = fn
	s.mu.Unlock()
}

// switchTo activates svc if it belongs to the switcher, firing the on-switch
// callback when the active service actually changes.
func (s *switcherState) switchTo(svc processor.Processor) bool {
	s.mu.Lock()
	if indexOf(s.services, svc) < 0 {
		s.mu.Unlock()
		return false
	}
	changed := s.active != svc
	s.active = svc
	cb := s.onSwitch
	s.mu.Unlock()
	if changed && cb != nil {
		cb(svc)
	}
	return true
}

// failover advances to the next service when failover is enabled and the named
// source is the active service.
func (s *switcherState) failover(sourceName string) {
	s.mu.Lock()
	if s.mode != SwitchFailover || len(s.services) <= 1 || s.active.Name() != sourceName {
		s.mu.Unlock()
		return
	}
	idx := indexOf(s.services, s.active)
	next := s.services[(idx+1)%len(s.services)]
	s.active = next
	cb := s.onSwitch
	s.mu.Unlock()
	if cb != nil {
		cb(next)
	}
}

// indexOf returns the position of svc in services, or -1.
func indexOf(services []processor.Processor, svc processor.Processor) int {
	for i, x := range services {
		if x == svc {
			return i
		}
	}
	return -1
}

// switchControl sits in front of the parallel pipeline: it consumes downstream
// SwitchServiceFrames and watches upstream non-fatal ErrorFrames to drive
// failover. Every other frame passes through untouched.
type switchControl struct {
	*processor.Base
	state *switcherState
}

func newSwitchControl(state *switcherState) *switchControl {
	c := &switchControl{state: state}
	c.Base = processor.New("ServiceSwitcher::Control", c)
	return c
}

// ProcessFrame intercepts switch requests and failover-triggering errors.
func (c *switchControl) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := c.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	switch dir {
	case processor.Downstream:
		if sf, ok := f.(*SwitchServiceFrame); ok {
			c.state.switchTo(sf.Service)
			return nil // The request is consumed, not forwarded.
		}
	case processor.Upstream:
		if ef, ok := f.(*frames.ErrorFrame); ok && !ef.Fatal && ef.Source != nil {
			c.state.failover(ef.Source.Name())
		}
	}
	return c.PushFrame(ctx, f, dir)
}
