package pipeline_test

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor"
	errs "github.com/gojargo/jargo/utils/errors"
	"github.com/gojargo/jargo/utils/events"
)

// errBoom is the non-fatal error a tagSvc raises to trigger failover.
var errBoom = errors.New("boom")

// tagSvc replaces each downstream TextFrame with one prefixed by its tag, so
// only the active branch of a switcher produces output. When a frame's text
// equals failOn it raises a non-fatal error instead, to drive failover.
//
// recoverable chooses between an error the service can carry on from and one
// that ends its usefulness; it ends it by default, which is the error worth
// failing over from.
type tagSvc struct {
	*processor.Base
	tag         string
	failOn      string
	recoverable bool
	category    errs.Category
}

func newTagSvc(tag, failOn string) *tagSvc {
	s := &tagSvc{tag: tag, failOn: failOn}
	s.Base = processor.New("TagSvc:"+tag, s)
	return s
}

func (s *tagSvc) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := s.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	tf, ok := f.(*frames.TextFrame)
	if !ok || dir != processor.Downstream {
		return s.PushFrame(ctx, f, dir)
	}
	if s.failOn != "" && tf.Text == s.failOn {
		s.PushError(ctx, "tag svc failed", errBoom, false, s.errorOptions()...)
		return nil
	}
	return s.PushFrame(ctx, frames.NewTextFrame(s.tag+tf.Text), dir)
}

// errorOptions describes the failure this service reports: the category it was
// given, and whether the service can carry on from it.
func (s *tagSvc) errorOptions() []processor.ErrorOption {
	var opts []processor.ErrorOption
	if s.category != errs.Unset {
		opts = append(opts, processor.WithErrorCategory(s.category))
	}
	if !s.recoverable {
		opts = append(opts, processor.ForceTreatAsPermanent())
	}
	return opts
}

// runCollector runs a task over proc, returning the task (to queue frames into),
// a channel of every downstream TextFrame text, and a stop function.
func runCollector(t *testing.T, proc processor.Processor) (*pipeline.Worker, <-chan string, func()) {
	t.Helper()
	out := make(chan string, 64)
	task := pipeline.NewWorker(pipeline.New(proc), pipeline.WorkerConfig{
		ReachedDownstreamFilter: pipeline.AnyFrame,
	})
	events.On(&task.Registry, pipeline.EventFrameReachedDownstream, func(_ context.Context, f frames.Frame) {
		if tf, ok := f.(*frames.TextFrame); ok {
			select {
			case out <- tf.Text:
			default:
			}
		}
	})
	done := make(chan struct{})
	go func() { _ = task.Run(context.Background()); close(done) }()
	return task, out, func() {
		task.Cancel(t.Context(), "")
		<-done
	}
}

// wantText waits for the next collected text and checks it.
func wantText(t *testing.T, out <-chan string, want string) {
	t.Helper()
	select {
	case got := <-out:
		if got != want {
			t.Errorf("downstream text = %q, want %q", got, want)
		}
	case <-time.After(2 * time.Second):
		t.Fatalf("timed out waiting for %q", want)
	}
}

func TestFunctionFilterGatesDirection(t *testing.T) {
	// A downstream filter drops "block" but passes "ok".
	allow := func(f frames.Frame) bool {
		tf, ok := f.(*frames.TextFrame)
		return !ok || tf.Text != "block"
	}
	down := processor.Downstream
	pipe := pipeline.New(processor.NewFunctionFilter("F", &down, allow), newEcho())
	task, out, stop := runCollector(t, pipe)
	defer stop()

	task.QueueFrame(frames.NewTextFrame("block"))
	task.QueueFrame(frames.NewTextFrame("ok"))
	wantText(t, out, "ok") // "block" was dropped, so "ok" arrives first
}

func TestServiceSwitcherRouting(t *testing.T) {
	a := newTagSvc("A:", "")
	b := newTagSvc("B:", "")
	sw, err := pipeline.NewServiceSwitcher([]processor.Processor{a, b}, pipeline.NewManualStrategy)
	if err != nil {
		t.Fatalf("NewServiceSwitcher: %v", err)
	}
	if sw.ActiveService() != a {
		t.Error("initial active service is not the first service")
	}

	task, out, stop := runCollector(t, sw)
	defer stop()

	task.QueueFrame(frames.NewTextFrame("one"))
	wantText(t, out, "A:one") // first service is active

	sw.SwitchTo(b)
	task.QueueFrame(frames.NewTextFrame("two"))
	wantText(t, out, "B:two") // routed to the new active service
}

func TestServiceSwitcherInBandSwitch(t *testing.T) {
	a := newTagSvc("A:", "")
	b := newTagSvc("B:", "")
	sw, err := pipeline.NewServiceSwitcher([]processor.Processor{a, b}, pipeline.NewManualStrategy)
	if err != nil {
		t.Fatalf("NewServiceSwitcher: %v", err)
	}

	task, out, stop := runCollector(t, sw)
	defer stop()

	// A SwitchServiceFrame queued in-band switches the active service before the
	// following text is routed.
	task.QueueFrame(frames.NewManuallySwitchServiceFrame(b))
	task.QueueFrame(frames.NewTextFrame("x"))
	wantText(t, out, "B:x")
}

func TestServiceSwitcherFailover(t *testing.T) {
	a := newTagSvc("A:", "FAIL") // errors on "FAIL"
	b := newTagSvc("B:", "")
	sw, err := pipeline.NewServiceSwitcher([]processor.Processor{a, b}, pipeline.NewFailoverStrategy)
	if err != nil {
		t.Fatalf("NewServiceSwitcher: %v", err)
	}

	switched := make(chan processor.Processor, 1)
	sw.OnSwitch(func(p processor.Processor) {
		select {
		case switched <- p:
		default:
		}
	})

	task, out, stop := runCollector(t, sw)
	defer stop()

	// The active service errors on this frame, which fails over to the next.
	task.QueueFrame(frames.NewTextFrame("FAIL"))
	select {
	case p := <-switched:
		if p != b {
			t.Errorf("failed over to %v, want service b", p.Name())
		}
	case <-time.After(2 * time.Second):
		t.Fatal("timed out waiting for failover")
	}

	task.QueueFrame(frames.NewTextFrame("hi"))
	wantText(t, out, "B:hi") // now served by the backup
}

func TestServiceSwitcherNoServices(t *testing.T) {
	if _, err := pipeline.NewServiceSwitcher(nil, pipeline.NewManualStrategy); err == nil {
		t.Fatal("NewServiceSwitcher(nil): want error, got nil")
	}
}

// settingsSvc records the settings updates that reach it, so a test can tell
// which member services of a switcher an update was delivered to.
type settingsSvc struct {
	*processor.Base
	mu       sync.Mutex
	received []*frames.LLMUpdateSettingsFrame
}

func newSettingsSvc(name string) *settingsSvc {
	s := &settingsSvc{}
	s.Base = processor.New("SettingsSvc:"+name, s)
	return s
}

func (s *settingsSvc) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := s.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	if uf, ok := f.(*frames.LLMUpdateSettingsFrame); ok && uf.TargetsService(s) {
		s.mu.Lock()
		s.received = append(s.received, uf)
		s.mu.Unlock()
	}
	return s.PushFrame(ctx, f, dir)
}

// updates returns how many updates this service applied.
func (s *settingsSvc) updates() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return len(s.received)
}

// switcherWithSettingsServices builds a three-service switcher and returns it
// with its members, the first of which is active.
func switcherWithSettingsServices(t *testing.T) (*pipeline.ServiceSwitcher, []*settingsSvc) {
	t.Helper()
	svcs := []*settingsSvc{newSettingsSvc("1"), newSettingsSvc("2"), newSettingsSvc("3")}
	procs := make([]processor.Processor, len(svcs))
	for i, s := range svcs {
		procs[i] = s
	}
	sw, err := pipeline.NewServiceSwitcher(procs, pipeline.NewManualStrategy)
	if err != nil {
		t.Fatalf("NewServiceSwitcher: %v", err)
	}
	return sw, svcs
}

// wantUpdates waits for each service to have applied the number of updates
// expected of it, and checks none applied more than that.
func wantUpdates(t *testing.T, svcs []*settingsSvc, want []int) {
	t.Helper()
	deadline := time.Now().Add(2 * time.Second)
	for {
		done := true
		for i, s := range svcs {
			if s.updates() != want[i] {
				done = false
			}
		}
		if done {
			return
		}
		if time.Now().After(deadline) {
			got := make([]int, len(svcs))
			for i, s := range svcs {
				got[i] = s.updates()
			}
			t.Fatalf("updates applied = %v, want %v", got, want)
		}
		time.Sleep(time.Millisecond)
	}
}

// An update that names no service and asks for nothing more reaches the service
// in use and no other: a setting is usually specific to one provider.
func TestServiceSwitcherSettingsUpdateReachesTheActiveServiceAlone(t *testing.T) {
	sw, svcs := switcherWithSettingsServices(t)
	task, _, stop := runCollector(t, sw)
	defer stop()

	task.QueueFrame(frames.NewLLMUpdateSettingsFrame(nil))
	wantUpdates(t, svcs, []int{1, 0, 0})
}

// An update that asks to reach inactive services reaches every service the
// switcher manages, so whichever becomes active later is already configured.
func TestServiceSwitcherSettingsUpdateReachesEveryService(t *testing.T) {
	sw, svcs := switcherWithSettingsServices(t)
	task, out, stop := runCollector(t, sw)
	defer stop()

	update := frames.NewLLMUpdateSettingsFrame(nil)
	update.ReachInactiveServices = true
	task.QueueFrame(update)
	wantUpdates(t, svcs, []int{1, 1, 1})

	// One copy leaves the switcher, not one per service: a text frame queued
	// behind the update is the next thing downstream sees.
	task.QueueFrame(frames.NewTextFrame("after"))
	wantText(t, out, "after")
}

// An update addressed to a service is applied by that service whether or not it
// is the one in use, and by no other.
func TestServiceSwitcherSettingsUpdateAddressedToAnInactiveService(t *testing.T) {
	sw, svcs := switcherWithSettingsServices(t)
	task, _, stop := runCollector(t, sw)
	defer stop()

	update := frames.NewLLMUpdateSettingsFrame(nil)
	update.Service = svcs[2]
	task.QueueFrame(update)
	wantUpdates(t, svcs, []int{0, 0, 1})
}

// An update addressed to a service another switcher manages passes through
// untouched, leaving this switcher's own inactive services out of it.
func TestServiceSwitcherSettingsUpdateForAnotherSwitcher(t *testing.T) {
	sw, svcs := switcherWithSettingsServices(t)
	outsider := newSettingsSvc("outsider")
	task, _, stop := runCollector(t, pipeline.New(sw, outsider))
	defer stop()

	update := frames.NewLLMUpdateSettingsFrame(nil)
	update.Service = outsider
	task.QueueFrame(update)

	wantUpdates(t, []*settingsSvc{outsider}, []int{1})
	wantUpdates(t, svcs, []int{0, 0, 0})
}

// upstreamRaiser sits behind the switcher and sends a held frame back up the
// pipeline when a text frame reaches it, standing in for a processor that raises
// a settings update of its own.
type upstreamRaiser struct {
	*processor.Base
	f frames.Frame
}

func newUpstreamRaiser(f frames.Frame) *upstreamRaiser {
	r := &upstreamRaiser{f: f}
	r.Base = processor.New("UpstreamRaiser", r)
	return r
}

func (r *upstreamRaiser) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := r.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	if _, ok := f.(*frames.TextFrame); ok && dir == processor.Downstream {
		return r.PushFrame(ctx, r.f, processor.Upstream)
	}
	return r.PushFrame(ctx, f, dir)
}

// An update traveling upstream is routed the same way as one going downstream.
func TestServiceSwitcherSettingsUpdateTravelingUpstream(t *testing.T) {
	sw, svcs := switcherWithSettingsServices(t)

	update := frames.NewLLMUpdateSettingsFrame(nil)
	update.ReachInactiveServices = true
	task, _, stop := runCollector(t, pipeline.New(sw, newUpstreamRaiser(update)))
	defer stop()

	task.QueueFrame(frames.NewTextFrame("tick")) // reaches the raiser, which replies
	wantUpdates(t, svcs, []int{1, 1, 1})
}

// metaSvc broadcasts a metadata frame naming itself whenever it is started or
// asked to, standing in for a real service describing itself to the pipeline.
type metaSvc struct {
	*processor.Base
	name string
}

func newMetaSvc(name string) *metaSvc {
	s := &metaSvc{name: name}
	s.Base = processor.New("MetaSvc:"+name, s)
	return s
}

func (s *metaSvc) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := s.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	if err := s.PushFrame(ctx, f, dir); err != nil {
		return err
	}
	switch f.(type) {
	case *frames.StartFrame, *frames.ServiceSwitcherRequestMetadataFrame:
		// A service names itself by its processor name, as the real ones do.
		return s.PushFrame(ctx, frames.NewServiceMetadataFrame(s.Name()), processor.Downstream)
	}
	return nil
}

// collectMetadata runs a switcher and returns the service names whose metadata
// escaped it.
func collectMetadata(t *testing.T, sw *pipeline.ServiceSwitcher) (*pipeline.Worker, func() []string, func()) {
	t.Helper()
	var (
		mu   sync.Mutex
		seen []string
	)
	task := pipeline.NewWorker(pipeline.New(sw), pipeline.WorkerConfig{
		ReachedDownstreamFilter: pipeline.AnyFrame,
	})
	events.On(&task.Registry, pipeline.EventFrameReachedDownstream, func(_ context.Context, f frames.Frame) {
		if mf, ok := f.(frames.ServiceMetadata); ok {
			mu.Lock()
			seen = append(seen, mf.Service())
			mu.Unlock()
		}
	})
	done := make(chan struct{})
	go func() { _ = task.Run(context.Background()); close(done) }()

	return task, func() []string {
			mu.Lock()
			defer mu.Unlock()
			return append([]string(nil), seen...)
		}, func() {
			task.Cancel(t.Context(), "")
			<-done
		}
}

// TestOnlyTheActiveServiceDescribesTheSwitcher checks the metadata leaving a
// switcher is the active service's alone. Every service is started, so every
// one of them broadcasts; letting them all out would leave whatever arrived
// last describing a service that is not in use.
func TestOnlyTheActiveServiceDescribesTheSwitcher(t *testing.T) {
	a, b := newMetaSvc("A"), newMetaSvc("B")
	sw, err := pipeline.NewServiceSwitcher([]processor.Processor{a, b}, pipeline.NewManualStrategy)
	if err != nil {
		t.Fatalf("NewServiceSwitcher: %v", err)
	}

	_, metadata, stop := collectMetadata(t, sw)
	defer stop()

	time.Sleep(300 * time.Millisecond)
	got := metadata()
	if len(got) != 1 || got[0] != a.Name() {
		t.Errorf("metadata leaving the switcher = %v, want just the active service %q", got, a.Name())
	}
}

// TestSwitchingAsksTheNewServiceToDescribeItself checks that after a switch the
// pipeline is told about the service now in use, rather than being left with
// what the one it replaced had said.
func TestSwitchingAsksTheNewServiceToDescribeItself(t *testing.T) {
	a, b := newMetaSvc("A"), newMetaSvc("B")
	sw, err := pipeline.NewServiceSwitcher([]processor.Processor{a, b}, pipeline.NewManualStrategy)
	if err != nil {
		t.Fatalf("NewServiceSwitcher: %v", err)
	}

	_, metadata, stop := collectMetadata(t, sw)
	defer stop()

	time.Sleep(200 * time.Millisecond)
	if !sw.SwitchTo(b) {
		t.Fatal("SwitchTo(b) = false, want the switch accepted")
	}
	time.Sleep(300 * time.Millisecond)

	got := metadata()
	if len(got) == 0 || got[len(got)-1] != b.Name() {
		t.Errorf("metadata leaving the switcher = %v, want it ending with the newly active %q", got, b.Name())
	}
}

// TestSwitchRequestForAnotherSwitcherTravelsOn checks a switcher leaves alone a
// request naming a service it does not manage, so the switcher that does manage
// it gets its turn. Swallowing the request would strand every switcher but the
// first.
func TestSwitchRequestForAnotherSwitcherTravelsOn(t *testing.T) {
	a1, a2 := newTagSvc("A1:", ""), newTagSvc("A2:", "")
	b1, b2 := newTagSvc("B1:", ""), newTagSvc("B2:", "")

	swA, err := pipeline.NewServiceSwitcher([]processor.Processor{a1, a2}, pipeline.NewManualStrategy)
	if err != nil {
		t.Fatalf("NewServiceSwitcher: %v", err)
	}
	swB, err := pipeline.NewServiceSwitcher([]processor.Processor{b1, b2}, pipeline.NewManualStrategy)
	if err != nil {
		t.Fatalf("NewServiceSwitcher: %v", err)
	}

	task, out, stop := runCollector(t, pipeline.New(swA, swB))
	defer stop()

	// b2 belongs to the second switcher, so the first must pass the request on.
	task.QueueFrame(frames.NewManuallySwitchServiceFrame(b2))
	time.Sleep(300 * time.Millisecond)

	if got := swB.ActiveService(); got != b2 {
		t.Errorf("the downstream switcher's active service = %v, want b2", got)
	}
	if got := swA.ActiveService(); got != a1 {
		t.Errorf("the upstream switcher's active service = %v, want a1 unchanged", got)
	}

	task.QueueFrame(frames.NewTextFrame("x"))
	wantText(t, out, "B2:A1:x")
}

// TestManualStrategyIgnoresErrors checks the strategy a switcher gets when
// switching is manual: an error from the active service is not a reason to move
// off it, since the caller decides when to switch.
func TestManualStrategyIgnoresErrors(t *testing.T) {
	a, b := newEcho(), newEcho()
	sw, err := pipeline.NewServiceSwitcher(
		[]processor.Processor{a, b}, pipeline.NewManualStrategy)
	if err != nil {
		t.Fatalf("NewServiceSwitcher: %v", err)
	}

	strategy := sw.Strategy()
	if strategy == nil {
		t.Fatal("Strategy() is nil, want the strategy choosing between the services")
	}
	if got := strategy.ActiveService(); got != processor.Processor(a) {
		t.Fatalf("ActiveService() = %v, want the first service", got)
	}

	ef := frames.NewErrorFrame("the provider failed")
	if got := strategy.HandleError(ef); got != nil {
		t.Errorf("HandleError switched to %v, want no switch", got)
	}
	if got := strategy.ActiveService(); got != processor.Processor(a) {
		t.Errorf("ActiveService() = %v, want it left on the first service", got)
	}
}
