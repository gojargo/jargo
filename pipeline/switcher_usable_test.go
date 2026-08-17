package pipeline_test

import (
	"context"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor"
	errs "github.com/gojargo/jargo/utils/errors"
	"github.com/gojargo/jargo/utils/events"
)

// plainSvc forwards everything, standing for a service that never fails.
type plainSvc struct{ *processor.Base }

func newPlainSvc(name string) *plainSvc {
	s := &plainSvc{}
	s.Base = processor.New("PlainSvc:"+name, s)
	return s
}

// threeServices builds a failover strategy over three services that never fail.
func threeServices(t *testing.T) (pipeline.SwitcherStrategy, []*plainSvc) {
	t.Helper()
	svcs := []*plainSvc{newPlainSvc("1"), newPlainSvc("2"), newPlainSvc("3")}
	procs := make([]processor.Processor, len(svcs))
	for i, s := range svcs {
		procs[i] = s
	}
	return pipeline.NewFailoverStrategy(procs), svcs
}

// failure is the error frame a service pushes, attributed to it.
func failure(p processor.Processor, message string) *frames.ErrorFrame {
	ef := frames.NewErrorFrame(message)
	ef.Source = p
	return ef
}

func TestErrorSwitchesToNextService(t *testing.T) {
	strategy, svcs := threeServices(t)
	ctx := context.Background()

	svcs[0].SetUsable(ctx, false)
	got := strategy.HandleError(failure(svcs[0], "connection lost"))

	if got != processor.Processor(svcs[1]) {
		t.Errorf("switched to %v, want the second service", got)
	}
	if strategy.ActiveService() != processor.Processor(svcs[1]) {
		t.Errorf("active is %v, want the second service", strategy.ActiveService())
	}
}

func TestRecoverableErrorDoesNotSwitch(t *testing.T) {
	strategy, svcs := threeServices(t)

	// The service can carry on, so there is nothing to fail over from.
	got := strategy.HandleError(failure(svcs[0], "transient failure"))

	if got != nil {
		t.Errorf("switched to %v, want no switch", got)
	}
	if strategy.ActiveService() != processor.Processor(svcs[0]) {
		t.Errorf("active is %v, want the first service", strategy.ActiveService())
	}
}

func TestConsecutiveErrorsCycleThroughServices(t *testing.T) {
	strategy, svcs := threeServices(t)
	ctx := context.Background()

	svcs[0].SetUsable(ctx, false)
	strategy.HandleError(failure(svcs[0], "error 1"))
	if strategy.ActiveService() != processor.Processor(svcs[1]) {
		t.Fatalf("active is %v, want the second service", strategy.ActiveService())
	}

	svcs[1].SetUsable(ctx, false)
	strategy.HandleError(failure(svcs[1], "error 2"))
	if strategy.ActiveService() != processor.Processor(svcs[2]) {
		t.Fatalf("active is %v, want the third service", strategy.ActiveService())
	}

	// Wraps around to the first, brought back in the meantime.
	svcs[0].SetUsable(ctx, true)
	svcs[2].SetUsable(ctx, false)
	strategy.HandleError(failure(svcs[2], "error 3"))
	if strategy.ActiveService() != processor.Processor(svcs[0]) {
		t.Fatalf("active is %v, want the first service", strategy.ActiveService())
	}
}

func TestSingleServiceHasNowhereToGo(t *testing.T) {
	only := newPlainSvc("only")
	strategy := pipeline.NewFailoverStrategy([]processor.Processor{only})
	only.SetUsable(context.Background(), false)

	if got := strategy.HandleError(failure(only, "error")); got != nil {
		t.Errorf("switched to %v, want no switch", got)
	}
}

func TestFailoverSkipsServicesThatCannotWork(t *testing.T) {
	strategy, svcs := threeServices(t)
	ctx := context.Background()

	svcs[1].SetUsable(ctx, false) // the one it would reach first
	svcs[0].SetUsable(ctx, false)
	got := strategy.HandleError(failure(svcs[0], "connection lost"))

	if got != processor.Processor(svcs[2]) {
		t.Errorf("switched to %v, want the third service", got)
	}
}

func TestManualSwitchRefusesAServiceThatCannotWork(t *testing.T) {
	first, second := newPlainSvc("first"), newPlainSvc("second")
	strategy := pipeline.NewManualStrategy([]processor.Processor{first, second})
	second.SetUsable(context.Background(), false)

	got := strategy.HandleFrame(frames.NewManuallySwitchServiceFrame(second), processor.Downstream)

	if got != nil {
		t.Errorf("switched to %v, want the request refused", got)
	}
	if strategy.ActiveService() != processor.Processor(first) {
		t.Errorf("active is %v, want the first service", strategy.ActiveService())
	}
}

func TestUsableServicesAreReportedInOrder(t *testing.T) {
	strategy, svcs := threeServices(t)
	svcs[1].SetUsable(context.Background(), false)

	got := strategy.UsableServices()

	if len(got) != 2 || got[0] != processor.Processor(svcs[0]) || got[1] != processor.Processor(svcs[2]) {
		t.Errorf("usable services = %v, want the first and third", got)
	}
}

func TestSwitcherIsUsableWhileAnyServiceIs(t *testing.T) {
	first, second := newPlainSvc("first"), newPlainSvc("second")
	sw, err := pipeline.NewServiceSwitcher([]processor.Processor{first, second}, pipeline.NewManualStrategy)
	if err != nil {
		t.Fatal(err)
	}
	ctx := context.Background()

	first.SetUsable(ctx, false)
	if !sw.Usable() {
		t.Error("a switcher with a service left should still be usable")
	}

	second.SetUsable(ctx, false)
	if sw.Usable() {
		t.Error("a switcher with nothing left should report itself unusable")
	}

	// Bringing one back brings the switcher back with it.
	second.SetUsable(ctx, true)
	if !sw.Usable() {
		t.Error("a switcher should come back with its service")
	}
}

func TestSwitcherAnnouncesItsOwnUsability(t *testing.T) {
	first, second := newPlainSvc("first"), newPlainSvc("second")
	sw, err := pipeline.NewServiceSwitcher([]processor.Processor{first, second}, pipeline.NewManualStrategy)
	if err != nil {
		t.Fatal(err)
	}
	ctx := context.Background()

	announced := make(chan bool, 8)
	events.On(sw.Events(), processor.EventUsableChanged, func(_ context.Context, usable bool) {
		announced <- usable
	})

	// Losing one service of two changes nothing the switcher cannot absorb.
	first.SetUsable(ctx, false)
	// Losing the last one does.
	second.SetUsable(ctx, false)
	if got := waitForAnnouncement(t, announced); got {
		t.Errorf("announced %t, want false", got)
	}

	// And getting one back brings the switcher back with it.
	first.SetUsable(ctx, true)
	if got := waitForAnnouncement(t, announced); !got {
		t.Errorf("announced %t, want true", got)
	}

	// Only those two: neither losing the first nor regaining the second moved
	// what the switcher reports.
	select {
	case extra := <-announced:
		t.Errorf("announced %t as well, want nothing further", extra)
	case <-time.After(200 * time.Millisecond):
	}
}

func waitForAnnouncement(t *testing.T, announced <-chan bool) bool {
	t.Helper()
	select {
	case got := <-announced:
		return got
	case <-time.After(2 * time.Second):
		t.Fatal("the switcher never announced its usability")
		return false
	}
}

func TestSettingTheSwitcherUsableIsIgnored(t *testing.T) {
	// The switcher reports a reading of its services, so setting it directly
	// would claim something the services do not say.
	first := newPlainSvc("first")
	sw, err := pipeline.NewServiceSwitcher([]processor.Processor{first}, pipeline.NewManualStrategy)
	if err != nil {
		t.Fatal(err)
	}
	ctx := context.Background()

	announced := make(chan bool, 4)
	events.On(sw.Events(), processor.EventUsableChanged, func(_ context.Context, usable bool) {
		announced <- usable
	})

	sw.SetUsable(ctx, false)

	if !sw.Usable() {
		t.Error("the switcher took a usability its service did not give it")
	}
	select {
	case got := <-announced:
		t.Errorf("announced %t, want nothing announced", got)
	case <-time.After(200 * time.Millisecond):
	}

	// And a service that cannot be given work is not overridden either.
	first.SetUsable(ctx, false)
	sw.SetUsable(ctx, true)
	if sw.Usable() {
		t.Error("the switcher overrode a service that can no longer work")
	}
}

// runSwitcher runs a switcher in a worker and returns the error frames that
// reached the start of the pipeline.
func runSwitcher(
	t *testing.T, sw *pipeline.ServiceSwitcher, feed []frames.Frame,
) []*frames.ErrorFrame {
	t.Helper()
	worker := pipeline.NewWorker(pipeline.New(sw), pipeline.WorkerConfig{})
	var mu sync.Mutex
	var got []*frames.ErrorFrame
	events.On(&worker.Registry, pipeline.EventPipelineError, func(_ context.Context, ef *frames.ErrorFrame) {
		mu.Lock()
		got = append(got, ef)
		mu.Unlock()
	})

	done := make(chan error, 1)
	go func() { done <- worker.Run(context.Background()) }()
	for _, f := range feed {
		worker.QueueFrame(f)
	}
	worker.StopWhenDone()
	select {
	case <-done:
	case <-time.After(10 * time.Second):
		t.Fatal("the worker never finished")
	}

	mu.Lock()
	defer mu.Unlock()
	return got
}

func TestFailoverAbsorbsTheError(t *testing.T) {
	// The switcher went on doing its job by moving work to another service, so
	// there is nothing left for the rest of the pipeline to act on.
	failing := newTagSvc("A:", "FAIL")
	backup := newTagSvc("B:", "")
	sw, err := pipeline.NewServiceSwitcher(
		[]processor.Processor{failing, backup}, pipeline.NewFailoverStrategy)
	if err != nil {
		t.Fatal(err)
	}

	got := runSwitcher(t, sw, []frames.Frame{frames.NewTextFrame("FAIL")})

	if len(got) != 0 {
		t.Errorf("reported %d errors, want none", len(got))
	}
	if sw.ActiveService() != processor.Processor(backup) {
		t.Errorf("active is %v, want the backup", sw.ActiveService())
	}
	if !sw.Usable() {
		t.Error("a switcher that recovered should still be usable")
	}
}

func TestAnErrorFromAServiceInReserveGoesNoFurther(t *testing.T) {
	// It is not being given work, so nothing about it bears on whether the
	// switcher can do its job.
	active := newTagSvc("A:", "")
	reserve := newTagSvc("B:", "")
	sw, err := pipeline.NewServiceSwitcher(
		[]processor.Processor{active, reserve}, pipeline.NewFailoverStrategy)
	if err != nil {
		t.Fatal(err)
	}
	worker := pipeline.NewWorker(pipeline.New(sw), pipeline.WorkerConfig{})
	var mu sync.Mutex
	var got []*frames.ErrorFrame
	events.On(&worker.Registry, pipeline.EventPipelineError, func(_ context.Context, ef *frames.ErrorFrame) {
		mu.Lock()
		got = append(got, ef)
		mu.Unlock()
	})

	done := make(chan error, 1)
	go func() { done <- worker.Run(context.Background()) }()
	// Reported from beside the frame path, the way a reconnect loop reports.
	time.Sleep(100 * time.Millisecond)
	reserve.PushError(context.Background(), "service connection lost", errBoom, false)
	time.Sleep(100 * time.Millisecond)
	worker.StopWhenDone()
	<-done

	mu.Lock()
	defer mu.Unlock()
	if len(got) != 0 {
		t.Errorf("reported %d errors, want none", len(got))
	}
	if sw.ActiveService() != processor.Processor(active) {
		t.Errorf("active is %v, want the first service", sw.ActiveService())
	}
}

func TestErrorWithNoServiceLeftIsReported(t *testing.T) {
	// With nowhere to move the work, the failed service stays active and the
	// error is the switcher's to report.
	only := newTagSvc("A:", "FAIL")
	sw, err := pipeline.NewServiceSwitcher([]processor.Processor{only}, pipeline.NewFailoverStrategy)
	if err != nil {
		t.Fatal(err)
	}

	got := runSwitcher(t, sw, []frames.Frame{frames.NewTextFrame("FAIL")})

	if len(got) != 1 {
		t.Fatalf("reported %d errors, want 1", len(got))
	}
	if got[0].Source != processor.Processor(sw) {
		t.Errorf("error names %v, want the switcher", got[0].Source)
	}
	if !strings.Contains(got[0].Error, "tag svc failed") {
		t.Errorf("error is %q, want the service's own message carried", got[0].Error)
	}
	if sw.Usable() {
		t.Error("a switcher with nothing left should report itself unusable")
	}
}

func TestALostServiceDoesNotWriteOffTheSwitcher(t *testing.T) {
	// The pipeline deals with the switcher, not with the services inside it, so
	// an error costing one service must not read as the switcher being spent
	// while it still has somewhere to send work. A manual strategy never
	// switches on its own, which is what leaves the pair in this state.
	failing := newTagSvc("A:", "FAIL")
	backup := newTagSvc("B:", "")
	sw, err := pipeline.NewServiceSwitcher(
		[]processor.Processor{failing, backup}, pipeline.NewManualStrategy)
	if err != nil {
		t.Fatal(err)
	}

	got := runSwitcher(t, sw, []frames.Frame{frames.NewTextFrame("FAIL")})

	if len(got) != 1 {
		t.Fatalf("reported %d errors, want 1", len(got))
	}
	if got[0].Source != processor.Processor(sw) {
		t.Errorf("error names %v, want the switcher", got[0].Source)
	}
	if !sw.Usable() {
		t.Error("a switcher with a service left should still be usable")
	}
	// The failing service is named, so the report still leads somewhere.
	if !strings.Contains(got[0].Error, failing.Name()) {
		t.Errorf("error is %q, want the failed service named", got[0].Error)
	}
}

func TestARejectedServiceDoesNotMisconfigureTheSwitcher(t *testing.T) {
	// Inheriting the category would write the switcher off for good, taking its
	// remaining services with it.
	failing := newTagSvc("A:", "FAIL")
	failing.category = errs.Authentication
	backup := newTagSvc("B:", "")
	sw, err := pipeline.NewServiceSwitcher(
		[]processor.Processor{failing, backup}, pipeline.NewManualStrategy)
	if err != nil {
		t.Fatal(err)
	}

	got := runSwitcher(t, sw, []frames.Frame{frames.NewTextFrame("FAIL")})

	if len(got) != 1 {
		t.Fatalf("reported %d errors, want 1", len(got))
	}
	if got[0].Category != errs.Unknown {
		t.Errorf("category: got %q, want %q", got[0].Category, errs.Unknown)
	}
	if !sw.Usable() {
		t.Error("a switcher should not inherit a service's rejected configuration")
	}
}

func TestRecoverableErrorDoesNotTriggerFailover(t *testing.T) {
	failing := newTagSvc("A:", "FAIL")
	failing.recoverable = true
	backup := newTagSvc("B:", "")
	sw, err := pipeline.NewServiceSwitcher(
		[]processor.Processor{failing, backup}, pipeline.NewFailoverStrategy)
	if err != nil {
		t.Fatal(err)
	}

	got := runSwitcher(t, sw, []frames.Frame{frames.NewTextFrame("FAIL")})

	if sw.ActiveService() != processor.Processor(failing) {
		t.Errorf("active is %v, want the service that can carry on", sw.ActiveService())
	}
	// An error the service carries on from travels upstream as usual.
	if len(got) != 1 {
		t.Fatalf("reported %d errors, want 1", len(got))
	}
	if got[0].Source != processor.Processor(failing) {
		t.Errorf("error names %v, want the service itself", got[0].Source)
	}
}

func TestStrategySeesEveryErrorFromTheActiveService(t *testing.T) {
	// Which errors are worth switching away from is the strategy's decision, so
	// it hears about them all, not only the ones that end a service.
	var mu sync.Mutex
	var seen []*frames.ErrorFrame
	recording := func(services []processor.Processor) pipeline.SwitcherStrategy {
		return &recordingStrategy{
			SwitcherStrategy: pipeline.NewManualStrategy(services),
			seen:             &seen,
			mu:               &mu,
		}
	}

	failing := newTagSvc("A:", "FAIL")
	failing.recoverable = true
	backup := newTagSvc("B:", "")
	sw, err := pipeline.NewServiceSwitcher([]processor.Processor{failing, backup}, recording)
	if err != nil {
		t.Fatal(err)
	}

	runSwitcher(t, sw, []frames.Frame{frames.NewTextFrame("FAIL")})

	mu.Lock()
	defer mu.Unlock()
	if len(seen) != 1 {
		t.Fatalf("the strategy saw %d errors, want 1", len(seen))
	}
	if seen[0].Source != processor.Processor(failing) {
		t.Errorf("error names %v, want the service itself", seen[0].Source)
	}
	if !failing.Usable() {
		t.Error("a recoverable error should leave the service usable")
	}
}

// recordingStrategy records the errors it is given and never switches.
type recordingStrategy struct {
	pipeline.SwitcherStrategy
	mu   *sync.Mutex
	seen *[]*frames.ErrorFrame
}

func (s *recordingStrategy) HandleError(ef *frames.ErrorFrame) processor.Processor {
	s.mu.Lock()
	*s.seen = append(*s.seen, ef)
	s.mu.Unlock()
	return nil
}
