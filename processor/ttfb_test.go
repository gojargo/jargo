package processor_test

import (
	"context"
	"testing"

	"github.com/gojargo/jargo/clock"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
)

// startBase hands base the StartFrame that carries the pipeline's reporting
// settings.
func startBase(t *testing.T, base *processor.Base, reportOnlyInitial bool) {
	t.Helper()
	start := frames.NewStartFrame()
	start.EnableMetrics = true
	start.ReportOnlyInitialTTFB = reportOnlyInitial
	if err := base.ProcessFrame(context.Background(), start, processor.Downstream); err != nil {
		t.Fatalf("start: %v", err)
	}
}

// startedBase is a processor set up and started, which is the state a service
// measures anything in.
func startedBase(t *testing.T, reportOnlyInitial bool) *processor.Base {
	t.Helper()
	ctx := context.Background()
	e := newEcho()
	if err := e.Setup(ctx, processor.Setup{Clock: clock.NewSystem()}); err != nil {
		t.Fatalf("setup: %v", err)
	}
	t.Cleanup(func() { _ = e.Cleanup(ctx) })
	startBase(t, e.Base, reportOnlyInitial)
	return e.Base
}

// Every measurement is armed when the pipeline wants them all, which is what a
// per-turn latency figure needs.
func TestBeginTTFBArmsEveryMeasurement(t *testing.T) {
	b := startedBase(t, false)

	for i := range 3 {
		if !b.BeginTTFB() {
			t.Fatalf("measurement %d was declined, want every one armed", i+1)
		}
	}
}

// Asked for the initial figure alone, the first measurement is the only one
// armed: the caller wants what the call opened with, not one reading per turn.
func TestBeginTTFBArmsOnlyTheFirstWhenAsked(t *testing.T) {
	b := startedBase(t, true)

	if !b.BeginTTFB() {
		t.Fatal("the initial measurement was declined")
	}
	for i := range 2 {
		if b.BeginTTFB() {
			t.Fatalf("measurement %d was armed after the initial one", i+2)
		}
	}
}

// A processor that has not been started yet arms what it is asked to. Declining
// on the strength of a restriction nothing has set would lose a measurement to
// the order the frames happened to arrive in.
func TestBeginTTFBArmsBeforeTheStartFrame(t *testing.T) {
	if !newEcho().Base.BeginTTFB() {
		t.Fatal("a measurement was declined before any StartFrame set the terms")
	}
}

// Starting again re-arms: a restriction spent on the run before must not carry
// into this one.
func TestBeginTTFBReArmsOnAStart(t *testing.T) {
	b := startedBase(t, true)
	b.BeginTTFB()

	startBase(t, b, true)
	if !b.BeginTTFB() {
		t.Fatal("the first measurement of the new run was declined")
	}
}
