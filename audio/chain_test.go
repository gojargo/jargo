package audio_test

import (
	"context"
	"errors"
	"testing"

	"github.com/gojargo/jargo/audio"
	"github.com/gojargo/jargo/frames"
)

// The failures the fake filters report.
//
//nolint:gochecknoglobals // sentinel errors
var (
	errBoom  = errors.New("boom")
	errEarly = errors.New("early")
)

// fake is a Filter that records its lifecycle calls, tags the audio it sees, and
// can be made to fail at any stage.
type fake struct {
	tag byte

	started    int
	stopped    int
	filtered   int
	controlled int
	startErr   error
	stopErr    error
	filterErr  error
	controlErr error
}

func (f *fake) ProcessFrame(context.Context, frames.FilterControlFrame) error {
	f.controlled++
	return f.controlErr
}

func (f *fake) Start(context.Context, int) error {
	f.started++
	return f.startErr
}

func (f *fake) Stop(context.Context) error {
	f.stopped++
	return f.stopErr
}

// Filter appends the filter's tag so the test can read the order filters ran in
// straight off the output.
func (f *fake) Filter(_ context.Context, pcm []byte) ([]byte, error) {
	f.filtered++
	if f.filterErr != nil {
		return nil, f.filterErr
	}
	return append(append([]byte{}, pcm...), f.tag), nil
}

// TestChainAppliesInOrder checks filters run front to back, which matters
// because denoising before gating is not the same as gating before denoising.
func TestChainAppliesInOrder(t *testing.T) {
	a, b, c := &fake{tag: 'a'}, &fake{tag: 'b'}, &fake{tag: 'c'}
	chain := audio.NewChain(a, b, c)

	out, err := chain.Filter(t.Context(), []byte("in:"))
	if err != nil {
		t.Fatalf("Filter: %v", err)
	}
	if string(out) != "in:abc" {
		t.Errorf("output = %q, want the filters applied in order", out)
	}
}

// TestChainEmpty checks a chain with no filters is a pass-through, so an app can
// build one from an optional list without special-casing the empty case.
func TestChainEmpty(t *testing.T) {
	chain := audio.NewChain()

	if err := chain.Start(t.Context(), 16000); err != nil {
		t.Errorf("Start: %v", err)
	}
	out, err := chain.Filter(t.Context(), []byte("pcm"))
	if err != nil {
		t.Fatalf("Filter: %v", err)
	}
	if string(out) != "pcm" {
		t.Errorf("output = %q, want the input unchanged", out)
	}
	if err := chain.Stop(t.Context()); err != nil {
		t.Errorf("Stop: %v", err)
	}
}

func TestChainStart(t *testing.T) {
	t.Run("starts every filter", func(t *testing.T) {
		a, b := &fake{}, &fake{}
		if err := audio.NewChain(a, b).Start(t.Context(), 16000); err != nil {
			t.Fatalf("Start: %v", err)
		}
		if a.started != 1 || b.started != 1 {
			t.Errorf("started a=%d b=%d, want 1 each", a.started, b.started)
		}
	})

	t.Run("stops at the first failure", func(t *testing.T) {
		a := &fake{startErr: errBoom}
		b := &fake{}

		err := audio.NewChain(a, b).Start(t.Context(), 16000)
		if !errors.Is(err, errBoom) {
			t.Fatalf("Start error = %v, want errBoom", err)
		}
		// A filter after a failed one must not be started; the chain is unusable.
		if b.started != 0 {
			t.Errorf("second filter started %d times, want 0", b.started)
		}
	})
}

// TestChainStopIsBestEffort checks every filter gets stopped even when an
// earlier one fails, so a single bad filter cannot leak the others' resources —
// and that the failure is still reported rather than masked by the successes
// that follow it.
func TestChainStopIsBestEffort(t *testing.T) {
	a := &fake{stopErr: errBoom}
	b := &fake{}

	err := audio.NewChain(a, b).Stop(t.Context())
	if !errors.Is(err, errBoom) {
		t.Errorf("Stop error = %v, want errBoom; a later success must not mask it", err)
	}
	if a.stopped != 1 || b.stopped != 1 {
		t.Errorf("stopped a=%d b=%d, want 1 each despite the failure", a.stopped, b.stopped)
	}
}

// TestChainStopReportsLastFailure checks that when several filters fail, the
// last failure is the one returned.
func TestChainStopReportsLastFailure(t *testing.T) {
	a := &fake{stopErr: errEarly}
	b := &fake{stopErr: errBoom}

	if err := audio.NewChain(a, b).Stop(t.Context()); !errors.Is(err, errBoom) {
		t.Errorf("Stop error = %v, want the last failure", err)
	}
}

// TestChainStopAllSucceed checks the happy path reports no error.
func TestChainStopAllSucceed(t *testing.T) {
	if err := audio.NewChain(&fake{}, &fake{}).Stop(t.Context()); err != nil {
		t.Errorf("Stop error = %v, want nil", err)
	}
}

// TestChainFilterShortCircuits checks a failing filter aborts the chain: passing
// half-processed audio downstream would be worse than dropping the chunk.
func TestChainFilterShortCircuits(t *testing.T) {
	a := &fake{tag: 'a'}
	b := &fake{filterErr: errBoom}
	c := &fake{tag: 'c'}

	out, err := audio.NewChain(a, b, c).Filter(t.Context(), []byte("in"))
	if !errors.Is(err, errBoom) {
		t.Fatalf("Filter error = %v, want errBoom", err)
	}
	if out != nil {
		t.Errorf("output = %q, want nil on failure", out)
	}
	if c.filtered != 0 {
		t.Errorf("the filter after the failure ran %d times, want 0", c.filtered)
	}
}
