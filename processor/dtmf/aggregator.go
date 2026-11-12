// Package dtmf aggregates the DTMF keypresses a caller makes into a string a
// language model can read. Aggregator collects the keys and flushes them on a
// terminator key or an idle timeout.
//
// Generating the tones for a key is a separate concern and lives in
// audio/dtmf, which is what an output transport sounds when it has to carry a
// keypress as audio.
package dtmf

import (
	"context"
	"strings"
	"sync"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
)

// DefaultTimeout is how long the aggregator waits after the last key before it
// flushes the digits collected without a terminator.
const DefaultTimeout = 2 * time.Second

// DefaultPrefix is what the emitted transcription starts with, so the model
// can tell a keypad entry from something the user said.
const DefaultPrefix = "DTMF: "

// AggregatorConfig configures a DTMF Aggregator. The zero value is the
// defaults: '#' ends an entry, two idle seconds flush one without it, and the
// transcription starts "DTMF: ".
type AggregatorConfig struct {
	// Terminator is the key that flushes the collected digits; empty uses '#'.
	Terminator frames.KeypadEntry
	// Timeout flushes the collected digits after this idle period even without a
	// terminator. Zero uses DefaultTimeout; a negative value turns the timeout
	// off, so only the terminator flushes.
	Timeout time.Duration
	// Prefix is prepended to the emitted transcription. Nil uses DefaultPrefix;
	// point it at "" for none.
	Prefix *string
}

// Aggregator collects the DTMF keys received from the transport into a string
// and emits it downstream as a TranscriptionFrame — so an LLM can react to the
// entered sequence — when the terminator key is pressed or the idle timeout
// elapses. It forwards the InputDTMFFrames it sees unchanged.
type Aggregator struct {
	*processor.Base
	cfg AggregatorConfig

	mu    sync.Mutex
	buf   strings.Builder
	timer *time.Timer
}

// NewAggregator builds a DTMF Aggregator.
func NewAggregator(cfg AggregatorConfig) *Aggregator {
	if cfg.Terminator == "" {
		cfg.Terminator = frames.KeypadPound
	}
	if cfg.Timeout == 0 {
		cfg.Timeout = DefaultTimeout
	}
	if cfg.Prefix == nil {
		cfg.Prefix = new(DefaultPrefix)
	}
	a := &Aggregator{cfg: cfg}
	a.Base = processor.New("DTMFAggregator", a)
	return a
}

// ProcessFrame aggregates DTMF keypresses and forwards every frame.
func (a *Aggregator) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := a.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	if df, ok := f.(*frames.InputDTMFFrame); ok {
		if err := a.PushFrame(ctx, f, dir); err != nil {
			return err
		}
		a.handle(ctx, df.Button)
		return nil
	}
	return a.PushFrame(ctx, f, dir)
}

// Cleanup stops the idle timer and tears down the processor.
func (a *Aggregator) Cleanup(ctx context.Context) error {
	a.mu.Lock()
	a.stopTimer()
	a.mu.Unlock()
	return a.Base.Cleanup(ctx)
}

// handle appends a key, and flushes the entry when the key is the terminator.
// The terminator is part of the entry: it is what the caller pressed.
func (a *Aggregator) handle(ctx context.Context, button frames.KeypadEntry) {
	a.mu.Lock()
	first := a.buf.Len() == 0
	a.buf.WriteString(string(button))
	a.mu.Unlock()

	if first {
		// A caller keying in an entry is taking the turn, so the bot stops
		// talking, as it does when the caller speaks.
		_ = a.BroadcastInterruption(ctx)
	}

	a.mu.Lock()
	if button != a.cfg.Terminator {
		a.armTimer(ctx)
		a.mu.Unlock()
		return
	}
	text := a.buf.String()
	a.buf.Reset()
	a.stopTimer()
	a.mu.Unlock()
	a.emit(ctx, text)
}

// armTimer restarts the idle-flush timer when a timeout is configured. It runs
// with the mutex held.
func (a *Aggregator) armTimer(ctx context.Context) {
	if a.cfg.Timeout <= 0 {
		return
	}
	a.stopTimer()
	a.timer = time.AfterFunc(a.cfg.Timeout, func() {
		a.mu.Lock()
		text := a.buf.String()
		a.buf.Reset()
		a.timer = nil
		a.mu.Unlock()
		a.emit(ctx, text)
	})
}

func (a *Aggregator) stopTimer() {
	if a.timer != nil {
		a.timer.Stop()
		a.timer = nil
	}
}

// emit pushes the collected digits downstream as a finalized transcription.
func (a *Aggregator) emit(ctx context.Context, text string) {
	if text == "" {
		return
	}
	tf := frames.NewTranscriptionFrame(*a.cfg.Prefix+text, "", frames.NowTimestamp())
	tf.Finalized = true
	_ = a.PushFrame(ctx, tf, processor.Downstream)
}
