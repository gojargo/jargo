package pipeline

import (
	"context"
	"log/slog"
	"os"
	"os/signal"
	"syscall"
)

// Runner runs a single task to completion and shuts it down gracefully on an
// interrupt signal (SIGINT or SIGTERM).
type Runner struct {
	handleSignals bool
}

// RunnerOption configures a Runner.
type RunnerOption func(*Runner)

// WithoutSignalHandling disables the SIGINT/SIGTERM handling, leaving shutdown
// entirely to the caller's context.
func WithoutSignalHandling() RunnerOption {
	return func(r *Runner) { r.handleSignals = false }
}

// NewRunner returns a Runner. By default it handles interrupt signals.
func NewRunner(opts ...RunnerOption) *Runner {
	r := &Runner{handleSignals: true}
	for _, opt := range opts {
		opt(r)
	}
	return r
}

// Run runs task until it finishes. When signal handling is enabled, the first
// interrupt signal cancels the task (so the pipeline drains the CancelFrame and
// shuts down); the task keeps running on the parent context so cleanup can
// complete. If the parent context is canceled the task stops the same way.
func (r *Runner) Run(parent context.Context, task *Task) error {
	if !r.handleSignals {
		return task.Run(parent)
	}

	sigCtx, stop := signal.NotifyContext(parent, os.Interrupt, syscall.SIGTERM)
	defer stop()

	runErr := make(chan error, 1)
	go func() { runErr <- task.Run(parent) }()

	select {
	case err := <-runErr:
		return err
	case <-sigCtx.Done():
		if parent.Err() == nil {
			slog.Info("interrupt received, stopping pipeline")
			task.Cancel()
		}
		return <-runErr
	}
}
