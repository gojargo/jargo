package eval

import (
	"bytes"
	"context"
	"log"
	"log/slog"
	"os"
	"path/filepath"
	"strings"
	"sync"
)

// The harness's per-run debug logs.
//
// The harness tags each of its pipelines' logs with a category through the
// context (see withEvalPipeline), and a run's logs with the run's id. While a
// capture is on, every record logged with a context carrying its run's id is
// buffered, then written as one <prefix>.debug.log, a section per category.
// Records reach a capture through the handler QuietLogs installs as the
// default, which the suite installs for the length of its run.

// logContextKey keys the log tags in a context.
type logContextKey int

const (
	// evalRunKey is the run a record belongs to.
	evalRunKey logContextKey = iota
	// evalPipelineKey is the harness pipeline a record comes from.
	evalPipelineKey
)

// withEvalRun tags the logs of ctx as run id's.
func withEvalRun(ctx context.Context, id string) context.Context {
	return context.WithValue(ctx, evalRunKey, id)
}

// withEvalPipeline tags the logs of ctx as coming from one of the harness's
// pipelines, one of PipelineLogCategories.
func withEvalPipeline(ctx context.Context, pipeline string) context.Context {
	return context.WithValue(ctx, evalPipelineKey, pipeline)
}

// logSink is one run's buffered logs, by category.
type logSink struct {
	mu      sync.Mutex
	buffers map[string][]string
}

func (s *logSink) add(category, line string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.buffers[category] = append(s.buffers[category], line)
}

// pipelineLogSinks are the captures on, by run id.
//
//nolint:gochecknoglobals // the default logger is process-wide, and so are its captures
var pipelineLogSinks = struct {
	mu    sync.Mutex
	sinks map[string]*logSink
}{sinks: map[string]*logSink{}}

// sinkFor is the capture of the run ctx belongs to, or nil.
func sinkFor(ctx context.Context) *logSink {
	if ctx == nil {
		return nil
	}
	id, _ := ctx.Value(evalRunKey).(string)
	if id == "" {
		return nil
	}
	pipelineLogSinks.mu.Lock()
	defer pipelineLogSinks.mu.Unlock()
	return pipelineLogSinks.sinks[id]
}

// CapturePipelineLogs captures the harness's logs for one run into a single
// <prefix>.debug.log under logsDir.
//
// The logs are buffered in memory and written by the returned func, one section
// per pipeline, headed with name. The returned context tags the run with
// prefix, and the capture takes only the records logged with a context
// carrying that tag, so concurrent runs never mix. It writes nothing unless
// enabled. Records reach it through the default logger once QuietLogs has
// installed its handler.
func CapturePipelineLogs(ctx context.Context, logsDir, prefix, name string, enabled bool) (context.Context, func()) {
	if !enabled {
		return ctx, func() {}
	}
	sink := &logSink{buffers: map[string][]string{}}
	pipelineLogSinks.mu.Lock()
	pipelineLogSinks.sinks[prefix] = sink
	pipelineLogSinks.mu.Unlock()
	finish := func() {
		pipelineLogSinks.mu.Lock()
		if pipelineLogSinks.sinks[prefix] == sink {
			delete(pipelineLogSinks.sinks, prefix)
		}
		pipelineLogSinks.mu.Unlock()
		sink.mu.Lock()
		defer sink.mu.Unlock()
		var sections []string
		for _, category := range PipelineLogCategories {
			if lines := sink.buffers[category]; len(lines) > 0 {
				sections = append(sections, "===== "+PipelineLogLabels[category]+": "+name+" =====\n\n"+
					strings.Join(lines, ""))
			}
		}
		if len(sections) == 0 {
			return
		}
		if err := os.MkdirAll(logsDir, 0o750); err != nil {
			return
		}
		_ = os.WriteFile(filepath.Join(logsDir, prefix+".debug.log"), []byte(strings.Join(sections, "\n")), 0o600)
	}
	return withEvalRun(ctx, prefix), finish
}

// QuietLogs makes the default logger write nothing but the per-run captures
// (see CapturePipelineLogs), keeping the output clean for a caller's display,
// until the returned func restores the default logger and the log package's
// output as they were.
func QuietLogs() (restore func()) {
	prev, w, flags := slog.Default(), log.Writer(), log.Flags()
	slog.SetDefault(slog.New(&captureHandler{inner: slog.DiscardHandler}))
	return func() {
		slog.SetDefault(prev)
		log.SetOutput(w)
		log.SetFlags(flags)
	}
}

// captureHandler hands each record logged under a run's capture to the
// capture, and every record to inner as inner would take it.
type captureHandler struct {
	inner slog.Handler
	// derive replays the attributes and groups added to the handler on the
	// one formatting a captured record.
	derive []func(slog.Handler) slog.Handler
}

// Enabled takes every record under a capture, and the rest as inner does.
func (h *captureHandler) Enabled(ctx context.Context, level slog.Level) bool {
	return sinkFor(ctx) != nil || h.inner.Enabled(ctx, level)
}

// Handle captures the record under a run's capture, and passes it to inner.
func (h *captureHandler) Handle(ctx context.Context, r slog.Record) error {
	if sink := sinkFor(ctx); sink != nil {
		category, _ := ctx.Value(evalPipelineKey).(string)
		if category == "" {
			category = PipelineHarness
		}
		var buf bytes.Buffer
		var f slog.Handler = slog.NewTextHandler(&buf, &slog.HandlerOptions{Level: slog.LevelDebug})
		for _, d := range h.derive {
			f = d(f)
		}
		if err := f.Handle(ctx, r); err == nil {
			sink.add(category, buf.String())
		}
	}
	if h.inner.Enabled(ctx, r.Level) {
		return h.inner.Handle(ctx, r)
	}
	return nil
}

// WithAttrs is the handler with attrs added, on inner and on the captures.
func (h *captureHandler) WithAttrs(attrs []slog.Attr) slog.Handler {
	return h.with(func(x slog.Handler) slog.Handler { return x.WithAttrs(attrs) })
}

// WithGroup is the handler with a group opened, on inner and on the captures.
func (h *captureHandler) WithGroup(name string) slog.Handler {
	return h.with(func(x slog.Handler) slog.Handler { return x.WithGroup(name) })
}

func (h *captureHandler) with(d func(slog.Handler) slog.Handler) slog.Handler {
	derive := make([]func(slog.Handler) slog.Handler, len(h.derive), len(h.derive)+1)
	copy(derive, h.derive)
	return &captureHandler{inner: d(h.inner), derive: append(derive, d)}
}
