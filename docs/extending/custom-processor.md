---
title: Writing a processor
weight: 1
---

# Writing a processor

Most things you want to add to a bot are a processor: logging a transcript,
redacting card numbers before they reach the LLM, hanging up on a keyword,
counting words. A processor is a struct embedding `*processor.Base` with one
method.

## The minimum

```go
package myproc

import (
    "context"

    "github.com/gojargo/jargo/frames"
    "github.com/gojargo/jargo/processor"
)

type Logger struct {
    *processor.Base
}

func NewLogger() *Logger {
    p := &Logger{}
    p.Base = processor.New("Logger", p)   // note: p, not nil
    return p
}

func (p *Logger) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
    if err := p.Base.ProcessFrame(ctx, f, dir); err != nil {
        return err
    }
    if t, ok := f.(*frames.TranscriptionFrame); ok {
        slog.Info("user said", "text", t.Text)
    }
    return p.PushFrame(ctx, f, dir)
}
```

Drop it in anywhere:

```go
pipeline.New(t.Input(), stt, myproc.NewLogger(), agg.User(), llm, tts, t.Output())
```

## The three rules

**1. Pass `self` to `processor.New`.** The base needs the concrete value to
dispatch to *your* `ProcessFrame`. Pass `nil` and you get a pass-through that
silently ignores your method.

**2. Call `p.Base.ProcessFrame` first.** The base handles `StartFrame`,
`InterruptionFrame` and `CancelFrame`. Skip it and your processor never starts its
process goroutine and never responds to barge-in.

**3. Always push, unless you mean to drop.** A frame you neither push nor
deliberately consume disappears. Dropping a `StartFrame` or `EndFrame` will hang
the pipeline, because the `Task` waits for those to complete the round trip.

## Transforming frames

Rewrite the payload and push the same frame on:

```go
func (p *Redactor) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
    if err := p.Base.ProcessFrame(ctx, f, dir); err != nil {
        return err
    }
    if t, ok := f.(*frames.TranscriptionFrame); ok {
        t.Text = cardNumbers.ReplaceAllString(t.Text, "[redacted]")
    }
    return p.PushFrame(ctx, f, dir)
}
```

Mutating before the push is fine; you own the frame until then. After pushing,
the frame belongs to the next processor; touching it is a data race.

## Emitting new frames

```go
func (p *Greeter) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
    if err := p.Base.ProcessFrame(ctx, f, dir); err != nil {
        return err
    }
    if _, ok := f.(*frames.StartFrame); ok {
        // Forward the StartFrame first; nothing may be pushed before it.
        if err := p.PushFrame(ctx, f, dir); err != nil {
            return err
        }
        return p.PushFrame(ctx, frames.NewTTSSpeakFrame("Hello."), processor.Downstream)
    }
    return p.PushFrame(ctx, f, dir)
}
```

Order matters here: `PushFrame` **drops anything pushed before the `StartFrame`**
and logs an error. Forward it, then emit.

## Dropping frames

Return without pushing. To drop only in one direction, use the built-in rather
than writing your own:

```go
processor.NewFunctionFilter("no-interim", processor.Downstream, func(f frames.Frame) bool {
    _, interim := f.(*frames.InterimTranscriptionFrame)
    return !interim
})
```

## Concurrency: the part that bites

**Your `ProcessFrame` runs on two different goroutines.** System frames are
handled on the input goroutine; data and control frames on the process goroutine.
Any state touched by both needs a mutex:

```go
type Counter struct {
    *processor.Base
    mu sync.Mutex
    n  int
}

func (p *Counter) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
    if err := p.Base.ProcessFrame(ctx, f, dir); err != nil {
        return err
    }
    p.mu.Lock()
    p.n++
    p.mu.Unlock()
    return p.PushFrame(ctx, f, dir)
}
```

Run your tests with `-race`. See [Processors](../concepts/processors.md#two-goroutines-per-processor).

## Honor `ctx`

`ctx` is canceled on interruption. Anything slow must respect it, or barge-in
stalls up to three seconds while the pipeline waits for you:

```go
select {
case res := <-p.slowCall(f):
    return p.PushFrame(ctx, res, dir)
case <-ctx.Done():
    return ctx.Err()          // abandon the work; the user is talking
}
```

## Setup and Cleanup

Override to own resources, and **always call the base**:

```go
func (p *Recorder) Setup(ctx context.Context, s processor.Setup) error {
    if err := p.Base.Setup(ctx, s); err != nil {
        return err
    }
    f, err := os.Create(p.path)
    if err != nil {
        return err
    }
    p.file = f
    return nil
}

func (p *Recorder) Cleanup(ctx context.Context) error {
    if p.file != nil {
        _ = p.file.Close()
    }
    return p.Base.Cleanup(ctx)
}
```

`Setup` runs before any frame arrives; `Clock()` is available after it. Do not
push frames from `Setup`, because the pipeline has not started. Use the `StartFrame`.

## Reporting errors

```go
p.PushError(ctx, "failed to reach the order service", err, false)
```

That builds an `ErrorFrame`, logs it, and pushes it upstream. Pass `true` for
fatal, which cancels the task. Returning an error from `ProcessFrame` does the
non-fatal version automatically.

## Ending the session

A processor cannot reach the `Task` directly. Push a worker frame:

```go
p.PushFrame(ctx, frames.NewEndWorkerFrame(), processor.Downstream)   // graceful
p.PushFrame(ctx, frames.NewCancelWorkerFrame(), processor.Downstream) // immediate
```

See [worker frames](../concepts/pipeline.md#worker-frames-talking-back-to-the-task).

## Direct mode

For a pure router that buffers nothing, skip the goroutines:

```go
p.Base = processor.New("Router", p, processor.WithDirectMode())
```

It then runs on the caller's goroutine and ignores interruptions. Correct for
`FunctionFilter`; **wrong** for anything holding queued work, since there would be
nothing to flush on barge-in.

## A custom frame

```go
type OrderLookupFrame struct {
    frames.BaseControlFrame
    OrderID string
}

func NewOrderLookupFrame(id string) *OrderLookupFrame {
    return &OrderLookupFrame{
        BaseControlFrame: frames.NewBaseControlFrame("OrderLookupFrame"),
        OrderID:          id,
    }
}
```

Pick the category deliberately: it decides whether an interruption drops your
frame. Add `frames.UninterruptibleMixin` if the work must survive one. See
[Frames](../concepts/frames.md#the-three-categories).

## Testing

Processors test well in isolation: link a sink, queue frames, assert what comes
out.

```go
func TestRedactor(t *testing.T) {
    p := NewRedactor()
    var got []frames.Frame
    sink := processor.NewSink("sink", func(_ context.Context, f frames.Frame, _ processor.Direction) error {
        got = append(got, f)
        return nil
    })
    p.Link(sink)

    ctx := context.Background()
    if err := p.Setup(ctx, processor.Setup{Clock: clock.NewSystem()}); err != nil {
        t.Fatal(err)
    }
    defer p.Cleanup(ctx)

    p.QueueFrame(ctx, frames.NewStartFrame(), processor.Downstream)
    p.QueueFrame(ctx, frames.NewTranscriptionFrame("my card is 4111111111111111", "user", ""),
        processor.Downstream)
    // … wait for the frame to arrive, then assert on got
}
```

Because `QueueFrame` is asynchronous, synchronize before asserting, with a channel the
sink signals, or `pipeline.Task.Flush` if you are driving a whole pipeline. See
`processor/*_test.go` for the patterns in use.
