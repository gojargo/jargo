package processor

import (
	"context"
	"sync"

	"github.com/gojargo/jargo/frames"
)

// FrameEncoder turns a frame into the bytes that stand for it on a wire, or nil
// for a frame it does not send.
//
// It is the part of a transport serializer this processor needs, declared here
// so the processor does not depend on the transport layer. Any serializer with
// the same method satisfies it.
type FrameEncoder interface {
	Serialize(f frames.Frame) ([]byte, error)
}

// AsyncGeneratorProcessor passes every frame through unchanged and, alongside,
// serializes it and hands the bytes to whoever is reading.
//
// It is how something outside the pipeline follows what the pipeline is saying:
// a caller ranges over Frames and receives each serialized frame as it goes by,
// while the pipeline itself is unaffected. The stream ends when the pipeline
// does.
type AsyncGeneratorProcessor struct {
	*Base
	encoder FrameEncoder

	mu     sync.Mutex
	items  [][]byte
	done   bool
	notify chan struct{}
}

// NewAsyncGeneratorProcessor builds a processor serializing what passes through
// it with encoder.
func NewAsyncGeneratorProcessor(name string, encoder FrameEncoder) *AsyncGeneratorProcessor {
	p := &AsyncGeneratorProcessor{encoder: encoder, notify: make(chan struct{}, 1)}
	p.Base = New(name, p)
	return p
}

// ProcessFrame pushes the frame on and queues its serialized form.
func (p *AsyncGeneratorProcessor) ProcessFrame(ctx context.Context, f frames.Frame, dir Direction) error {
	if err := p.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	if err := p.PushFrame(ctx, f, dir); err != nil {
		return err
	}

	switch f.(type) {
	case *frames.EndFrame, *frames.CancelFrame:
		// The pipeline is over, so the stream is too. The frame itself is not
		// serialized: it is the end signal, not something to send on.
		p.finish()
		return nil
	}

	data, err := p.encoder.Serialize(f)
	if err != nil {
		return err
	}
	if len(data) > 0 {
		p.put(data)
	}
	return nil
}

// Frames yields each serialized frame as it passes, ending when the pipeline
// does or when ctx is canceled. It never blocks the frame path: a reader falling
// behind queues rather than holding the pipeline up.
func (p *AsyncGeneratorProcessor) Frames(ctx context.Context) func(func([]byte) bool) {
	return func(yield func([]byte) bool) {
		for {
			data, ok := p.get(ctx)
			if !ok {
				return
			}
			if !yield(data) {
				return
			}
		}
	}
}

// put appends a serialized frame and wakes the reader. It never blocks.
func (p *AsyncGeneratorProcessor) put(data []byte) {
	p.mu.Lock()
	p.items = append(p.items, data)
	p.mu.Unlock()
	p.wake()
}

// finish marks the stream ended, so a reader that has drained what is queued
// stops rather than waiting for more.
func (p *AsyncGeneratorProcessor) finish() {
	p.mu.Lock()
	p.done = true
	p.mu.Unlock()
	p.wake()
}

func (p *AsyncGeneratorProcessor) wake() {
	select {
	case p.notify <- struct{}{}:
	default:
	}
}

// get returns the next serialized frame, waiting for one. It reports false once
// the stream has ended and everything queued has been read, or when ctx ends.
func (p *AsyncGeneratorProcessor) get(ctx context.Context) ([]byte, bool) {
	for {
		p.mu.Lock()
		if len(p.items) > 0 {
			data := p.items[0]
			p.items = p.items[1:]
			p.mu.Unlock()
			return data, true
		}
		done := p.done
		p.mu.Unlock()
		if done {
			return nil, false
		}

		select {
		case <-p.notify:
		case <-ctx.Done():
			return nil, false
		}
	}
}
