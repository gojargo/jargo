package processor

import (
	"context"
	"sync"

	"github.com/gojargo/jargo/frames"
)

// FrameTransformer rewrites a frame on its way from a producer to a consumer.
type FrameTransformer func(frames.Frame) frames.Frame

// IdentityTransformer passes a frame along unchanged. It is what a producer or
// consumer built without a transformer uses.
func IdentityTransformer(f frames.Frame) frames.Frame { return f }

// frameQueue is an unbounded first-in-first-out queue of frames with a single
// consumer. Producers never block, so a slow consumer cannot stall the frame
// path the producer sits on.
type frameQueue struct {
	mu     sync.Mutex
	items  []frames.Frame
	notify chan struct{}
}

func newFrameQueue() *frameQueue {
	return &frameQueue{notify: make(chan struct{}, 1)}
}

// put appends a frame. It never blocks.
func (q *frameQueue) put(f frames.Frame) {
	q.mu.Lock()
	q.items = append(q.items, f)
	q.mu.Unlock()
	select {
	case q.notify <- struct{}{}:
	default:
	}
}

// get returns the next frame, blocking until one arrives or ctx ends. It reports
// false only when ctx ended.
func (q *frameQueue) get(ctx context.Context) (frames.Frame, bool) {
	for {
		q.mu.Lock()
		if len(q.items) > 0 {
			f := q.items[0]
			q.items = q.items[1:]
			q.mu.Unlock()
			return f, true
		}
		q.mu.Unlock()

		select {
		case <-q.notify:
		case <-ctx.Done():
			return nil, false
		}
	}
}

// ProducerProcessor picks frames out of the stream passing through it and hands
// copies to any number of consumers elsewhere in the pipeline.
//
// It is how a frame reaches a part of the pipeline it does not flow through: a
// branch of a ParallelPipeline can watch what another branch is saying without
// being wired to it. The frames it picks are chosen by a predicate, and may be
// rewritten on the way out.
type ProducerProcessor struct {
	*Base
	filter      func(frames.Frame) bool
	transformer FrameTransformer
	passthrough bool

	mu        sync.Mutex
	consumers []*frameQueue
}

// ProducerOption configures a ProducerProcessor.
type ProducerOption func(*ProducerProcessor)

// WithProducerTransformer rewrites each frame on its way to the consumers. The
// frame carrying on down the pipeline is not the rewritten one.
func WithProducerTransformer(t FrameTransformer) ProducerOption {
	return func(p *ProducerProcessor) { p.transformer = t }
}

// WithoutPassthrough keeps a frame the predicate picked from carrying on down
// the pipeline, so it reaches the consumers only. A frame the predicate did not
// pick still passes.
func WithoutPassthrough() ProducerOption {
	return func(p *ProducerProcessor) { p.passthrough = false }
}

// NewProducerProcessor builds a producer handing every frame filter picks to its
// consumers. By default the frame carries on down the pipeline as well.
func NewProducerProcessor(
	name string, filter func(frames.Frame) bool, opts ...ProducerOption,
) *ProducerProcessor {
	p := &ProducerProcessor{
		filter:      filter,
		transformer: IdentityTransformer,
		passthrough: true,
	}
	for _, opt := range opts {
		opt(p)
	}
	p.Base = New(name, p, WithDirectMode())
	return p
}

// AddConsumer registers a consumer and returns the queue its frames arrive on.
// A ConsumerProcessor calls it for you when the pipeline starts.
func (p *ProducerProcessor) AddConsumer() *frameQueue {
	q := newFrameQueue()
	p.mu.Lock()
	p.consumers = append(p.consumers, q)
	p.mu.Unlock()
	return q
}

// ProcessFrame implements Processor.
func (p *ProducerProcessor) ProcessFrame(
	ctx context.Context, frame frames.Frame, dir Direction,
) error {
	if err := p.Base.ProcessFrame(ctx, frame, dir); err != nil {
		return err
	}
	if p.filter(frame) {
		p.produce(frame)
		if !p.passthrough {
			return nil
		}
	}
	return p.PushFrame(ctx, frame, dir)
}

// produce hands the frame to every consumer, rewritten for each of them.
func (p *ProducerProcessor) produce(frame frames.Frame) {
	p.mu.Lock()
	consumers := make([]*frameQueue, len(p.consumers))
	copy(consumers, p.consumers)
	p.mu.Unlock()

	for _, q := range consumers {
		q.put(p.transformer(frame))
	}
}

// ConsumerProcessor puts the frames a ProducerProcessor picked into the pipeline
// at the point it sits, while passing everything reaching it along untouched.
type ConsumerProcessor struct {
	*Base
	producer    *ProducerProcessor
	transformer FrameTransformer
	dir         Direction

	mu     sync.Mutex
	cancel context.CancelFunc
	wg     sync.WaitGroup
}

// ConsumerOption configures a ConsumerProcessor.
type ConsumerOption func(*ConsumerProcessor)

// WithConsumerTransformer rewrites each frame before it is put into the
// pipeline.
func WithConsumerTransformer(t FrameTransformer) ConsumerOption {
	return func(c *ConsumerProcessor) { c.transformer = t }
}

// WithConsumerDirection sets which way the consumed frames travel. They go
// downstream by default.
func WithConsumerDirection(d Direction) ConsumerOption {
	return func(c *ConsumerProcessor) { c.dir = d }
}

// NewConsumerProcessor builds a consumer of the frames producer picks.
func NewConsumerProcessor(
	name string, producer *ProducerProcessor, opts ...ConsumerOption,
) *ConsumerProcessor {
	c := &ConsumerProcessor{
		producer:    producer,
		transformer: IdentityTransformer,
		dir:         Downstream,
	}
	for _, opt := range opts {
		opt(c)
	}
	c.Base = New(name, c)
	return c
}

// ProcessFrame implements Processor.
func (c *ConsumerProcessor) ProcessFrame(
	ctx context.Context, frame frames.Frame, dir Direction,
) error {
	if err := c.Base.ProcessFrame(ctx, frame, dir); err != nil {
		return err
	}
	switch frame.(type) {
	case *frames.StartFrame:
		c.start()
	case *frames.EndFrame, *frames.CancelFrame:
		c.stop()
	}
	return c.PushFrame(ctx, frame, dir)
}

// Cleanup implements Processor.
func (c *ConsumerProcessor) Cleanup(ctx context.Context) error {
	c.stop()
	return c.Base.Cleanup(ctx)
}

// start registers with the producer and begins reading what it hands over.
func (c *ConsumerProcessor) start() {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.cancel != nil {
		return
	}
	queue := c.producer.AddConsumer()
	ctx, cancel := context.WithCancel(context.Background())
	c.cancel = cancel
	c.wg.Add(1)
	go c.consume(ctx, queue)
}

// stop ends the reading goroutine and waits for it to finish.
func (c *ConsumerProcessor) stop() {
	c.mu.Lock()
	cancel := c.cancel
	c.cancel = nil
	c.mu.Unlock()
	if cancel == nil {
		return
	}
	cancel()
	c.wg.Wait()
}

// consume reads what the producer hands over and queues it into the pipeline.
func (c *ConsumerProcessor) consume(ctx context.Context, queue *frameQueue) {
	defer c.wg.Done()
	for {
		frame, ok := queue.get(ctx)
		if !ok {
			return
		}
		// Queued rather than pushed, so a consumed frame takes the same path
		// through this processor as one that arrived on the pipeline, and lands
		// in order with them.
		_ = c.QueueFrame(ctx, c.transformer(frame), c.dir)
	}
}
