package bus_test

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/bus"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
)

// Upstream ships no test for the bridge processors, so these are ours. They
// cover the filtering rules, which is where all of the logic is: what may cross
// the bus, what must stay local, and which arriving frames a bridge accepts.

// sink records the frames pushed past it.
type sink struct {
	*processor.Base
	mu  sync.Mutex
	got []frames.Frame
}

// frames returns what reached the sink, once delivery has settled: the
// processors carry frames on goroutines of their own.
func (s *sink) frames() []frames.Frame {
	last := -1
	for range 100 {
		s.mu.Lock()
		n := len(s.got)
		s.mu.Unlock()
		if n == last {
			break
		}
		last = n
		time.Sleep(5 * time.Millisecond)
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]frames.Frame(nil), s.got...)
}

// reset drops what has been recorded so far.
func (s *sink) reset() {
	s.mu.Lock()
	s.got = nil
	s.mu.Unlock()
}

func newSink() *sink {
	s := &sink{}
	s.Base = processor.New("Sink", s)
	return s
}

func (s *sink) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := s.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	s.mu.Lock()
	s.got = append(s.got, f)
	s.mu.Unlock()
	return s.PushFrame(ctx, f, dir)
}

// busSpy records everything published, so a test can tell what crossed.
type busSpy struct {
	*bus.Bus
	sent []bus.Message
}

func newBusSpy() *busSpy {
	b := &busSpy{}
	b.Bus = bus.New(b)
	return b
}

func (b *busSpy) Publish(_ context.Context, m bus.Message) {
	b.sent = append(b.sent, m)
	b.OnMessageReceived(m)
}

// bridgeUnder builds a bridge with a sink behind it, started as a pipeline
// would start it.
func bridgeUnder(t *testing.T, cfg bus.BridgeConfig) (*bus.BridgeProcessor, *sink) {
	t.Helper()
	p := bus.NewBridgeProcessor(cfg)
	out := newSink()
	p.Link(out)
	for _, x := range []processor.Processor{p, out} {
		if err := x.Setup(t.Context(), processor.Setup{}); err != nil {
			t.Fatalf("Setup: %v", err)
		}
	}
	t.Cleanup(func() {
		for _, x := range []processor.Processor{p, out} {
			_ = x.Cleanup(context.Background())
		}
	})
	if err := p.ProcessFrame(t.Context(), frames.NewStartFrame(), processor.Downstream); err != nil {
		t.Fatalf("StartFrame: %v", err)
	}
	out.frames()
	out.reset()
	return p, out
}

// A frame reaching the bridge leaves the local pipeline and crosses the bus.
func TestBridgeSendsFramesAcrossTheBus(t *testing.T) {
	b := newBusSpy()
	p, out := bridgeUnder(t, bus.BridgeConfig{Bus: b.Bus, WorkerName: "worker_a"})

	if err := p.ProcessFrame(t.Context(), frames.NewLLMTextFrame("hello"), processor.Downstream); err != nil {
		t.Fatalf("ProcessFrame: %v", err)
	}

	if len(out.frames()) != 0 {
		t.Errorf("%d frames continued locally, want none: the bridge consumes them", len(out.frames()))
	}
	if len(b.sent) != 1 {
		t.Fatalf("%d messages crossed the bus, want 1", len(b.sent))
	}
	fm, ok := b.sent[0].(*bus.FrameMessage)
	if !ok || fm.Source() != "worker_a" {
		t.Errorf("the bus carried %v, want a frame message from worker_a", b.sent[0])
	}
}

// A pipeline's lifecycle is its own: those frames never cross, or a worker
// would shut down with another.
func TestBridgeKeepsLifecycleFramesLocal(t *testing.T) {
	// Each gets a bridge of its own: an end or a cancel shuts the processor
	// down, so they cannot be sent one after another through the same one.
	lifecycle := map[string]func() frames.Frame{
		"end":    func() frames.Frame { return frames.NewEndFrame() },
		"stop":   func() frames.Frame { return frames.NewStopFrame() },
		"cancel": func() frames.Frame { return frames.NewCancelFrame() },
	}

	for name, build := range lifecycle {
		t.Run(name, func(t *testing.T) {
			b := newBusSpy()
			p, out := bridgeUnder(t, bus.BridgeConfig{Bus: b.Bus, WorkerName: "worker_a"})

			if err := p.ProcessFrame(t.Context(), build(), processor.Downstream); err != nil {
				t.Fatalf("ProcessFrame: %v", err)
			}

			if len(out.frames()) != 1 {
				t.Errorf("%d frames continued locally, want the lifecycle frame forwarded", len(out.frames()))
			}
			if len(b.sent) != 0 {
				t.Errorf("%d lifecycle frames crossed the bus, want none", len(b.sent))
			}
		})
	}
}

// An urgent transport message has to reach the transport even when no child
// worker is active to carry it, so it goes straight on.
func TestBridgePassesUrgentTransportMessagesThrough(t *testing.T) {
	b := newBusSpy()
	p, out := bridgeUnder(t, bus.BridgeConfig{Bus: b.Bus, WorkerName: "worker_a"})

	f := frames.NewOutputTransportMessageUrgentFrame(nil)
	if err := p.ProcessFrame(t.Context(), f, processor.Downstream); err != nil {
		t.Fatalf("ProcessFrame: %v", err)
	}

	if len(out.frames()) != 1 {
		t.Errorf("%d frames continued locally, want the urgent message to pass through", len(out.frames()))
	}
	if len(b.sent) != 0 {
		t.Errorf("%d frames crossed the bus, want none", len(b.sent))
	}
}

func TestBridgeKeepsExcludedFramesLocal(t *testing.T) {
	b := newBusSpy()
	p, out := bridgeUnder(t, bus.BridgeConfig{
		Bus:           b.Bus,
		WorkerName:    "worker_a",
		ExcludeFrames: []frames.Frame{&frames.LLMTextFrame{}},
	})

	if err := p.ProcessFrame(t.Context(), frames.NewLLMTextFrame("hello"), processor.Downstream); err != nil {
		t.Fatalf("ProcessFrame: %v", err)
	}
	if err := p.ProcessFrame(t.Context(), frames.NewTextFrame("other"), processor.Downstream); err != nil {
		t.Fatalf("ProcessFrame: %v", err)
	}

	if len(out.frames()) != 1 {
		t.Errorf("%d frames continued locally, want only the excluded one", len(out.frames()))
	}
	if len(b.sent) != 1 {
		t.Errorf("%d frames crossed the bus, want only the one not excluded", len(b.sent))
	}
}

// A frame arriving from another worker is pushed into this pipeline.
func TestBridgeInjectsFramesArrivingFromTheBus(t *testing.T) {
	b := newBusSpy()
	p, out := bridgeUnder(t, bus.BridgeConfig{Bus: b.Bus, WorkerName: "worker_a"})

	m := &bus.FrameMessage{Frame: frames.NewLLMTextFrame("hi"), Direction: processor.Downstream}
	m.From = "worker_b"
	p.OnBusMessage(t.Context(), m)

	if len(out.frames()) != 1 {
		t.Fatalf("%d frames were injected, want 1", len(out.frames()))
	}
	if tf, ok := out.frames()[0].(*frames.LLMTextFrame); !ok || tf.Text != "hi" {
		t.Errorf("the injected frame was %v, want the one that arrived", out.frames()[0])
	}
}

// The rules that decide which arriving frames a bridge accepts.
func TestBridgeIgnoresFramesItShouldNotAccept(t *testing.T) {
	cases := []struct {
		name string
		cfg  bus.BridgeConfig
		msg  func() *bus.FrameMessage
	}{
		{
			name: "its own frames come back",
			cfg:  bus.BridgeConfig{WorkerName: "worker_a"},
			msg: func() *bus.FrameMessage {
				m := &bus.FrameMessage{Frame: frames.NewLLMTextFrame("x"), Direction: processor.Downstream}
				m.From = "worker_a"
				return m
			},
		},
		{
			name: "tagged for another bridge",
			cfg:  bus.BridgeConfig{WorkerName: "worker_a", Bridge: "mine"},
			msg: func() *bus.FrameMessage {
				m := &bus.FrameMessage{
					Frame: frames.NewLLMTextFrame("x"), Direction: processor.Downstream, Bridge: "other",
				}
				m.From = "worker_b"
				return m
			},
		},
		{
			name: "from a worker this bridge does not pair with",
			cfg:  bus.BridgeConfig{WorkerName: "worker_a", TargetWorker: "worker_b"},
			msg: func() *bus.FrameMessage {
				m := &bus.FrameMessage{Frame: frames.NewLLMTextFrame("x"), Direction: processor.Downstream}
				m.From = "worker_c"
				return m
			},
		},
		{
			name: "addressed to someone else",
			cfg:  bus.BridgeConfig{WorkerName: "worker_a"},
			msg: func() *bus.FrameMessage {
				m := &bus.FrameMessage{Frame: frames.NewLLMTextFrame("x"), Direction: processor.Downstream}
				m.From, m.To = "worker_b", "worker_c"
				return m
			},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			b := newBusSpy()
			cfg := tc.cfg
			cfg.Bus = b.Bus
			p, out := bridgeUnder(t, cfg)

			p.OnBusMessage(t.Context(), tc.msg())

			if len(out.frames()) != 0 {
				t.Errorf("%d frames were injected, want none", len(out.frames()))
			}
		})
	}
}

// A message that is not a frame message is not a bridge's business.
func TestBridgeIgnoresOtherMessages(t *testing.T) {
	b := newBusSpy()
	p, out := bridgeUnder(t, bus.BridgeConfig{Bus: b.Bus, WorkerName: "worker_a"})

	m := &bus.EndMessage{}
	m.From = "runner"
	p.OnBusMessage(t.Context(), m)

	if len(out.frames()) != 0 {
		t.Errorf("%d frames were injected for a non-frame message, want none", len(out.frames()))
	}
}
