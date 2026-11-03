package novasonic

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/clock"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
)

// errFakeClosed is what a fake stream's Recv returns once it is closed.
var errFakeClosed = errors.New("fake stream closed")

// fakeStream is a session stream recording what the service sends and
// delivering what a test has the model send.
type fakeStream struct {
	mu          sync.Mutex
	sent        []map[string]any
	inputClosed bool
	in          chan []byte
	done        chan struct{}
	closeOnce   sync.Once
}

func newFakeStream() *fakeStream {
	return &fakeStream{in: make(chan []byte, 64), done: make(chan struct{})}
}

func (f *fakeStream) Send(_ context.Context, data []byte) error {
	var m map[string]any
	if err := json.Unmarshal(data, &m); err != nil {
		return err
	}
	f.mu.Lock()
	f.sent = append(f.sent, m)
	f.mu.Unlock()
	return nil
}

func (f *fakeStream) Recv() ([]byte, error) {
	select {
	case data := <-f.in:
		return data, nil
	case <-f.done:
		return nil, errFakeClosed
	}
}

func (f *fakeStream) CloseInput() error {
	f.mu.Lock()
	f.inputClosed = true
	f.mu.Unlock()
	return nil
}

func (f *fakeStream) Close() error {
	f.closeOnce.Do(func() { close(f.done) })
	return nil
}

// end has the model end the stream.
func (f *fakeStream) end() { f.in <- nil }

// events are the events sent, as their names and bodies.
func (f *fakeStream) events() []sentEvent {
	f.mu.Lock()
	defer f.mu.Unlock()
	out := make([]sentEvent, 0, len(f.sent))
	for _, m := range f.sent {
		envelope, _ := m["event"].(map[string]any)
		for name, body := range envelope {
			b, _ := body.(map[string]any)
			out = append(out, sentEvent{name: name, body: b})
		}
	}
	return out
}

// names are the names of the events sent, in order.
func (f *fakeStream) names() []string {
	var out []string
	for _, e := range f.events() {
		out = append(out, e.name)
	}
	return out
}

// sentEvent is one event the service sent.
type sentEvent struct {
	name string
	body map[string]any
}

// sentText is one text block the service sent: its role, its text and whether
// it asked the model to answer.
type sentText struct {
	role        string
	text        string
	interactive bool
}

// texts are the text blocks sent, in order.
func (f *fakeStream) texts() []sentText {
	var out []sentText
	var role string
	var interactive bool
	for _, e := range f.events() {
		switch e.name {
		case "contentStart":
			if e.body["type"] == "TEXT" {
				role, _ = e.body["role"].(string)
				interactive, _ = e.body["interactive"].(bool)
			}
		case "textInput":
			text, _ := e.body[keyContent].(string)
			out = append(out, sentText{role: role, text: text, interactive: interactive})
		}
	}
	return out
}

// fakeClient opens fake streams.
type fakeClient struct {
	mu      sync.Mutex
	streams []*fakeStream
	model   string
}

func (c *fakeClient) OpenStream(_ context.Context, model string) (sessionStream, error) {
	st := newFakeStream()
	c.mu.Lock()
	c.streams = append(c.streams, st)
	c.model = model
	c.mu.Unlock()
	return st, nil
}

// stream is the i-th stream opened, waiting for it.
func (c *fakeClient) stream(t *testing.T, i int) *fakeStream {
	t.Helper()
	var st *fakeStream
	waitFor(t, func() bool {
		c.mu.Lock()
		defer c.mu.Unlock()
		if len(c.streams) > i {
			st = c.streams[i]
			return true
		}
		return false
	})
	return st
}

// opened is how many streams have been opened.
func (c *fakeClient) opened() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return len(c.streams)
}

// recorder records the frames pushed to it.
type recorder struct {
	*processor.Base
	mu  sync.Mutex
	got []frames.Frame
}

func newRecorder() *recorder {
	r := &recorder{}
	r.Base = processor.New("Recorder", r, processor.WithDirectMode())
	return r
}

func (r *recorder) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := r.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	r.mu.Lock()
	r.got = append(r.got, f)
	r.mu.Unlock()
	return nil
}

func (r *recorder) frames() []frames.Frame {
	r.mu.Lock()
	defer r.mu.Unlock()
	return append([]frames.Frame(nil), r.got...)
}

// names are the names of the frames recorded, leaving out the ones every
// pipeline carries.
func (r *recorder) names() []string {
	var out []string
	for _, f := range r.frames() {
		switch f.(type) {
		case *frames.MetricsFrame, *frames.StartFrame, *frames.LLMContextFrame:
			continue
		}
		name, _, _ := strings.Cut(f.Name(), "#")
		out = append(out, name)
	}
	return out
}

// harness is a service set up against a fake client, with recorders on either
// side.
type harness struct {
	s      *Service
	client *fakeClient
	down   *recorder
	up     *recorder
}

// newHarness builds a service from cfg against a fake client and sets it up,
// which opens the first session.
func newHarness(t *testing.T, cfg Config) *harness {
	t.Helper()
	h := &harness{client: &fakeClient{}, down: newRecorder(), up: newRecorder()}
	h.s = New(cfg)
	h.s.newClient = func(context.Context, Config) (sessionClient, error) { return h.client, nil }
	h.up.Link(h.s)
	h.s.Link(h.down)
	ctx := context.Background()
	if err := h.s.Setup(ctx, processor.Setup{
		Clock: clock.NewSystem(), EnableMetrics: true, EnableUsageMetrics: true,
	}); err != nil {
		t.Fatalf("setup: %v", err)
	}
	t.Cleanup(func() { _ = h.s.Cleanup(ctx) })
	return h
}

// seed reports a conversation to the service, which sets the session up.
func (h *harness) seed(t *testing.T, convo *frames.LLMContext) {
	t.Helper()
	if err := h.s.ProcessFrame(context.Background(), frames.NewLLMContextFrame(convo), processor.Downstream); err != nil {
		t.Fatalf("context: %v", err)
	}
}

// drive has the service handle one event the model sent.
func (h *harness) drive(t *testing.T, name string, body map[string]any) {
	t.Helper()
	raw, err := json.Marshal(body)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if err := h.s.handleEvent(context.Background(), map[string]json.RawMessage{name: raw}); err != nil {
		t.Fatalf("event %s: %v", name, err)
	}
}

// waitFor polls cond until it holds or a deadline passes.
func waitFor(t *testing.T, cond func() bool) {
	t.Helper()
	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		if cond() {
			return
		}
		time.Sleep(5 * time.Millisecond)
	}
	t.Fatal("condition not met in time")
}

// The content blocks the model sends, as event bodies.

func contentStartBody(typ, role, stage string) map[string]any {
	body := map[string]any{"type": typ, "role": role}
	if stage != "" {
		body["additionalModelFields"] = `{"generationStage":"` + stage + `"}`
	}
	return body
}

func textOutputBody(text string) map[string]any { return map[string]any{"content": text} }

func contentEndBody(stopReason string) map[string]any {
	return map[string]any{"stopReason": stopReason}
}

// waitTimeout fires after the time a test waits for something.
func waitTimeout() <-chan time.Time { return time.After(3 * time.Second) }

// errorsAs is errors.As.
func errorsAs(err error, target any) bool { return errors.As(err, target) }
