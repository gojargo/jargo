package bus_test

// Ported from upstream's bus serializer suite and its serializer source: every
// field of a frame crosses a bus that leaves the process, its base state
// (id, name, timestamp, metadata, transport routing) and a settings delta
// included, and a number keeps its kind.

import (
	"context"
	"reflect"
	"testing"
	"time"

	"github.com/gojargo/jargo/adapter"
	"github.com/gojargo/jargo/adapter/openai"
	"github.com/gojargo/jargo/bus"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/service/settings"
)

// frameMessage wraps a frame the way a bridge sends it.
func frameMessage(f frames.Frame) *bus.FrameMessage {
	m := &bus.FrameMessage{Frame: f, Direction: processor.Downstream}
	m.From = "controller"
	return m
}

// crossedFrame sends f across the serializer and returns what arrived.
func crossedFrame[T frames.Frame](t *testing.T, f frames.Frame) T {
	t.Helper()
	back, ok := roundTrip(t, frameMessage(f)).(*bus.FrameMessage)
	if !ok {
		t.Fatal("the message came back as something else")
	}
	got, ok := back.Frame.(T)
	if !ok {
		t.Fatalf("the frame came back as %T", back.Frame)
	}
	return got
}

// A frame arrives as itself: its id, name, timestamp, metadata and transport
// routing cross with it.
func TestFrameBaseStateCrosses(t *testing.T) {
	f := frames.NewTextFrame("hello")
	f.SetPTS(1234)
	f.Metadata()["config"] = "value"
	f.SetTransportSource("src-a")
	f.SetTransportDestination("dest-a")
	f.SetBroadcastSiblingID(77)
	f.SetInterruptible(false)

	got := crossedFrame[*frames.TextFrame](t, f)
	if got.ID() != f.ID() || got.Name() != f.Name() {
		t.Errorf("identity = %d %q, want %d %q", got.ID(), got.Name(), f.ID(), f.Name())
	}
	if pts, ok := got.PTS(); !ok || pts != 1234 {
		t.Errorf("pts = %d (%v), want 1234", pts, ok)
	}
	if got.Metadata()["config"] != "value" {
		t.Errorf("metadata = %v, want config=value", got.Metadata())
	}
	if got.TransportSource() != "src-a" || got.TransportDestination() != "dest-a" {
		t.Errorf("transport = %q -> %q, want src-a -> dest-a", got.TransportSource(), got.TransportDestination())
	}
	if sib, ok := got.BroadcastSiblingID(); !ok || sib != 77 {
		t.Errorf("broadcast sibling = %d (%v), want 77", sib, ok)
	}
	if frames.Interruptible(got) {
		t.Error("the frame set uninterruptible arrived interruptible")
	}
	if got.Text != "hello" {
		t.Errorf("text = %q, want hello", got.Text)
	}
}

// Two frames keep their own ids, so a parallel pipeline on the far side, which
// tells frames apart by id, delivers every one of them.
func TestBridgedFramesThroughAParallelPipeline(t *testing.T) {
	var arrived []frames.Frame
	for _, text := range []string{"one", "two", "three"} {
		arrived = append(arrived, crossedFrame[*frames.TextFrame](t, frames.NewTextFrame(text)))
	}
	if arrived[0].ID() == arrived[1].ID() {
		t.Fatalf("two frames arrived sharing id %d", arrived[0].ID())
	}

	sink := &textSink{}
	sink.Base = processor.New("Sink", sink)
	par, err := pipeline.NewParallel(
		[]processor.Processor{processor.NewIdentityFilter("A")},
		[]processor.Processor{processor.NewIdentityFilter("B")},
	)
	if err != nil {
		t.Fatal(err)
	}
	w := pipeline.NewWorker(pipeline.New(par, sink), pipeline.WorkerConfig{})
	done := make(chan error, 1)
	go func() { done <- w.Run(context.Background()) }()
	for _, f := range arrived {
		w.QueueFrame(f)
	}
	w.StopWhenDone()
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("pipeline did not end")
	}
	if want := []string{"one", "two", "three"}; !reflect.DeepEqual(sink.texts, want) {
		t.Errorf("sink saw %v, want %v", sink.texts, want)
	}
}

// textSink records the text frames reaching it.
type textSink struct {
	*processor.Base
	texts []string
}

func (s *textSink) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := s.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	if tf, ok := f.(*frames.TextFrame); ok {
		s.texts = append(s.texts, tf.Text)
	}
	return s.PushFrame(ctx, f, dir)
}

// test_llm_delta_clears_settings: a typed delta crosses typed, with the fields
// it cleared, and still applies.
func TestLLMDeltaClearsSettings(t *testing.T) {
	delta := &settings.LLM{Temperature: settings.Cleared[float64](), MaxTokens: settings.Cleared[int]()}
	got := crossedFrame[*frames.LLMUpdateSettingsFrame](t, frames.NewLLMUpdateSettingsFrame(delta))
	received, ok := got.Delta.(*settings.LLM)
	if !ok {
		t.Fatalf("delta arrived as %T, want *settings.LLM", got.Delta)
	}
	if received.Model.IsGiven() {
		t.Error("the model arrived given, want it left alone")
	}
	current := &settings.LLM{Temperature: settings.Set(0.7), MaxTokens: settings.Set(256)}
	current.Model = settings.Set("example-model")
	if _, err := settings.Apply(current, received); err != nil {
		t.Fatalf("apply: %v", err)
	}
	if _, ok := current.Temperature.Value(); ok {
		t.Error("temperature was not cleared")
	}
	if _, ok := current.MaxTokens.Value(); ok {
		t.Error("max_tokens was not cleared")
	}
	if current.Model.Or("") != "example-model" {
		t.Errorf("model = %q, want it untouched", current.Model.Or(""))
	}
}

// test_stt_delta_clears_language: a cleared language crosses as cleared.
func TestSTTDeltaClearsLanguage(t *testing.T) {
	delta := &settings.STT{Language: settings.Cleared[string]()}
	got := crossedFrame[*frames.STTUpdateSettingsFrame](t, frames.NewSTTUpdateSettingsFrame(delta))
	received, ok := got.Delta.(*settings.STT)
	if !ok {
		t.Fatalf("delta arrived as %T, want *settings.STT", got.Delta)
	}
	given, _ := settings.Given(received)
	if !reflect.DeepEqual(given, map[string]any{"language": nil}) {
		t.Errorf("given = %v, want only the language, cleared", given)
	}
}

// test_falsy_and_unspecified_settings: a zero is a value like any other.
func TestFalsyAndUnspecifiedSettings(t *testing.T) {
	delta := &settings.LLM{
		Temperature: settings.Set(0.0), MaxTokens: settings.Set(0),
		UseProviderThinkingDefault: settings.Set(false),
	}
	got := crossedFrame[*frames.LLMUpdateSettingsFrame](t, frames.NewLLMUpdateSettingsFrame(delta))
	want, _ := settings.Given(delta)
	given, err := settings.Given(got.Delta)
	if err != nil {
		t.Fatalf("delta arrived as %T: %v", got.Delta, err)
	}
	if !reflect.DeepEqual(given, want) {
		t.Errorf("given = %v, want %v", given, want)
	}
}

// A provider's own settings type the far end may not have crosses as the plain
// settings it gives, and is applied to the receiving service's own settings.
func TestUnregisteredSettingsDeltaApplies(t *testing.T) {
	type providerSettings struct {
		settings.LLM
		Effort settings.Opt[string] `settings:"effort"`
	}
	delta := &providerSettings{Effort: settings.Set("low")}
	delta.Temperature = settings.Cleared[float64]()
	got := crossedFrame[*frames.LLMUpdateSettingsFrame](t, frames.NewLLMUpdateSettingsFrame(delta))

	store := &providerSettings{}
	store.Temperature = settings.Set(0.5)
	resolved, ok, err := settings.Resolve(&got.ServiceUpdateSettingsFrame, store)
	if err != nil || !ok {
		t.Fatalf("resolve = %v, %v", ok, err)
	}
	if _, err := settings.Apply(store, resolved); err != nil {
		t.Fatalf("apply: %v", err)
	}
	if store.Effort.Or("") != "low" {
		t.Errorf("effort = %q, want low", store.Effort.Or(""))
	}
	if _, ok := store.Temperature.Value(); ok {
		t.Error("temperature was not cleared")
	}
}

// A VAD frame's time crosses, rather than failing the whole message.
func TestVADFrameCrosses(t *testing.T) {
	at := time.Date(2026, 11, 12, 10, 0, 0, 123456789, time.UTC)
	got := crossedFrame[*frames.VADUserStartedSpeakingFrame](t, frames.NewVADUserStartedSpeakingFrame(0.2, at))
	if !got.Timestamp.Equal(at) {
		t.Errorf("timestamp = %v, want %v", got.Timestamp, at)
	}
	if got.StartSecs != 0.2 {
		t.Errorf("start secs = %v, want 0.2", got.StartSecs)
	}
}

// A provider-native message written as its adapter's own type crosses as the
// object it is, and its adapter still reads it.
func TestSpecificMessageCrosses(t *testing.T) {
	c := frames.NewLLMContext("")
	c.AddMessage(frames.NewLLMSpecificMessage("openai", openai.Message{Role: "user", Content: "hi"}))
	got := crossedFrame[*frames.LLMContextFrame](t, frames.NewLLMContextFrame(c)).Context.Messages()
	if len(got) != 1 {
		t.Fatalf("messages = %+v, want the one sent", got)
	}
	native, err := adapter.NativeMessage[openai.Message](got[0])
	if err != nil {
		t.Fatalf("the adapter cannot read the native message (%T): %v", got[0].Native, err)
	}
	if native.Role != "user" || native.Content != "hi" {
		t.Errorf("native = %+v, want the message sent", native)
	}
}

// A whole number in a payload stays an int, as it does on the local bus.
func TestPayloadIntStaysInt(t *testing.T) {
	m := &bus.JobRequestMessage{JobID: "t-1", Payload: map[string]any{"count": 3, "ratio": 0.5}}
	m.From = "a"
	back, ok := roundTrip(t, m).(*bus.JobRequestMessage)
	if !ok {
		t.Fatal("the message came back as something else")
	}
	if n, ok := back.Payload["count"].(int); !ok || n != 3 {
		t.Errorf("count = %v (%T), want the int 3", back.Payload["count"], back.Payload["count"])
	}
	if back.Payload["ratio"] != 0.5 {
		t.Errorf("ratio = %v, want 0.5", back.Payload["ratio"])
	}
}

// A nested value of a type this process does not know is left out, rather
// than losing the message it sat in.
func TestUnknownNestedTypeIsLeftOut(t *testing.T) {
	s := bus.NewJSONSerializer(nil)
	back, err := s.Deserialize([]byte(`{"__type__":"JobRequestMessage","__data__":{"JobID":"j",` +
		`"Payload":{"x":{"__type__":"NoSuchType","__data__":{}},"y":1}}}`))
	if err != nil {
		t.Fatalf("Deserialize: %v", err)
	}
	got, ok := back.(*bus.JobRequestMessage)
	if !ok || got.JobID != "j" || got.Payload["y"] != 1 {
		t.Errorf("message = %+v, want it delivered without the unknown value", back)
	}
}
