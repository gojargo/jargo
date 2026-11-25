package llm_test

import (
	"context"
	"encoding/json"
	"slices"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/event"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/service/llm"
	"github.com/gojargo/jargo/service/settings"
)

// How an LLM service runs its LLM extensions, with the service running in a
// pipeline. The tests of attaching and composing are in
// extension_internal_test.go.

// processedFrame is a frame an extension was handed, with its direction.
type processedFrame struct {
	frame frames.Frame
	dir   processor.Direction
}

// frameRecorder is an extension that records what it sees and does what it is
// told.
type frameRecorder struct {
	llm.BaseExtension
	order  int
	filter func(frames.Frame) []frames.Frame

	mu        sync.Mutex
	processed []processedFrame
	filtered  []frames.Frame
}

func (r *frameRecorder) Order() int { return r.order }

func (r *frameRecorder) ProcessFrame(_ context.Context, f frames.Frame, dir processor.Direction) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.processed = append(r.processed, processedFrame{f, dir})
	return nil
}

func (r *frameRecorder) FilterPushFrame(
	_ context.Context, f frames.Frame, _ processor.Direction,
) ([]frames.Frame, error) {
	r.mu.Lock()
	r.filtered = append(r.filtered, f)
	r.mu.Unlock()
	if r.filter != nil {
		return r.filter(f), nil
	}
	return []frames.Frame{f}, nil
}

func (r *frameRecorder) filteredFrames() []frames.Frame {
	r.mu.Lock()
	defer r.mu.Unlock()
	return slices.Clone(r.filtered)
}

// alphaRecorder and betaRecorder are recorders of types of their own, for a
// service that holds several: one of each type.
type (
	alphaRecorder struct{ *frameRecorder }
	betaRecorder  struct{ *frameRecorder }
)

// appendSuffix is a filter that appends suffix to the text of every
// LLMTextFrame.
func appendSuffix(suffix string) func(frames.Frame) []frames.Frame {
	return func(f frames.Frame) []frames.Frame {
		if tf, ok := f.(*frames.LLMTextFrame); ok {
			return []frames.Frame{frames.NewLLMTextFrame(tf.Text + suffix)}
		}
		return []frames.Frame{f}
	}
}

// dropText is a filter that holds back every LLMTextFrame.
func dropText(f frames.Frame) []frames.Frame {
	if _, ok := f.(*frames.LLMTextFrame); ok {
		return nil
	}
	return []frames.Frame{f}
}

// llmTexts are the texts of the LLMTextFrames among fs, in order.
func llmTexts(fs []frames.Frame) []string {
	var out []string
	for _, f := range fs {
		if tf, ok := f.(*frames.LLMTextFrame); ok {
			out = append(out, tf.Text)
		}
	}
	return out
}

// passingGen stands in for a model that is never asked to generate.
type passingGen struct{}

func (passingGen) Generate(context.Context, *frames.LLMContext, llm.Emit) error { return nil }

// extendedService builds a service with the extensions attached.
func extendedService(t *testing.T, extensions ...llm.Extension) *llm.Base {
	t.Helper()
	svc := llm.New("FakeLLM", passingGen{})
	for _, extension := range extensions {
		if err := svc.AddExtension(extension); err != nil {
			t.Fatalf("AddExtension: %v", err)
		}
	}
	return svc
}

// runExtended runs p alone in a pipeline, sends it each frame in order (a
// time.Duration sleeps that long instead), lets it finish, and returns the
// frames that came out of it downstream and upstream, without the StartFrame,
// the EndFrame and the service's metadata.
func runExtended(t *testing.T, p processor.Processor, send ...any) (down, up []frames.Frame) {
	t.Helper()
	var mu sync.Mutex
	noRTVI := false
	task := pipeline.NewWorker(pipeline.New(p), pipeline.WorkerConfig{
		EnableRTVI:              &noRTVI,
		ReachedDownstreamFilter: pipeline.AnyFrame,
		ReachedUpstreamFilter:   pipeline.AnyFrame,
	})
	keep := func(f frames.Frame) bool {
		switch f.(type) {
		case *frames.StartFrame, *frames.EndFrame, *frames.LLMServiceMetadataFrame:
			return false
		}
		return true
	}
	event.On(&task.Registry, pipeline.EventFrameReachedDownstream, func(_ context.Context, f frames.Frame) {
		if keep(f) {
			mu.Lock()
			down = append(down, f)
			mu.Unlock()
		}
	})
	event.On(&task.Registry, pipeline.EventFrameReachedUpstream, func(_ context.Context, f frames.Frame) {
		if keep(f) {
			mu.Lock()
			up = append(up, f)
			mu.Unlock()
		}
	})
	done := make(chan error, 1)
	go func() { done <- task.Run(context.Background()) }()
	for _, s := range send {
		switch v := s.(type) {
		case time.Duration:
			time.Sleep(v)
		case frames.Frame:
			task.QueueFrame(v)
		default:
			t.Fatalf("cannot send %T", s)
		}
	}
	task.StopWhenDone()
	select {
	case err := <-done:
		if err != nil {
			t.Fatal(err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("the pipeline did not finish")
	}
	mu.Lock()
	defer mu.Unlock()
	return slices.Clone(down), slices.Clone(up)
}

func TestExtensionsSeeEveryFrameTheServiceProcesses(t *testing.T) {
	extension := &frameRecorder{}
	svc := extendedService(t, extension)

	runExtended(t, svc, frames.NewTextFrame("hi"))

	extension.mu.Lock()
	defer extension.mu.Unlock()
	var seen []string
	for _, p := range extension.processed {
		if p.dir != processor.Downstream {
			t.Errorf("%s was processed going %v, want downstream", p.frame.Name(), p.dir)
		}
		switch p.frame.(type) {
		case *frames.StartFrame:
			seen = append(seen, "start")
		case *frames.TextFrame:
			seen = append(seen, "text")
		case *frames.EndFrame:
			seen = append(seen, "end")
		}
	}
	if !slices.Equal(seen, []string{"start", "text", "end"}) {
		t.Errorf("processed %v, want [start text end]", seen)
	}
}

func TestExtensionsFilterWhatTheServicePushesInOrder(t *testing.T) {
	later := &betaRecorder{&frameRecorder{order: 1, filter: appendSuffix("-later")}}
	earlier := &alphaRecorder{&frameRecorder{order: 0, filter: appendSuffix("-earlier")}}
	// Attached in the other order: the order decides.
	svc := extendedService(t, later, earlier)

	down, _ := runExtended(t, svc, frames.NewLLMTextFrame("x"))

	if got := llmTexts(down); !slices.Equal(got, []string{"x-earlier-later"}) {
		t.Errorf("pushed %v, want [x-earlier-later]", got)
	}
}

func TestFramesAnExtensionAddsContinueThroughTheOnesAfterIt(t *testing.T) {
	addOne := func(f frames.Frame) []frames.Frame {
		if _, ok := f.(*frames.LLMTextFrame); ok {
			return []frames.Frame{f, frames.NewLLMTextFrame("added")}
		}
		return []frames.Frame{f}
	}
	adder := &alphaRecorder{&frameRecorder{order: 0, filter: addOne}}
	after := &betaRecorder{&frameRecorder{order: 1}}
	svc := extendedService(t, adder, after)

	down, _ := runExtended(t, svc, frames.NewLLMTextFrame("x"))

	if got := llmTexts(after.filteredFrames()); !slices.Equal(got, []string{"x", "added"}) {
		t.Errorf("the extension after it filtered %v, want [x added]", got)
	}
	if got := llmTexts(down); !slices.Equal(got, []string{"x", "added"}) {
		t.Errorf("pushed %v, want [x added]", got)
	}
}

// announcer pushes a frame of its own as it filters the text "first".
type announcer struct{ llm.BaseExtension }

func (a *announcer) FilterPushFrame(
	ctx context.Context, f frames.Frame, _ processor.Direction,
) ([]frames.Frame, error) {
	if tf, ok := f.(*frames.LLMTextFrame); ok && tf.Text == "first" {
		if err := a.PushFrame(ctx, frames.NewLLMTextFrame("announced"), processor.Downstream); err != nil {
			return nil, err
		}
	}
	return []frames.Frame{f}, nil
}

// A frame one extension adds goes through the rest and out before the next one.
func TestFramesGoOutInTheOrderTheExtensionsProduceThem(t *testing.T) {
	split := func(f frames.Frame) []frames.Frame {
		if _, ok := f.(*frames.LLMTextFrame); ok {
			return []frames.Frame{frames.NewLLMTextFrame("first"), frames.NewLLMTextFrame("second")}
		}
		return []frames.Frame{f}
	}
	svc := extendedService(t, &frameRecorder{filter: split}, &announcer{})

	down, _ := runExtended(t, svc, frames.NewLLMTextFrame("x"))

	if got := llmTexts(down); !slices.Equal(got, []string{"announced", "first", "second"}) {
		t.Errorf("pushed %v, want [announced first second]", got)
	}
}

func TestAnExtensionCanHoldAFrameBack(t *testing.T) {
	svc := extendedService(t, &frameRecorder{filter: dropText})

	down, _ := runExtended(t, svc, frames.NewLLMTextFrame("x"))

	if got := llmTexts(down); len(got) != 0 {
		t.Errorf("pushed %v, want nothing", got)
	}
}

// speaker pushes and broadcasts frames of its own when it processes a
// TextFrame.
type speaker struct{ llm.BaseExtension }

func (s *speaker) Order() int { return 1 }

func (s *speaker) ProcessFrame(ctx context.Context, f frames.Frame, _ processor.Direction) error {
	if _, ok := f.(*frames.TextFrame); !ok {
		return nil
	}
	if err := s.PushFrame(ctx, frames.NewLLMTextFrame("spoken"), processor.Downstream); err != nil {
		return err
	}
	if err := s.PushFrame(ctx, frames.NewLLMTextFrame("up"), processor.Upstream); err != nil {
		return err
	}
	return s.BroadcastFrame(ctx, func() frames.Frame { return frames.NewUserTurnInferenceCompletedFrame() })
}

func countOfType[T frames.Frame](fs []frames.Frame) int {
	n := 0
	for _, f := range fs {
		if _, ok := f.(T); ok {
			n++
		}
	}
	return n
}

func TestWhatAnExtensionPushesPassesTheExtensionsOutsideIt(t *testing.T) {
	// A filter that would drop the text, were it handed it.
	inside := &alphaRecorder{&frameRecorder{order: 0, filter: dropText}}
	outside := &betaRecorder{&frameRecorder{order: 2, filter: appendSuffix("-outside")}}
	svc := extendedService(t, inside, &speaker{}, outside)

	down, up := runExtended(t, svc, frames.NewTextFrame("hi"))

	if got := llmTexts(down); !slices.Equal(got, []string{"spoken-outside"}) {
		t.Errorf("pushed downstream %v, want [spoken-outside]", got)
	}
	if got := llmTexts(up); !slices.Equal(got, []string{"up-outside"}) {
		t.Errorf("pushed upstream %v, want [up-outside]", got)
	}
	if n := countOfType[*frames.UserTurnInferenceCompletedFrame](down); n != 1 {
		t.Errorf("broadcast downstream %d times, want once", n)
	}
	if n := countOfType[*frames.UserTurnInferenceCompletedFrame](up); n != 1 {
		t.Errorf("broadcast upstream %d times, want once", n)
	}
	if got := llmTexts(inside.filteredFrames()); len(got) != 0 {
		t.Errorf("the extension inside filtered %v, want none of the speaker's frames", got)
	}
	if got := llmTexts(outside.filteredFrames()); !slices.Equal(got, []string{"spoken", "up"}) {
		t.Errorf("the extension outside filtered %v, want [spoken up]", got)
	}
	if n := countOfType[*frames.UserTurnInferenceCompletedFrame](outside.filteredFrames()); n != 2 {
		t.Errorf("the extension outside filtered %d broadcast frames, want 2", n)
	}
	if n := countOfType[*frames.UserTurnInferenceCompletedFrame](inside.filteredFrames()); n != 0 {
		t.Errorf("the extension inside filtered %d broadcast frames, want none", n)
	}
}

// responder answers a TextFrame with a frame of its own, and logs what it is
// handed.
type responder struct {
	llm.BaseExtension
	name  string
	order int
	log   *eventLog
}

// eventLog is a log shared by several extensions.
type eventLog struct {
	mu      sync.Mutex
	entries []string
}

func (l *eventLog) add(entry string) {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.entries = append(l.entries, entry)
}

func (r *responder) Order() int { return r.order }

func (r *responder) ProcessFrame(ctx context.Context, f frames.Frame, _ processor.Direction) error {
	if _, ok := f.(*frames.TextFrame); !ok {
		return nil
	}
	r.log.add(r.name + " processes")
	return r.PushFrame(ctx, frames.NewLLMTextFrame("from "+r.name), processor.Downstream)
}

func (r *responder) FilterPushFrame(_ context.Context, f frames.Frame, _ processor.Direction) ([]frames.Frame, error) {
	if tf, ok := f.(*frames.LLMTextFrame); ok {
		r.log.add(r.name + " filters " + tf.Text)
	}
	return []frames.Frame{f}, nil
}

type (
	responderA struct{ *responder }
	responderB struct{ *responder }
	responderC struct{ *responder }
)

// Each extension processes a frame before it filters any frame pushed in
// response.
func TestExtensionsProcessAFrameFromTheOutsideIn(t *testing.T) {
	log := &eventLog{}
	svc := extendedService(t,
		&responderA{&responder{name: "A", order: 0, log: log}},
		&responderB{&responder{name: "B", order: 1, log: log}},
		&responderC{&responder{name: "C", order: 2, log: log}},
	)

	down, _ := runExtended(t, svc, frames.NewTextFrame("hi"))

	want := []string{
		"C processes",
		"B processes",
		"C filters from B",
		"A processes",
		"B filters from A",
		"C filters from A",
	}
	log.mu.Lock()
	defer log.mu.Unlock()
	if !slices.Equal(log.entries, want) {
		t.Errorf("log = %q, want %q", log.entries, want)
	}
	if got := llmTexts(down); !slices.Equal(got, []string{"from C", "from B", "from A"}) {
		t.Errorf("pushed %v, want [from C, from B, from A]", got)
	}
}

// closedGate stands in for the gate predictive delegation installs: it lets no
// text out.
type closedGate struct{}

func (closedGate) ProcessFrame(context.Context, *llm.Base, frames.Frame) error { return nil }

func (closedGate) AllowPush(f frames.Frame) bool {
	_, isText := f.(*frames.LLMTextFrame)
	return !isText
}

func (closedGate) AllowFunctionCalls() bool { return true }

// gateSpeaker pushes text of its own when it processes a
// UserTurnInferenceCompletedFrame.
type gateSpeaker struct{ llm.BaseExtension }

func (s *gateSpeaker) ProcessFrame(ctx context.Context, f frames.Frame, _ processor.Direction) error {
	if _, ok := f.(*frames.UserTurnInferenceCompletedFrame); ok {
		return s.PushFrame(ctx, frames.NewLLMTextFrame("from the extension"), processor.Downstream)
	}
	return nil
}

func TestWhatAnExtensionPushesIsPastTheOutputGate(t *testing.T) {
	svc := extendedService(t, &gateSpeaker{})
	svc.SetOutputGate(closedGate{})

	down, _ := runExtended(t, svc,
		frames.NewUserTurnInferenceCompletedFrame(), frames.NewLLMTextFrame("from the service"))

	if got := llmTexts(down); !slices.Equal(got, []string{"from the extension"}) {
		t.Errorf("pushed %v, want only the extension's text", got)
	}
}

// companion adds a built-in tool for every registered tool whose name starts
// with "go".
type companion struct {
	llm.BaseExtension
	handler llm.FunctionCallHandler
}

func (c *companion) BuiltinTools(registered map[string]llm.FunctionCallRegistryItem) []llm.ExtensionTool {
	async := false
	var tools []llm.ExtensionTool
	for name := range registered {
		if len(name) >= 2 && name[:2] == "go" {
			tools = append(tools, llm.ExtensionTool{
				Schema: frames.Tool{
					Name: "stop_" + name, Parameters: json.RawMessage(`{"type":"object"}`), Handler: c.handler,
				},
				CancelOnInterruption: &async,
			})
		}
	}
	return tools
}

// A tool an extension returns is built in; one it stops returning goes.
func TestAnExtensionsToolsAreAdvertisedAndWithdrawnWithTheRegistry(t *testing.T) {
	gen := &advertisingGen{}
	svc := llm.New("FakeLLM", gen)
	if err := svc.AddExtension(&companion{handler: noop}); err != nil {
		t.Fatal(err)
	}
	svc.RegisterFunction("go_fetch", noop)
	syncTools(svc)

	if got := advertised(gen); !slices.Equal(got, []string{"stop_go_fetch"}) {
		t.Fatalf("advertised %v, want [stop_go_fetch]", got)
	}
	if !svc.FunctionIsAsync("stop_go_fetch") {
		t.Error("the tool should run as an asynchronous one, as the extension asked")
	}
	// The name is reserved while the tool is built in.
	var replaced bool
	svc.RegisterFunction("stop_go_fetch", func(context.Context, llm.FunctionCallParams) error {
		replaced = true
		return nil
	})
	if !svc.FunctionIsAsync("stop_go_fetch") || replaced {
		t.Error("registering over the built-in tool's name should be refused")
	}

	svc.UnregisterFunction("go_fetch")
	syncTools(svc)

	if got := advertised(gen); len(got) != 0 {
		t.Errorf("advertised %v, want nothing once the tool it follows goes", got)
	}
	if svc.HasFunction("stop_go_fetch") {
		t.Error("the withdrawn tool still has a handler")
	}
}

// helperTool is an extension that adds a built-in tool named "helper".
type helperTool struct{ llm.BaseExtension }

func (*helperTool) BuiltinTools(map[string]llm.FunctionCallRegistryItem) []llm.ExtensionTool {
	return []llm.ExtensionTool{{Schema: frames.Tool{Name: "helper", Handler: llm.FunctionCallHandler(noop)}}}
}

func TestAUsersToolKeepsItsNameOverAnExtensions(t *testing.T) {
	gen := &advertisingGen{}
	svc := llm.New("FakeLLM", gen)
	if err := svc.AddExtension(&helperTool{}); err != nil {
		t.Fatal(err)
	}
	svc.RegisterFunction("helper", noop, llm.WithCancelOnInterruption(false))
	syncTools(svc)

	if got := advertised(gen); len(got) != 0 {
		t.Errorf("advertised %v, want the application's tool left alone", got)
	}
	if !svc.FunctionIsAsync("helper") {
		t.Error("the application's own registration was replaced")
	}
}

// announcing adds two built-in tools: one quiet, one whose calls are announced.
type announcing struct {
	llm.BaseExtension
	handler llm.FunctionCallHandler
}

func (a *announcing) BuiltinTools(map[string]llm.FunctionCallRegistryItem) []llm.ExtensionTool {
	tool := func(name string) frames.Tool {
		return frames.Tool{Name: name, Parameters: json.RawMessage(`{"type":"object"}`), Handler: a.handler}
	}
	return []llm.ExtensionTool{
		{Schema: tool("quiet")},
		{Schema: tool("announced"), AnnounceCallStart: true},
	}
}

// Either way the call is run, and the tool's name is reserved.
func TestAnExtensionToolsCallStartIsAnnouncedOnlyIfTheToolAsks(t *testing.T) {
	ran := make(chan string, 2)
	handler := func(ctx context.Context, p llm.FunctionCallParams) error {
		ran <- p.FunctionName
		return p.Result(ctx, "ok", nil)
	}
	gen := &scriptedGen{turns: [][]frames.ToolCall{{
		{ID: "quiet", Name: "quiet", Args: json.RawMessage(`{}`)},
		{ID: "announced", Name: "announced", Args: json.RawMessage(`{}`)},
	}}}
	svc := llm.New("FakeLLM", gen)
	if err := svc.AddExtension(&announcing{handler: handler}); err != nil {
		t.Fatal(err)
	}
	var mu sync.Mutex
	var notified []string
	svc.OnFunctionCallsStarted(func(_ context.Context, calls []frames.ToolCall) {
		mu.Lock()
		defer mu.Unlock()
		for _, c := range calls {
			notified = append(notified, c.Name)
		}
	})

	down, _ := runExtended(t, svc,
		frames.NewLLMContextFrame(toolConvo("anything")), 300*time.Millisecond)

	var started []string
	for _, f := range down {
		if fr, ok := f.(*frames.FunctionCallsStartedFrame); ok {
			for _, c := range fr.Calls {
				started = append(started, c.Name)
			}
		}
	}
	if !slices.Equal(started, []string{"announced"}) {
		t.Errorf("FunctionCallsStartedFrame announced %v, want [announced]", started)
	}
	mu.Lock()
	if !slices.Equal(notified, []string{"announced"}) {
		t.Errorf("OnFunctionCallsStarted was told of %v, want [announced]", notified)
	}
	mu.Unlock()
	var run []string
	for range 2 {
		select {
		case name := <-ran:
			run = append(run, name)
		case <-time.After(time.Second):
			t.Fatal("a call never ran")
		}
	}
	slices.Sort(run)
	if !slices.Equal(run, []string{"announced", "quiet"}) {
		t.Errorf("ran %v, want both calls", run)
	}
	for _, name := range []string{"quiet", "announced"} {
		svc.RegisterFunction(name, noop, llm.WithCancelOnInterruption(false))
		if svc.FunctionIsAsync(name) {
			t.Errorf("registering over the built-in %s was not refused", name)
		}
	}
}

// labeled is an extension that contributes an instruction.
type labeled struct {
	llm.BaseExtension
	instruction string
}

func (l *labeled) SystemInstruction() string { return l.instruction }

type (
	firstLabel  struct{ *labeled }
	secondLabel struct{ *labeled }
)

func TestASettingsUpdateBuildsAndAddsExtensionsWhileRunning(t *testing.T) {
	first := &firstLabel{&labeled{instruction: "FIRST"}}
	svc := extendedService(t, first)

	var delta settings.LLM
	delta.Extensions = settings.Set(settings.Extensions{
		func() llm.Extension { return &secondLabel{&labeled{instruction: "SECOND"}} },
	})
	runExtended(t, svc, frames.NewLLMUpdateSettingsFrame(&delta))

	if _, ok := llm.GetExtension[*secondLabel](svc); !ok {
		t.Fatal("the update did not attach the extension it built")
	}
	if got, _ := llm.GetExtension[*firstLabel](svc); got != first {
		t.Error("the extension held before the update is gone")
	}
	if got := svc.SystemInstruction(); got != "FIRST\n\nSECOND" {
		t.Errorf("SystemInstruction() = %q, want FIRST then SECOND", got)
	}
}

// A tool handler that acts on the calls in flight, such as an extension's
// cancel tool, sees them in RunningFunctionCalls and cancels one by its
// tool_call_id with CancelFunctionCall, whatever it was registered with.
func TestRunningFunctionCallsAndCancelFunctionCall(t *testing.T) {
	svc := llm.New("FakeToolLLM", cancelOnInterruptionGen{})
	running := make(chan struct{})
	canceled := make(chan struct{})
	svc.RegisterFunction("get_weather", func(ctx context.Context, _ llm.FunctionCallParams) error {
		close(running)
		<-ctx.Done()
		close(canceled)
		return nil
	}, llm.WithCancelOnInterruption(false), llm.WithCancellableByLLM(true))

	var mu sync.Mutex
	cancels := 0
	probe := newProbe(func(f frames.Frame) {
		if _, ok := f.(*frames.FunctionCallCancelFrame); ok {
			mu.Lock()
			cancels++
			mu.Unlock()
		}
	})
	task := pipeline.NewWorker(pipeline.New(svc, probe), pipeline.WorkerConfig{})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()
	task.QueueFrame(frames.NewLLMContextFrame(toolConvo("get_weather")))
	select {
	case <-running:
	case <-time.After(3 * time.Second):
		t.Fatal("the handler never ran")
	}

	calls := svc.RunningFunctionCalls()
	if len(calls) != 1 {
		t.Fatalf("running calls = %+v, want the one call", calls)
	}
	if call := calls[0]; call.FunctionName != "get_weather" || call.ToolCallID != "call_1" ||
		call.RegistryItem.CancelOnInterruption || !call.RegistryItem.CancellableByLLM {
		t.Errorf("running call = %+v, want get_weather/call_1, asynchronous and cancellable", call)
	}

	svc.CancelFunctionCall(context.Background(), "call_1", "test")
	select {
	case <-canceled:
	case <-time.After(3 * time.Second):
		t.Fatal("CancelFunctionCall did not cancel the call's context")
	}
	if calls := svc.RunningFunctionCalls(); len(calls) != 0 {
		t.Errorf("running calls = %+v after canceling, want none", calls)
	}
	time.Sleep(50 * time.Millisecond)
	task.StopWhenDone()
	<-runDone

	mu.Lock()
	defer mu.Unlock()
	if cancels == 0 {
		t.Error("the cancellation was never announced")
	}
}
