package realtime

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"maps"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/processor/aggregators"
	"github.com/gojargo/jargo/processor/turns"
	"github.com/gojargo/jargo/transport"
	"github.com/gojargo/jargo/utils/events"
)

// waitTimeout bounds every wait of the tests.
const waitTimeout = 3 * time.Second

// object is a JSON object as the peer reads and writes it.
type object = map[string]any

// peer is a local Prism protocol peer: it accepts the session's requests and
// lets each test control the response events and the failures.
type peer struct {
	t           *testing.T
	server      *httptest.Server
	unversioned bool
	// refuseUpgrade, when set, answers an upgrade attempt with an HTTP refusal
	// instead: it returns the status and the Retry-After header, or 0 to
	// accept the attempt.
	refuseUpgrade func(attempt int) (status int, retryAfter *string)

	writeMu sync.Mutex
	mu      sync.Mutex
	conn    *websocket.Conn
	// connCtx is the context of the connection's handler.
	connCtx  context.Context
	attempts []time.Time
	events   []object
	requests chan object
	history  []object
	settings object
	revision int
	serial   int
	gates    map[string]chan struct{}
	refuse   map[string]string
	// historyFailureAt is the history mutation that is refused; 0 refuses none.
	historyFailureAt int
	historyMutations int
	historySerial    int
}

func newPeer(t *testing.T, unversioned bool) *peer {
	t.Helper()
	p := &peer{
		t:           t,
		unversioned: unversioned,
		requests:    make(chan object, 4096),
		settings:    object{},
		gates:       map[string]chan struct{}{},
		refuse:      map[string]string{},
	}
	p.server = httptest.NewServer(http.HandlerFunc(p.handle))
	t.Cleanup(p.server.Close)
	return p
}

// endpoint is the peer's Prism endpoint.
func (p *peer) endpoint() string {
	return "ws" + strings.TrimPrefix(p.server.URL, "http") + "/v1/realtime"
}

func (p *peer) handle(w http.ResponseWriter, req *http.Request) {
	p.mu.Lock()
	p.attempts = append(p.attempts, time.Now())
	attempt := len(p.attempts)
	refuse := p.refuseUpgrade
	p.mu.Unlock()
	if refuse != nil {
		if status, retryAfter := refuse(attempt); status != 0 {
			if retryAfter != nil {
				w.Header().Set("Retry-After", *retryAfter)
			}
			http.Error(w, "Temporary test refusal", status)
			return
		}
	}
	conn, err := websocket.Accept(w, req, nil)
	if err != nil {
		return
	}
	conn.SetReadLimit(-1)
	p.mu.Lock()
	p.conn, p.connCtx = conn, req.Context()
	p.mu.Unlock()
	defer func() { _ = conn.CloseNow() }()
	p.emit("session.created", object{"session": object{"id": "session-1"}})
	var handlers sync.WaitGroup
	defer handlers.Wait()
	for {
		_, data, err := conn.Read(req.Context())
		if err != nil {
			return
		}
		var event object
		if json.Unmarshal(data, &event) != nil {
			continue
		}
		p.mu.Lock()
		p.events = append(p.events, event)
		gate, gated := p.gates[event["type"].(string)]
		p.mu.Unlock()
		if !gated {
			p.handleRequest(event)
			continue
		}
		// A gated request waits apart, and those after it go on.
		handlers.Go(func() {
			select {
			case <-gate:
				p.handleRequest(event)
			case <-req.Context().Done():
			}
		})
	}
}

// emit sends an event the way Prism does.
func (p *peer) emit(kind string, body object) {
	event := object{"type": kind, "event_id": "peer"}
	maps.Copy(event, body)
	data, err := json.Marshal(event)
	if err != nil {
		p.t.Errorf("peer: %v", err)
		return
	}
	p.mu.Lock()
	conn, ctx := p.conn, p.connCtx
	p.mu.Unlock()
	if conn == nil {
		return
	}
	p.writeMu.Lock()
	defer p.writeMu.Unlock()
	_ = conn.Write(ctx, websocket.MessageText, data)
}

// closeSocket closes the connection the way a server going away does.
func (p *peer) closeSocket() {
	p.mu.Lock()
	conn := p.conn
	p.mu.Unlock()
	_ = conn.Close(websocket.StatusNormalClosure, "")
}

// gate holds the requests of a kind until the returned function is called.
func (p *peer) gate(kind string) func() {
	ch := make(chan struct{})
	p.mu.Lock()
	p.gates[kind] = ch
	p.mu.Unlock()
	var once sync.Once
	open := func() { once.Do(func() { close(ch) }) }
	p.t.Cleanup(open)
	return open
}

func (p *peer) setRefusal(kind, code string) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.refuse[kind] = code
}

// failHistoryAt refuses the history mutation that comes n after the ones made.
func (p *peer) failHistoryAt(n int) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.historyFailureAt = p.historyMutations + n
}

func (p *peer) handleRequest(event object) {
	kind, _ := event["type"].(string)
	p.mu.Lock()
	if kind == kindItemCreate || kind == kindItemDelete {
		p.historyMutations++
	}
	refusalKey := kind
	if session, ok := event["session"].(object); ok && kind == kindSessionUpdate {
		if _, labels := session[keyLabels]; labels {
			refusalKey = "session.labels"
		}
	}
	refusal := p.refuse[refusalKey]
	delete(p.refuse, refusalKey)
	failing := p.historyFailureAt != 0 && p.historyFailureAt == p.historyMutations
	if refusal != "" || failing {
		p.historyFailureAt = 0
		p.mu.Unlock()
		code := refusal
		if code == "" {
			code = "invalid_request"
		}
		p.emit("error", object{"error": object{
			"code":    code,
			"message": "Peer refused request",
			"scope":   "event",
			"owner":   object{"kind": "event", "event_id": event["event_id"]},
		}})
		if kind == kindResponseNew {
			p.emit("prism.typed_input.ready", nil)
		}
		return
	}
	switch kind {
	case kindSessionUpdate:
		session, _ := event["session"].(object)
		// Prism merges partial turn settings into the current ones.
		turn := object{}
		if current, ok := p.settings[keyTurn].(object); ok {
			maps.Copy(turn, current)
		}
		if update, ok := session[keyTurn].(object); ok {
			maps.Copy(turn, update)
		}
		maps.Copy(p.settings, session)
		if len(turn) > 0 {
			p.settings[keyTurn] = turn
		}
		p.revision++
		reply := maps.Clone(p.settings)
		reply["id"], reply["voice"], reply["request_id"] = "session-1", "test", "actual-peer-request"
		if !p.unversioned {
			reply[keyRevision] = p.revision
		}
		interrupt := true
		if value, ok := turn["interrupt_response"].(bool); ok {
			interrupt = value
		}
		reply["prism_effective_interrupt_response"] = interrupt
		p.mu.Unlock()
		p.emit("session.updated", object{"prism_request_event_id": event["event_id"], "session": reply})
		p.emit("prism.typed_input.ready", nil)
	case kindItemCreate:
		p.historySerial++
		item := object{"id": fmt.Sprintf("history-%d", p.historySerial)}
		maps.Copy(item, asObject(event["item"]))
		p.insertLocked(item, event["previous_item_id"])
		p.mu.Unlock()
		p.emit("conversation.item.created", object{"prism_request_event_id": event["event_id"], "item": item})
	case kindItemDelete:
		id := event["item_id"]
		var removed object
		for _, value := range p.history {
			if value["id"] == id {
				removed = value
			}
		}
		p.history = slices.DeleteFunc(p.history, func(value object) bool {
			if value["id"] == id {
				return true
			}
			return removed != nil && removed["call_id"] != nil && value["call_id"] == removed["call_id"]
		})
		p.mu.Unlock()
		p.emit("conversation.item.deleted", object{"prism_request_event_id": event["event_id"], "item_id": id})
	case kindHistoryGet:
		items := slices.Clone(p.history)
		p.mu.Unlock()
		p.emit("prism.conversation", object{"prism_request_event_id": event["event_id"], "items": items})
	case "prism.playback.drained":
		p.mu.Unlock()
		p.emit("prism.typed_input.ready", nil)
	case kindResponseStop:
		p.mu.Unlock()
		p.end(text(event["response_id"]), "cancelled", true, nil) //nolint:misspell // the wire's spelling
	case kindItemTruncate:
		p.mu.Unlock()
		p.emit("conversation.item.truncated", object{
			"item_id":       event["item_id"],
			"content_index": event["content_index"],
			"audio_end_ms":  event["audio_end_ms"],
		})
	case kindAudioClear:
		p.mu.Unlock()
		p.emit("input_audio_buffer.cleared", nil)
	default:
		p.mu.Unlock()
		p.requests <- event
	}
}

// insertLocked files an item in the history where Prism does.
func (p *peer) insertLocked(item object, anchor any) {
	index := len(p.history)
	switch {
	case anchor == "root":
		index = 0
	case anchor != nil:
		index = slices.IndexFunc(p.history, func(value object) bool { return value["id"] == anchor }) + 1
	case item["type"] == "function_call_output":
		if at := slices.IndexFunc(p.history, func(value object) bool {
			return value["call_id"] == item["call_id"]
		}); at >= 0 {
			index = at + 1
		}
	}
	p.history = slices.Insert(p.history, index, item)
}

// next is the next request of a kind the peer leaves to the test.
func (p *peer) next(kind string) object {
	p.t.Helper()
	deadline := time.After(waitTimeout)
	for {
		select {
		case event := <-p.requests:
			if event["type"] == kind {
				return event
			}
		case <-deadline:
			p.t.Fatalf("peer: no %s request arrived", kind)
		}
	}
}

// nextReply is the next request for a reply.
func (p *peer) nextReply() object {
	p.t.Helper()
	return p.next(kindResponseNew)
}

// start starts a reply, the one request asks for when there is one.
func (p *peer) start(request object, inputIDs []string) string {
	p.mu.Lock()
	p.serial++
	ref := fmt.Sprintf("response-%d", p.serial)
	response := object{"id": ref, "metadata": object{"prism_cause": "user_audio"}}
	if !p.unversioned {
		response[keyRevision] = p.revision
	}
	if request != nil {
		response["metadata"] = at(request, "response", "metadata")
	}
	if inputIDs != nil {
		response["prism_input_item_ids"] = inputIDs
	}
	if request != nil {
		if text, ok := request["response"].(object)["prism_input_text"]; ok {
			id := fmt.Sprintf("typed-%d", p.serial)
			response["prism_input_item_ids"] = []string{id}
			p.history = append(p.history, object{"id": id, "type": "message", "role": "user", "text": text})
		}
	}
	p.mu.Unlock()
	body := object{"response": response}
	if request != nil {
		body["prism_request_event_id"] = request["event_id"]
	}
	p.emit("response.created", body)
	return ref
}

// messageOptions shape a message of a reply.
type messageOptions struct {
	itemID  string
	text    string
	samples int
	open    bool
}

// message streams one message of a reply: its text and its audio.
func (p *peer) message(ref string, o messageOptions) object {
	if o.itemID == "" {
		o.itemID = "message-" + ref
	}
	if o.text == "" {
		o.text = "Hello"
	}
	p.emit("response.output_item.added", object{
		"response_id": ref, "output_index": 0, "item": object{"id": o.itemID, "type": "message"},
	})
	common := object{"response_id": ref, "output_index": 0, "item_id": o.itemID, "content_index": 0}
	with := func(key string, value any) object {
		body := maps.Clone(common)
		if key != "" {
			body[key] = value
		}
		return body
	}
	p.emit("response.text.delta", with("delta", o.text))
	p.emit("response.text.done", with("text", o.text))
	if o.samples > 0 {
		pcm := make([]byte, 0, 2*o.samples)
		for range o.samples {
			pcm = append(pcm, 0x10, 0x00)
		}
		p.emit("response.audio.delta", with("delta", base64.StdEncoding.EncodeToString(pcm)))
	}
	if !o.open {
		p.emit("response.audio.done", with("", nil))
	}
	return common
}

// say streams one message with the default samples.
func (p *peer) say(ref, text string) {
	p.message(ref, messageOptions{text: text, samples: 480})
}

// end ends a reply. The server admits another turn once its playback estimate
// expires, which ready stands for.
func (p *peer) end(ref, status string, ready bool, usage object) {
	response := object{"id": ref, "status": status, "metadata": object{"prism_drain_token": "drain-" + ref}}
	if usage != nil {
		response["usage"] = usage
	}
	p.emit("response.done", object{"response": response})
	if ready {
		p.emit("prism.typed_input.ready", nil)
	}
}

// complete ends a reply as completed.
func (p *peer) complete(ref string) { p.end(ref, "completed", true, nil) }

// emitCall emits a call of a reply.
func (p *peer) emitCall(ref, callID, name string) {
	p.emit("response.function_call_arguments.done", object{
		"response_id": ref, "item_id": "item-" + callID, "output_index": 0,
		"call_id": callID, "name": name, "arguments": "{}",
	})
}

// sent are the requests of a kind the service sent.
func (p *peer) sent(kind string) []object {
	p.mu.Lock()
	defer p.mu.Unlock()
	var out []object
	for _, event := range p.events {
		if event["type"] == kind {
			out = append(out, event)
		}
	}
	return out
}

// historyTexts are the texts of the history items.
func (p *peer) historyTexts() []string {
	p.mu.Lock()
	defer p.mu.Unlock()
	out := []string{}
	for _, item := range p.history {
		text, _ := item["text"].(string)
		out = append(out, text)
	}
	return out
}

// createdTexts are the texts of the items the service created.
func (p *peer) createdTexts() []string {
	out := []string{}
	for _, event := range p.sent(kindItemCreate) {
		text, _ := event["item"].(object)["text"].(string)
		out = append(out, text)
	}
	return out
}

// createdItems are the items the service created.
func (p *peer) createdItems() []object {
	var out []object
	for _, event := range p.sent(kindItemCreate) {
		out = append(out, asObject(event["item"]))
	}
	return out
}

func (p *peer) setting(key string) any {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.settings[key]
}

// toolNames are the names of the tools Prism holds.
func (p *peer) toolNames() []string {
	tools, _ := p.setting(keyTools).([]any)
	names := []string{}
	for _, tool := range tools {
		names = append(names, text(at(tool, "function", "name")))
	}
	return names
}

func (p *peer) attemptTimes() []time.Time {
	p.mu.Lock()
	defer p.mu.Unlock()
	return slices.Clone(p.attempts)
}

// capture records the frames that travel one way past it.
type capture struct {
	*processor.Base
	direction processor.Direction
	mu        sync.Mutex
	seen      []frames.Frame
	// transform, when set, changes a frame before it travels on.
	transform func(frames.Frame)
}

func newCapture(direction processor.Direction) *capture {
	c := &capture{direction: direction}
	c.Base = processor.New("Capture", c)
	return c
}

func (c *capture) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := c.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	if dir == c.direction {
		c.mu.Lock()
		transform := c.transform
		c.mu.Unlock()
		if transform != nil {
			transform(f)
		}
		c.mu.Lock()
		c.seen = append(c.seen, f)
		c.mu.Unlock()
	}
	return c.PushFrame(ctx, f, dir)
}

func (c *capture) frames() []frames.Frame {
	c.mu.Lock()
	defer c.mu.Unlock()
	return slices.Clone(c.seen)
}

// captured are the frames of type T two captures saw, in order.
func captured[T frames.Frame](captures ...*capture) []T {
	var out []T
	for _, c := range captures {
		for _, f := range c.frames() {
			if v, ok := f.(T); ok {
				out = append(out, v)
			}
		}
	}
	return out
}

func has[T frames.Frame](captures ...*capture) bool { return len(captured[T](captures...)) > 0 }

// output captures the audio written through the standard output transport.
type testOutput struct {
	*transport.BaseOutput
	mu     sync.Mutex
	writes [][]byte
	// gate, when set, runs before each write is recorded.
	gate func(ctx context.Context, index int)
	n    int
}

func newTestOutput(enabled bool) *testOutput {
	params := transport.DefaultParams()
	params.AudioOutEnabled = enabled
	params.AudioOutSampleRate = outputSampleRate
	params.AudioOutChannels = 1
	o := &testOutput{}
	o.BaseOutput = transport.NewBaseOutput("Output", params, o)
	return o
}

func (o *testOutput) WriteAudio(ctx context.Context, f frames.OutputAudioFrame) (bool, error) {
	o.mu.Lock()
	gate, index := o.gate, o.n
	o.n++
	o.mu.Unlock()
	if gate != nil {
		gate(ctx, index)
	}
	if ctx.Err() != nil {
		return false, ctx.Err()
	}
	o.mu.Lock()
	o.writes = append(o.writes, slices.Clone(f.AudioData().Audio))
	o.mu.Unlock()
	return true, nil
}

func (o *testOutput) written() int {
	o.mu.Lock()
	defer o.mu.Unlock()
	return len(o.writes)
}

// holdAudio keeps the bot speaking until the returned function is called.
func (o *testOutput) holdAudio(t *testing.T) func() {
	t.Helper()
	release := make(chan struct{})
	var once sync.Once
	open := func() { once.Do(func() { close(release) }) }
	t.Cleanup(open)
	o.mu.Lock()
	o.gate = func(ctx context.Context, _ int) {
		select {
		case <-release:
		case <-ctx.Done():
		}
	}
	o.mu.Unlock()
	return open
}

// gateWrites holds each of the first count writes until its release is called,
// and reports when each starts.
func (o *testOutput) gateWrites(t *testing.T, count int) (writing []chan struct{}, release []func()) {
	t.Helper()
	gates := make([]chan struct{}, count)
	for i := range count {
		writing = append(writing, make(chan struct{}))
		gates[i] = make(chan struct{})
		var once sync.Once
		gate := gates[i]
		open := func() { once.Do(func() { close(gate) }) }
		t.Cleanup(open)
		release = append(release, open)
	}
	o.mu.Lock()
	o.gate = func(ctx context.Context, index int) {
		if index < count {
			close(writing[index])
			select {
			case <-gates[index]:
			case <-ctx.Done():
			}
		}
	}
	o.mu.Unlock()
	return writing, release
}

// until waits for a condition.
func until(t *testing.T, what string, condition func() bool) {
	t.Helper()
	deadline := time.Now().Add(waitTimeout)
	for !condition() {
		if time.Now().After(deadline) {
			t.Fatalf("timed out waiting for %s", what)
		}
		time.Sleep(5 * time.Millisecond)
	}
}

func waitFor(t *testing.T, what string, ch <-chan struct{}) {
	t.Helper()
	select {
	case <-ch:
	case <-time.After(waitTimeout):
		t.Fatalf("timed out waiting for %s", what)
	}
}

// harness is one conversation running against the peer.
type harness struct {
	t      *testing.T
	peer   *peer
	svc    *Service
	worker *pipeline.Worker
	convo  *frames.LLMContext
	output *testOutput
	// up sees the frames traveling upstream on either side of the service,
	// down the frames that reached the assistant aggregator.
	up, upAfter, down *capture
	done              chan error
	closeOnce         sync.Once
}

// harnessOptions shape a conversation.
type harnessOptions struct {
	config      func(*Config)
	convo       *frames.LLMContext
	unversioned bool
	params      pipeline.Params
	// opening starts the conversation; nil queues an LLMRunFrame.
	opening func(h *harness)
	// setup runs on the service before the pipeline starts.
	setup func(*Service)
}

// newService builds a service for the peer.
func newService(p *peer, configure func(*Config)) *Service {
	cfg := Config{APIKey: "local-peer", Endpoint: p.endpoint()}
	if configure != nil {
		configure(&cfg)
	}
	svc := New(cfg)
	svc.shutdownTimeout = 400 * time.Millisecond
	return svc
}

// conversation runs a conversation through the standard aggregators and
// output, and answers its opening reply.
func converse(t *testing.T, o harnessOptions) *harness {
	t.Helper()
	h := &harness{t: t, peer: newPeer(t, o.unversioned), done: make(chan error, 1)}
	h.svc = newService(h.peer, o.config)
	if o.setup != nil {
		o.setup(h.svc)
	}
	h.convo = o.convo
	if h.convo == nil {
		h.convo = frames.NewLLMContext("")
	}
	// The external strategies the service recommends, given up front: the
	// defaults would load a local turn model before the recommendation arrives.
	pair := aggregators.New(h.convo, aggregators.WithTurns(turns.Config{
		Strategies: turns.ExternalStrategies(turns.ExternalStrategiesConfig{}),
	}))
	h.output = newTestOutput(true)
	h.up, h.upAfter = newCapture(processor.Upstream), newCapture(processor.Upstream)
	h.down = newCapture(processor.Downstream)
	h.worker = pipeline.NewWorker(pipeline.New(
		h.up, pair.User(), h.svc, h.upAfter, h.output, h.down, pair.Assistant(),
	), pipeline.WorkerConfig{Params: o.params})
	started := make(chan struct{})
	events.On(&h.worker.Registry, pipeline.EventPipelineStarted, func(context.Context, *frames.StartFrame) {
		close(started)
	})
	go func() { h.done <- h.worker.Run(context.Background()) }()
	t.Cleanup(h.close)
	waitFor(t, "the pipeline to start", started)
	until(t, "the service metadata", func() bool {
		return has[*frames.LLMServiceMetadataFrame](h.up, h.upAfter)
	})
	if o.opening != nil {
		o.opening(h)
	} else {
		h.worker.QueueFrame(frames.NewLLMRunFrame())
	}
	initial := h.peer.nextReply()
	h.peer.complete(h.peer.start(initial, nil))
	until(t, "the opening reply to end", func() bool { return has[*frames.LLMFullResponseEndFrame](h.down) })
	return h
}

// close ends the conversation.
func (h *harness) close() {
	h.closeOnce.Do(func() {
		h.peer.mu.Lock()
		gates := slices.Collect(maps.Values(h.peer.gates))
		h.peer.gates = map[string]chan struct{}{}
		h.peer.mu.Unlock()
		for _, gate := range gates {
			select {
			case <-gate:
			default:
				close(gate)
			}
		}
		h.worker.QueueFrame(frames.NewEndFrame())
		select {
		case <-h.done:
		case <-time.After(waitTimeout):
			h.t.Error("the pipeline did not end")
			h.worker.Cancel(context.Background(), "")
			<-h.done
		}
	})
}

func (h *harness) queue(fs ...frames.Frame) { h.worker.QueueFrames(fs) }

// errors are the errors that reached the start of the pipeline.
func (h *harness) errors() []*frames.ErrorFrame { return captured[*frames.ErrorFrame](h.up) }

func (h *harness) failed() bool { return len(h.errors()) > 0 }

// hasMessage reports whether the conversation holds a message.
func (h *harness) hasMessage(role frames.Role, text string) bool {
	return h.countMessage(role, text) > 0
}

func (h *harness) countMessage(role frames.Role, text string) int {
	count := 0
	for _, m := range h.convo.Messages() {
		if m.Role == role && m.StandardText() == text {
			count++
		}
	}
	return count
}

// answer starts a reply to a request, says text and ends it.
func (h *harness) answer(request object, text string, samples int) string {
	if request == nil {
		request = h.peer.nextReply()
	}
	ref := h.peer.start(request, nil)
	h.peer.message(ref, messageOptions{text: text, samples: samples})
	h.peer.complete(ref)
	return ref
}

// speak has the bot speak a reply, and waits for it to start speaking.
func (h *harness) speak() string {
	ref := h.peer.start(nil, nil)
	h.peer.message(ref, messageOptions{text: "Speaking", samples: 2400})
	h.peer.complete(ref)
	until(h.t, "the bot to start speaking", func() bool { return has[*frames.BotStartedSpeakingFrame](h.down) })
	return ref
}

// userSpeaks has Prism report the user's speech, and waits for the
// interruption the speech starts to have traveled the pipeline, so it does not
// race the events that follow it.
func (h *harness) userSpeaks() {
	h.peer.emit("input_audio_buffer.speech_started", object{"item_id": "speech-1", "audio_start_ms": 0})
	until(h.t, "the user's turn to start", func() bool { return has[*frames.InterruptionFrame](h.down) })
	h.peer.emit("input_audio_buffer.speech_stopped", object{"item_id": "speech-1", "audio_end_ms": 100})
	until(h.t, "the user's turn to end", func() bool { return has[*frames.UserStoppedSpeakingFrame](h.down) })
}

// appendUser queues an LLMMessagesAppendFrame of one user message.
func appendUser(text string, runLLM bool) *frames.LLMMessagesAppendFrame {
	f := frames.NewLLMMessagesAppendFrame([]frames.Message{{Role: frames.RoleUser, Text: text}})
	f.RunLLM = runLLM
	return f
}

// usable reports whether the service can still be given work.
func (h *harness) usable() bool { return h.svc.Usable() }

// responseOf is the conversation's state of a reply.
func (h *harness) responseOf(ref string) (*response, bool) {
	h.svc.mu.Lock()
	defer h.svc.mu.Unlock()
	value, ok := h.svc.conversation.responses[ref]
	return value, ok
}

func (h *harness) callOf(id string) (*tool, bool) {
	h.svc.mu.Lock()
	defer h.svc.mu.Unlock()
	value, ok := h.svc.conversation.calls[id]
	return value, ok
}

func (h *harness) requestsPending() int {
	h.svc.mu.Lock()
	defer h.svc.mu.Unlock()
	return h.svc.requests
}

// playingResponses are the replies of the outputs playing, in order.
func (h *harness) playingResponses() []string {
	h.svc.mu.Lock()
	defer h.svc.mu.Unlock()
	out := []string{}
	for _, value := range h.svc.playing {
		out = append(out, value.ref.Response.ResponseID)
	}
	return out
}

func (h *harness) live() Settings {
	return *h.svc.snapshot()
}
