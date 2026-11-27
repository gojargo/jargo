package rtvi_test

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"maps"
	"reflect"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/event"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/processor/rtvi"
	"github.com/gojargo/jargo/processor/rtvi/spec"
)

// Tests that the server speaks RTVI as its specification says.
//
// Each protocol version's document (spec/<version>/rtvi.asyncapi.json) is the
// contract. The real Processor and Observer run a session with a client of
// that version, and every message they put on the wire is checked against the
// document: it must be a message the server sends, its payload must match the
// document's schema, and the client must be new enough to receive it. Messages
// a client sends are checked the other way: the server accepts the document's
// examples, and answers what the document forbids with an error-response. Each
// of the document's x-rtvi-behaviors has a test here of its own.

// What the documents describe that the server does not implement, and so
// neither sends nor handles, with the reason. The tests that the server sends
// and accepts every message leave these out, and fail when one of them is
// implemented, so the lists stay true.
//
//nolint:gochecknoglobals // fixed tables, read only
var (
	unimplementedServerMessages = map[string]string{
		"system-log":        "the server does not forward its log to the client",
		"llm-function-call": "deprecated in favor of llm-function-call-in-progress, which the server sends",
		"ui-command":        "the server has no UI worker to drive the client's UI",
		"ui-job-group":      "the server has no UI worker to report job groups",
	}
	unimplementedClientMessages = map[string]string{
		"send-file":           "the server does not take files from the client",
		"ui-event":            "the server has no UI worker to take the client's UI events",
		"ui-snapshot":         "the server has no UI worker to take the client's UI snapshots",
		"ui-cancel-job-group": "the server has no UI worker whose job groups a client cancels",
	}
	unimplementedSettings = map[string]string{
		"system_logs_enabled": "the server does not send system-log",
	}
)

// lenient is what the server accepts although the document forbids it, and
// why.
//
//nolint:gochecknoglobals // fixed table, read only
var lenient = map[[2]string]string{
	{rtvi.TypeClientReady, "no about"}:                "the session starts without knowing the client's library",
	{rtvi.TypeClientReady, "about of the wrong type"}: "an invalid about is ignored, as a missing one",
}

// observerSwitch is an observer setting a document names in
// x-rtvi-enabled-by: how to turn it on or off, and whether it is on by
// default.
type observerSwitch struct {
	set func(p *rtvi.ObserverParams, on bool)
	get func(p rtvi.ObserverParams) bool
}

// onUnlessOff reads a setting that is on unless turned off.
func onUnlessOff(p *bool) bool { return p == nil || *p }

// observerSwitches are the observer settings, by the name the documents give
// them.
//
//nolint:gochecknoglobals,staticcheck // fixed table, read only; the deprecated setting is still honored
var observerSwitches = map[string]observerSwitch{
	"metrics_enabled": {
		func(p *rtvi.ObserverParams, on bool) { p.MetricsEnabled = &on },
		func(p rtvi.ObserverParams) bool { return onUnlessOff(p.MetricsEnabled) },
	},
	"user_speaking_enabled": {
		func(p *rtvi.ObserverParams, on bool) { p.UserSpeakingEnabled = &on },
		func(p rtvi.ObserverParams) bool { return onUnlessOff(p.UserSpeakingEnabled) },
	},
	"vad_user_speaking_enabled": {
		func(p *rtvi.ObserverParams, on bool) { p.VADUserSpeakingEnabled = on },
		func(p rtvi.ObserverParams) bool { return p.VADUserSpeakingEnabled },
	},
	"user_mute_enabled": {
		func(p *rtvi.ObserverParams, on bool) { p.UserMuteEnabled = &on },
		func(p rtvi.ObserverParams) bool { return onUnlessOff(p.UserMuteEnabled) },
	},
	"bot_speaking_enabled": {
		func(p *rtvi.ObserverParams, on bool) { p.BotSpeakingEnabled = &on },
		func(p rtvi.ObserverParams) bool { return onUnlessOff(p.BotSpeakingEnabled) },
	},
	"user_audio_level_enabled": {
		func(p *rtvi.ObserverParams, on bool) { p.UserAudioLevelEnabled = on },
		func(p rtvi.ObserverParams) bool { return p.UserAudioLevelEnabled },
	},
	"bot_audio_level_enabled": {
		func(p *rtvi.ObserverParams, on bool) { p.BotAudioLevelEnabled = on },
		func(p rtvi.ObserverParams) bool { return p.BotAudioLevelEnabled },
	},
	"user_input_enabled": {
		func(p *rtvi.ObserverParams, on bool) { p.UserInputEnabled = &on },
		func(p rtvi.ObserverParams) bool { return onUnlessOff(p.UserInputEnabled) },
	},
	"user_llm_enabled": {
		func(p *rtvi.ObserverParams, on bool) { p.UserLLMEnabled = &on },
		func(p rtvi.ObserverParams) bool { return onUnlessOff(p.UserLLMEnabled) },
	},
	"stt_raw_text_enabled": {
		func(p *rtvi.ObserverParams, on bool) { p.STTRawTextEnabled = on },
		func(p rtvi.ObserverParams) bool { return p.STTRawTextEnabled },
	},
	"user_transcription_enabled": {
		func(p *rtvi.ObserverParams, on bool) { p.UserTranscriptionEnabled = &on },
		func(p rtvi.ObserverParams) bool { return onUnlessOff(p.UserTranscriptionEnabled) },
	},
	"bot_output_enabled": {
		func(p *rtvi.ObserverParams, on bool) { p.BotOutputEnabled = &on },
		func(p rtvi.ObserverParams) bool { return onUnlessOff(p.BotOutputEnabled) },
	},
	"bot_llm_enabled": {
		func(p *rtvi.ObserverParams, on bool) { p.BotLLMEnabled = &on },
		func(p rtvi.ObserverParams) bool { return onUnlessOff(p.BotLLMEnabled) },
	},
	"bot_tts_enabled": {
		func(p *rtvi.ObserverParams, on bool) { p.BotTTSEnabled = &on },
		func(p rtvi.ObserverParams) bool { return onUnlessOff(p.BotTTSEnabled) },
	},
	"llm_raw_text_enabled": {
		func(p *rtvi.ObserverParams, on bool) { p.LLMRawTextEnabled = on },
		func(p rtvi.ObserverParams) bool { return p.LLMRawTextEnabled },
	},
	"tts_raw_text_enabled": {
		func(p *rtvi.ObserverParams, on bool) { p.TTSRawTextEnabled = on },
		func(p rtvi.ObserverParams) bool { return p.TTSRawTextEnabled },
	},
	"bot_llm_marker_enabled": {
		func(p *rtvi.ObserverParams, on bool) { p.BotLLMMarkerEnabled = on },
		func(p rtvi.ObserverParams) bool { return p.BotLLMMarkerEnabled },
	},
}

// parseVersion reads a protocol version's numbers.
func parseVersion(t *testing.T, version string) []int {
	t.Helper()
	var out []int
	for part := range strings.SplitSeq(version, ".") {
		n, err := strconv.Atoi(part)
		if err != nil {
			t.Fatalf("version %q: %v", version, err)
		}
		out = append(out, n)
	}
	return out
}

// supportedVersions are the versions with a document whose clients the server
// still serves: the current major version and LegacySupportedMajor. The
// processor rejects any other.
func supportedVersions(t *testing.T) []string {
	t.Helper()
	majors := []int{parseVersion(t, rtvi.ProtocolVersion)[0], rtvi.LegacySupportedMajor}
	var out []string
	for _, version := range spec.Versions() {
		if slices.Contains(majors, parseVersion(t, version)[0]) {
			out = append(out, version)
		}
	}
	return out
}

// at reads the value at keys in nested decoded JSON objects, or nil when
// there is none.
func at(node any, keys ...string) any {
	for _, key := range keys {
		obj, ok := node.(map[string]any)
		if !ok {
			return nil
		}
		node = obj[key]
	}
	return node
}

// asObject reads a decoded JSON object, or nil for anything else.
func asObject(node any) map[string]any {
	obj, _ := node.(map[string]any)
	return obj
}

// asList reads a decoded JSON array, or nil for anything else.
func asList(node any) []any {
	items, _ := node.([]any)
	return items
}

// asString reads a decoded JSON string, or "" for anything else.
func asString(node any) string {
	str, _ := node.(string)
	return str
}

// decodeJSON decodes JSON with its numbers as json.Number, as the schema
// checks read them.
func decodeJSON(t *testing.T, raw []byte) any {
	t.Helper()
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.UseNumber()
	var v any
	if err := dec.Decode(&v); err != nil {
		t.Fatalf("decoding %s: %v", raw, err)
	}
	return v
}

// wire is a value as it goes on the wire, decoded.
func wire(t *testing.T, v any) map[string]any {
	t.Helper()
	raw, err := json.Marshal(v)
	if err != nil {
		t.Fatal(err)
	}
	m, ok := decodeJSON(t, raw).(map[string]any)
	if !ok {
		t.Fatalf("%s is not an object", raw)
	}
	return m
}

// clientWire is a client message as it arrives on the wire. Nil data leaves
// data out.
func clientWire(msgType string, data any, id string) map[string]any {
	m := map[string]any{"label": rtvi.MessageLabel, "type": msgType, "id": id}
	if data != nil {
		m["data"] = data
	}
	return m
}

// specDoc is one protocol version's document, and checks of messages against
// it.
type specDoc struct {
	version     string
	document    map[string]any
	messages    map[string]map[string]any
	clientTypes []string
	serverTypes []string
	checker     schemaChecker
}

// loadSpec reads a protocol version's document.
func loadSpec(t *testing.T, version string) *specDoc {
	t.Helper()
	raw, err := spec.Document(version)
	if err != nil {
		t.Fatal(err)
	}
	doc, ok := decodeJSON(t, raw).(map[string]any)
	if !ok {
		t.Fatalf("spec/%s is not an object", version)
	}
	s := &specDoc{version: version, document: doc, messages: map[string]map[string]any{}, checker: schemaChecker{doc}}
	components := asObject(at(doc, "components", "messages"))
	for _, m := range components {
		msg := asObject(m)
		s.messages[asString(msg["name"])] = msg
	}
	types := func(operation string) []string {
		var out []string
		for _, ref := range asList(at(doc, "operations", operation, "messages")) {
			target := asString(at(ref, "$ref"))
			key := target[strings.LastIndex(target, "/")+1:]
			out = append(out, asString(at(components, key, "name")))
		}
		slices.Sort(out)
		return out
	}
	s.clientTypes = types("receiveClientMessages")
	s.serverTypes = types("sendServerMessages")
	return s
}

// serverTypesFor are the messages the server may send to a client of
// clientVersion.
func (s *specDoc) serverTypesFor(t *testing.T, clientVersion string) []string {
	t.Helper()
	var out []string
	for _, msgType := range s.serverTypes {
		minimum, _ := s.messages[msgType]["x-rtvi-min-client-version"].(string)
		if minimum == "" {
			minimum = "0.0.0"
		}
		if slices.Compare(parseVersion(t, minimum), parseVersion(t, clientVersion)) <= 0 {
			out = append(out, msgType)
		}
	}
	return out
}

// payload is the schema of whole messages of msgType.
func (s *specDoc) payload(msgType string) map[string]any {
	return asObject(at(s.messages[msgType], "payload"))
}

// example is the document's example of a whole message of msgType.
func (s *specDoc) example(msgType string) map[string]any {
	examples := asList(at(s.messages[msgType], "examples"))
	if len(examples) == 0 {
		return nil
	}
	return asObject(at(examples[0], "payload"))
}

// validate is what is wrong with a whole message of msgType.
func (s *specDoc) validate(msgType string, message any) []string {
	return s.checker.errors(s.payload(msgType), message, "")
}

// resolve is the schema a $ref points to, also from an allOf of one, or the
// schema itself.
func (s *specDoc) resolve(schema map[string]any) map[string]any {
	for {
		if allOf, ok := schema["allOf"].([]any); ok && len(allOf) == 1 {
			schema = asObject(allOf[0])
			continue
		}
		ref, ok := schema["$ref"].(string)
		if !ok {
			return schema
		}
		name := ref[strings.LastIndex(ref, "/")+1:]
		schema = asObject(at(s.document, "components", "schemas", name))
	}
}

// jsonTypes are the JSON types a schema allows, or none when it doesn't say.
func (s *specDoc) jsonTypes(schema map[string]any) map[string]bool {
	schema = s.resolve(schema)
	alternatives, ok := schema["anyOf"].([]any)
	if !ok {
		alternatives, ok = schema["oneOf"].([]any)
	}
	if ok {
		union := map[string]bool{}
		for _, alternative := range alternatives {
			part := s.jsonTypes(asObject(alternative))
			if len(part) == 0 {
				return map[string]bool{}
			}
			maps.Copy(union, part)
		}
		return union
	}
	if c, ok := schema["const"]; ok {
		return map[string]bool{constType(c): true}
	}
	types := map[string]bool{}
	switch t := schema["type"].(type) {
	case nil:
		if _, ok := schema["properties"]; ok {
			types["object"] = true
		}
		return types
	case string:
		types[t] = true
	case []any:
		for _, name := range t {
			types[asString(name)] = true
		}
	}
	if types["integer"] {
		types["number"] = true
	}
	return types
}

// constType is the JSON type of a const's value.
func constType(v any) string {
	switch n := v.(type) {
	case string:
		return "string"
	case bool:
		return "boolean"
	case nil:
		return "null"
	case json.Number:
		if _, err := n.Int64(); err == nil {
			return "integer"
		}
		return "number"
	default:
		return "object"
	}
}

// wrongValues are a value of each JSON type, for a field that doesn't allow
// that type.
//
//nolint:gochecknoglobals // fixed table, read only
var wrongValues = []struct {
	jsonType string
	value    any
}{
	{"object", map[string]any{"unexpected": true}},
	{"array", []any{"unexpected"}},
	{"string", "unexpected"},
	{"number", json.Number("12345")},
}

// variant is data for a client message, with what is wrong with it. Nil data
// is a message without data.
type variant struct {
	problem string
	data    any
}

// invalidData is data the document doesn't allow for a client message of
// msgType, by what's wrong.
//
// It is built from the message's schema: no data when it's required, and data
// missing each required field or with a field of the wrong type.
func (s *specDoc) invalidData(msgType string) []variant {
	payload := s.payload(msgType)
	var variants []variant
	if slices.Contains(asList(payload["required"]), any("data")) {
		variants = append(variants, variant{"no data", nil})
	}
	dataSchema := asObject(at(payload, "properties", "data"))
	if dataSchema == nil {
		dataSchema = map[string]any{}
	}
	dataSchema = s.resolve(dataSchema)
	example, isObject := s.example(msgType)["data"].(map[string]any)
	if dataSchema["type"] != "object" || !isObject {
		return s.rejected(msgType, variants)
	}
	required, _ := dataSchema["required"].([]any)
	for _, field := range required {
		without := maps.Clone(example)
		delete(without, asString(field))
		variants = append(variants, variant{"no " + asString(field), without})
	}
	properties, _ := dataSchema["properties"].(map[string]any)
	for _, field := range slices.Sorted(maps.Keys(properties)) {
		allowed := s.jsonTypes(asObject(properties[field]))
		if len(allowed) == 0 {
			continue
		}
		for _, wrong := range wrongValues {
			if !allowed[wrong.jsonType] {
				with := maps.Clone(example)
				with[field] = wrong.value
				variants = append(variants, variant{field + " of the wrong type", with})
				break
			}
		}
	}
	return s.rejected(msgType, variants)
}

// rejected keeps only the variants the document rejects: a field of any type
// has none.
func (s *specDoc) rejected(msgType string, variants []variant) []variant {
	var out []variant
	for _, v := range variants {
		if !s.checker.valid(s.payload(msgType), clientWire(msgType, v.data, "under-test")) {
			out = append(out, v)
		}
	}
	return out
}

// violations are what is wrong with a message the server sent to a client of
// clientVersion.
func (s *specDoc) violations(t *testing.T, message map[string]any, clientVersion string) []string {
	t.Helper()
	msgType, _ := message["type"].(string)
	if !slices.Contains(s.serverTypes, msgType) {
		return []string{fmt.Sprintf("%q isn't a message the server sends in %s", msgType, s.version)}
	}
	var problems []string
	if !slices.Contains(s.serverTypesFor(t, clientVersion), msgType) {
		problems = append(problems, fmt.Sprintf("%q needs a %v client, sent to %s",
			msgType, s.messages[msgType]["x-rtvi-min-client-version"], clientVersion))
	}
	for _, problem := range s.validate(msgType, message) {
		problems = append(problems, fmt.Sprintf("%q at %s", msgType, problem))
	}
	return problems
}

// allViolations are the violations of every message sent.
func (s *specDoc) allViolations(t *testing.T, sent []map[string]any, clientVersion string) []string {
	t.Helper()
	var problems []string
	for _, m := range sent {
		problems = append(problems, s.violations(t, m, clientVersion)...)
	}
	return problems
}

// specSession is a real Processor and Observer in a running pipeline, with the
// wire messages they send.
type specSession struct {
	t    *testing.T
	proc *rtvi.Processor
	task *pipeline.Worker
	done chan error

	mu sync.Mutex
	// sent are the messages sent to the client, as they went on the wire.
	sent []map[string]any
	// pushed are the other frames that reached the end of the pipeline.
	pushed []frames.Frame
}

// newSpecSession starts a session. Nil params use the observer's defaults.
func newSpecSession(t *testing.T, params *rtvi.ObserverParams) *specSession {
	t.Helper()
	p := rtvi.DefaultObserverParams()
	if params != nil {
		p = *params
	}
	proc := rtvi.NewProcessor()
	// The pipeline ends in a stand-in for the transport's output end, which
	// the bot's output is reported from.
	task := pipeline.NewWorker(pipeline.New(proc, newPlayback()), pipeline.WorkerConfig{
		Observers:               []processor.Observer{rtvi.NewObserverWithParams(proc, p)},
		ReachedDownstreamFilter: pipeline.AnyFrame,
	})
	s := &specSession{t: t, proc: proc, task: task, done: make(chan error, 1)}
	event.On(&task.Registry, pipeline.EventFrameReachedDownstream, func(_ context.Context, f frames.Frame) {
		if m, ok := f.(*frames.OutputTransportMessageUrgentFrame); ok {
			if msg, ok := m.Message.(rtvi.Message); ok {
				w := wire(t, msg)
				s.mu.Lock()
				s.sent = append(s.sent, w)
				s.mu.Unlock()
				return
			}
		}
		s.mu.Lock()
		s.pushed = append(s.pushed, f)
		s.mu.Unlock()
	})
	go func() { s.done <- task.Run(context.Background()) }()
	return s
}

// receive hands the server a message from the client. Nil data leaves data
// out.
func (s *specSession) receive(msgType string, data any, id string) {
	s.receiveRaw(clientWire(msgType, data, id))
}

// receiveRaw hands the server a transport message.
func (s *specSession) receiveRaw(message map[string]any) {
	raw, err := json.Marshal(message)
	if err != nil {
		s.t.Fatal(err)
	}
	s.task.QueueFrame(frames.NewInputTransportMessageFrame(raw))
}

// settle waits for the server to have handled every message the client sent:
// the server handles them in order, so once it has answered one more, unknown
// to it, it has handled the ones before. A session that ended has nothing more
// to handle.
func (s *specSession) settle() {
	const id = "settle"
	s.receive("not-a-message-type", nil, id)
	deadline := time.After(3 * time.Second)
	for {
		if len(s.answersTo(id)) > 0 {
			return
		}
		select {
		case err := <-s.done:
			s.done <- err
			return
		case <-deadline:
			s.t.Fatal("timed out waiting for the server to handle the client's messages")
		case <-time.After(5 * time.Millisecond):
		}
	}
}

// waitFor waits until the server has sent a message of msgType.
func (s *specSession) waitFor(msgType string) {
	deadline := time.After(3 * time.Second)
	for !slices.Contains(s.sentTypes(), msgType) {
		select {
		case <-deadline:
			s.t.Fatalf("timed out waiting for a %s message", msgType)
		case <-time.After(5 * time.Millisecond):
		}
	}
}

// push shows the pipeline a frame, giving it time to travel: the events
// bracketing playback are system frames and would otherwise overtake the data
// frames they bracket.
func (s *specSession) push(f frames.Frame) {
	s.task.QueueFrame(f)
	time.Sleep(10 * time.Millisecond)
}

// ready completes the handshake as a client of version, or of no version when
// it is empty.
func (s *specSession) ready(version string) {
	data := map[string]any{}
	if version != "" {
		data["version"] = version
	}
	s.receive(rtvi.TypeClientReady, data, "ready")
	s.waitFor(rtvi.TypeBotReady)
}

// messages are the messages sent so far.
func (s *specSession) messages() []map[string]any {
	s.mu.Lock()
	defer s.mu.Unlock()
	return slices.Clone(s.sent)
}

// sentTypes are the types of the messages sent so far, in order.
func (s *specSession) sentTypes() []string {
	var out []string
	for _, m := range s.messages() {
		out = append(out, asString(m["type"]))
	}
	return out
}

// clear forgets the messages sent so far.
func (s *specSession) clear() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.sent = nil
}

// answersTo are the messages carrying id, in order.
func (s *specSession) answersTo(id string) []map[string]any {
	var out []map[string]any
	for _, m := range s.messages() {
		if m["id"] == id {
			out = append(out, m)
		}
	}
	return out
}

// responsesTo are the types of the answers to the message with id, leaving
// out the bot-ready that completes a handshake.
func (s *specSession) responsesTo(id string) []string {
	var out []string
	for _, m := range s.answersTo(id) {
		if m["type"] != rtvi.TypeBotReady {
			out = append(out, asString(m["type"]))
		}
	}
	return out
}

// close ends the session, once what is in flight has arrived, and returns the
// messages the server sent.
func (s *specSession) close() []map[string]any {
	time.Sleep(200 * time.Millisecond)
	s.task.StopWhenDone()
	if err := <-s.done; err != nil {
		s.t.Fatal(err)
	}
	return s.messages()
}

// everything are observer settings that turn on every message the observer
// can send.
func everything() rtvi.ObserverParams {
	p := rtvi.DefaultObserverParams()
	for _, name := range slices.Sorted(maps.Keys(observerSwitches)) {
		observerSwitches[name].set(&p, true)
	}
	// Every user and bot audio frame reports its level.
	var every time.Duration
	p.AudioLevelPeriod = &every
	p.FunctionCallReportLevel = map[string]rtvi.FunctionCallReportLevel{"*": rtvi.ReportFull}
	return p
}

// runSession holds a session that makes the server send every message it can,
// with a client of clientVersion, and returns the messages the server sent.
func runSession(t *testing.T, clientVersion string, params rtvi.ObserverParams) []map[string]any {
	t.Helper()
	s := newSpecSession(t, &params)
	ctx := context.Background()

	// Session: handshake, errors, and replies to client messages.
	s.receive(rtvi.TypeClientReady,
		map[string]any{"version": clientVersion, "about": map[string]any{"library": "conformance-test"}}, "ready")
	s.waitFor(rtvi.TypeBotReady)
	s.receive("not-a-message-type", nil, "unknown")
	s.push(frames.NewErrorFrame("The LLM service failed"))
	if err := s.proc.SendServerMessage(ctx, map[string]any{"event": "order-updated"}); err != nil {
		t.Fatal(err)
	}
	clientMsg := rtvi.NewClientMessageFrame("cm", "get-weather", json.RawMessage(`{}`))
	s.push(rtvi.NewServerResponseFrame(clientMsg, map[string]any{"t": 21}))
	s.push(rtvi.NewServerErrorResponseFrame(clientMsg, "No city"))
	s.push(rtvi.NewServerMessageFrame(map[string]any{"event": "order-shipped"}))
	s.push(rtvi.NewFlowNodeEnteredFrame(rtvi.FlowNodeEnteredData{
		Node:             "verify.check_birthday",
		PreviousNode:     "greet",
		Function:         "wants_to_book",
		Metadata:         map[string]any{"step": 2},
		FunctionMetadata: map[string]any{"exit": "booking"},
		Parent:           "verify",
	}))
	// Every kind of metric, so the document's metrics schema is checked whole.
	llm := frames.BaseMetricsData{Processor: "OpenAILLM", Model: "gpt-4.1"}
	tts := frames.BaseMetricsData{Processor: "CartesiaTTS"}
	s.push(frames.NewMetricsFrame(
		frames.TTFBMetricsData{BaseMetricsData: llm, Value: 400 * time.Millisecond},
		frames.TTFAMetricsData{
			BaseMetricsData: tts, TTFA: 600 * time.Millisecond, TTFB: 200 * time.Millisecond,
			LeadingSilence: 400 * time.Millisecond,
		},
		frames.TTFATMetricsData{
			BaseMetricsData: llm, TTFAT: 900 * time.Millisecond, TTFB: 400 * time.Millisecond,
			ThinkingTime: 500 * time.Millisecond,
		},
		frames.ProcessingMetricsData{BaseMetricsData: llm, Value: 50 * time.Millisecond},
		frames.LLMUsageMetricsData{
			BaseMetricsData: llm,
			Value:           frames.LLMTokenUsage{PromptTokens: 10, CompletionTokens: 20, TotalTokens: 30},
		},
		frames.STTUsageMetricsData{Processor: "DeepgramSTT", Value: frames.STTUsage{AudioSeconds: 1.5}},
		frames.TTSUsageMetricsData{BaseMetricsData: tts, Value: 42},
	))

	// The user speaks.
	s.push(frames.NewVADUserStartedSpeakingFrame(0.2, time.Now()))
	s.push(frames.NewUserStartedSpeakingFrame())
	s.push(frames.NewInputAudioRawFrame(bytes.Repeat([]byte{1, 0}, 160), 16000, 1))
	s.push(frames.NewTranscriptionFrame("What's the weather?", "user_1", "now"))
	s.push(frames.NewVADUserStoppedSpeakingFrame(0.8, time.Now()))
	s.push(frames.NewUserStoppedSpeakingFrame())
	s.push(frames.NewUserMuteStartedFrame())
	s.push(frames.NewUserMuteStoppedFrame())
	convo := frames.NewLLMContext("system")
	convo.AddUserMessage("What's the weather?")
	s.push(frames.NewLLMContextFrame(convo))

	// The bot answers, with a function call.
	s.push(frames.NewLLMFullResponseStartFrame())
	s.push(frames.NewLLMTextFrame("It's sunny."))
	s.push(frames.NewLLMMarkerResponseFrame("✓ It's sunny.", "✓", "", nil))
	s.push(frames.NewLLMFullResponseEndFrame())
	s.push(frames.NewFunctionCallsStartedFrame([]frames.ToolCall{
		{ID: "call_1", Name: "get_weather", Args: json.RawMessage(`{"city":"Lisbon"}`)},
	}))
	s.push(frames.NewFunctionCallInProgressFrame("call_1", "get_weather", json.RawMessage(`{}`), true, "g1"))
	s.push(frames.NewFunctionCallResultFrame("call_1", "get_weather", json.RawMessage(`{}`), `{"temperature":21}`))
	s.push(frames.NewTTSStartedFrame())
	s.push(frames.NewBotStartedSpeakingFrame())
	s.push(frames.NewTTSAudioRawFrame(bytes.Repeat([]byte{1, 0}, 160), 16000, 1))
	s.push(segment("It's sunny.", frames.TextTypeSentence))
	s.push(frames.NewTTSTextFrame("It's sunny.", frames.TextTypeSentence))
	s.push(frames.NewInterruptionFrame())
	s.push(frames.NewBotStoppedSpeakingFrame())
	s.push(frames.NewTTSStoppedFrame())

	return s.close()
}

// typeSet is the set of the types of messages.
func typeSet(sent []map[string]any) map[string]bool {
	out := map[string]bool{}
	for _, m := range sent {
		out[asString(m["type"])] = true
	}
	return out
}

// implemented leaves out of types the server messages the server does not
// send.
func implemented(types []string) []string {
	return slices.DeleteFunc(slices.Clone(types), func(msgType string) bool {
		_, ok := unimplementedServerMessages[msgType]
		return ok
	})
}

// TestEveryMessageObeysItsVersion checks what the server sends obeys each
// protocol version's document.
func TestEveryMessageObeysItsVersion(t *testing.T) {
	for _, version := range supportedVersions(t) {
		t.Run(version, func(t *testing.T) {
			s := loadSpec(t, version)
			if problems := s.allViolations(t, runSession(t, version, everything()), version); len(problems) != 0 {
				t.Errorf("violations:\n%s", strings.Join(problems, "\n"))
			}
		})
	}
}

// TestTheSessionSendsEveryMessageOfTheCurrentVersion checks the session
// covers the document: a message the document lists but the session never
// sends goes unchecked. Add what makes the server send it to runSession.
func TestTheSessionSendsEveryMessageOfTheCurrentVersion(t *testing.T) {
	s := loadSpec(t, rtvi.ProtocolVersion)
	sent := typeSet(runSession(t, rtvi.ProtocolVersion, everything()))
	var missing, unexpected []string
	for _, msgType := range s.serverTypes {
		_, unimplemented := unimplementedServerMessages[msgType]
		switch {
		case unimplemented && sent[msgType]:
			unexpected = append(unexpected, msgType)
		case !unimplemented && !sent[msgType]:
			missing = append(missing, msgType)
		}
	}
	if len(missing) != 0 {
		t.Errorf("never sent: %v", missing)
	}
	if len(unexpected) != 0 {
		t.Errorf("sent, though listed as not implemented: %v", unexpected)
	}
	for msgType := range unimplementedServerMessages {
		if !slices.Contains(s.serverTypes, msgType) {
			t.Errorf("%s is listed as not implemented, but the document has no such message", msgType)
		}
	}
}

// TestOlderClientsOnlyGetTheMessagesForTheirVersion checks a client just
// below each message's minimum version gets everything else, and not those.
func TestOlderClientsOnlyGetTheMessagesForTheirVersion(t *testing.T) {
	s := loadSpec(t, rtvi.ProtocolVersion)
	minimums := map[string]bool{}
	for _, m := range s.messages {
		if minimum, ok := m["x-rtvi-min-client-version"].(string); ok {
			minimums[minimum] = true
		}
	}
	sorted := slices.Collect(maps.Keys(minimums))
	slices.SortFunc(sorted, func(a, b string) int { return slices.Compare(parseVersion(t, a), parseVersion(t, b)) })
	for _, minimum := range sorted {
		// The newest client that is still too old for these messages.
		parts := parseVersion(t, minimum)
		older := fmt.Sprintf("%d.%d.0", parts[0], parts[1]-1)
		if parts[1] == 0 {
			older = fmt.Sprintf("%d.0.0", parts[0]-1)
		}
		t.Run(older, func(t *testing.T) {
			sent := runSession(t, older, everything())
			if problems := s.allViolations(t, sent, older); len(problems) != 0 {
				t.Errorf("violations:\n%s", strings.Join(problems, "\n"))
			}
			got := slices.Sorted(maps.Keys(typeSet(sent)))
			if want := implemented(s.serverTypesFor(t, older)); !slices.Equal(got, want) {
				t.Errorf("sent %v, want %v", got, want)
			}
		})
	}
}

// receiveOne hands a fresh session one client message, with id under-test,
// checks what the server sent obeys the document of version, and returns the
// session.
func receiveOne(t *testing.T, version, msgType string, data any) *specSession {
	t.Helper()
	sess := newSpecSession(t, nil)
	sess.receive(msgType, data, "under-test")
	sess.settle()
	sent := sess.close()
	s := loadSpec(t, version)
	if problems := s.allViolations(t, sent, version); len(problems) != 0 {
		t.Errorf("violations:\n%s", strings.Join(problems, "\n"))
	}
	return sess
}

// TestExamplesMatchTheDocument checks each message's example, of the client's
// messages and the server's, is a message its schema allows, with data exactly
// when the message carries some.
func TestExamplesMatchTheDocument(t *testing.T) {
	for _, version := range spec.Versions() {
		s := loadSpec(t, version)
		for _, msgType := range slices.Sorted(maps.Keys(s.messages)) {
			t.Run(version+"/"+msgType, func(t *testing.T) {
				example := s.example(msgType)
				if problems := s.validate(msgType, example); len(problems) != 0 {
					t.Errorf("example %s: %v", show(example), problems)
				}
				_, hasData := s.payload(msgType)["properties"].(map[string]any)["data"]
				if _, exampleData := example["data"]; exampleData != hasData {
					t.Errorf("example has data: %v, message carries data: %v", exampleData, hasData)
				}
			})
		}
	}
}

// TestMessagesTheDocumentForbidsGetAnErrorResponse checks what the current
// document requires of a client message, the server enforces: it answers a
// message without required data, missing a required field, or with a field
// of the wrong type with an error-response. Older documents aren't checked: a
// later version may accept what they forbid, and their clients never send it.
func TestMessagesTheDocumentForbidsGetAnErrorResponse(t *testing.T) {
	version := rtvi.ProtocolVersion
	s := loadSpec(t, version)
	for _, msgType := range s.clientTypes {
		if _, ok := unimplementedClientMessages[msgType]; ok {
			continue
		}
		for _, v := range s.invalidData(msgType) {
			if _, ok := lenient[[2]string{msgType, v.problem}]; ok {
				continue
			}
			t.Run(msgType+"/"+v.problem, func(t *testing.T) {
				sess := receiveOne(t, version, msgType, v.data)
				if got := sess.responsesTo("under-test"); !slices.Equal(got, []string{rtvi.TypeErrorResponse}) {
					t.Errorf("answered %v, want one error-response", got)
				}
			})
		}
	}
}

// TestClientReadyIsLenientAboutTheClient checks what the client says about
// itself is advice: without it, or with one that does not describe a client,
// the session starts as with a client of an unknown library.
func TestClientReadyIsLenientAboutTheClient(t *testing.T) {
	for problem, data := range map[string]map[string]any{
		"no about":                {"version": rtvi.ProtocolVersion},
		"about of the wrong type": {"version": rtvi.ProtocolVersion, "about": []any{"unexpected"}},
		"about without a library": {"version": rtvi.ProtocolVersion, "about": map[string]any{"platform": "iOS"}},
	} {
		t.Run(problem, func(t *testing.T) {
			sess := receiveOne(t, rtvi.ProtocolVersion, rtvi.TypeClientReady, data)
			got := typesOf(sess.answersTo("under-test"))
			if !slices.Equal(got, []string{rtvi.TypeBotReady}) {
				t.Errorf("answered %v, want a bot-ready alone", got)
			}
		})
	}
}

// TestExamplesAreAccepted checks the server takes every client message as
// each document it still serves describes it, answering none with an error.
func TestExamplesAreAccepted(t *testing.T) {
	for _, version := range supportedVersions(t) {
		s := loadSpec(t, version)
		for _, msgType := range s.clientTypes {
			if _, ok := unimplementedClientMessages[msgType]; ok {
				continue
			}
			t.Run(version+"/"+msgType, func(t *testing.T) {
				sess := receiveOne(t, version, msgType, s.example(msgType)["data"])
				if got := sess.responsesTo("under-test"); len(got) != 0 {
					t.Errorf("answered %v, want nothing", got)
				}
			})
		}
	}
}

// TestUnimplementedClientMessagesAreRefused checks the client messages the
// server does not implement are in the document, and are answered with an
// error-response rather than handled in part.
func TestUnimplementedClientMessagesAreRefused(t *testing.T) {
	s := loadSpec(t, rtvi.ProtocolVersion)
	for _, msgType := range slices.Sorted(maps.Keys(unimplementedClientMessages)) {
		if !slices.Contains(s.clientTypes, msgType) {
			t.Errorf("%s is listed as not implemented, but the document has no such message", msgType)
			continue
		}
		t.Run(msgType, func(t *testing.T) {
			sess := receiveOne(t, rtvi.ProtocolVersion, msgType, s.example(msgType)["data"])
			if got := sess.responsesTo("under-test"); !slices.Equal(got, []string{rtvi.TypeErrorResponse}) {
				t.Errorf("answered %v, want one error-response", got)
			}
		})
	}
}

// TestUnknownFieldsAreIgnored checks a field the server does not know is
// ignored: a later protocol version can add one to a client message's data.
func TestUnknownFieldsAreIgnored(t *testing.T) {
	for _, version := range supportedVersions(t) {
		s := loadSpec(t, version)
		for _, msgType := range s.clientTypes {
			if _, ok := unimplementedClientMessages[msgType]; ok {
				continue
			}
			data, ok := s.example(msgType)["data"].(map[string]any)
			if !ok {
				continue
			}
			t.Run(version+"/"+msgType, func(t *testing.T) {
				withUnknown := maps.Clone(data)
				withUnknown["x_unknown"] = json.Number("1")
				sess := receiveOne(t, version, msgType, withUnknown)
				if got := sess.responsesTo("under-test"); len(got) != 0 {
					t.Errorf("answered %v, want nothing", got)
				}
			})
		}
	}
}

// typesOf are the types of messages, in order.
func typesOf(msgs []map[string]any) []string {
	var out []string
	for _, m := range msgs {
		out = append(out, asString(m["type"]))
	}
	return out
}

// TestEnabledByNamesAnObserverSetting checks each setting a document says
// turns a message on is one of the observer's, on by default exactly when the
// document says so.
func TestEnabledByNamesAnObserverSetting(t *testing.T) {
	defaults := rtvi.DefaultObserverParams()
	for _, version := range spec.Versions() {
		s := loadSpec(t, version)
		for _, msgType := range slices.Sorted(maps.Keys(s.messages)) {
			m := s.messages[msgType]
			setting, ok := m["x-rtvi-enabled-by"].(string)
			if !ok {
				continue
			}
			if _, ok := unimplementedSettings[setting]; ok {
				if _, ok := unimplementedServerMessages[msgType]; !ok {
					t.Errorf("spec/%s: %s is turned on by %s, which is listed as not implemented",
						version, msgType, setting)
				}
				continue
			}
			sw, ok := observerSwitches[setting]
			if !ok {
				t.Errorf("spec/%s: %s is turned on by %s, which is not an observer setting", version, msgType, setting)
				continue
			}
			summary, _ := m["summary"].(string)
			onByDefault := strings.Contains(summary, "On by default;")
			if sw.get(defaults) != onByDefault {
				t.Errorf("spec/%s: %s is on by default: %v, the document says %v",
					version, setting, sw.get(defaults), onByDefault)
			}
		}
	}
}

// constants are the string constants declared in the package's source files,
// by name.
func constants(t *testing.T, files ...string) map[string]string {
	t.Helper()
	out := map[string]string{}
	fset := token.NewFileSet()
	for _, name := range files {
		file, err := parser.ParseFile(fset, name, nil, 0)
		if err != nil {
			t.Fatal(err)
		}
		for _, decl := range file.Decls {
			gen, ok := decl.(*ast.GenDecl)
			if !ok || gen.Tok != token.CONST {
				continue
			}
			for _, s := range gen.Specs {
				vs, ok := s.(*ast.ValueSpec)
				if !ok {
					continue
				}
				for i, ident := range vs.Names {
					if i >= len(vs.Values) {
						continue
					}
					if lit, ok := vs.Values[i].(*ast.BasicLit); ok && lit.Kind == token.STRING {
						value, err := strconv.Unquote(lit.Value)
						if err != nil {
							t.Fatal(err)
						}
						out[ident.Name] = value
					}
				}
			}
		}
	}
	return out
}

// TestEveryMessageTypeIsInTheSpec checks each message type the package
// declares is one the current document describes: a message the server sends
// or accepts needs its entry in the document, with a description and an
// example, even before anything sends it.
func TestEveryMessageTypeIsInTheSpec(t *testing.T) {
	s := loadSpec(t, rtvi.ProtocolVersion)
	declared := constants(t, "messages.go")
	var missing []string
	for name, value := range declared {
		if !strings.HasPrefix(name, "Type") {
			continue
		}
		if _, ok := s.messages[value]; !ok {
			missing = append(missing, value)
		}
	}
	if len(missing) == 0 && len(declared) == 0 {
		t.Fatal("found no message type in messages.go")
	}
	slices.Sort(missing)
	if len(missing) != 0 {
		t.Errorf("not in the document: %v", missing)
	}
}

// handledTypes are the message types Processor.handleMessage has a case for.
func handledTypes(t *testing.T) []string {
	t.Helper()
	declared := constants(t, "messages.go")
	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, "processor.go", nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	var types []string
	for _, decl := range file.Decls {
		fn, ok := decl.(*ast.FuncDecl)
		if !ok || fn.Name.Name != "handleMessage" || fn.Recv == nil {
			continue
		}
		ast.Inspect(fn.Body, func(n ast.Node) bool {
			sw, ok := n.(*ast.SwitchStmt)
			if !ok || types == nil && !isSelector(sw.Tag, "in", "Type") {
				return true
			}
			for _, stmt := range sw.Body.List {
				clause, ok := stmt.(*ast.CaseClause)
				if !ok {
					continue
				}
				for _, expr := range clause.List {
					switch e := expr.(type) {
					case *ast.Ident:
						types = append(types, declared[e.Name])
					case *ast.BasicLit:
						value, _ := strconv.Unquote(e.Value)
						types = append(types, value)
					}
				}
			}
			return false
		})
	}
	slices.Sort(types)
	return types
}

// isSelector reports whether expr is x.sel.
func isSelector(expr ast.Expr, x, sel string) bool {
	s, ok := expr.(*ast.SelectorExpr)
	if !ok || s.Sel.Name != sel {
		return false
	}
	id, ok := s.X.(*ast.Ident)
	return ok && id.Name == x
}

// TestEveryHandledClientMessageIsInTheSpec checks a message the processor
// handles is one the document describes, and a client message the document
// describes has its case in handleMessage, unless it is listed as not
// implemented.
func TestEveryHandledClientMessageIsInTheSpec(t *testing.T) {
	handled := handledTypes(t)
	if len(handled) == 0 {
		t.Fatal("found no case in handleMessage")
	}
	s := loadSpec(t, rtvi.ProtocolVersion)
	want := slices.DeleteFunc(slices.Clone(s.clientTypes), func(msgType string) bool {
		_, ok := unimplementedClientMessages[msgType]
		return ok
	})
	if !slices.Equal(handled, want) {
		t.Errorf("handleMessage handles %v, want %v", handled, want)
	}
}

// behaviorTests are the tests of the documents' behaviors, by id. The test of
// a behavior is named after its id: TestBehavior and the id in camel case.
//
//nolint:gochecknoglobals // fixed table, read only
var behaviorTests = map[string]func(t *testing.T){
	"bot-ready-answers-client-ready":      TestBehaviorBotReadyAnswersClientReady,
	"bot-ready-version":                   TestBehaviorBotReadyVersion,
	"current-version-before-client-ready": TestBehaviorCurrentVersionBeforeClientReady,
	"client-ready-version-check":          TestBehaviorClientReadyVersionCheck,
	"failed-request-gets-error-response":  TestBehaviorFailedRequestGetsErrorResponse,
	"invalid-message-gets-error":          TestBehaviorInvalidMessageGetsError,
	"client-message-answer":               TestBehaviorClientMessageAnswer,
	"disconnect-bot-ends-session":         TestBehaviorDisconnectBotEndsSession,
	"messages-turned-off":                 TestBehaviorMessagesTurnedOff,
	"spoken-output-waits-for-speech":      TestBehaviorSpokenOutputWaitsForSpeech,
	"interruption-drops-unspoken-output":  TestBehaviorInterruptionDropsUnspokenOutput,
}

// TestEveryBehaviorHasATest checks each behavior of a document the server
// still serves has its test.
func TestEveryBehaviorHasATest(t *testing.T) {
	for _, version := range supportedVersions(t) {
		for _, b := range asList(loadSpec(t, version).document["x-rtvi-behaviors"]) {
			id := asString(at(b, "id"))
			test, ok := behaviorTests[id]
			if !ok {
				t.Errorf("spec/%s: behavior %q has no test", version, id)
				continue
			}
			name := "TestBehavior"
			for part := range strings.SplitSeq(id, "-") {
				name += strings.ToUpper(part[:1]) + part[1:]
			}
			if got := runtime.FuncForPC(reflect.ValueOf(test).Pointer()).Name(); !strings.HasSuffix(got, "."+name) {
				t.Errorf("behavior %q is tested by %s, want %s", id, got, name)
			}
		}
	}
}

// TestBehaviorBotReadyAnswersClientReady checks the bot-ready-answers-client-ready behavior.
func TestBehaviorBotReadyAnswersClientReady(t *testing.T) {
	s := newSpecSession(t, nil)
	s.ready(rtvi.ProtocolVersion)
	var ids []any
	for _, m := range s.close() {
		if m["type"] == rtvi.TypeBotReady {
			ids = append(ids, m["id"])
		}
	}
	if !reflect.DeepEqual(ids, []any{"ready"}) {
		t.Errorf("bot-ready ids = %v, want [ready]", ids)
	}
}

// TestBehaviorBotReadyVersion checks the bot-ready-version behavior.
func TestBehaviorBotReadyVersion(t *testing.T) {
	major, _, _ := strings.Cut(rtvi.ProtocolVersion, ".")
	for _, tc := range []struct{ client, want string }{
		{rtvi.ProtocolVersion, rtvi.ProtocolVersion},
		{major + ".0.0", rtvi.ProtocolVersion},
		{"1.2.0", "1.2.0"},
	} {
		t.Run(tc.client, func(t *testing.T) {
			s := newSpecSession(t, nil)
			s.ready(tc.client)
			var versions []any
			for _, m := range s.close() {
				if m["type"] == rtvi.TypeBotReady {
					versions = append(versions, at(m, "data", "version"))
				}
			}
			if !reflect.DeepEqual(versions, []any{tc.want}) {
				t.Errorf("bot-ready versions = %v, want [%s]", versions, tc.want)
			}
		})
	}
}

// TestBehaviorCurrentVersionBeforeClientReady checks the current-version-before-client-ready behavior.
func TestBehaviorCurrentVersionBeforeClientReady(t *testing.T) {
	// llm-raw-text is for clients of 2.2.0 and later.
	params := rtvi.DefaultObserverParams()
	params.LLMRawTextEnabled = true
	s := newSpecSession(t, &params)
	s.push(frames.NewLLMTextFrame("Hello."))
	time.Sleep(50 * time.Millisecond)
	if !slices.Contains(s.sentTypes(), rtvi.TypeLLMRawText) {
		t.Errorf("before client-ready sent %v, want an llm-raw-text", s.sentTypes())
	}
	s.clear()
	s.ready("2.1.0")
	s.push(frames.NewLLMTextFrame("Hello."))
	if got := typesOf(s.close()); slices.Contains(got, rtvi.TypeLLMRawText) {
		t.Errorf("after a 2.1.0 client-ready sent %v, want no llm-raw-text", got)
	}
}

// TestBehaviorClientReadyVersionCheck checks the client-ready-version-check behavior.
func TestBehaviorClientReadyVersionCheck(t *testing.T) {
	major := parseVersion(t, rtvi.ProtocolVersion)[0]
	for _, tc := range []struct {
		version   string
		complains bool
	}{
		{"", true},
		{"2.x", true},
		{"2.3", true},
		{strconv.Itoa(major+1) + ".0.0", true},
		{rtvi.ProtocolVersion, false},
		{"1.0.0", false},
	} {
		t.Run(tc.version, func(t *testing.T) {
			s := newSpecSession(t, nil)
			s.ready(tc.version)
			s.close()
			want := []string{rtvi.TypeBotReady}
			if tc.complains {
				want = []string{rtvi.TypeErrorResponse, rtvi.TypeBotReady}
			}
			if got := typesOf(s.answersTo("ready")); !slices.Equal(got, want) {
				t.Errorf("answered %v, want %v", got, want)
			}
		})
	}
}

// TestBehaviorFailedRequestGetsErrorResponse checks the failed-request-gets-error-response behavior.
func TestBehaviorFailedRequestGetsErrorResponse(t *testing.T) {
	s := loadSpec(t, rtvi.ProtocolVersion)
	malformed := []string{"not-a-message-type"}
	for _, msgType := range s.clientTypes {
		if slices.Contains(asList(s.payload(msgType)["required"]), any("data")) {
			malformed = append(malformed, msgType)
		}
	}
	for _, msgType := range malformed {
		t.Run(msgType, func(t *testing.T) {
			sess := newSpecSession(t, nil)
			sess.receive(msgType, nil, "under-test")
			sess.settle()
			sess.close()
			if got := sess.responsesTo("under-test"); !slices.Equal(got, []string{rtvi.TypeErrorResponse}) {
				t.Errorf("answered %v, want one error-response", got)
			}
		})
	}
}

// TestBehaviorInvalidMessageGetsError checks the invalid-message-gets-error behavior.
func TestBehaviorInvalidMessageGetsError(t *testing.T) {
	// A message labeled as RTVI that is not an RTVI message, such as one
	// without an id or with data that is not an object, gets an error, since
	// there is no id to answer.
	for name, raw := range map[string]map[string]any{
		"no id": {"label": rtvi.MessageLabel, "type": rtvi.TypeSendText, "data": map[string]any{"content": "Hi"}},
		"no type": {
			"label": rtvi.MessageLabel, "id": "m1", "data": map[string]any{"content": "Hi"},
		},
		"data that is not an object": {
			"label": rtvi.MessageLabel, "type": rtvi.TypeSendText, "id": "m1", "data": []any{"Hi"},
		},
	} {
		t.Run(name, func(t *testing.T) {
			s := newSpecSession(t, nil)
			s.receiveRaw(raw)
			s.settle()
			var got []string
			for _, m := range s.close() {
				if m["id"] != "settle" {
					got = append(got, asString(m["type"]))
				}
			}
			if !slices.Equal(got, []string{rtvi.TypeError}) {
				t.Errorf("sent %v, want one error", got)
			}
		})
	}

	// A message without the label is not one for the server.
	s := newSpecSession(t, nil)
	s.receiveRaw(map[string]any{"label": "other", "type": rtvi.TypeSendText, "id": "1"})
	s.settle()
	for _, m := range s.close() {
		if m["id"] != "settle" {
			t.Errorf("a message for another protocol drew %v", m)
		}
	}
}

// TestBehaviorClientMessageAnswer checks the client-message-answer behavior.
func TestBehaviorClientMessageAnswer(t *testing.T) {
	s := newSpecSession(t, nil)
	s.receive(rtvi.TypeClientMessage, map[string]any{"t": "get-weather", "d": map[string]any{}}, "cm")
	s.settle()
	var request *rtvi.ClientMessageFrame
	s.mu.Lock()
	for _, f := range s.pushed {
		if cm, ok := f.(*rtvi.ClientMessageFrame); ok {
			request = cm
		}
	}
	s.mu.Unlock()
	if request == nil {
		s.close()
		t.Fatal("the client-message reached no one in the pipeline")
	}
	s.push(rtvi.NewServerResponseFrame(request, map[string]any{"temperature": 21}))
	s.push(rtvi.NewServerErrorResponseFrame(request, "No city"))
	ctx := context.Background()
	if err := s.proc.SendServerResponse(ctx, request, map[string]any{"temperature": 21}); err != nil {
		t.Fatal(err)
	}
	time.Sleep(10 * time.Millisecond)
	if err := s.proc.SendErrorResponse(ctx, request, "No city"); err != nil {
		t.Fatal(err)
	}
	s.close()
	answers := s.answersTo("cm")
	want := []string{
		rtvi.TypeServerResponse, rtvi.TypeErrorResponse, rtvi.TypeServerResponse, rtvi.TypeErrorResponse,
	}
	if got := typesOf(answers); !slices.Equal(got, want) {
		t.Errorf("answered %v, want %v", got, want)
	}
	for _, answer := range answers {
		if answer["type"] == rtvi.TypeServerResponse && at(answer, "data", "t") != "get-weather" {
			t.Errorf("server-response t = %v, want get-weather", answer["data"])
		}
	}
}

// TestBehaviorDisconnectBotEndsSession checks the disconnect-bot-ends-session behavior.
func TestBehaviorDisconnectBotEndsSession(t *testing.T) {
	s := newSpecSession(t, nil)
	s.receive(rtvi.TypeDisconnectBot, nil, "msg_1")
	s.settle()
	s.close()
	s.mu.Lock()
	defer s.mu.Unlock()
	if !slices.ContainsFunc(s.pushed, func(f frames.Frame) bool {
		_, ok := f.(*frames.EndWorkerFrame)
		return ok
	}) {
		t.Error("disconnect-bot did not end the worker")
	}
}

// TestBehaviorMessagesTurnedOff checks the messages-turned-off behavior.
func TestBehaviorMessagesTurnedOff(t *testing.T) {
	s := loadSpec(t, rtvi.ProtocolVersion)
	// The setting that turns each message on.
	switches := map[string]string{}
	for name, m := range s.messages {
		if setting, ok := m["x-rtvi-enabled-by"].(string); ok {
			if _, unimplemented := unimplementedServerMessages[name]; !unimplemented {
				switches[name] = setting
			}
		}
	}
	settings := map[string]bool{}
	for _, setting := range switches {
		settings[setting] = true
	}
	for _, setting := range slices.Sorted(maps.Keys(settings)) {
		t.Run(setting, func(t *testing.T) {
			params := everything()
			observerSwitches[setting].set(&params, false)
			sent := typeSet(runSession(t, rtvi.ProtocolVersion, params))
			for msgType, by := range switches {
				switch {
				case by == setting && sent[msgType]:
					t.Errorf("%s sent with %s off", msgType, setting)
				case by != setting && !sent[msgType]:
					// Turning one off leaves the rest on.
					t.Errorf("%s not sent with only %s off", msgType, setting)
				}
			}
		})
	}
}

// TestBehaviorSpokenOutputWaitsForSpeech checks the spoken-output-waits-for-speech behavior.
func TestBehaviorSpokenOutputWaitsForSpeech(t *testing.T) {
	s := newSpecSession(t, nil)
	s.ready(rtvi.ProtocolVersion)
	s.push(segment("It's sunny.", frames.TextTypeSentence))
	s.push(segment("In Lisbon.", frames.TextTypeSentence))
	time.Sleep(50 * time.Millisecond)
	if slices.Contains(s.sentTypes(), rtvi.TypeBotOutput) {
		t.Errorf("sent %v before the bot spoke, want no bot-output", s.sentTypes())
	}
	s.push(speaking())
	var after []string
	for _, m := range s.close() {
		switch m["type"] {
		case rtvi.TypeBotStartedSpeaking:
			after = append(after, rtvi.TypeBotStartedSpeaking)
		case rtvi.TypeBotOutput:
			after = append(after, rtvi.TypeBotOutput+" "+asString(at(m, "data", "text")))
		}
	}
	want := []string{rtvi.TypeBotStartedSpeaking, "bot-output It's sunny.", "bot-output In Lisbon."}
	if !slices.Equal(after, want) {
		t.Errorf("sent %q, want %q", after, want)
	}
}

// TestBehaviorInterruptionDropsUnspokenOutput checks the interruption-drops-unspoken-output behavior.
func TestBehaviorInterruptionDropsUnspokenOutput(t *testing.T) {
	s := newSpecSession(t, nil)
	s.ready(rtvi.ProtocolVersion)
	s.push(segment("It's sunny.", frames.TextTypeSentence))
	s.push(frames.NewInterruptionFrame())
	s.push(speaking())
	got := typesOf(s.close())
	if !slices.Contains(got, rtvi.TypeBotInterrupted) {
		t.Errorf("sent %v, want a bot-interrupted", got)
	}
	if slices.Contains(got, rtvi.TypeBotOutput) {
		t.Errorf("sent %v, want no bot-output", got)
	}
}
