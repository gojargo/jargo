// Package eval is a behavioral eval harness for jargo bots. It drives a real
// bot over RTVI, plays scripted conversation turns, and asserts on the semantic
// events the bot emits — a level above unit tests, which check one processor at
// a time.
//
// A scenario is a YAML file of turns and the events each turn should produce.
// The same scenario runs from a Go test (the bot hosted in-process, see Run) or
// from the command line against a running bot. This first iteration covers
// text-mode scenarios: each user turn is delivered as RTVI send-text and the
// assertions read the bot's LLM output and tool calls. Audio-mode scenarios
// (synthesized user speech, transcription of the bot's audio) build on the same
// core later.
//
// # Expectation fields
//
//	event: <name>            required: the event to match
//	within_ms: <int>         latency budget, measured from the turn's user input
//	text_contains: <str>     substring check on the event's text, case-sensitive
//	eval: <str>              criterion an LLM judge checks the bot's reply against
//	name: <str>              for function_call: the tool the call must be to
//	args: <mapping>          for function_call: an argument subset the call must carry
//	calls: <list>            for function_call: several calls, in any order
//	absent: true             assert the event does NOT arrive
//
// # Turn fields
//
//	user: <str>              what the user says, sent as text or synthesized
//	dtmf: <str>              keypad keys the user presses; quote it, # is a comment
//	expect: <list>           the events to match, in order
//	send_after: <mapping>    when to send this turn's input (see below)
//
// user and dtmf are mutually exclusive, and both are optional: a turn with
// neither only waits and asserts, which is what a bot-first scenario needs to
// check an opening greeting. expect is optional too, for a turn that only sends.
//
// # Scenario fields
//
//	name: <str>              required: the scenario's name
//	turns: <list>            the turns, played in order
//	context: <list>          messages the bot's context starts from
//
// Any value can come from a separate file with !include, resolved against the
// scenario file's directory, so scenarios can share a block they all need:
//
//	turns: !include shared_turns.yaml
//
// A turn often answers in more than one response: an interim filler ("Let me
// check on that.") and then the answer. So a content check on llm_response
// aggregates. It accumulates the text of successive responses and re-checks on
// each, until the check passes, the judge rejects, or within_ms expires. A
// missing substring is not a failure on its own, because more text may follow,
// which is why an assertion on text the bot never produces waits out its whole
// budget. Set within_ms on one to keep that short.
//
// A judge grades the conversation rather than one reply on its own: it is given
// each user turn and each segment of the bot's reply, so a terse answer is read
// against the question it answers. It may also answer "continue", meaning the
// reply so far is only filler and the criterion should be judged again once
// more arrives.
//
// A function_call expectation holds the set of calls the turn should make.
// They are matched by name in any order and the expectation passes only when
// every one is found. Write a single call with the name/args shorthand:
//
//	expect:
//	  - event: function_call
//	    name: get_weather
//	    args: {city: Paris}
//
// and several under calls:, where an entry is a bare tool name or a mapping:
//
//	expect:
//	  - event: function_call
//	    calls:
//	      - get_weather
//	      - {name: get_restaurants, args: {city: Paris}}
//
// args is a subset check: every argument listed must be present with that
// value, and any further argument the model passed is ignored. A bare
// function_call, with neither name nor calls, matches whatever the bot calls.
// Asserting on a call's name or its arguments requires the bot to report them,
// which the harness arranges for itself (see Handler).
//
// absent: true inverts an expectation. It passes only when no event of that
// type arrives before the within_ms budget expires, and fails as soon as one
// does, which is how "must not answer twice" is written:
//
//	expect:
//	  - event: llm_response
//	    eval: "answers the question"
//	  - event: llm_response
//	    absent: true
//	    within_ms: 5000
//
// It matches on event type alone, so no content or call check may accompany it,
// and it waits out its whole budget: set within_ms explicitly to keep the quiet
// window short.
//
// # Scheduling a turn
//
// A turn's input goes out as soon as the previous turn finishes, unless the turn
// carries send_after. That waits for an event to have been seen, then waits
// delay_ms longer, which is how an interruption is written:
//
//	turns:
//	  - user: "tell me about Paris"
//	    expect:
//	      - event: llm_started
//	  - user: "actually, tell me about Tokyo"
//	    send_after: {event: llm_started, delay_ms: 500}
//	    expect:
//	      - event: bot_interrupted
//
// An event seen earlier in the run anchors the delay at that earlier sighting,
// so the turn may fire at once. The wait for the event is bounded at 30s, and a
// turn whose schedule never fires reports the schedule rather than any of its
// expectations.
//
// event is optional. On its own, delay_ms is a pure time delay measured from the
// previous turn's send, for pacing turns where there is no event to anchor on.
//
// # Diagnosing a run
//
// A Result carries more than its failures: how long the run took, every event
// the bot emitted whether or not the scenario asserted on it, and a timestamped
// trace of what the harness itself decided. Together they are what makes a
// scenario that failed once and passed the next time readable. Options.OnProgress
// reports the same as it happens, for a caller watching a long run.
package eval

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/gojargo/jargo/frames"
	"gopkg.in/yaml.v3"
)

// Validation errors. Callers surface these to the operator; they are sentinels
// so a message change does not silently break a caller matching on them.
//
//nolint:gochecknoglobals // sentinel errors
var (
	errNoName          = errors.New("scenario has no name")
	errNoTurns         = errors.New("scenario has no turns")
	errUserAndDTMF     = errors.New("a turn takes user or dtmf, not both")
	errDTMFNotKeys     = errors.New("dtmf must be a string of keypad keys")
	errDTMFKey         = errors.New("invalid keypad key")
	errScheduleNoInput = errors.New("send_after needs a user or dtmf turn to schedule")
	errUnknownEvent    = errors.New("unknown event")
	errFieldForEvent   = errors.New("field not valid for this event")
	errNegativeBudget  = errors.New("within_ms must not be negative")
	errEmptyCalls      = errors.New("calls must not be empty")
	errAbsentWithCheck = errors.New("absent cannot be combined with a content or call check")
	errNegativeDelay   = errors.New("send_after delay_ms must be non-negative")
	errEmptySendAfter  = errors.New("send_after needs an event or a positive delay_ms")
	errIncludeNotPath  = errors.New("!include expects a file path")
	errIncludeTooDeep  = errors.New("!include nested too deep, check for a cycle")
)

// Event names a scenario can assert on. These are the friendly names scenarios
// use; the harness maps the bot's RTVI server messages onto them.
const (
	// EventUserStartedSpeaking fires when the bot's VAD detects the user's speech
	// beginning (audio mode only).
	EventUserStartedSpeaking = "user_started_speaking"
	// EventUserStoppedSpeaking fires when the user's turn ends (audio mode only).
	EventUserStoppedSpeaking = "user_stopped_speaking"
	// EventUserTranscription carries the bot's STT transcription of the user's
	// speech (audio mode only).
	EventUserTranscription = "user_transcription"
	// EventLLMStarted fires when the bot begins generating a response.
	EventLLMStarted = "llm_started"
	// EventLLMResponse carries the bot's LLM text, joined across the response.
	EventLLMResponse = "llm_response"
	// EventFunctionCall fires when the bot invokes a tool.
	EventFunctionCall = "function_call"
	// EventBotInterrupted fires when the bot's in-flight output is cut off, by a
	// barge-in or by a turn sent with send_after. It is the event a barge-in
	// scenario asserts on.
	EventBotInterrupted = "bot_interrupted"
	// EventVADUserStartedSpeaking is the raw VAD signal, ungated by turn
	// detection (audio mode only). Useful as a timing anchor when a turn strategy
	// gates or defers the turn-level user_stopped_speaking.
	EventVADUserStartedSpeaking = "vad_user_started_speaking"
	// EventVADUserStoppedSpeaking is the raw VAD signal for the end of speech
	// (audio mode only).
	EventVADUserStoppedSpeaking = "vad_user_stopped_speaking"
)

// knownEvents is the set of event names a scenario may assert on. The user_*
// events only fire in audio mode (see Options.UserTTS).
//
//nolint:gochecknoglobals // fixed lookup table
var knownEvents = map[string]bool{
	EventUserStartedSpeaking:    true,
	EventUserStoppedSpeaking:    true,
	EventUserTranscription:      true,
	EventLLMStarted:             true,
	EventLLMResponse:            true,
	EventFunctionCall:           true,
	EventBotInterrupted:         true,
	EventVADUserStartedSpeaking: true,
	EventVADUserStoppedSpeaking: true,
}

// vadEvents are the events the bot reports only when asked to. The harness asks
// when a scenario references one, and not otherwise.
//
//nolint:gochecknoglobals // fixed lookup table
var vadEvents = map[string]bool{
	EventVADUserStartedSpeaking: true,
	EventVADUserStoppedSpeaking: true,
}

// Scenario is a scripted conversation and the events it should produce.
type Scenario struct {
	// Name identifies the scenario in reports; required.
	Name string `yaml:"name"`
	// Turns are played in order.
	Turns []Turn `yaml:"turns"`
	// Context is the conversation the bot's context should start from. When set,
	// the harness sends it once the bot is ready, replacing whatever context the
	// bot built for itself. Leave it out to test the bot's own opening state.
	Context []frames.Message `yaml:"context,omitempty"`
}

// Turn is one step of a scenario: input for the bot, expectations about what it
// does, or both.
//
// A turn drives the bot one of two ways: the harness sends a user utterance, or
// it sends a sequence of keypad presses. The two are mutually exclusive, and
// both are optional. A turn with neither only waits and asserts, which is what a
// bot-first scenario needs: nothing to say, only an opening greeting to check.
type Turn struct {
	// User is the text the user "says" this turn (sent as RTVI send-text).
	User string `yaml:"user,omitempty"`
	// DTMF is a keypad sequence the user presses, one frame per key. Mutually
	// exclusive with User. Quote it in YAML: an unquoted # starts a comment.
	DTMF string `yaml:"dtmf,omitempty"`
	// Expect lists the events to match, in order, after the turn's input.
	// Optional: a turn may just send input, with the assertion on a later turn.
	Expect []Expectation `yaml:"expect,omitempty"`
	// SendAfter, when set, schedules when the turn's input is sent rather than
	// sending it as soon as the previous turn finishes.
	SendAfter *SendAfter `yaml:"send_after,omitempty"`
}

// UnmarshalYAML decodes a turn, normalizing a keypad sequence written as a bare
// number. YAML reads `dtmf: 123` as an integer, and the digits matter.
func (t *Turn) UnmarshalYAML(node *yaml.Node) error {
	type raw Turn // a distinct type, so decoding does not recurse here
	var r raw
	if err := node.Decode(&r); err != nil {
		// A bare numeric dtmf fails to decode into a string, so retry with the
		// keys the strict decode choked on read as scalars.
		return decodeTurnWithNumericDTMF(node, t)
	}
	*t = Turn(r)
	return nil
}

// decodeTurnWithNumericDTMF re-reads a turn whose dtmf: was written unquoted, so
// `dtmf: 123` and `dtmf: "123#"` mean the same thing.
func decodeTurnWithNumericDTMF(node *yaml.Node, t *Turn) error {
	type raw struct {
		User      string        `yaml:"user,omitempty"`
		DTMF      any           `yaml:"dtmf,omitempty"`
		Expect    []Expectation `yaml:"expect,omitempty"`
		SendAfter *SendAfter    `yaml:"send_after,omitempty"`
	}
	var r raw
	if err := node.Decode(&r); err != nil {
		return err
	}
	*t = Turn{User: r.User, Expect: r.Expect, SendAfter: r.SendAfter}
	switch v := r.DTMF.(type) {
	case nil:
	case string:
		t.DTMF = v
	case int:
		t.DTMF = strconv.Itoa(v)
	default:
		return fmt.Errorf("%w: %v", errDTMFNotKeys, r.DTMF)
	}
	return nil
}

// SendAfter schedules when a turn's input is sent. The harness waits for Event
// to have been seen, either earlier in the run or arriving now, then waits
// DelayMS longer before sending. That is how a barge-in is written:
// `send_after: {event: llm_started, delay_ms: 500}` interrupts 500ms after the
// bot started responding.
//
// Event is optional. On its own, DelayMS is a pure time delay measured from the
// previous turn's send, with no event to anchor on.
type SendAfter struct {
	// Event is the event to schedule from, or empty for a pure delay.
	Event string `yaml:"event,omitempty"`
	// DelayMS is how long to wait after the event was seen, or, with no event,
	// after the previous turn's send.
	DelayMS int `yaml:"delay_ms,omitempty"`
}

// FunctionCall is one expected tool call within a function_call expectation.
type FunctionCall struct {
	// Name is the tool name to match. An empty name matches any call, which is
	// what a bare function_call expectation asserts.
	Name string `yaml:"name,omitempty"`
	// Args, when set, is a subset check on the call's arguments: every listed
	// key must be present with the listed value, and any further argument the
	// model passed is ignored.
	Args map[string]any `yaml:"args,omitempty"`
}

// Expectation is one assertion about an event the bot emits.
type Expectation struct {
	// Event is the friendly event name to match (see the Event constants).
	Event string `yaml:"event"`
	// TextContains, when set, requires the event's text to contain this
	// substring (case-insensitive). Applies to llm_response.
	TextContains string `yaml:"text_contains,omitempty"`
	// Eval, when set, is a natural-language criterion an LLM judge checks the
	// bot's reply against. Applies to llm_response.
	Eval string `yaml:"eval,omitempty"`
	// Name is the single-call shorthand for Calls: the tool name a function_call
	// event must match.
	Name string `yaml:"name,omitempty"`
	// Args is the single-call shorthand for Calls: the argument subset the
	// matched call must carry.
	Args map[string]any `yaml:"args,omitempty"`
	// Calls is the set of tool calls the turn should make, for a function_call
	// event. They are matched by name in any order and the expectation passes
	// only when all of them are found. Built from `calls:`, or from the
	// `name:`/`args:` shorthand, by normalizeCalls.
	Calls []FunctionCall `yaml:"calls,omitempty"`
	// Absent inverts the expectation: it passes only when no event of this type
	// arrives before the WithinMS budget expires, and fails as soon as one does.
	// It matches on event type only, so no content check may accompany it.
	Absent bool `yaml:"absent,omitempty"`
	// WithinMS is the latency budget for the event, measured from the user input;
	// zero uses the harness default.
	WithinMS int `yaml:"within_ms,omitempty"`

	// hasCalls records that the scenario wrote a `calls:` key, which is how an
	// empty list is told from an omitted one.
	hasCalls bool `yaml:"-"`
}

// includeDepth bounds how deep !include may nest, so a file that includes
// itself is reported rather than exhausting the stack.
const includeDepth = 16

// Load reads and validates a scenario YAML file.
func Load(path string) (*Scenario, error) {
	node, err := loadNode(path, includeDepth)
	if err != nil {
		return nil, err
	}
	var s Scenario
	if err := node.Decode(&s); err != nil {
		return nil, fmt.Errorf("eval: parse %s: %w", path, err)
	}
	if err := s.validate(); err != nil {
		return nil, fmt.Errorf("eval: %s: %w", path, err)
	}
	return &s, nil
}

// loadNode parses a YAML file and resolves the !include tags in it.
func loadNode(path string, depth int) (*yaml.Node, error) {
	data, err := os.ReadFile(path) //nolint:gosec // scenario path is operator-supplied
	if err != nil {
		return nil, fmt.Errorf("eval: read scenario: %w", err)
	}
	var doc yaml.Node
	if err := yaml.Unmarshal(data, &doc); err != nil {
		return nil, fmt.Errorf("eval: parse %s: %w", path, err)
	}
	if err := resolveIncludes(&doc, filepath.Dir(path), depth); err != nil {
		return nil, fmt.Errorf("eval: %s: %w", path, err)
	}
	return &doc, nil
}

// resolveIncludes replaces every `!include <path>` node in the tree with the
// contents of the file it names, resolved against dir. Included files are parsed
// the same way, so an include may itself include.
//
// It is what lets scenarios share a block they all need, rather than each
// repeating it.
func resolveIncludes(node *yaml.Node, dir string, depth int) error {
	if node.Tag == "!include" {
		if depth <= 0 {
			return errIncludeTooDeep
		}
		if node.Kind != yaml.ScalarNode {
			return errIncludeNotPath
		}
		included, err := loadNode(filepath.Join(dir, node.Value), depth-1)
		if err != nil {
			return err
		}
		// A parsed file is a document node wrapping the value it holds.
		if included.Kind == yaml.DocumentNode && len(included.Content) == 1 {
			included = included.Content[0]
		}
		*node = *included
		return nil
	}
	for _, child := range node.Content {
		if err := resolveIncludes(child, dir, depth); err != nil {
			return err
		}
	}
	return nil
}

// validate checks the scenario is well-formed and only references known events.
func (s *Scenario) validate() error {
	if strings.TrimSpace(s.Name) == "" {
		return errNoName
	}
	if len(s.Turns) == 0 {
		return fmt.Errorf("%w: %q", errNoTurns, s.Name)
	}
	for i, turn := range s.Turns {
		if err := turn.validate(); err != nil {
			return fmt.Errorf("turn %d: %w", i+1, err)
		}
		for j := range turn.Expect {
			// By pointer: validate normalizes the expectation's tool calls, and
			// the scenario must carry the normalized form.
			if err := turn.Expect[j].validate(); err != nil {
				return fmt.Errorf("turn %d expectation %d: %w", i+1, j+1, err)
			}
		}
	}
	return nil
}

// input is what the turn sends, for a report: the utterance, the keys, or a
// note that the turn only observes.
func (t Turn) input() string {
	switch {
	case t.User != "":
		return t.User
	case t.DTMF != "":
		return "dtmf " + t.DTMF
	default:
		return "(observe only)"
	}
}

// validate checks a turn is well-formed. Both kinds of input are optional, but
// a turn takes one kind or the other, and a schedule needs input to schedule.
func (t *Turn) validate() error {
	if t.User != "" && t.DTMF != "" {
		return errUserAndDTMF
	}
	for _, key := range t.DTMF {
		if !frames.KeypadEntry(key).Valid() {
			return fmt.Errorf("%w %q", errDTMFKey, string(key))
		}
	}
	if t.SendAfter != nil && t.User == "" && t.DTMF == "" {
		return errScheduleNoInput
	}
	return t.SendAfter.validate()
}

// validate checks a turn's schedule is well-formed. A nil schedule is valid:
// the turn's input goes out as soon as the previous turn finishes.
func (s *SendAfter) validate() error {
	if s == nil {
		return nil
	}
	if s.DelayMS < 0 {
		return errNegativeDelay
	}
	// With no event to anchor on, a zero delay would be a schedule that
	// schedules nothing.
	if s.Event == "" && s.DelayMS == 0 {
		return errEmptySendAfter
	}
	return nil
}

// UnmarshalYAML decodes an expectation, recording whether `calls:` was written
// out. The distinction matters: a missing `calls:` falls back to the
// `name:`/`args:` shorthand, whereas an empty one is a mistake.
func (e *Expectation) UnmarshalYAML(node *yaml.Node) error {
	type raw Expectation // a distinct type, so decoding does not recurse here
	var r raw
	if err := node.Decode(&r); err != nil {
		return err
	}
	*e = Expectation(r)
	e.hasCalls = hasKey(node, "calls")
	return nil
}

// hasKey reports whether a YAML mapping node has the named key.
func hasKey(node *yaml.Node, key string) bool {
	if node.Kind != yaml.MappingNode {
		return false
	}
	for i := 0; i+1 < len(node.Content); i += 2 {
		if node.Content[i].Value == key {
			return true
		}
	}
	return false
}

// UnmarshalYAML accepts either a bare tool name or a {name, args} mapping, so a
// calls: list can mix the two.
func (c *FunctionCall) UnmarshalYAML(node *yaml.Node) error {
	if node.Kind == yaml.ScalarNode {
		return node.Decode(&c.Name)
	}
	type raw FunctionCall
	var r raw
	if err := node.Decode(&r); err != nil {
		return err
	}
	*c = FunctionCall(r)
	return nil
}

// validate checks a single expectation is well-formed for its event type, and
// normalizes its expected tool calls.
func (e *Expectation) validate() error {
	if !knownEvents[e.Event] {
		return fmt.Errorf("%w %q", errUnknownEvent, e.Event)
	}
	if err := e.validateFields(); err != nil {
		return err
	}
	if e.WithinMS < 0 {
		return errNegativeBudget
	}
	return e.normalizeCalls()
}

// validateFields checks each content assertion against the event it was written
// on, and against absence.
func (e *Expectation) validateFields() error {
	callFields := e.Name != "" || e.Args != nil || e.hasCalls
	if callFields && e.Event != EventFunctionCall {
		return fmt.Errorf("%w: name, args and calls need a %s event", errFieldForEvent, EventFunctionCall)
	}
	if e.Eval != "" && e.Event != EventLLMResponse {
		return fmt.Errorf("%w: eval is only valid on a %s event", errFieldForEvent, EventLLMResponse)
	}
	if e.TextContains != "" && e.Event != EventLLMResponse && e.Event != EventUserTranscription {
		return fmt.Errorf("%w: text_contains needs %s or %s", errFieldForEvent, EventLLMResponse, EventUserTranscription)
	}
	// An absent expectation matches on event type only: a content or call check
	// describes an event that must arrive, which contradicts absence.
	if e.Absent && (e.TextContains != "" || e.Eval != "" || callFields) {
		return errAbsentWithCheck
	}
	return nil
}

// normalizeCalls resolves an expectation's expected tool calls into Calls: the
// `calls:` list as written, or the single `name:`/`args:` shorthand, or one
// nameless call for a bare function_call, which matches whatever the bot calls.
func (e *Expectation) normalizeCalls() error {
	if e.Event != EventFunctionCall || e.Absent {
		e.Calls = nil
		return nil
	}
	if !e.hasCalls {
		e.Calls = []FunctionCall{{Name: e.Name, Args: e.Args}}
		return nil
	}
	if len(e.Calls) == 0 {
		return errEmptyCalls
	}
	return nil
}
