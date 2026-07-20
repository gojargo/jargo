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
package eval

import (
	"errors"
	"fmt"
	"os"
	"strings"

	"gopkg.in/yaml.v3"
)

// Validation errors. Callers surface these to the operator; they are sentinels
// so a message change does not silently break a caller matching on them.
//
//nolint:gochecknoglobals // sentinel errors
var (
	errNoName         = errors.New("scenario has no name")
	errNoTurns        = errors.New("scenario has no turns")
	errNoUser         = errors.New("turn has no user input")
	errNoExpect       = errors.New("turn has no expectations")
	errUnknownEvent   = errors.New("unknown event")
	errFieldForEvent  = errors.New("field not valid for this event")
	errNegativeBudget = errors.New("within_ms must not be negative")
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
)

// knownEvents is the set of event names a scenario may assert on. The user_*
// events only fire in audio mode (see Options.UserTTS).
//
//nolint:gochecknoglobals // fixed lookup table
var knownEvents = map[string]bool{
	EventUserStartedSpeaking: true,
	EventUserStoppedSpeaking: true,
	EventUserTranscription:   true,
	EventLLMStarted:          true,
	EventLLMResponse:         true,
	EventFunctionCall:        true,
}

// Scenario is a scripted conversation and the events it should produce.
type Scenario struct {
	// Name identifies the scenario in reports; required.
	Name string `yaml:"name"`
	// Turns are played in order.
	Turns []Turn `yaml:"turns"`
}

// Turn is one user utterance and the events expected in response.
type Turn struct {
	// User is the text the user "says" this turn (sent as RTVI send-text).
	User string `yaml:"user"`
	// Expect lists the events to match, in order, after the user input.
	Expect []Expectation `yaml:"expect"`
}

// Expectation is one assertion about an event the bot emits.
type Expectation struct {
	// Event is the friendly event name to match (see the Event constants).
	Event string `yaml:"event"`
	// TextContains, when set, requires the event's text to contain this
	// substring (case-insensitive). Applies to llm_response.
	TextContains string `yaml:"text_contains,omitempty"`
	// Judge, when set, is a natural-language criterion an LLM judge checks the
	// event's text against. Applies to llm_response.
	Judge string `yaml:"judge,omitempty"`
	// Function, when set, is the tool name a function_call event must match.
	Function string `yaml:"function,omitempty"`
	// WithinMS is the latency budget for the event, measured from the user input;
	// zero uses the harness default.
	WithinMS int `yaml:"within_ms,omitempty"`
}

// Load reads and validates a scenario YAML file.
func Load(path string) (*Scenario, error) {
	data, err := os.ReadFile(path) //nolint:gosec // scenario path is operator-supplied
	if err != nil {
		return nil, fmt.Errorf("eval: read scenario: %w", err)
	}
	var s Scenario
	if err := yaml.Unmarshal(data, &s); err != nil {
		return nil, fmt.Errorf("eval: parse %s: %w", path, err)
	}
	if err := s.validate(); err != nil {
		return nil, fmt.Errorf("eval: %s: %w", path, err)
	}
	return &s, nil
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
		if strings.TrimSpace(turn.User) == "" {
			return fmt.Errorf("turn %d: %w", i+1, errNoUser)
		}
		if len(turn.Expect) == 0 {
			return fmt.Errorf("turn %d: %w", i+1, errNoExpect)
		}
		for j, exp := range turn.Expect {
			if err := exp.validate(); err != nil {
				return fmt.Errorf("turn %d expectation %d: %w", i+1, j+1, err)
			}
		}
	}
	return nil
}

// validate checks a single expectation is well-formed for its event type.
func (e Expectation) validate() error {
	if !knownEvents[e.Event] {
		return fmt.Errorf("%w %q", errUnknownEvent, e.Event)
	}
	if e.Function != "" && e.Event != EventFunctionCall {
		return fmt.Errorf("%w: function is only valid on a %s event", errFieldForEvent, EventFunctionCall)
	}
	if e.Judge != "" && e.Event != EventLLMResponse {
		return fmt.Errorf("%w: judge is only valid on a %s event", errFieldForEvent, EventLLMResponse)
	}
	if e.TextContains != "" && e.Event != EventLLMResponse && e.Event != EventUserTranscription {
		return fmt.Errorf("%w: text_contains needs %s or %s", errFieldForEvent, EventLLMResponse, EventUserTranscription)
	}
	if e.WithinMS < 0 {
		return errNegativeBudget
	}
	return nil
}
