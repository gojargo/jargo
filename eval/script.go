package eval

import (
	"fmt"
	"log/slog"
	"maps"
	"path/filepath"
	"regexp"
	"slices"
	"strconv"
	"strings"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor/rtvi"
	"gopkg.in/yaml.v3"
)

// MaxTurnPauseMS is the most a turn's pauses can add up to, in milliseconds. A
// caller silent for longer has stopped taking part, and the turn's audio,
// silence included, is built in memory.
const MaxTurnPauseMS = 60_000

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
	// EventResponse is the bot's reply whatever the modality: the transcription
	// of its actual synthesized audio in audio modality, the LLM text in text
	// modality, where a scenario's response expectations are read as
	// llm_response when it is loaded.
	EventResponse = "response"
	// EventLLMResponse carries the bot's LLM text, joined across the response.
	EventLLMResponse = "llm_response"
	// EventTTSResponse carries the text the bot's TTS reports speaking, one
	// segment as each arrives (audio modality only: in text modality the bot
	// skips TTS, so no segment is ever produced).
	EventTTSResponse = "tts_response"
	// EventFunctionCall fires when the bot invokes a tool.
	EventFunctionCall = "function_call"
	// EventFunctionCallStopped fires when a tool call ends, its args saying
	// how it ended: stopped, or run to completion.
	EventFunctionCallStopped = "function_call_stopped"
	// EventImage is an image the bot output, such as a generated one. The bot
	// reports images only to a scenario that expects one.
	EventImage = "image"
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
	// EventLLMMarker carries the sideband marker the bot's model emitted in a
	// response, such as a turn-completion marker. It arrives when the response
	// ends, and carries the response's raw text with it. Markers never reach a
	// client by default; a scenario asserting on one asks the bot to report them.
	EventLLMMarker = "llm_marker"
	// EventExpression is the expression the bot's TTS performed in an
	// expressive conversation (alpha), from bot-output, in the tags the LLM
	// writes: <<excited>> where the delivery of its spoken text changes
	// (<<neutral>> back to the baseline), [[laughter]] for a sound,
	// [[pause 600]] for a pause. A tag the TTS can't perform is never reported.
	EventExpression = "expression"
)

// The meanings a marker expectation can name, which are the emitter's own
// vocabulary rather than the marker text, since that is configurable.
const (
	// MarkerComplete means the user's turn was finished and the bot answers.
	MarkerComplete = "complete"
	// MarkerShort means the user was cut off and the bot waits.
	MarkerShort = "short"
	// MarkerLong means the user asked for time and the bot waits longer.
	MarkerLong = "long"
	// MarkerIncomplete matches either of the two above, for a scenario that
	// cares that the bot held the turn open rather than which reason it read.
	MarkerIncomplete = "incomplete"
)

// Where an expectation checks for its event from.
const (
	// CheckFromPrevious looks only from where the expectation above it was
	// met. It is the default.
	CheckFromPrevious = "previous"
	// CheckFromTurnStart looks at the events since the beginning of the turn
	// that the expectations above it did not use.
	CheckFromTurnStart = "turn_start"
)

// Fixed tables of the names a scenario may use.
//
//nolint:gochecknoglobals // fixed lookup tables, read only
var (
	// JudgeableEvents are the events whose payloads carry bot-generated text a
	// judge can sensibly evaluate, including the expression the bot's TTS
	// performed. An eval: on anything else but a function call (a user
	// transcript, an interruption signal) draws a warning when the scenario is
	// loaded: the test controls the user's input, so judging it costs a round
	// trip and tells you nothing. response is the modality-agnostic name,
	// resolved to one of the others when the scenario is loaded.
	JudgeableEvents = []string{EventExpression, EventLLMResponse, EventResponse, EventTTSResponse}
	// FunctionCallEvents are the events carrying a function call, matched by
	// name and arguments rather than by text: function_call when one starts,
	// function_call_stopped when it ends.
	FunctionCallEvents = []string{EventFunctionCall, EventFunctionCallStopped}
	// MarkerKinds are what a marker: may name on an llm_marker expectation.
	// The first three are the kinds the turn-completion markers carry;
	// incomplete accepts short or long.
	MarkerKinds = []string{MarkerComplete, MarkerShort, MarkerLong, MarkerIncomplete}
	// CheckFrom are the values check_from: may take.
	CheckFrom = []string{CheckFromPrevious, CheckFromTurnStart}
	// vadEvents are the raw VAD speaking events, which the bot reports only when
	// asked to.
	vadEvents = []string{EventVADUserStartedSpeaking, EventVADUserStoppedSpeaking}
)

// keypadEntries are the keys a dtmf: sequence may press, as a message lists
// them.
const keypadEntries = "1, 2, 3, 4, 5, 6, 7, 8, 9, 0, #, *"

// FunctionCall is one expected tool call within a function call expectation.
type FunctionCall struct {
	// Name is the tool name to match. An empty name matches any call, which is
	// what a bare function_call expectation asserts.
	Name string
	// Args, when set, is a subset check on the call's arguments: every listed
	// key must be present with the listed value, and any further argument the
	// model passed is ignored.
	Args map[string]any
}

// Signature is a short label for the call: name(arg=value, ...).
func (c FunctionCall) Signature() string {
	name := c.Name
	if name == "" {
		name = "any function"
	}
	if len(c.Args) == 0 {
		return name
	}
	args := make([]string, 0, len(c.Args))
	for _, k := range slices.Sorted(maps.Keys(c.Args)) {
		args = append(args, k+"="+pyRepr(c.Args[k]))
	}
	return name + "(" + strings.Join(args, ", ") + ")"
}

// Patterns are regular expressions an event's text must each match. The
// patterns are RE2, as Go's regexp reads them, so a lookaround or a
// backreference is rejected when the scenario is loaded.
type Patterns []string

// firstUnmatched is the first pattern content does not match, and whether
// there is one.
func (p Patterns) firstUnmatched(content string) (string, bool) {
	for _, pattern := range p {
		// The patterns were compiled when the scenario was loaded.
		if !regexp.MustCompile(pattern).MatchString(content) {
			return pattern, true
		}
	}
	return "", false
}

// Expectation is a single expected event in a scenario turn.
type Expectation struct {
	// Event is the friendly event name to match (see the Event constants).
	Event string
	// WithinMS is the latency budget, measured from the turn's user send: all
	// of a turn's expectations share that one anchor, so a stalled turn fails
	// within a single budget rather than one per expectation. Zero uses the
	// harness default.
	WithinMS int
	// TextContains, when set, requires the event's text to contain this
	// substring, whatever the spacing.
	TextContains string
	// TextExcludes is the mirror of TextContains: the event's text must not hold
	// this substring. With TextContains or Eval, which accumulate a reply, it is
	// checked on the reply accumulated up to the match.
	TextExcludes string
	// TextMatches are regular expressions the event's text must each match,
	// checked on the same text as TextExcludes.
	TextMatches Patterns
	// Calls is the set of calls expected in the turn, for a function call
	// event. They are matched by name in any order and the expectation passes
	// only when all of them are found. Built from calls:, or from the single
	// name:/args: shorthand; nil for any other event.
	Calls []FunctionCall
	// Eval, when set, is a natural-language criterion an LLM judge checks the
	// event's text against. Meaningful on the bot-generated text events
	// (response, llm_response, tts_response, and the expression its TTS
	// performed) and on function_call, where it is about each matched call's
	// name and arguments rather than text.
	Eval string
	// Marker, on an llm_marker, is the meaning the marker must have: one of
	// MarkerKinds, where incomplete accepts short or long.
	Marker string
	// MarkerFirst, on an llm_marker, is whether the first marker in the
	// response's raw text must come before any text.
	MarkerFirst *bool
	// Markers, on an llm_marker, is how many markers the response's raw text
	// must hold.
	Markers *int
	// TextAfter, on an llm_marker, is whether text must (true) or must not
	// (false) follow the first marker in the raw text.
	TextAfter *bool
	// CheckFrom is where the expectation checks for its event from: previous
	// (the default, also when empty), where the expectation above it was met;
	// or turn_start, the beginning of the turn, at events the expectations
	// above it did not use.
	CheckFrom string
	// Absent inverts the expectation: it passes only when no event of this type
	// arrives before the WithinMS budget expires, and fails as soon as one does.
	// It matches on event type only, so no content, call or marker check may
	// accompany it.
	Absent bool
}

// Aggregates reports whether the check accumulates text across events rather
// than matching one. A reply with a content check accumulates the bot's
// segments, and an expression check the expression performed in the turn; a
// user_transcription with TextContains accumulates an STT's pieces.
func (e *Expectation) Aggregates() bool {
	switch e.Event {
	case EventResponse, EventLLMResponse, EventTTSResponse, EventExpression:
		return e.TextContains != "" || e.Eval != ""
	case EventUserTranscription:
		return e.TextContains != "" && e.Eval == ""
	default:
		return false
	}
}

// SendAfter schedules when a turn's input (user or dtmf) is sent. The harness
// waits for Event to have been seen, either earlier in the run or arriving now,
// then waits DelayMS longer before sending. That is how a barge-in is written:
// `send_after: {event: llm_started, delay_ms: 500}` interrupts 500ms after the
// bot started responding.
//
// Event is optional. On its own, DelayMS is a pure time delay measured from the
// previous turn's send, with no event to anchor on.
type SendAfter struct {
	// Event is the event to schedule from, or empty for a pure delay.
	Event string
	// DelayMS is how long to wait after the event was seen, or, with no event,
	// after the previous turn's send.
	DelayMS int
}

// TurnPart is one item of a turn that pauses: text to speak, or a pause.
type TurnPart struct {
	// Text is the text to speak, or empty for a pause.
	Text string
	// PauseMS is the pause in milliseconds, or zero for text.
	PauseMS int
}

// ScriptTurn is one turn in a scripted scenario.
//
// A turn drives the bot one of three ways: the harness sends a user utterance
// (the person speaks), it sends a dtmf keypress sequence (the person presses
// keys), or it is observation-only (neither, which is what a bot-first scenario
// needs to check an opening greeting). User and DTMF are mutually exclusive.
type ScriptTurn struct {
	// User is the text the harness sends as the user's turn: an RTVI send-text
	// in text modality, or synthesized speech in audio modality. Empty for a
	// turn that only waits and asserts.
	User string
	// Expect are the expected events, in the order they should arrive.
	// Optional: a turn may just send input, with the assertion on a later turn.
	Expect []Expectation
	// Parts are the turn's text and pauses, in order, when it pauses; User then
	// holds the text alone. Requires user.modality: audio, and the pauses can
	// total at most MaxTurnPauseMS.
	Parts []TurnPart
	// Audio is the path of an audio file, resolved against the scenario file,
	// played as this turn instead of synthesizing User. User is required
	// alongside it and is what the recording says: the judge reads it as the
	// turn's input and text_contains matches against it. Requires
	// user.modality: audio, and is mutually exclusive with DTMF.
	Audio string
	// DTMF is a keypad sequence the user presses, one frame per key. Quote it
	// in YAML: an unquoted # starts a comment.
	DTMF string
	// SendAfter, when set, schedules when the turn's input is sent rather than
	// sending it once the bot has finished speaking.
	SendAfter *SendAfter
	// Image is the path of an image, resolved against the scenario file,
	// registered for this turn: when the bot requests a user image during the
	// turn, it is served this one. It stays registered until a later turn
	// provides a different image.
	Image string
}

// input is what the turn sends, for a report: the utterance, the keys, or a
// note that the turn only observes.
func (t ScriptTurn) input() string {
	switch {
	case t.User != "":
		return t.User
	case t.DTMF != "":
		return "dtmf " + t.DTMF
	default:
		return "(observe only)"
	}
}

// Turn is a scripted turn.
//
// Deprecated: Use ScriptTurn instead.
type Turn = ScriptTurn

// ScriptScenario is a scripted scenario: a conversation and the events it
// should produce.
type ScriptScenario struct {
	// Name identifies the scenario in reports.
	Name string
	// Turns are played in order.
	Turns []ScriptTurn
	// Context is the conversation the bot's context should start from. When
	// set, the harness sends it once the bot is ready, replacing whatever
	// context the bot built for itself. Leave it out to test the bot's own
	// opening state.
	Context []frames.Message
	// Configured is the scenario's user: and judge: blocks.
	Configured
	// TriggerDisconnect is the trigger_disconnect: key.
	//
	// Deprecated: No replacement. A bot fires its client-disconnected handlers
	// whenever the harness disconnects.
	TriggerDisconnect bool
	// StopOnFailure is whether the first failed turn ends the scenario; nil is
	// true. A failed turn leaves the conversation in an unknown state, so the
	// remaining turns usually just burn a budget each. Set it false for a
	// scenario whose turns are scored independently; give those turns an
	// explicit within_ms, since with the 60s default a silent bot costs one full
	// budget per turn.
	StopOnFailure *bool
	// SourcePath is the file the scenario was loaded from.
	SourcePath string
}

// Scenario is a scripted scenario.
//
// Deprecated: Use ScriptScenario instead.
type Scenario = ScriptScenario

// ScenarioName is the scenario's name.
func (s *ScriptScenario) ScenarioName() string { return s.Name }

func (*ScriptScenario) loadedScenario() {}

// stopsOnFailure reports whether the first failed turn ends the scenario.
func (s *ScriptScenario) stopsOnFailure() bool {
	return s.StopOnFailure == nil || *s.StopOnFailure
}

// WantsResponse reports whether any expectation asserts on the transcription
// of the bot's audio.
func (s *ScriptScenario) WantsResponse() bool {
	return s.anyExpectation(func(e *Expectation) bool { return e.Event == EventResponse })
}

// RequiredReportLevel is the function-call report level the scenario's
// assertions need: full for arguments, name for names, or "" when they need
// nothing beyond the default. A call with an eval: needs full too, since the
// judge is asked about the call's arguments.
func (s *ScriptScenario) RequiredReportLevel() rtvi.FunctionCallReportLevel {
	needsName := false
	for _, turn := range s.Turns {
		for _, exp := range turn.Expect {
			if !slices.Contains(FunctionCallEvents, exp.Event) {
				continue
			}
			if exp.Eval != "" {
				return rtvi.ReportFull
			}
			// Name and args live in Calls: the parser normalizes the single
			// name:/args: shorthand into it too.
			for _, call := range exp.Calls {
				if call.Args != nil {
					return rtvi.ReportFull
				}
				if call.Name != "" {
					needsName = true
				}
			}
		}
	}
	if needsName {
		return rtvi.ReportName
	}
	return ""
}

// NeedsVADEvents reports whether the scenario uses the raw VAD speaking events,
// which the bot emits only on request.
func (s *ScriptScenario) NeedsVADEvents() bool {
	for _, turn := range s.Turns {
		if turn.SendAfter != nil && slices.Contains(vadEvents, turn.SendAfter.Event) {
			return true
		}
		for _, exp := range turn.Expect {
			if slices.Contains(vadEvents, exp.Event) {
				return true
			}
		}
	}
	return false
}

// NeedsBotImages reports whether the scenario asserts on the images the bot
// outputs, which it reports only on request.
func (s *ScriptScenario) NeedsBotImages() bool {
	return s.anyExpectation(func(e *Expectation) bool { return e.Event == EventImage })
}

// NeedsMarkerEvents reports whether the scenario asserts on the model's
// markers, which the bot emits only on request.
func (s *ScriptScenario) NeedsMarkerEvents() bool {
	return s.anyExpectation(func(e *Expectation) bool { return e.Event == EventLLMMarker })
}

// anyExpectation reports whether any of the scenario's expectations satisfies
// match.
func (s *ScriptScenario) anyExpectation(match func(*Expectation) bool) bool {
	for i := range s.Turns {
		for j := range s.Turns[i].Expect {
			if match(&s.Turns[i].Expect[j]) {
				return true
			}
		}
	}
	return false
}

// parseScript parses a scripted scenario's mapping, as read from path. Error
// messages name path, and the turns' audio: and image: paths resolve relative
// to it.
func parseScript(data map[string]any, path string) (*ScriptScenario, error) {
	name, ok := data[keyName].(string)
	if !ok || name == "" {
		return nil, invalidf("%s: missing or invalid 'name:' field", path)
	}
	rawTurns, ok := data["turns"].([]any)
	if !ok {
		return nil, invalidf("%s: 'turns:' must be a list", path)
	}
	turns := make([]ScriptTurn, 0, len(rawTurns))
	for idx, t := range rawTurns {
		turn, err := parseTurn(t, path, idx)
		if err != nil {
			return nil, err
		}
		turns = append(turns, turn)
	}

	context, err := parseContext(data["context"], path)
	if err != nil {
		return nil, err
	}

	// user: {modality: audio|text, speech: {...}}. Audio synthesizes each user
	// turn through a TTS, exercising the bot's STT; text sends it as text.
	userAudio, userSpeech, err := parseUserBlock(data["user"], path)
	if err != nil {
		return nil, err
	}
	if err = checkUserAudio(turns, userAudio, userSpeech, path); err != nil {
		return nil, err
	}

	// judge: {modality: audio|text, eval: {...}, transcription: {...}}. Audio
	// means the bot speaks and the judge evaluates the transcription of its
	// actual audio; text means the bot's LLM text directly, the bot skipping
	// TTS.
	botAudio, transcriber, judge, err := parseJudgeBlock(data["judge"], path)
	if err != nil {
		return nil, err
	}

	// Resolve the modality-agnostic response event, and check event and
	// modality agree, now that the judge modality is known.
	if err = resolveResponseEvents(turns, botAudio, path); err != nil {
		return nil, err
	}

	stopOnFailure := true
	if v, set := data["stop_on_failure"]; set {
		stopOnFailure = truthy(v)
	}
	return &ScriptScenario{
		Name:              name,
		Turns:             turns,
		Context:           context,
		UserAudio:         userAudio,
		UserSpeech:        userSpeech,
		BotAudio:          botAudio,
		Transcriber:       transcriber,
		Judge:             judge,
		TriggerDisconnect: parseTriggerDisconnect(data, path),
		StopOnFailure:     &stopOnFailure,
		SourcePath:        path,
	}, nil
}

// parseContext reads the context: list as messages.
func parseContext(raw any, path string) ([]frames.Message, error) {
	if raw == nil {
		return nil, nil
	}
	list, ok := raw.([]any)
	if !ok {
		return nil, invalidf("%s: 'context:' must be a list of message dicts", path)
	}
	encoded, err := yaml.Marshal(list)
	if err != nil {
		return nil, invalidf("%s: 'context:' must be a list of message dicts", path)
	}
	var messages []frames.Message
	if err := yaml.Unmarshal(encoded, &messages); err != nil {
		return nil, invalidf("%s: 'context:' must be a list of message dicts", path)
	}
	return messages, nil
}

// checkUserAudio checks the turns against the user: block they are delivered
// by.
func checkUserAudio(turns []ScriptTurn, userAudio bool, speech *SpeechConfig, path string) error {
	for idx, turn := range turns {
		if turn.Audio != "" && !userAudio {
			return invalidf("%s: turn #%d has 'audio:' but the scenario is text modality; "+
				"set 'user.modality: audio' to play it to the bot", path, idx)
		}
		if turn.Parts != nil && !userAudio {
			return invalidf("%s: turn #%d pauses in 'user:' but the scenario is text modality; "+
				"set 'user.modality: audio' to speak it to the bot", path, idx)
		}
	}

	// A turn naming a file is played as is; only the rest need a voice.
	if !userAudio || speech != nil {
		return nil
	}
	var spoken []string
	for i, t := range turns {
		if t.User != "" && t.Audio == "" {
			spoken = append(spoken, strconv.Itoa(i))
		}
	}
	if len(spoken) > 0 {
		return invalidf("%s: 'user.modality: audio' requires a 'user.speech:' block "+
			"(TTS service + voice) to synthesize turn(s) [%s], or give each of them an 'audio:' file",
			path, strings.Join(spoken, ", "))
	}
	return nil
}

// resolveResponseEvents resolves response for the modality and checks
// consistency. In audio modality response is the transcription of the bot's
// audio; in text modality it becomes llm_response. tts_response needs the bot
// to speak, so it is an error in text modality.
func resolveResponseEvents(turns []ScriptTurn, botAudio bool, path string) error {
	if botAudio {
		return nil
	}
	for ti := range turns {
		for ei := range turns[ti].Expect {
			exp := &turns[ti].Expect[ei]
			switch exp.Event {
			case EventResponse:
				exp.Event = EventLLMResponse
			case EventTTSResponse:
				return invalidf("%s: turn #%d asserts 'tts_response' but 'judge.modality' is text "+
					"(the bot doesn't speak). Use 'response'/'llm_response', or set "+
					"'judge.modality: audio'.", path, ti)
			}
		}
	}
	return nil
}

// parseTurn parses one entry of the turns: list.
func parseTurn(raw any, path string, idx int) (ScriptTurn, error) {
	t, ok := raw.(map[string]any)
	if !ok {
		return ScriptTurn{}, invalidf("%s: turn #%d must be a mapping", path, idx)
	}
	user, parts, err := parseUser(t["user"], path, idx)
	if err != nil {
		return ScriptTurn{}, err
	}
	dtmf, err := parseDTMF(t["dtmf"], path, idx)
	if err != nil {
		return ScriptTurn{}, err
	}
	if user != "" && dtmf != "" {
		return ScriptTurn{}, invalidf("%s: turn #%d has both 'user:' and 'dtmf:'; a turn is one or the other",
			path, idx)
	}

	expect, err := parseExpect(t, path, idx)
	if err != nil {
		return ScriptTurn{}, err
	}

	var sendAfter *SendAfter
	if v, set := t["send_after"]; set {
		if sendAfter, err = parseSendAfter(v, path, idx); err != nil {
			return ScriptTurn{}, err
		}
		if user == "" && dtmf == "" {
			return ScriptTurn{}, invalidf("%s: turn #%d has 'send_after:' but no 'user:' or 'dtmf:'; "+
				"send_after only schedules when the turn's input gets sent", path, idx)
		}
	}

	audio, err := parseTurnAudio(t["audio"], user, dtmf, parts, path, idx)
	if err != nil {
		return ScriptTurn{}, err
	}
	// Image paths resolve relative to the scenario file, so a scenario is
	// portable.
	var image string
	if v := t["image"]; v != nil {
		s, isPath := v.(string)
		if !isPath {
			return ScriptTurn{}, invalidf("%s: turn #%d 'image:' must be a path string", path, idx)
		}
		image = resolveAgainst(path, s)
	}
	return ScriptTurn{
		User: user, Parts: parts, DTMF: dtmf, Expect: expect, SendAfter: sendAfter, Image: image, Audio: audio,
	}, nil
}

// parseExpect parses a turn's expect: list. It is optional: a turn may just
// send input (paced keypresses, say) or just wait, with the assertion living on
// another turn.
func parseExpect(t map[string]any, path string, idx int) ([]Expectation, error) {
	var rawExpect []any
	if v, set := t["expect"]; set {
		var ok bool
		if rawExpect, ok = v.([]any); !ok {
			return nil, invalidf("%s: turn #%d 'expect:' must be a list if present", path, idx)
		}
	}
	expect := make([]Expectation, 0, len(rawExpect))
	for ei, e := range rawExpect {
		exp, err := parseExpectation(e, path, idx, ei)
		if err != nil {
			return nil, err
		}
		expect = append(expect, exp)
	}
	return expect, nil
}

// parseTurnAudio checks a turn's audio: against the rest of the turn, and
// resolves it relative to the scenario file, so a scenario is portable.
func parseTurnAudio(raw any, user, dtmf string, parts []TurnPart, path string, idx int) (string, error) {
	if raw == nil {
		return "", nil
	}
	audio, ok := raw.(string)
	if !ok {
		return "", invalidf("%s: turn #%d 'audio:' must be a path string", path, idx)
	}
	if dtmf != "" {
		return "", invalidf("%s: turn #%d has both 'audio:' and 'dtmf:'; a turn is one or the other", path, idx)
	}
	if user == "" {
		return "", invalidf("%s: turn #%d has 'audio:' but no 'user:'; give the text the "+
			"recording says, so the judge and 'text_contains' have the turn's input", path, idx)
	}
	if parts != nil {
		return "", invalidf("%s: turn #%d has both 'audio:' and pauses in 'user:'; "+
			"a recording has its own pauses", path, idx)
	}
	return resolveAgainst(path, audio), nil
}

// resolveAgainst is p resolved against the directory of the file at path, as an
// absolute path. An absolute p is kept as it is.
func resolveAgainst(path, p string) string {
	if !filepath.IsAbs(p) {
		p = filepath.Join(filepath.Dir(path), p)
	}
	if abs, err := filepath.Abs(p); err == nil {
		return abs
	}
	return filepath.Clean(p)
}

// parseUser parses a turn's user:: its text, and its text and pauses when it
// pauses.
func parseUser(user any, path string, idx int) (string, []TurnPart, error) {
	switch v := user.(type) {
	case nil:
		return "", nil, nil
	case string:
		return v, nil, nil
	case []any:
		parts := make([]TurnPart, 0, len(v))
		var texts []string
		pauseMS := 0
		for _, item := range v {
			part, err := parseUserPart(item, path, idx)
			if err != nil {
				return "", nil, err
			}
			parts = append(parts, part)
			if part.Text != "" {
				texts = append(texts, part.Text)
			}
			pauseMS += part.PauseMS
		}
		if len(texts) == 0 {
			return "", nil, invalidf("%s: turn #%d 'user:' has pauses but no text", path, idx)
		}
		if pauseMS > MaxTurnPauseMS {
			return "", nil, invalidf("%s: turn #%d pauses for %d ms in total; "+
				"a turn can pause for at most %d ms", path, idx, pauseMS, MaxTurnPauseMS)
		}
		return strings.Join(texts, " "), parts, nil
	default:
		return "", nil, invalidf("%s: turn #%d 'user:' must be a string, or a list of text and pauses", path, idx)
	}
}

// parseUserPart parses one item of a pausing user:: text, or a pause's
// milliseconds.
func parseUserPart(part any, path string, idx int) (TurnPart, error) {
	switch v := part.(type) {
	case string:
		if text := strings.TrimSpace(v); text != "" {
			return TurnPart{Text: text}, nil
		}
	case map[string]any:
		if len(v) == 1 {
			if ms, ok := isInt(v["pause_ms"]); ok && ms > 0 {
				return TurnPart{PauseMS: ms}, nil
			}
		}
	}
	return TurnPart{}, invalidf("%s: turn #%d 'user:' items must be text or 'pause_ms: <positive int>'", path, idx)
}

// parseDTMF parses and validates a turn's dtmf: keypad sequence.
func parseDTMF(raw any, path string, idx int) (string, error) {
	if raw == nil {
		return "", nil
	}
	// An unquoted digit sequence reads as an integer (`dtmf: 123`); take its
	// digits, so `dtmf: 123` and `dtmf: "123#"` work the same.
	if n, ok := isInt(raw); ok {
		raw = strconv.Itoa(n)
	}
	dtmf, ok := raw.(string)
	if !ok || dtmf == "" {
		return "", invalidf("%s: turn #%d 'dtmf:' must be a non-empty string of keypad entries", path, idx)
	}
	for _, ch := range dtmf {
		if !frames.KeypadEntry(string(ch)).Valid() {
			return "", invalidf("%s: turn #%d 'dtmf:' has invalid keypad entry %s (valid entries: %s)",
				path, idx, pyRepr(string(ch)), keypadEntries)
		}
	}
	return dtmf, nil
}

// parseSendAfter parses a send_after: block.
func parseSendAfter(raw any, path string, idx int) (*SendAfter, error) {
	s, ok := raw.(map[string]any)
	if !ok {
		return nil, invalidf("%s: turn #%d 'send_after:' must be a mapping", path, idx)
	}
	var event string
	if v := s["event"]; v != nil {
		if event, ok = v.(string); !ok {
			return nil, invalidf("%s: turn #%d 'send_after.event' must be a string if present", path, idx)
		}
	}
	delay := 0
	if v, set := s["delay_ms"]; set {
		if delay, ok = isInt(v); !ok || delay < 0 {
			return nil, invalidf("%s: turn #%d 'send_after.delay_ms' must be a non-negative int", path, idx)
		}
	}
	// With no event to anchor on, a zero delay would schedule nothing.
	if event == "" && delay == 0 {
		return nil, invalidf("%s: turn #%d 'send_after:' needs an 'event:' or a positive 'delay_ms:'", path, idx)
	}
	return &SendAfter{Event: event, DelayMS: delay}, nil
}

// The keys of an llm_marker expectation's checks.
const (
	keyMarker      = "marker"
	keyMarkerFirst = "marker_first"
	keyMarkers     = "markers"
	keyTextAfter   = "text_after"
)

// absentConflicts are the keys an absent expectation may not carry: content,
// call and marker checks describe an event that must arrive.
//
//nolint:gochecknoglobals // fixed list, read only
var absentConflicts = []string{
	"text_contains", "text_excludes", "text_matches", keyEval, "calls", keyName, "args",
	keyMarker, keyMarkerFirst, keyMarkers, keyTextAfter,
}

// markerKeys are the keys that only apply to an llm_marker expectation.
//
//nolint:gochecknoglobals // fixed list, read only
var markerKeys = []string{keyMarker, keyMarkerFirst, keyMarkers, keyTextAfter}

// parseExpectation parses one entry of a turn's expect: list.
func parseExpectation(raw any, path string, ti, ei int) (Expectation, error) {
	where := fmt.Sprintf("%s: turn #%d expectation #%d", path, ti, ei)
	e, ok := raw.(map[string]any)
	if !ok {
		return Expectation{}, invalidf("%s must be a mapping", where)
	}
	event, ok := e["event"].(string)
	if !ok || event == "" {
		return Expectation{}, invalidf("%s missing or invalid 'event:'", where)
	}

	criterion, hasCriterion := e[keyEval]
	hasCriterion = hasCriterion && criterion != nil
	if hasCriterion && event == EventFunctionCallStopped {
		return Expectation{}, invalidf("%s: 'eval:' on 'function_call_stopped': a stopped call carries "+
			"no arguments to judge; put the 'eval:' on the 'function_call' instead", where)
	}
	if hasCriterion && !slices.Contains(JudgeableEvents, event) && event != EventFunctionCall {
		slog.Warn(fmt.Sprintf("%s: 'eval:' on event %s; a judge only makes sense on bot-generated text "+
			"events (%s) and 'function_call'. Will run but is unlikely to be meaningful.",
			where, pyRepr(event), strings.Join(JudgeableEvents, ", ")))
	}

	absent, checkFrom, err := parseAbsentAndCheckFrom(e, where)
	if err != nil {
		return Expectation{}, err
	}
	exp := Expectation{Event: event, Absent: absent, CheckFrom: checkFrom}
	if !absent {
		if exp.Calls, err = parseFunctionCalls(e, event, where); err != nil {
			return Expectation{}, err
		}
	}
	if err = parseMarkerChecks(e, &exp, where); err != nil {
		return Expectation{}, err
	}
	if exp.TextMatches, err = parseTextMatches(e["text_matches"], where); err != nil {
		return Expectation{}, err
	}
	return exp, parseTextChecks(e, &exp, where)
}

// parseAbsentAndCheckFrom reads an expectation's absent: and check_from:, and
// refuses an absent expectation anything that describes an event that arrives.
func parseAbsentAndCheckFrom(e map[string]any, where string) (bool, string, error) {
	absent := false
	if v, set := e["absent"]; set {
		var ok bool
		if absent, ok = v.(bool); !ok {
			return false, "", invalidf("%s 'absent:' must be a boolean", where)
		}
	}
	checkFrom := CheckFromPrevious
	v, hasCheckFrom := e["check_from"]
	if hasCheckFrom {
		s, ok := v.(string)
		if !ok || !slices.Contains(CheckFrom, s) {
			return false, "", invalidf("%s 'check_from:' must be one of %s, not %s",
				where, strings.Join(CheckFrom, ", "), pyRepr(v))
		}
		checkFrom = s
	}
	if !absent {
		return false, checkFrom, nil
	}
	if hasCheckFrom {
		// An absence is about what arrives after the expectations before it;
		// looking back for it has no meaning.
		return false, "", invalidf("%s 'absent: true' cannot be combined with 'check_from'", where)
	}
	// An absent expectation matches on event type only: content and call
	// checks describe an event that must arrive, which contradicts absence.
	var conflicting []string
	for _, key := range absentConflicts {
		if _, set := e[key]; set {
			conflicting = append(conflicting, pyRepr(key))
		}
	}
	if len(conflicting) > 0 {
		return false, "", invalidf("%s 'absent: true' cannot be combined with %s",
			where, strings.Join(conflicting, ", "))
	}
	return true, checkFrom, nil
}

// parseMarkerChecks reads an llm_marker expectation's marker checks, refusing
// them on any other event.
func parseMarkerChecks(e map[string]any, exp *Expectation, where string) error {
	var written []string
	for _, key := range markerKeys {
		if _, set := e[key]; set {
			written = append(written, pyRepr(key)+":")
		}
	}
	if len(written) > 0 && exp.Event != EventLLMMarker {
		return invalidf("%s %s only applies to the 'llm_marker' event, not %s",
			where, strings.Join(written, ", "), pyRepr(exp.Event))
	}
	if v := e[keyMarker]; v != nil {
		s, ok := v.(string)
		if !ok || !slices.Contains(MarkerKinds, s) {
			return invalidf("%s 'marker:' must be one of %s, not %s", where, strings.Join(MarkerKinds, ", "), pyRepr(v))
		}
		exp.Marker = s
	}
	for _, f := range []struct {
		key string
		dst **bool
	}{{keyMarkerFirst, &exp.MarkerFirst}, {keyTextAfter, &exp.TextAfter}} {
		if v := e[f.key]; v != nil {
			b, ok := v.(bool)
			if !ok {
				return invalidf("%s '%s:' must be a bool", where, f.key)
			}
			*f.dst = &b
		}
	}
	if v := e[keyMarkers]; v != nil {
		n, ok := isInt(v)
		if !ok {
			return invalidf("%s 'markers:' must be a int", where)
		}
		exp.Markers = &n
	}
	return nil
}

// parseTextMatches reads text_matches:, one pattern or a list of them, and
// compiles each.
func parseTextMatches(raw any, where string) (Patterns, error) {
	if raw == nil {
		return nil, nil
	}
	var patterns Patterns
	switch v := raw.(type) {
	case string:
		patterns = Patterns{v}
	case []any:
		for _, p := range v {
			s, ok := p.(string)
			if !ok {
				return nil, invalidf("%s 'text_matches:' must be a pattern or a list of patterns", where)
			}
			patterns = append(patterns, s)
		}
	default:
		return nil, invalidf("%s 'text_matches:' must be a pattern or a list of patterns", where)
	}
	for _, pattern := range patterns {
		if _, err := regexp.Compile(pattern); err != nil {
			return nil, invalidf("%s 'text_matches:' pattern %s: %v", where, pyRepr(pattern), err)
		}
	}
	return patterns, nil
}

// parseTextChecks reads an expectation's budget, its text checks and its judge
// criterion.
func parseTextChecks(e map[string]any, exp *Expectation, where string) error {
	if v := e["within_ms"]; v != nil {
		ms, ok := isNumber(v)
		if !ok {
			return invalidf("%s 'within_ms:' must be a number", where)
		}
		exp.WithinMS = int(ms)
	}
	for _, f := range []stringField{
		{"text_contains", &exp.TextContains}, {"text_excludes", &exp.TextExcludes}, {keyEval, &exp.Eval},
	} {
		switch v := e[f.key].(type) {
		case nil:
		case map[string]any, []any:
			return invalidf("%s '%s:' must be a string", where, f.key)
		default:
			*f.dst = pyStr(v)
		}
	}
	return nil
}

// parseFunctionCalls normalizes a function call expectation's calls into a
// list: a calls: list (names, or {name, args} mappings), the single
// name:/args: shorthand, or nothing, which matches any one call. Nil for other
// events.
func parseFunctionCalls(e map[string]any, event, where string) ([]FunctionCall, error) {
	if !slices.Contains(FunctionCallEvents, event) {
		return nil, nil
	}
	rawCalls := e["calls"]
	if rawCalls == nil {
		call, err := functionCall(e, where)
		if err != nil {
			return nil, err
		}
		return []FunctionCall{call}, nil
	}
	list, ok := rawCalls.([]any)
	if !ok || len(list) == 0 {
		return nil, invalidf("%s: 'calls:' must be a non-empty list", where)
	}
	out := make([]FunctionCall, 0, len(list))
	for _, c := range list {
		switch v := c.(type) {
		case string:
			out = append(out, FunctionCall{Name: v})
		case map[string]any:
			call, err := functionCall(v, where)
			if err != nil {
				return nil, err
			}
			out = append(out, call)
		default:
			return nil, invalidf("%s: each 'calls:' entry must be a name or a mapping", where)
		}
	}
	return out, nil
}

// functionCall reads a call's name: and args: from a mapping.
func functionCall(m map[string]any, where string) (FunctionCall, error) {
	var call FunctionCall
	if v := m[keyName]; v != nil {
		name, ok := v.(string)
		if !ok {
			return FunctionCall{}, invalidf("%s: a call's 'name:' must be a string", where)
		}
		call.Name = name
	}
	if v := m["args"]; v != nil {
		args, ok := v.(map[string]any)
		if !ok {
			return FunctionCall{}, invalidf("%s: a call's 'args:' must be a mapping", where)
		}
		call.Args = args
	}
	return call, nil
}
