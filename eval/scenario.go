// Package eval is a behavioral eval harness for jargo bots. It drives a real
// bot over RTVI, plays scripted conversation turns, and asserts on the semantic
// events the bot emits. That is a level above unit tests, which check one
// processor at a time.
//
// A scenario describes one conversation to hold with a bot and how to judge it.
// A YAML file holds one or more under `scenarios:`, each with a `name:`, and a
// scenario's keys say which kind it is:
//
//	turns:     a scripted scenario: the user's turns are written out, each
//	           with the events expected back from the bot
//	persona:   a simulated scenario, a simulation for short: an LLM plays a
//	           caller with a goal, and a judge reads the whole conversation
//	success:   with neither of the above, a rubric: what a simulation is
//	           judged on, without the caller, to judge a conversation the
//	           harness didn't drive; it has nothing to run
//
// The same scenario runs from a Go test (the bot hosted in-process, see Run) or
// from the command line against a running bot. In text modality each user turn
// is delivered as RTVI send-text, so the audio processors sit idle; in audio
// modality it is synthesized and streamed as microphone audio, so the bot's own
// VAD, turn detection and STT run for real.
//
// # Scenario files
//
// Any key a scenario can have may also sit at the top of the file, where it is
// the default for every scenario in it. A scenario that sets the same key
// replaces the whole value; nothing is merged, so a `context:` is written out in
// full, never added to. A scenario is named `<file name>/<scenario name>`:
//
//	name: turn_completion
//	judge: !include ../judge_text.yaml
//	scenarios:
//	  - name: short_answer
//	    turns:
//	      - user: "Japan."
//	        expect:
//	          - event: response
//
// A file holding a scenario's own keys at its top level and no `scenarios:` is
// deprecated; it still loads, as that one scenario under the file's `name:`,
// with a warning.
//
// Any value can come from a separate file with !include, resolved against the
// directory of the file that names it, so scenarios can share a block they all
// need, the user: and judge: blocks above all:
//
//	user: !include user_audio.yaml
//	judge: !include judge_audio.yaml
//
// Integers are plain decimal only, so a DTMF `012` keeps its digits rather than
// being read as octal.
//
// # Expectation fields
//
//	event: <name>            required: the event to match
//	within_ms: <int>         latency budget, measured from the turn's user input
//	text_contains: <str>     substring check on the event's text, whatever the spacing
//	text_excludes: <str>     the reverse of text_contains: the text must not hold it
//	text_matches: <regex> | [<regex>, ...]
//	                         regular expressions (RE2, as Go's regexp reads them)
//	                         the text must each match; the failure names the first
//	                         pattern that did not match
//	eval: <str>              criterion an LLM judge checks the bot's reply against
//	                         (response, llm_response, tts_response, and the
//	                         expression its TTS performed), or, on function_call,
//	                         the call itself by name and args
//	marker: <str>            for llm_marker: what the marker meant, one of
//	                         complete, short, long, or incomplete for either of
//	                         the last two
//	marker_first: <bool>     for llm_marker: the marker came before any text
//	markers: <int>           for llm_marker: how many markers the response holds
//	text_after: <bool>       for llm_marker: text follows the first marker
//	metric: <str>            for metrics: the kind of metric the report must hold
//	                         (ttfb, processing, tokens, ...)
//	processor: <str>         for metrics: text the name of the processor that
//	                         measured it must contain
//	name: <str>              for function_call: the tool the call must be to
//	args: <mapping>          for function_call: an argument subset the call must carry
//	calls: <list>            for function_call: several calls, in any order
//	check_from: <str>        previous (default) or turn_start (see below)
//	absent: true             assert the event does NOT arrive
//
// # Turn fields
//
//	user: <str> | <list>     what the user says, sent as text or synthesized
//	dtmf: <str>              keypad keys the user presses; quote it, # is a comment
//	audio: <path>            in audio modality, a recording played instead of
//	                         synthesizing user, which still says what it says
//	image: <path>            an image served when the bot requests one
//	expect: <list>           the events to match, in order
//	send_after: <mapping>    when to send this turn's input (see below)
//
// user and dtmf are mutually exclusive, and both are optional: a turn with
// neither only waits and asserts, which is what a bot-first scenario needs to
// check an opening greeting. expect is optional too, for a turn that only sends.
// audio and image paths resolve against the scenario file.
//
// In audio modality a spoken turn may pause, as a person does mid-thought: user
// is then a list of text and pause_ms silences, played to the bot as one stretch
// of audio. The judge and text_contains see the text alone. A turn's pauses can
// total at most MaxTurnPauseMS:
//
//	turns:
//	  - user:
//	      - "So last week I went to"
//	      - pause_ms: 2000
//	      - "Spain with my family."
//
// # Scenario fields
//
//	name: <str>              required: the scenario's name
//	turns: <list>            the turns, played in order
//	context: <list>          messages the bot's context starts from
//	stop_on_failure: <bool>  whether the first failed turn ends the scenario
//	                         (default true); false drives every turn, for a
//	                         scenario whose turns are scored independently
//	user: <mapping>          how user turns are delivered
//	judge: <mapping>         what the judge evaluates, and what decides the verdicts
//
// The user: block:
//
//	user:
//	  modality: audio          # audio | text (default text)
//	  speech:                  # needed unless every spoken turn has audio:
//	    service: kokoro        # the TTS that synthesizes the user turns
//	    voice: af_heart        # voices are language-specific
//	    language: en           # optional; must match the voice
//	    speed: 1.0             # optional
//	    sample_rate: 16000     # optional
//
// The judge: block:
//
//	judge:
//	  modality: audio          # audio | text (default text)
//	  eval:                    # what judges (default ollama)
//	    service: ollama
//	    model: gemma4:12b
//	  transcription:           # required when modality is audio
//	    service: moonshine     # STT for the bot's audio
//	    language: en           # optional; the language the bot speaks
//
// audio makes the bot speak and judges the transcription of its actual audio;
// text (the default) skips TTS and judges the LLM text, which is faster and
// silent. A factory: key in any service block names a registered builder that
// makes the service from the block instead.
//
// # Asserting on the bot's reply
//
// Three events carry the bot's own words:
//
//	response       the transcription of the bot's actual synthesized audio in
//	               audio modality, or the LLM text in text modality: the real
//	               end-to-end check, to prefer
//	llm_response   the text the model produced (bot-llm-text), in both modalities
//	tts_response   the text the TTS reports speaking (bot-tts-text), audio
//	               modality only: a text modality scenario asserting on it is
//	               refused when it is loaded
//
// In an expressive conversation (alpha), expression is the expression the bot's
// TTS performed, from bot-output, in the tags the LLM writes: <<excited>> where
// the delivery of its spoken text changes (<<neutral>> back to the baseline),
// [[laughter]] for a sound, [[pause 600]] for a pause. A tag the TTS can't
// perform is never reported. A content check on it accumulates the expression
// performed in the turn.
//
// A turn often answers in more than one response: an interim filler ("Let me
// check on that.") and then the answer. So a content check on a reply
// aggregates. It accumulates the text of successive segments and re-checks on
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
// # Asserting on the markers the model emits
//
// A model reading the turn-completion protocol begins each response with a
// marker saying whether the user's turn was finished. The markers are a
// diagnostic and never reach a client, so a scenario asserting on one asks the
// bot to report them:
//
//	turns:
//	  - user: "Let me think about it, hmmm"
//	    expect:
//	      - event: llm_marker
//	        marker: incomplete        # the bot held the turn open
//	  - user: "I think I'd go to Japan."
//	    expect:
//	      - event: llm_marker
//	        marker: complete          # ... and answered this one
//
// The event carries the response as the model produced it, before anything was
// held back, so a scenario can check the model against the protocol it was
// given: marker_first that nothing came before the marker, markers how many the
// response holds, and text_after whether anything follows the first one, which a
// complete turn should have and an incomplete one should not.
//
// # Asserting on the bot's metrics and its own messages
//
// metrics is a report of what the bot's processors measured, such as a
// service's TTFB, its processing time or its usage. A bot sends them when its
// pipeline has metrics on, grouped by kind (ttfb, processing, tokens, ...).
// metric: names the kind the report must hold and processor: text the
// measuring processor's name must contain; a bare metrics takes any report. The
// zero a pipeline reports for every processor as it starts is not a
// measurement and never matches. A report that arrived while an expectation
// above was waiting for its own event is behind this one, so look back for it:
//
//	turns:
//	  - user: "What is the capital of Germany?"
//	    expect:
//	      - event: user_transcription
//	      - event: metrics
//	        metric: ttfb
//	        processor: STT
//	        check_from: turn_start
//
// Every report of a run is kept with its result (ScriptResult.BotMetrics and
// SimulationResult.BotMetrics, and bot_metrics in a suite's results.jsonl, for
// passed and failed runs alike), expected or not, so an expectation is only
// needed to hold a turn open until one arrives.
//
// server_message is a message of the bot's own making: whatever data it pushed
// in an RTVI server message frame. It is how a bot built for an eval reports
// what only it can know, such as a result it computed or that it has finished
// its work on a turn. The event's text is the data as JSON (or the data itself
// when it is a string), with a space after each comma and colon, so the content
// checks read it and a matched expectation records it in the run's result:
//
//	turns:
//	  - user: "What is the capital of Germany?"
//	    expect:
//	      - event: server_message
//	        text_contains: '"done": true'
//
// # Asserting on function calls
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
// function_call_stopped takes the same calls: shape, and its args say how the
// call ended, which is how a scenario tells work that was stopped from work
// that finished on its own. A stopped call carries no arguments to judge, so it
// takes no eval:.
//
// # Absent events
//
// absent: true inverts an expectation. It passes only when no event of that
// type arrives before the within_ms budget expires, and fails as soon as one
// does, which is how "must not answer twice" is written:
//
//	expect:
//	  - event: response
//	    eval: "answers the question"
//	  - event: response
//	    absent: true
//	    within_ms: 5000
//
// It matches on event type alone, so no content, call, marker or metric check may
// accompany it, and it waits out its whole budget: set within_ms explicitly to
// keep the quiet window short.
//
// # Where an expectation looks
//
// Expectations are met in order: each is looked for only once the one above it
// has been met, and an event of another type that arrives in the meantime is
// passed over. check_from: turn_start lets an expectation run against events
// since the beginning of the turn that were not used by the expectations above
// it. Use it for an event whose timing the bot does not control, such as a call
// a backend makes at its own pace, or a reply some models give before the tool
// call that others give after.
//
// # Scheduling a turn
//
// A turn's input goes out once the bot has finished speaking, unless the turn
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
// A result carries more than its failures: how long the run took, every event
// the bot emitted whether or not the scenario asserted on it, and a timestamped
// trace of what the harness itself decided. Together they are what makes a
// scenario that failed once and passed the next time readable. Options.OnProgress
// reports the same as it happens, for a caller watching a long run.
package eval

import (
	"log/slog"
	"maps"
	"slices"
	"strings"
)

// Kind is the kind of scenario, as a run, a session and a results record name
// it.
type Kind string

// The kinds of scenario that run.
const (
	// KindScript is a scripted scenario.
	KindScript Kind = "script"
	// KindSimulation is a simulation.
	KindSimulation Kind = "simulation"
)

// LoadedScenario is one scenario a file holds, of whichever kind:
// *ScriptScenario, *SimulationScenario or *Rubric.
type LoadedScenario interface {
	// ScenarioName is the scenario's name, `<file name>/<scenario name>` for
	// one in a file's scenarios: list.
	ScenarioName() string
	loadedScenario()
}

// ScenarioFile is a scenario file: what it is called, where it is, and the
// scenarios it holds.
//
// Manifests and the command line load files through LoadFile, so every kind of
// scenario mixes in one list.
type ScenarioFile struct {
	// Name is the file's own `name:`.
	Name string
	// Path is the file it was read from.
	Path string
	// Scenarios are the scenarios it holds, in file order.
	Scenarios []LoadedScenario
}

// LoadFile reads a scenario file, parsing each scenario as whichever kind it
// is.
//
// A file in the deprecated shape, a scenario's own keys at the top level and no
// scenarios:, loads as that one scenario under the file's name: and warns.
//
// The error is a *ScenarioError when the file is malformed, or a scenario is of
// no kind, claims to be two, or is invalid for its kind.
func LoadFile(path string) (*ScenarioFile, error) {
	data, err := loadMapping(path)
	if err != nil {
		return nil, err
	}
	name, ok := data[keyName].(string)
	if !ok || name == "" {
		return nil, invalidf("%s: missing or invalid 'name:' field", path)
	}

	rawEntries := data["scenarios"]
	if rawEntries == nil {
		slog.Warn("eval: `top-level turns: or persona:` is deprecated and will be removed. "+
			"Use a `scenarios:` list instead.", "path", path)
		scenario, err := scenarioFromMapping(data, path, nil)
		if err != nil {
			return nil, err
		}
		return &ScenarioFile{Name: name, Path: path, Scenarios: []LoadedScenario{scenario}}, nil
	}

	entries, ok := rawEntries.([]any)
	if !ok || len(entries) == 0 {
		return nil, invalidf("%s: 'scenarios:' must be a non-empty list", path)
	}
	defaults := maps.Clone(data)
	delete(defaults, "scenarios")

	scenarios := make([]LoadedScenario, 0, len(entries))
	for idx, raw := range entries {
		scenario, err := scenarioFromEntry(raw, defaults, name, path, idx)
		if err != nil {
			return nil, err
		}
		scenarios = append(scenarios, scenario)
	}
	if err := checkDuplicateNames(scenarios, path); err != nil {
		return nil, err
	}
	return &ScenarioFile{Name: name, Path: path, Scenarios: scenarios}, nil
}

// scenarioFromEntry parses one entry of a file's scenarios: list, over the
// file's own keys as defaults, naming it under the file.
func scenarioFromEntry(raw any, defaults map[string]any, fileName, path string, idx int) (LoadedScenario, error) {
	entry, ok := raw.(map[string]any)
	if !ok {
		return nil, invalidf("%s: scenario #%d must be a mapping", path, idx)
	}
	if _, nested := entry["scenarios"]; nested {
		return nil, invalidf("%s: scenario #%d cannot hold a 'scenarios:' list of its own", path, idx)
	}
	entryName, ok := entry[keyName].(string)
	if !ok || entryName == "" {
		return nil, invalidf("%s: scenario #%d needs a 'name:'", path, idx)
	}
	merged := maps.Clone(defaults)
	maps.Copy(merged, entry)
	merged[keyName] = fileName + "/" + entryName
	return scenarioFromMapping(merged, path, entry)
}

// checkDuplicateNames refuses a file whose scenarios share a name.
func checkDuplicateNames(scenarios []LoadedScenario, path string) error {
	counts := make(map[string]int, len(scenarios))
	for _, s := range scenarios {
		counts[s.ScenarioName()]++
	}
	var duplicates []string
	for n, c := range counts {
		if c > 1 {
			duplicates = append(duplicates, n)
		}
	}
	if len(duplicates) > 0 {
		slices.Sort(duplicates)
		return invalidf("%s: duplicate scenario names: %s", path, strings.Join(duplicates, ", "))
	}
	return nil
}

// Scenario is the scenario called name, or a *ScenarioError naming the ones the
// file holds.
func (f *ScenarioFile) Scenario(name string) (LoadedScenario, error) {
	names := make([]string, 0, len(f.Scenarios))
	for _, s := range f.Scenarios {
		if s.ScenarioName() == name {
			return s, nil
		}
		names = append(names, s.ScenarioName())
	}
	return nil, invalidf("%s: no scenario called %s (has %s)", f.Path, pyRepr(name), strings.Join(names, ", "))
}

// LoadScenarioFile loads a file holding one scenario, as whichever kind it is.
//
// Deprecated: Use LoadFile instead, which returns every scenario a file holds.
func LoadScenarioFile(path string) (LoadedScenario, error) {
	slog.Warn("eval: `LoadScenarioFile` is deprecated and will be removed. Use `LoadFile` instead.")
	file, err := LoadFile(path)
	if err != nil {
		return nil, err
	}
	if len(file.Scenarios) != 1 {
		return nil, invalidf("%s: holds %d scenarios; use LoadFile()", path, len(file.Scenarios))
	}
	return file.Scenarios[0], nil
}

// Load reads a file holding one scripted scenario.
//
// Deprecated: Use LoadFile instead, which returns every scenario a file holds,
// of every kind.
func Load(path string) (*ScriptScenario, error) {
	file, err := LoadFile(path)
	if err != nil {
		return nil, err
	}
	if len(file.Scenarios) != 1 {
		return nil, invalidf("%s: holds %d scenarios; use LoadFile()", path, len(file.Scenarios))
	}
	script, ok := file.Scenarios[0].(*ScriptScenario)
	if !ok {
		return nil, invalidf("%s: holds a scenario that is not scripted; use LoadFile()", path)
	}
	return script, nil
}

// scenarioFromMapping parses one scenario's mapping as a simulation (persona:),
// a script (turns:), or a rubric (success:).
//
// own is the entry's keys without the file's defaults, nil when the mapping is
// all the entry's own. A rubric that takes its metrics: from the file leaves out
// the measured ones, which only a simulation can score; a measured metric the
// rubric lists itself is refused.
func scenarioFromMapping(data map[string]any, path string, own map[string]any) (LoadedScenario, error) {
	_, hasPersona := data["persona"]
	_, hasTurns := data["turns"]
	_, hasSuccess := data["success"]
	switch {
	case hasPersona && hasTurns:
		return nil, invalidf("%s: a scenario is scripted ('turns:') or a simulation ('persona:'), not both", path)
	case hasPersona:
		return parseSimulation(data, path)
	case hasTurns:
		return parseScript(data, path)
	case hasSuccess:
		if _, ownMetrics := own["metrics"]; own != nil && !ownMetrics {
			data = maps.Clone(data)
			data["metrics"] = judgedMetrics(data["metrics"])
		}
		return RubricFromMap(data, path)
	default:
		return nil, invalidf("%s: a scenario needs 'turns:' (scripted), 'persona:' (a simulation), or "+
			"'success:' (a rubric)", path)
	}
}

// judgedMetrics are a metrics: list's judged entries, without its measured
// ones.
func judgedMetrics(raw any) []any {
	list, _ := raw.([]any)
	judged := []any{}
	for _, m := range list {
		if entry, ok := m.(map[string]any); ok {
			if _, measured := entry["measure"]; measured {
				continue
			}
		}
		judged = append(judged, m)
	}
	return judged
}

// IsScenarioFile reports whether a YAML file is a scenario of any kind rather
// than a fragment one includes.
//
// Every scenario file has a `name:`; a fragment a scenario includes with
// !include (a block of turns, a judge: block) has none. A file that does not
// parse counts as a scenario, so loading it reports the error instead of a
// directory run skipping it silently.
func IsScenarioFile(path string) bool {
	data, err := loadMapping(path)
	if err != nil {
		return true
	}
	_, named := data[keyName]
	return named
}
