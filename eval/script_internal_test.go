package eval

import (
	"bytes"
	"errors"
	"log/slog"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/gojargo/jargo/processor/rtvi"
)

// writeYAML writes body to a scenario file in a fresh directory and returns its
// path.
func writeYAML(t *testing.T, body string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "scenario.yaml")
	if err := os.WriteFile(path, []byte(body), 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

// parseScriptFile parses a file holding one scripted scenario's own keys at its
// top level.
func parseScriptFile(path string) (*ScriptScenario, error) {
	data, err := loadMapping(path)
	if err != nil {
		return nil, err
	}
	return parseScript(data, path)
}

// script parses body as one scripted scenario's own keys, failing the test on
// an error.
func script(t *testing.T, body string) *ScriptScenario {
	t.Helper()
	s, err := parseScriptFile(writeYAML(t, body))
	if err != nil {
		t.Fatal(err)
	}
	return s
}

// scriptErr parses body and returns the error it is rejected with, failing the
// test when it is accepted.
func scriptErr(t *testing.T, body string) error {
	t.Helper()
	_, err := parseScriptFile(writeYAML(t, body))
	if err == nil {
		t.Fatalf("expected an error for:\n%s", body)
	}
	if _, ok := errors.AsType[*ScenarioError](err); !ok {
		t.Fatalf("error %v is not a *ScenarioError", err)
	}
	return err
}

// wantErrContains asserts err's message holds each of wants.
func wantErrContains(t *testing.T, err error, wants ...string) {
	t.Helper()
	for _, want := range wants {
		if !strings.Contains(err.Error(), want) {
			t.Fatalf("error %q does not contain %q", err, want)
		}
	}
}

// captureWarnings runs fn with the default logger writing to a buffer, and
// returns the warnings it logged.
func captureWarnings(t *testing.T, fn func()) []string {
	t.Helper()
	var buf bytes.Buffer
	prev := slog.Default()
	slog.SetDefault(slog.New(slog.NewTextHandler(&buf, &slog.HandlerOptions{Level: slog.LevelWarn})))
	defer slog.SetDefault(prev)
	fn()
	var lines []string
	for line := range strings.Lines(buf.String()) {
		lines = append(lines, strings.TrimSpace(line))
	}
	return lines
}

func TestScriptMinimalValid(t *testing.T) {
	s := script(t, `
name: minimal
turns:
  - user: "hello"
    expect:
      - event: user_started_speaking
`)
	if s.Name != "minimal" || len(s.Turns) != 1 || s.Turns[0].User != "hello" {
		t.Fatalf("unexpected scenario: %+v", s)
	}
	if s.Turns[0].Expect[0].Event != EventUserStartedSpeaking || s.Turns[0].SendAfter != nil {
		t.Fatalf("unexpected turn: %+v", s.Turns[0])
	}
	// bot_audio defaults to false: evals are text and silent unless they opt in.
	if s.BotAudio || s.Transcriber != nil {
		t.Fatalf("want text modality by default, got %+v", s.Configured)
	}
}

func TestScriptJudgeAudioModalityEnablesTranscriber(t *testing.T) {
	s := script(t, "name: a\n"+
		"judge:\n"+
		"  modality: audio\n"+
		"  transcription: {service: whisper, model: base}\n"+
		"turns: [{user: hi, expect: [{event: tts_response, eval: ok}]}]\n")
	if !s.BotAudio || s.Transcriber == nil {
		t.Fatalf("want audio modality with a transcriber, got %+v", s.Configured)
	}
	want := map[string]any{"service": "whisper", "model": "base"}
	if s.Transcriber.Service != "whisper" || s.Transcriber.Model != "base" || !reflect.DeepEqual(s.Transcriber.Raw, want) {
		t.Fatalf("transcriber = %+v, want %v", s.Transcriber, want)
	}
}

func TestScriptJudgeAudioRequiresTranscription(t *testing.T) {
	err := scriptErr(t, "name: a\njudge: {modality: audio}\n"+
		"turns: [{user: hi, expect: [{event: tts_response, eval: ok}]}]\n")
	wantErrContains(t, err, "transcription")
}

func TestScriptJudgeInvalidModalityRejected(t *testing.T) {
	_ = scriptErr(t, "name: a\njudge: {modality: bogus}\nturns: [{user: hi, expect: [{event: llm_started}]}]\n")
}

func TestScriptUserAudioModality(t *testing.T) {
	s := script(t, "name: a\n"+
		"user:\n"+
		"  modality: audio\n"+
		"  speech: {service: cartesia, voice: v1}\n"+
		"turns: [{user: hi, expect: [{event: llm_started}]}]\n")
	if !s.UserAudio || s.UserSpeech == nil {
		t.Fatalf("want audio modality with speech, got %+v", s.Configured)
	}
	want := map[string]any{"service": "cartesia", "voice": "v1"}
	if s.UserSpeech.Service != "cartesia" || s.UserSpeech.Voice != "v1" || !reflect.DeepEqual(s.UserSpeech.Raw, want) {
		t.Fatalf("speech = %+v, want %v", s.UserSpeech, want)
	}
}

func TestScriptUserTurnWithPauses(t *testing.T) {
	s := script(t, "name: a\n"+
		"user: {modality: audio, speech: {service: kokoro, voice: v1}}\n"+
		"turns:\n"+
		"  - user: ['So last week I went to', {pause_ms: 2000}, 'Spain.']\n")
	turn := s.Turns[0]
	if turn.User != "So last week I went to Spain." {
		t.Fatalf("user = %q", turn.User)
	}
	want := []TurnPart{{Text: "So last week I went to"}, {PauseMS: 2000}, {Text: "Spain."}}
	if !reflect.DeepEqual(turn.Parts, want) {
		t.Fatalf("parts = %+v, want %+v", turn.Parts, want)
	}
}

func TestScriptATurnCanPauseUpToTheLimit(t *testing.T) {
	s := script(t, "name: a\n"+
		"user: {modality: audio, speech: {service: kokoro, voice: v1}}\n"+
		"turns: [{user: ['Hi', {pause_ms: 30000}, 'so', {pause_ms: 30000}]}]\n")
	want := []TurnPart{{Text: "Hi"}, {PauseMS: 30000}, {Text: "so"}, {PauseMS: 30000}}
	if !reflect.DeepEqual(s.Turns[0].Parts, want) {
		t.Fatalf("parts = %+v, want %+v", s.Turns[0].Parts, want)
	}
}

func TestScriptUserTurnPausesAreValidated(t *testing.T) {
	audio := "user: {modality: audio, speech: {service: kokoro, voice: v1}}\n"
	cases := map[string]string{
		"pauses but no text": audio + "turns: [{user: [{pause_ms: 500}]}]\n",
		"items must be text": audio + "turns: [{user: ['Hi', {pause_ms: 0}]}]\n",
		"text modality":      "turns: [{user: ['Hi', {pause_ms: 500}, 'there']}]\n",
		"60001 ms in total":  audio + "turns: [{user: ['Hi', {pause_ms: 60001}]}]\n",
		"60500 ms in total":  audio + "turns: [{user: ['Hi', {pause_ms: 30000}, 'so', {pause_ms: 30500}]}]\n",
	}
	for message, body := range cases {
		t.Run(message, func(t *testing.T) {
			wantErrContains(t, scriptErr(t, "name: a\n"+body), message)
		})
	}
}

func TestScriptUserAudioRequiresSpeech(t *testing.T) {
	err := scriptErr(t, "name: a\nuser: {modality: audio}\nturns: [{user: hi, expect: [{event: llm_started}]}]\n")
	wantErrContains(t, err, "speech")
}

func TestScriptResponseEventResolvesToModality(t *testing.T) {
	// judge.modality audio: response stays response, the audio transcription.
	audio := script(t, "name: a\n"+
		"judge: {modality: audio, transcription: {service: whisper}}\n"+
		"turns: [{user: hi, expect: [{event: response, eval: ok}]}]\n")
	if got := audio.Turns[0].Expect[0].Event; got != EventResponse {
		t.Fatalf("audio modality event = %q, want response", got)
	}
	// text (the default): response falls back to llm_response.
	text := script(t, "name: a\nturns: [{user: hi, expect: [{event: response, eval: ok}]}]\n")
	if got := text.Turns[0].Expect[0].Event; got != EventLLMResponse {
		t.Fatalf("text modality event = %q, want llm_response", got)
	}
}

func TestScriptTTSResponseInTextModalityRejected(t *testing.T) {
	err := scriptErr(t, "name: a\nturns: [{user: hi, expect: [{event: tts_response, eval: ok}]}]\n")
	wantErrContains(t, err, "tts_response")
}

func TestScriptAllExpectationFields(t *testing.T) {
	s := script(t, `
name: all_fields
turns:
  - user: "x"
    expect:
      - event: llm_response
        within_ms: 500
        text_contains: "bar"
        text_excludes: "baz"
        eval: "is friendly"
`)
	exp := s.Turns[0].Expect[0]
	if exp.WithinMS != 500 || exp.TextContains != "bar" || exp.TextExcludes != "baz" || exp.Eval != "is friendly" {
		t.Fatalf("unexpected expectation: %+v", exp)
	}
	if exp.Calls != nil || exp.Absent {
		t.Fatalf("want no calls and not absent, got %+v", exp)
	}
}

func TestScriptTextMatchesParsed(t *testing.T) {
	s := script(t, `
name: matches
turns:
  - user: "x"
    expect:
      - event: llm_response
        text_matches: "^Hi"
      - event: llm_response
        text_matches: ["^Hi", "!$"]
`)
	if got := s.Turns[0].Expect[0].TextMatches; !reflect.DeepEqual(got, Patterns{"^Hi"}) {
		t.Fatalf("text_matches = %v", got)
	}
	if got := s.Turns[0].Expect[1].TextMatches; !reflect.DeepEqual(got, Patterns{"^Hi", "!$"}) {
		t.Fatalf("text_matches = %v", got)
	}
}

func TestScriptTextMatchesRejectsBadPatterns(t *testing.T) {
	for _, value := range []string{`"(unclosed"`, "[1, 2]"} {
		_ = scriptErr(t, "name: bad_matches\nturns:\n  - user: \"x\"\n    expect:\n"+
			"      - event: llm_response\n        text_matches: "+value+"\n")
	}
}

func TestScriptAbsentExpectationParsed(t *testing.T) {
	s := script(t, `
name: absent
turns:
  - user: "x"
    expect:
      - event: llm_response
        eval: "answers"
      - event: llm_response
        absent: true
        within_ms: 5000
`)
	exp := s.Turns[0].Expect[1]
	if !exp.Absent || exp.WithinMS != 5000 {
		t.Fatalf("unexpected expectation: %+v", exp)
	}
}

func TestScriptAbsentRejectsContentChecks(t *testing.T) {
	for _, extra := range []string{
		`eval: "repeats itself"`, `text_contains: "again"`, `text_excludes: "x"`, `text_matches: "x"`,
	} {
		_ = scriptErr(t, "name: bad_absent\nturns:\n  - user: \"x\"\n    expect:\n"+
			"      - event: llm_response\n        absent: true\n        "+extra+"\n")
	}
}

func TestScriptAbsentMustBeBoolean(t *testing.T) {
	_ = scriptErr(t, "name: bad_absent_type\nturns:\n  - user: \"x\"\n    expect:\n"+
		"      - event: llm_response\n        absent: \"yes please\"\n")
}

// A single function_call uses the name:/args: shorthand.
func TestScriptFunctionCallNameArgsShorthand(t *testing.T) {
	s := script(t, `
name: one_call
turns:
  - user: "weather?"
    expect:
      - event: function_call
        name: get_weather
        args: {city: Paris}
`)
	want := []FunctionCall{{Name: "get_weather", Args: map[string]any{"city": "Paris"}}}
	if got := s.Turns[0].Expect[0].Calls; !reflect.DeepEqual(got, want) {
		t.Fatalf("calls = %+v, want %+v", got, want)
	}
}

// Several calls in a turn go under calls:, matched in any order.
func TestScriptFunctionCallCallsListAnyOrder(t *testing.T) {
	s := script(t, `
name: two_calls
turns:
  - user: "weather and food?"
    expect:
      - event: function_call
        calls:
          - get_weather
          - {name: get_restaurants, args: {city: Paris}}
`)
	want := []FunctionCall{{Name: "get_weather"}, {Name: "get_restaurants", Args: map[string]any{"city": "Paris"}}}
	if got := s.Turns[0].Expect[0].Calls; !reflect.DeepEqual(got, want) {
		t.Fatalf("calls = %+v, want %+v", got, want)
	}
}

// A bare function_call, with no name or calls, matches any single call.
func TestScriptBareFunctionCallMatchesAny(t *testing.T) {
	s := script(t, "name: bare\nturns: [{user: hi, expect: [{event: function_call}]}]\n")
	if got := s.Turns[0].Expect[0].Calls; !reflect.DeepEqual(got, []FunctionCall{{}}) {
		t.Fatalf("calls = %+v, want one nameless call", got)
	}
}

func TestScriptLLMMarkerExpectation(t *testing.T) {
	s := script(t, `
name: markers
turns:
  - user: "Let me think, hmmm"
    expect:
      - event: llm_marker
        marker: incomplete
  - user: "Japan."
    expect:
      - event: llm_marker
        marker: complete
      - event: response
`)
	if s.Turns[0].Expect[0].Marker != MarkerIncomplete || s.Turns[1].Expect[0].Marker != MarkerComplete {
		t.Fatalf("unexpected markers: %+v", s.Turns)
	}
	if !s.NeedsMarkerEvents() {
		t.Fatal("a scenario asserting on a marker should ask for markers")
	}
}

func TestScriptLLMMarkerFormatFields(t *testing.T) {
	s := script(t, `
name: markers
turns:
  - user: "I'd go to Japan because"
    expect:
      - event: llm_marker
        marker: short
        marker_first: true
        markers: 1
        text_after: false
`)
	exp := s.Turns[0].Expect[0]
	if exp.MarkerFirst == nil || !*exp.MarkerFirst || exp.Markers == nil || *exp.Markers != 1 ||
		exp.TextAfter == nil || *exp.TextAfter {
		t.Fatalf("unexpected marker checks: %+v", exp)
	}
	for _, tc := range []struct{ bad, message string }{
		{"expect: [{event: response, markers: 1}]", "only applies to the 'llm_marker' event"},
		{"expect: [{event: llm_marker, markers: yes}]", "'markers:' must be a int"},
		{"expect: [{event: llm_marker, marker_first: 1}]", "'marker_first:' must be a bool"},
		{"expect: [{event: llm_marker, absent: true, text_after: false}]", "cannot be combined"},
	} {
		err := scriptErr(t, "name: bad\nturns:\n  - user: \"x\"\n    "+tc.bad+"\n")
		wantErrContains(t, err, tc.message)
	}
}

func TestScriptLLMMarkerUnknownKindRejected(t *testing.T) {
	err := scriptErr(t, "name: bad_marker\nturns:\n  - user: \"x\"\n    expect: [{event: llm_marker, marker: done}]\n")
	wantErrContains(t, err, "'marker:' must be one of")
}

func TestScriptMarkerOnOtherEventRejected(t *testing.T) {
	err := scriptErr(t, "name: bad\nturns:\n  - user: \"x\"\n    expect: [{event: response, marker: complete}]\n")
	wantErrContains(t, err, "only applies to the 'llm_marker' event")
}

func TestScriptAbsentRejectsMarker(t *testing.T) {
	_ = scriptErr(t, "name: absent_marker\nturns:\n  - user: \"x\"\n"+
		"    expect: [{event: llm_marker, absent: true, marker: complete}]\n")
}

func TestScriptSendAfterParsed(t *testing.T) {
	s := script(t, `
name: with_send_after
turns:
  - user: "first"
    expect: [{event: user_stopped_speaking}]
  - send_after: {event: user_stopped_speaking, delay_ms: 200}
    user: "second"
    expect: [{event: user_stopped_speaking}]
`)
	if s.Turns[0].SendAfter != nil {
		t.Fatalf("the first turn has no schedule, got %+v", s.Turns[0].SendAfter)
	}
	if got := s.Turns[1].SendAfter; got == nil || *got != (SendAfter{Event: EventUserStoppedSpeaking, DelayMS: 200}) {
		t.Fatalf("send_after = %+v", got)
	}
}

// A turn without user: is observation-only, for a bot-first scenario.
func TestScriptExpectOnlyTurn(t *testing.T) {
	s := script(t, "name: bot_first\nturns:\n  - expect:\n      - event: llm_response\n")
	if s.Turns[0].User != "" || s.Turns[0].Expect[0].Event != EventLLMResponse {
		t.Fatalf("unexpected turn: %+v", s.Turns[0])
	}
}

func TestScriptDTMFTurnParsed(t *testing.T) {
	s := script(t, `
name: dtmf
turns:
  - dtmf: "123#"
    expect:
      - event: user_transcription
        text_contains: "DTMF: 123#"
`)
	if s.Turns[0].DTMF != "123#" || s.Turns[0].User != "" {
		t.Fatalf("unexpected turn: %+v", s.Turns[0])
	}
}

// An unquoted digit sequence reads as an integer; its digits are taken.
func TestScriptDTMFUnquotedIntNormalized(t *testing.T) {
	if got := script(t, "name: dtmf\nturns: [{dtmf: 123}]\n").Turns[0].DTMF; got != "123" {
		t.Fatalf("dtmf = %q", got)
	}
}

// A leading zero stays literal digits rather than being read as octal, which
// would silently send the wrong keys.
func TestScriptDTMFUnquotedLeadingZeroPreserved(t *testing.T) {
	for _, seq := range []string{"012", "010", "007", "0420", "000"} {
		if got := script(t, "name: dtmf\nturns: [{dtmf: "+seq+"}]\n").Turns[0].DTMF; got != seq {
			t.Fatalf("dtmf = %q, want %q", got, seq)
		}
	}
}

// A hex-looking token isn't read as a number; its x fails validation.
func TestScriptDTMFUnquotedHexRejected(t *testing.T) {
	wantErrContains(t, scriptErr(t, "name: bad\nturns: [{dtmf: 0x10}]\n"), "invalid keypad entry")
}

func TestScriptDTMFInvalidEntryRejected(t *testing.T) {
	wantErrContains(t, scriptErr(t, "name: bad\nturns: [{dtmf: \"1A\"}]\n"), "invalid keypad entry 'A'")
}

func TestScriptDTMFAndUserMutuallyExclusive(t *testing.T) {
	wantErrContains(t, scriptErr(t, "name: bad\nturns: [{user: hi, dtmf: \"1\"}]\n"), "one or the other")
}

func TestScriptSendAfterAllowedOnDTMFTurn(t *testing.T) {
	s := script(t, "name: bargein\nturns:\n  - dtmf: \"0\"\n    send_after: {event: llm_started, delay_ms: 300}\n")
	if s.Turns[0].SendAfter == nil || s.Turns[0].SendAfter.Event != EventLLMStarted {
		t.Fatalf("send_after = %+v", s.Turns[0].SendAfter)
	}
}

func TestScriptJudgeEvalPreserved(t *testing.T) {
	s := script(t, `
name: with_judge
judge:
  eval:
    service: openai
    model: gpt-4o-mini
    endpoint: http://custom-endpoint
turns:
  - user: "hi"
    expect: [{event: user_stopped_speaking}]
`)
	want := map[string]any{"service": "openai", "model": "gpt-4o-mini", "endpoint": "http://custom-endpoint"}
	j := s.Judge
	if !reflect.DeepEqual(j.Raw, want) || j.Service != "openai" || j.Model != "gpt-4o-mini" ||
		j.Endpoint != "http://custom-endpoint" || j.Extra != nil {
		t.Fatalf("judge = %+v, want %v", j, want)
	}
}

func TestScriptJudgeBlockDefaultsToOllama(t *testing.T) {
	s := script(t, "name: e\nturns: [{user: hi, expect: [{event: user_stopped_speaking}]}]\n")
	want := map[string]any{"service": "ollama", "model": "gemma4:12b", "extra": map[string]any{"reasoning_effort": "none"}}
	j := s.Judge
	if !reflect.DeepEqual(j.Raw, want) || j.Service != "ollama" || j.Model != "gemma4:12b" ||
		!reflect.DeepEqual(j.Extra, want["extra"]) {
		t.Fatalf("judge = %+v, want %v", j, want)
	}
}

func TestScriptJudgeBlockNonMappingRejected(t *testing.T) {
	err := scriptErr(t, "name: bad\njudge: not_a_mapping\nturns: [{user: hi, expect: [{event: x}]}]\n")
	wantErrContains(t, err, "'judge:'")
}

func TestScriptJudgeEvalExplainerAndKnobs(t *testing.T) {
	s := script(t, `
name: knobs
judge:
  eval:
    factory: my_evals.judge
    explainer: {service: ollama, model: explainer}
    explain_below: 0.5
    allow_continue: false
turns: []
`)
	j := s.Judge
	e := j.Explainer
	if j.Factory != "my_evals.judge" || e.LLM == nil || e.LLM.Model != "explainer" || e.Disabled {
		t.Fatalf("judge = %+v", j)
	}
	if !reflect.DeepEqual(e.LLM.Raw, map[string]any{"service": "ollama", "model": "explainer"}) {
		t.Fatalf("explainer raw = %v", e.LLM.Raw)
	}
	if j.ExplainBelow == nil || *j.ExplainBelow != 0.5 || j.AllowContinue == nil || *j.AllowContinue {
		t.Fatalf("judge knobs = %+v", j)
	}
	off := script(t, "name: disabled\njudge: {eval: {explainer: false}}\nturns: []\n")
	if !off.Judge.Explainer.Disabled || off.Judge.Explainer.LLM != nil {
		t.Fatalf("explainer: false = %+v", off.Judge)
	}
}

func TestScriptMissingNameField(t *testing.T) {
	wantErrContains(t, scriptErr(t, "turns: []\n"), "'name:'")
}

func TestScriptSendAfterWithoutUserRejected(t *testing.T) {
	err := scriptErr(t, "name: bad\nturns:\n  - send_after: {event: x, delay_ms: 100}\n    expect: [{event: y}]\n")
	wantErrContains(t, err, "send_after", "no 'user:'")
}

func TestScriptInvalidSendAfterDelayRejected(t *testing.T) {
	err := scriptErr(t, "name: bad\nturns:\n  - send_after: {event: x, delay_ms: -1}\n    user: \"y\"\n"+
		"    expect: [{event: z}]\n")
	wantErrContains(t, err, "non-negative")
}

func TestScriptSendAfterWithoutEventIsPureDelay(t *testing.T) {
	s := script(t, `
name: paced
turns:
  - user: "first"
    expect: [{event: llm_started}]
  - user: "second"
    send_after: {delay_ms: 500}
    expect: [{event: llm_started}]
`)
	if got := s.Turns[1].SendAfter; got == nil || *got != (SendAfter{DelayMS: 500}) {
		t.Fatalf("send_after = %+v", got)
	}
}

func TestScriptSendAfterWithoutEventOrDelayRejected(t *testing.T) {
	err := scriptErr(t, "name: bad\nturns: [{user: hi, send_after: {delay_ms: 0}, expect: [{event: x}]}]\n")
	wantErrContains(t, err, "positive 'delay_ms:'")
}

// A turn without expect: just sends or waits.
func TestScriptMissingExpectDefaultsToEmpty(t *testing.T) {
	if got := script(t, "name: ok\nturns: [{user: hi}]\n").Turns[0].Expect; len(got) != 0 {
		t.Fatalf("expect = %+v, want none", got)
	}
}

func TestScriptExpectNonListRejected(t *testing.T) {
	wantErrContains(t, scriptErr(t, "name: bad\nturns: [{user: hi, expect: nope}]\n"), "expect")
}

func TestScriptExpectationMissingEventRejected(t *testing.T) {
	wantErrContains(t, scriptErr(t, "name: bad\nturns: [{user: hi, expect: [{within_ms: 100}]}]\n"), "event")
}

// An expectation built directly keeps its zero defaults.
func TestScriptExpectationDefaults(t *testing.T) {
	e := Expectation{Event: "foo"}
	if e.WithinMS != 0 || e.TextContains != "" || e.Eval != "" {
		t.Fatalf("unexpected defaults: %+v", e)
	}
}

// eval: on a user-side event draws a warning: the user transcript is
// deterministic, so judging it adds cost without signal. The scenario still
// loads, with the criterion.
func TestScriptEvalOnNonBotEventWarns(t *testing.T) {
	var s *ScriptScenario
	warnings := captureWarnings(t, func() {
		s = script(t, `
name: misused_eval
turns:
  - user: "hi"
    expect:
      - event: user_stopped_speaking
        eval: "is a greeting"
`)
	})
	if s.Turns[0].Expect[0].Eval != "is a greeting" {
		t.Fatalf("eval = %q", s.Turns[0].Expect[0].Eval)
	}
	if len(warnings) != 1 || !strings.Contains(warnings[0], "'eval:' on event 'user_stopped_speaking'") {
		t.Fatalf("warnings = %q, want one about the eval", warnings)
	}
}

// eval: on a function call judges the matched calls, so it is not a misuse.
func TestScriptEvalOnFunctionCallIsAccepted(t *testing.T) {
	var s *ScriptScenario
	warnings := captureWarnings(t, func() {
		s = script(t, `
name: judged_call
turns:
  - user: "suggest a session"
    expect:
      - event: function_call
        calls: [{name: submit_session_suggestion}]
        eval: "the suggestion is about tracing"
`)
	})
	exp := s.Turns[0].Expect[0]
	if exp.Event != EventFunctionCall || exp.Eval != "the suggestion is about tracing" ||
		!reflect.DeepEqual(exp.Calls, []FunctionCall{{Name: "submit_session_suggestion"}}) {
		t.Fatalf("unexpected expectation: %+v", exp)
	}
	if len(warnings) != 0 {
		t.Fatalf("warnings = %q, want none", warnings)
	}
}

// A stopped call carries no arguments, so there is nothing for a judge to see.
func TestScriptEvalOnAStoppedCallIsRejected(t *testing.T) {
	err := scriptErr(t, `
name: judged_stop
turns:
  - user: "suggest a session"
    expect:
      - event: function_call_stopped
        calls: [{name: submit_session_suggestion}]
        eval: "the suggestion is about tracing"
`)
	wantErrContains(t, err, "'function_call_stopped'")
}

// A turn built directly, as a programmatic scenario does.
func TestScriptTurnConstruction(t *testing.T) {
	turn := ScriptTurn{User: "hi", Expect: []Expectation{{Event: "x"}}}
	if turn.User != "hi" || turn.SendAfter != nil || turn.Image != "" {
		t.Fatalf("unexpected turn: %+v", turn)
	}
}

func TestScriptTurnImageResolvedRelativeToScenario(t *testing.T) {
	p := writeYAML(t, "name: t\nturns: [{user: hi, image: pics/cat.jpg, expect: [{event: llm_response}]}]\n")
	s, err := parseScriptFile(p)
	if err != nil {
		t.Fatal(err)
	}
	want, err := filepath.Abs(filepath.Join(filepath.Dir(p), "pics/cat.jpg"))
	if err != nil {
		t.Fatal(err)
	}
	if s.Turns[0].Image != want {
		t.Fatalf("image = %q, want %q", s.Turns[0].Image, want)
	}
}

func TestScriptTurnImageNonStringRejected(t *testing.T) {
	wantErrContains(t, scriptErr(t, "name: t\nturns: [{user: hi, image: 5, expect: [{event: x}]}]\n"), "image")
}

func TestScriptContextDefaultsToEmpty(t *testing.T) {
	s := script(t, "name: e\nturns: [{user: hi, expect: [{event: user_stopped_speaking}]}]\n")
	if got := s.Context; len(got) != 0 {
		t.Fatalf("context = %+v, want none", got)
	}
}

// A context is a list of messages, in jargo's message form (text:).
func TestScriptContextParsedAsList(t *testing.T) {
	s := script(t, `
name: with_context
context:
  - role: system
    text: "You are a helpful assistant."
turns:
  - user: hi
    expect: [{event: user_stopped_speaking}]
`)
	if len(s.Context) != 1 || s.Context[0].Role != "system" || s.Context[0].Text != "You are a helpful assistant." {
		t.Fatalf("context = %+v", s.Context)
	}
}

func TestScriptContextNonListRejected(t *testing.T) {
	err := scriptErr(t, "name: bad\ncontext: not_a_list\nturns: [{user: hi, expect: [{event: x}]}]\n")
	wantErrContains(t, err, "'context:'")
}

// judge: !include and user: !include let scenarios share config. Includes
// resolve relative to the scenario file's directory.
func TestScriptIncludeResolvesJudgeAndUserBlocks(t *testing.T) {
	d := t.TempDir()
	write := func(name, body string) {
		if err := os.WriteFile(filepath.Join(d, name), []byte(body), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	write("judge_audio.yaml", "modality: audio\n"+
		"eval: {service: ollama, model: llama3:latest}\n"+
		"transcription: {service: whisper, model: base}\n")
	write("user_audio.yaml", "modality: audio\nspeech: {service: kokoro, voice: af_heart}\n")
	write("math.yaml", "name: math\n"+
		"user: !include user_audio.yaml\n"+
		"judge: !include judge_audio.yaml\n"+
		"turns: [{user: hi, expect: [{event: response, eval: ok}]}]\n")

	s, err := parseScriptFile(filepath.Join(d, "math.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	if !s.BotAudio || !reflect.DeepEqual(s.Transcriber.Raw, map[string]any{"service": "whisper", "model": "base"}) {
		t.Fatalf("transcriber = %+v", s.Transcriber)
	}
	if !reflect.DeepEqual(s.Judge.Raw, map[string]any{"service": "ollama", "model": "llama3:latest"}) {
		t.Fatalf("judge = %+v", s.Judge)
	}
	if !s.UserAudio || !reflect.DeepEqual(s.UserSpeech.Raw, map[string]any{"service": "kokoro", "voice": "af_heart"}) {
		t.Fatalf("speech = %+v", s.UserSpeech)
	}
}

func TestScriptStopOnFailure(t *testing.T) {
	if s := script(t, "name: a\nturns: [{user: hi, expect: [{event: llm_response}]}]\n"); !s.stopsOnFailure() {
		t.Fatal("stop_on_failure should default to true")
	}
	s := script(t, "name: a\nstop_on_failure: false\nturns: [{user: hi, expect: [{event: llm_response}]}]\n")
	if s.stopsOnFailure() {
		t.Fatal("stop_on_failure: false should drive every turn")
	}
}

// A turn can name an audio file to play instead of synthesizing its text.
const audioUser = "user:\n  modality: audio\n  speech: {service: kokoro, voice: af_heart}\n"

func TestTurnAudioPathResolvesAgainstTheScenario(t *testing.T) {
	d := t.TempDir()
	if err := os.MkdirAll(filepath.Join(d, "clips"), 0o750); err != nil {
		t.Fatal(err)
	}
	p := filepath.Join(d, "s.yaml")
	body := "name: a\n" + audioUser + "turns: [{user: hi, audio: clips/hi.wav}]\n"
	if err := os.WriteFile(p, []byte(body), 0o600); err != nil {
		t.Fatal(err)
	}
	s, err := parseScriptFile(p)
	if err != nil {
		t.Fatal(err)
	}
	want, _ := filepath.Abs(filepath.Join(d, "clips", "hi.wav"))
	if s.Turns[0].Audio != want {
		t.Fatalf("audio = %q, want %q", s.Turns[0].Audio, want)
	}
	// The text stays the turn's input for the judge and text_contains.
	if s.Turns[0].User != "hi" {
		t.Fatalf("user = %q", s.Turns[0].User)
	}
}

func TestTurnAudioWithoutUserIsRejected(t *testing.T) {
	wantErrContains(t, scriptErr(t, "name: a\n"+audioUser+"turns: [{audio: hi.wav}]\n"), "no 'user:'")
}

func TestTurnAudioWithDTMFIsRejected(t *testing.T) {
	err := scriptErr(t, "name: a\n"+audioUser+"turns: [{user: hi, audio: hi.wav, dtmf: \"1\"}]\n")
	wantErrContains(t, err, "one or the other")
}

func TestTurnAudioNeedsAudioModality(t *testing.T) {
	wantErrContains(t, scriptErr(t, "name: a\nturns: [{user: hi, audio: hi.wav}]\n"), "text modality")
}

func TestTurnAudioMustBeAPath(t *testing.T) {
	wantErrContains(t, scriptErr(t, "name: a\n"+audioUser+"turns: [{user: hi, audio: 3}]\n"), "must be a path string")
}

// Nothing is synthesized, so the scenario need not name a TTS.
func TestTurnAudioFileOnlyScenarioNeedsNoSpeechConfig(t *testing.T) {
	s := script(t, "name: a\nuser: {modality: audio}\nturns: [{user: hi, audio: hi.wav}]\n")
	if !s.UserAudio || s.UserSpeech != nil {
		t.Fatalf("configured = %+v", s.Configured)
	}
}

func TestTurnAudioASynthesizedTurnStillNeedsSpeech(t *testing.T) {
	err := scriptErr(t, "name: a\nuser: {modality: audio}\n"+
		"turns: [{user: recorded, audio: hi.wav}, {user: synthesized}]\n")
	wantErrContains(t, err, "turn(s) [1]")
}

func TestDeprecatedTriggerDisconnectIsStillRead(t *testing.T) {
	var s *ScriptScenario
	warnings := captureWarnings(t, func() {
		s = script(t, "name: t\ntrigger_disconnect: true\nturns: [{user: hi}]\n")
	})
	if !s.TriggerDisconnect {
		t.Fatal("trigger_disconnect: true should still be read")
	}
	if len(warnings) != 1 || !strings.Contains(warnings[0], "`trigger_disconnect:`") {
		t.Fatalf("warnings = %q, want the deprecation", warnings)
	}
}

func TestDeprecatedTriggerDisconnectAbsentDoesNotWarn(t *testing.T) {
	var s *ScriptScenario
	warnings := captureWarnings(t, func() { s = script(t, "name: t\nturns: [{user: hi}]\n") })
	if s.TriggerDisconnect || len(warnings) != 0 {
		t.Fatalf("trigger_disconnect = %v, warnings = %q", s.TriggerDisconnect, warnings)
	}
}

// scriptLevel is the report level a one-turn scenario asks the bot for.
func scriptLevel(expect ...Expectation) rtvi.FunctionCallReportLevel {
	s := &ScriptScenario{Name: "t", Turns: []ScriptTurn{{User: "x", Expect: expect}}}
	return s.RequiredReportLevel()
}

// The minimal function-call report level the harness asks the bot for.
func TestScriptRequiredReportLevel(t *testing.T) {
	cases := map[string]struct {
		expect []Expectation
		want   rtvi.FunctionCallReportLevel
	}{
		"none without function call": {[]Expectation{{Event: EventLLMResponse}}, ""},
		// Asserting the call happened needs no name or args.
		"none for bare function call": {[]Expectation{{Event: EventFunctionCall}}, ""},
		"name when only name asserted": {
			[]Expectation{{Event: EventFunctionCall, Calls: []FunctionCall{{Name: "get_weather"}}}}, rtvi.ReportName,
		},
		"full when args asserted": {
			[]Expectation{{Event: EventFunctionCall, Calls: []FunctionCall{
				{Name: "get_weather", Args: map[string]any{"city": "P"}},
			}}}, rtvi.ReportFull,
		},
		// The judge reads the call's arguments, so names alone are not enough.
		"full when a call is judged": {
			[]Expectation{{
				Event: EventFunctionCall, Calls: []FunctionCall{{Name: "submit"}},
				Eval: "submitted for the right person",
			}}, rtvi.ReportFull,
		},
		// A reply is judged on the spoken conversation only.
		"a judged reply after a call does not need args": {
			[]Expectation{
				{Event: EventFunctionCall, Calls: []FunctionCall{{Name: "submit"}}},
				{Event: EventLLMResponse, Eval: "confirms what it submitted"},
			}, rtvi.ReportName,
		},
		// A stopped call is matched by name and args too.
		"full when a stopped call's args are asserted": {
			[]Expectation{{Event: EventFunctionCallStopped, Calls: []FunctionCall{
				{Name: "write_report", Args: map[string]any{"cancelled": true}}, //nolint:misspell // the reported key
			}}}, rtvi.ReportFull,
		},
	}
	for name, tc := range cases {
		if got := scriptLevel(tc.expect...); got != tc.want {
			t.Errorf("%s: level = %q, want %q", name, got, tc.want)
		}
	}
}

// The harness enables raw VAD events only when a scenario references them.
func TestScriptNeedsVADEvents(t *testing.T) {
	needs := func(turn ScriptTurn) bool {
		return (&ScriptScenario{Name: "t", Turns: []ScriptTurn{turn}}).NeedsVADEvents()
	}
	if needs(ScriptTurn{User: "x", Expect: []Expectation{{Event: EventResponse}}}) {
		t.Error("false without vad events")
	}
	if !needs(ScriptTurn{User: "x", Expect: []Expectation{{Event: EventVADUserStartedSpeaking}}}) {
		t.Error("true when expected")
	}
	if !needs(ScriptTurn{
		User: "x", Expect: []Expectation{{Event: EventResponse}},
		SendAfter: &SendAfter{Event: EventVADUserStoppedSpeaking, DelayMS: 2000},
	}) {
		t.Error("true when used as a send_after anchor")
	}
}

// The harness asks for the model's markers only when a scenario asserts on one.
func TestScriptNeedsMarkerEvents(t *testing.T) {
	needs := func(expect ...Expectation) bool {
		return (&ScriptScenario{Name: "t", Turns: []ScriptTurn{{User: "x", Expect: expect}}}).NeedsMarkerEvents()
	}
	if needs(Expectation{Event: EventResponse}) {
		t.Error("false without a marker expectation")
	}
	if !needs(Expectation{Event: EventLLMMarker, Marker: MarkerComplete}) {
		t.Error("true when expected")
	}
}

// The harness asks for the bot's images only when a scenario expects one.
func TestScriptNeedsBotImages(t *testing.T) {
	needs := func(expect ...Expectation) bool {
		return (&ScriptScenario{Name: "t", Turns: []ScriptTurn{{Expect: expect}}}).NeedsBotImages()
	}
	if needs(Expectation{Event: EventResponse}) {
		t.Error("false without an image expectation")
	}
	if !needs(Expectation{Event: EventImage}) {
		t.Error("true when expected")
	}
}

// WantsResponse is whether a scenario asserts on the transcription of the bot's
// audio, which only an audio modality scenario keeps as response.
func TestScriptWantsResponse(t *testing.T) {
	audio := script(t, "name: a\njudge: {modality: audio, transcription: {service: whisper}}\n"+
		"turns: [{user: hi, expect: [{event: response}]}]\n")
	text := script(t, "name: a\nturns: [{user: hi, expect: [{event: response}]}]\n")
	if !audio.WantsResponse() || text.WantsResponse() {
		t.Fatalf("wants response: audio %v, text %v", audio.WantsResponse(), text.WantsResponse())
	}
}

// A call's signature is how a matched call is reported.
func TestFunctionCallSignature(t *testing.T) {
	for _, tc := range []struct {
		call FunctionCall
		want string
	}{
		{FunctionCall{}, "any function"},
		{FunctionCall{Name: "get_weather"}, "get_weather"},
		{FunctionCall{Name: "book", Args: map[string]any{"party": 2, "name": "Jo"}}, "book(name='Jo', party=2)"},
	} {
		if got := tc.call.Signature(); got != tc.want {
			t.Errorf("Signature() = %q, want %q", got, tc.want)
		}
	}
}

// A content check on a reply accumulates the bot's segments; a user
// transcription accumulates an STT's pieces only for text_contains.
func TestExpectationAggregates(t *testing.T) {
	for _, tc := range []struct {
		exp  Expectation
		want bool
	}{
		{Expectation{Event: EventResponse, Eval: "x"}, true},
		{Expectation{Event: EventLLMResponse, TextContains: "x"}, true},
		{Expectation{Event: EventLLMResponse}, false},
		{Expectation{Event: EventUserTranscription, TextContains: "x"}, true},
		{Expectation{Event: EventUserTranscription, TextContains: "x", Eval: "y"}, false},
		{Expectation{Event: EventFunctionCall, Eval: "x"}, false},
	} {
		if got := tc.exp.Aggregates(); got != tc.want {
			t.Errorf("%+v: Aggregates() = %v, want %v", tc.exp, got, tc.want)
		}
	}
}
