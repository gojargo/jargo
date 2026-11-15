package eval

import (
	"fmt"

	"github.com/gojargo/jargo/service/stt"
)

// ScriptSession runs one ScriptScenario against a bot.
//
// Build one with ScriptSessionFromScenario, which constructs the judge, the user
// TTS and the bot STT the scenario needs, then call Run:
//
//	url := "ws://localhost:7860/ws"
//	session, err := eval.ScriptSessionFromScenario(scenario, url, eval.SessionParams{}, nil, nil, nil)
//	if err != nil { … }
//	result, err := session.Run(ctx)
//
// Per-turn outcomes are in result.Turns, for a scenario scored a turn at a
// time.
type ScriptSession struct {
	*Session[ScriptResult]

	scenario *ScriptScenario
	script   *ScriptDriver
}

// NewScriptSession builds the session of scenario against the bot at botURL.
//
// The services come pre-built, or nil where the scenario has no use for them;
// ScriptSessionFromScenario constructs the ones a scenario needs. judge answers
// eval: assertions, or is nil when the scenario has none; userTTS speaks the
// user's turns, or is nil for text turns; botSTT transcribes the bot's audio
// for the response event, or is nil when the scenario has none.
func NewScriptSession(
	scenario *ScriptScenario, botURL string, params SessionParams, judge Judge,
	userTTS *CachingTTSService, botSTT *stt.SegmentService,
) (*ScriptSession, error) {
	if err := params.validate(); err != nil {
		return nil, err
	}
	s := &ScriptSession{
		Session:  newSession[ScriptResult](KindScript, scenario.Name, botURL, params),
		scenario: scenario,
	}
	// The bot's output as events: fed by the client's pipeline, read by the
	// driver.
	s.stream = NewEventStream(scenario.BotAudio, s.trace)
	// The connection to the bot: the eval pipeline and the user's sends,
	// asking the bot for what the scenario's assertions need.
	s.client = NewClient(botURL, ClientParams{
		BotAudio:          scenario.BotAudio,
		UserAudio:         scenario.UserAudio,
		UserSpeech:        scenario.UserSpeech,
		CaptureBotAudio:   scenario.WantsResponse(),
		CaptureBotImages:  scenario.NeedsBotImages(),
		ReportLevel:       scenario.RequiredReportLevel(),
		VADEvents:         scenario.NeedsVADEvents(),
		MarkerEvents:      scenario.NeedsMarkerEvents(),
		Context:           scenario.Context,
		TriggerDisconnect: scenario.TriggerDisconnect, //nolint:staticcheck // the deprecated key still asks
	}, s.params, s.stream, s.trace, userTTS, botSTT, nil)
	// What the user says next and how the outcome is scored: a scenario is
	// played by the script driver.
	s.script = newScriptDriver(scenario, s.params.DefaultTimeoutMS, s.client, s.stream, judge, s.trace,
		func(p Progress) { s.progress(p) })
	s.driver = s.script
	s.assemble = s.script.result
	s.describe = func() string { return DescribeConfig(&scenario.Configured, false) }
	s.skipReason = s.scriptSkipReason
	return s, nil
}

// ScriptSessionFromScenario builds a ready-to-run session from scenario,
// constructing the services it needs. Pass judge, userTTS or botSTT to use your
// own: otherwise the judge is built from the scenario's judge: block when it has
// eval: assertions, the user TTS from its user: block's speech in audio mode,
// and the bot STT from its judge: block's transcription when it asserts
// response in audio mode.
//
// It fails when a service the scenario's blocks name cannot be built, a
// transcription factory returning anything but a *stt.SegmentService included.
func ScriptSessionFromScenario(
	scenario *ScriptScenario, botURL string, params SessionParams, judge Judge,
	userTTS *CachingTTSService, botSTT *stt.SegmentService,
) (*ScriptSession, error) {
	if judge == nil && scenario.anyExpectation(func(e *Expectation) bool { return e.Eval != "" }) {
		built, err := NewEvalJudgeFromConfig(&scenario.Judge)
		if err != nil {
			return nil, fmt.Errorf("eval: building the judge of %q: %w", scenario.Name, err)
		}
		judge = built
	}
	if userTTS == nil && scenario.UserSpeech != nil {
		built, err := TTSServiceFromConfig(scenario.UserSpeech, params.CacheDir, params.useCache())
		if err != nil {
			return nil, fmt.Errorf("eval: building the user speech of %q: %w", scenario.Name, err)
		}
		userTTS = built
	}
	if botSTT == nil && scenario.WantsResponse() && scenario.BotAudio {
		built, err := STTServiceFromConfig(scenario.Transcriber)
		if err != nil {
			return nil, fmt.Errorf("eval: building the transcription of %q: %w", scenario.Name, err)
		}
		botSTT = built
	}
	return NewScriptSession(scenario, botURL, params, judge, userTTS, botSTT)
}

// Driver is the session's script driver, whose turn records fill in as it
// runs.
func (s *ScriptSession) Driver() *ScriptDriver { return s.script }

// scriptSkipReason is why the scenario cannot be driven at all, or "".
func (s *ScriptSession) scriptSkipReason() string {
	// The response transcription needs the bot's actual audio; without audio
	// mode there is nothing to transcribe, so skip rather than fail. Loading a
	// scenario resolves response to llm_response in text modality, so this
	// guards scenarios built directly.
	if s.scenario.WantsResponse() && !s.scenario.BotAudio {
		return "asserts 'response' transcription but judge modality is text (no audio)"
	}
	return ""
}
