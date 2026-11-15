package eval

import (
	"errors"
	"fmt"

	"github.com/gojargo/jargo/processor/rtvi"
	"github.com/gojargo/jargo/service/stt"
)

// errNoPersonaLLM is a simulation session built without the LLM that plays the
// caller.
var errNoPersonaLLM = errors.New("eval: a simulation needs a persona LLM")

// SimulationSession runs one SimulationScenario against a bot.
//
// The persona LLM answers the bot on its own inside the client's pipeline, and
// the judge decides the goal and the criteria at the end. Build one with
// SimulationSessionFromScenario, which constructs the persona LLM, the judge,
// and in audio mode the user TTS and the STT, then call Run:
//
//	file, _ := eval.LoadFile("scenarios/simulated/curious_caller.yaml")
//	for _, loaded := range file.Scenarios {
//		scenario, ok := loaded.(*eval.SimulationScenario)
//		if !ok {
//			continue
//		}
//		session, err := eval.SimulationSessionFromScenario(scenario, "ws://localhost:7860/ws",
//			eval.SessionParams{}, nil, nil, nil, nil)
//		if err != nil { … }
//		result, err := session.Run(ctx)
//		…
//	}
type SimulationSession struct {
	*Session[*SimulationResult]

	scenario   *SimulationScenario
	simulation *SimulationDriver
}

// NewSimulationSession builds the session of scenario against the bot at
// botURL.
//
// personaLLM is the persona LLM service, run inside the eval pipeline; it is
// required. judge decides the goal and the quality criteria, or is nil (the run
// then reports no verdict). userTTS speaks the persona's replies, or is nil for
// text mode; botSTT transcribes the bot's audio, or is nil in text mode.
func NewSimulationSession(
	scenario *SimulationScenario, botURL string, params SessionParams, personaLLM PersonaLLM,
	judge RunJudge, userTTS *CachingTTSService, botSTT *stt.SegmentService,
) (*SimulationSession, error) {
	if err := params.validate(); err != nil {
		return nil, err
	}
	if personaLLM == nil {
		return nil, fmt.Errorf("%w: %q", errNoPersonaLLM, scenario.Name)
	}
	s := &SimulationSession{
		Session:  newSession[*SimulationResult](KindSimulation, scenario.Name, botURL, params),
		scenario: scenario,
	}
	persona := NewPersona(scenario.Persona, scenario.Goal, personaLLM)
	s.stream = NewEventStream(scenario.BotAudio, s.trace)
	// The persona hears the bot, and the judge sees its tool calls.
	s.client = NewClient(botURL, ClientParams{
		BotAudio:        scenario.BotAudio,
		UserAudio:       scenario.UserAudio,
		UserSpeech:      scenario.UserSpeech,
		CaptureBotAudio: scenario.BotAudio,
		// The bot's function calls, with their arguments, are the judge's
		// evidence of what the bot actually did.
		ReportLevel:       rtvi.ReportFull,
		TriggerDisconnect: scenario.TriggerDisconnect, //nolint:staticcheck // the deprecated key still asks
	}, s.params, s.stream, s.trace, userTTS, botSTT, persona)
	s.simulation = newSimulationDriver(scenario, persona, s.client, s.stream, judge, s.trace,
		func(p SimulationProgress) { s.progress(p) })
	s.driver = s.simulation
	s.assemble = s.simulation.result
	s.describe = func() string { return DescribeSimulation(scenario, false) }
	s.skipReason = func() string { return "" }
	return s, nil
}

// SimulationSessionFromScenario builds a ready-to-run session from scenario,
// constructing the services it needs. Pass personaLLM, judge, userTTS or botSTT
// to use your own: otherwise the persona LLM is built from the simulation's
// simulator: block, the judge from its judge: block, the user TTS from its
// user: block's speech in audio mode, and the bot STT from its judge: block's
// transcription in audio mode.
//
// It fails when a service the simulation's blocks name cannot be built.
func SimulationSessionFromScenario(
	scenario *SimulationScenario, botURL string, params SessionParams, personaLLM PersonaLLM,
	judge RunJudge, userTTS *CachingTTSService, botSTT *stt.SegmentService,
) (*SimulationSession, error) {
	if personaLLM == nil {
		built, err := PersonaLLMFromConfig(&scenario.Simulator)
		if err != nil {
			return nil, fmt.Errorf("eval: building the persona of %q: %w", scenario.Name, err)
		}
		personaLLM = built
	}
	if judge == nil {
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
	if botSTT == nil && scenario.BotAudio {
		built, err := STTServiceFromConfig(scenario.Transcriber)
		if err != nil {
			return nil, fmt.Errorf("eval: building the transcription of %q: %w", scenario.Name, err)
		}
		botSTT = built
	}
	return NewSimulationSession(scenario, botURL, params, personaLLM, judge, userTTS, botSTT)
}

// PersonaLLMFromConfig builds the persona LLM a simulator: block names: an LLM
// service the eval pipeline runs.
//
// A factory builds it; otherwise the block names an LLM, as
// LLMServiceFromConfig reads it, Ollama by default. It fails when the block
// names an unknown service, or its factory is not registered or builds
// something a pipeline cannot run as an LLM service.
func PersonaLLMFromConfig(cfg *LLMServiceConfig) (PersonaLLM, error) {
	var built any
	if cfg != nil && cfg.Factory != "" {
		f, err := factory("simulator", cfg.Factory)
		if err != nil {
			return nil, err
		}
		if built, err = f(cfg.raw()); err != nil {
			return nil, err
		}
	} else {
		service, err := LLMServiceFromConfig(cfg, "simulator")
		if err != nil {
			return nil, err
		}
		built = service
	}
	service, ok := built.(PersonaLLM)
	if !ok {
		where := "simulator"
		if cfg != nil && cfg.Factory != "" {
			where = fmt.Sprintf("simulator.factory %q", cfg.Factory)
		}
		return nil, fmt.Errorf("%w: %s returned %T, not an LLM service", ErrFactory, where, built)
	}
	return service, nil
}

// Driver is the session's simulation driver.
func (s *SimulationSession) Driver() *SimulationDriver { return s.simulation }
