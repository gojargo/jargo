package realtime

import (
	"reflect"
	"slices"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/service/settings"
)

// The names of the settings the service acts on.
const (
	settingModel             = "model"
	settingSystemInstruction = "system_instruction"
	settingTools             = "tools"
	settingVoice             = "voice"
	settingInterruptOnSpeech = "interrupt_on_speech"
	settingToolFillers       = "tool_fillers"
	settingTruncation        = "truncation"
	settingResponseMetadata  = "response_metadata"
	settingSpeed             = "speed"
)

// interruptionRules are the settings that make up the interruption rules.
//
//nolint:gochecknoglobals // a fixed list
var interruptionRules = []string{keyInterruptMin, keyInterruptPhr, keyAckPhrases}

// sessionFields are the settings sent to Prism under the same value.
//
//nolint:gochecknoglobals // a fixed list
var sessionFields = []string{settingVoice, settingInterruptOnSpeech, settingTruncation, settingSpeed}

// supportedSettings are the settings Prism takes a value for.
//
//nolint:gochecknoglobals // a fixed list
var supportedSettings = []string{
	settingModel, settingSystemInstruction, settingTools, settingVoice, settingInterruptOnSpeech,
	settingToolFillers, settingTruncation, settingResponseMetadata, settingSpeed,
	keyInterruptMin, keyInterruptPhr, keyAckPhrases,
	"filter_incomplete_user_turns", "use_provider_thinking_default", "extensions",
}

// Settings is the Prism configuration, and the changes an
// LLMUpdateSettingsFrame makes to it while the conversation runs. A field left
// not given in an update keeps its value.
//
// Prism runs its own model and detects the turns itself, so the model stays
// "prism", incomplete user turns are not filtered, and the sampling settings
// take no value. The system instruction, the voice, the interruption on
// speech, the truncation, the speed and the interruption rules cannot be
// cleared by an update, which Prism would read as keeping them.
type Settings struct {
	settings.LLM

	// Tools are the tools advertised. An empty toolset removes all tools.
	Tools settings.Opt[frames.ToolsSchema] `settings:"tools"`
	// Voice is the Prism voice.
	Voice settings.Opt[string] `settings:"voice"`
	// InterruptOnSpeech is the server's interruption setting. The local turn
	// strategies control the pipeline's interruptions independently.
	InterruptOnSpeech settings.Opt[bool] `settings:"interrupt_on_speech"`
	// ToolFillers are what Prism says while a tool runs, by tool name.
	ToolFillers settings.Opt[map[string]Filler] `settings:"tool_fillers"`
	// Truncation is the policy for a history that outgrows the model's
	// capacity.
	Truncation settings.Opt[Truncation] `settings:"truncation"`
	// ResponseMetadata are the labels replies carry. They are applied apart
	// from the other settings.
	ResponseMetadata settings.Opt[map[string]string] `settings:"response_metadata"`
	// Speed is the speech speed, from 0.85 to 1.3. Unset uses the server's
	// default.
	Speed settings.Opt[float64] `settings:"speed"`
	// InterruptMinWords is how many words, from 1 to 10, the user says over a
	// reply before Prism interrupts it. Unset uses the server's default.
	InterruptMinWords settings.Opt[int] `settings:"interrupt_min_words"`
	// InterruptPhrases are phrases that interrupt a reply whatever
	// InterruptMinWords says. An empty list clears them.
	InterruptPhrases settings.Opt[[]string] `settings:"interrupt_phrases"`
	// AcknowledgmentPhrases are phrases, such as "uh huh", that do not
	// interrupt a reply. An empty list clears them.
	AcknowledgmentPhrases settings.Opt[[]string] `settings:"acknowledgment_phrases"`
}

// defaultSettings is the state a service starts from before the settings it
// was built with are applied.
func defaultSettings() *Settings {
	s := &Settings{}
	s.Model = settings.Set(modelPrism)
	s.SystemInstruction = settings.Cleared[string]()
	s.Temperature = settings.Cleared[float64]()
	s.MaxTokens = settings.Cleared[int]()
	s.TopP = settings.Cleared[float64]()
	s.TopK = settings.Cleared[int]()
	s.FrequencyPenalty = settings.Cleared[float64]()
	s.PresencePenalty = settings.Cleared[float64]()
	s.Seed = settings.Cleared[int]()
	s.FilterIncompleteUserTurns = settings.Set(false) //nolint:staticcheck // the store holds the deprecated setting
	s.UseProviderThinkingDefault = settings.Set(false)
	s.Tools = settings.Cleared[frames.ToolsSchema]()
	s.Voice = settings.Cleared[string]()
	s.InterruptOnSpeech = settings.Set(true)
	s.ToolFillers = settings.Set(map[string]Filler{})
	s.Truncation = settings.Cleared[Truncation]()
	s.ResponseMetadata = settings.Cleared[map[string]string]()
	s.Speed = settings.Cleared[float64]()
	s.InterruptMinWords = settings.Cleared[int]()
	s.InterruptPhrases = settings.Cleared[[]string]()
	s.AcknowledgmentPhrases = settings.Cleared[[]string]()
	return s
}

// newSettings is the settings a configuration starts the service with.
func newSettings(cfg Config) (*Settings, error) {
	s := defaultSettings()
	initial := cfg.Settings
	if _, err := settings.Apply(s, &initial); err != nil {
		return nil, err
	}
	if td := cfg.TurnDetection; td != nil {
		if err := td.validate(); err != nil {
			return nil, settingsError("%s", err.Error())
		}
		if err := joinOpeningRules(s, td.TurnDetectionUpdate); err != nil {
			return nil, err
		}
	}
	if err := validateSettings(s); err != nil {
		return nil, err
	}
	return s, nil
}

// joinOpeningRules takes the interruption rules the opening turn detection
// gives into the settings, which can change them later. A rule is given in one
// of the two places.
func joinOpeningRules(s *Settings, rules TurnDetectionUpdate) error {
	both := func(name string) error {
		return settingsError("Set %s in turn_detection or settings, not both", name)
	}
	if rules.InterruptMinWords != nil {
		if _, ok := s.InterruptMinWords.Value(); ok {
			return both(keyInterruptMin)
		}
		s.InterruptMinWords = settings.Set(*rules.InterruptMinWords)
	}
	if rules.InterruptPhrases != nil {
		if _, ok := s.InterruptPhrases.Value(); ok {
			return both(keyInterruptPhr)
		}
		s.InterruptPhrases = settings.Set(slices.Clone(rules.InterruptPhrases))
	}
	if rules.AcknowledgmentPhrases != nil {
		if _, ok := s.AcknowledgmentPhrases.Value(); ok {
			return both(keyAckPhrases)
		}
		s.AcknowledgmentPhrases = settings.Set(slices.Clone(rules.AcknowledgmentPhrases))
	}
	return nil
}

// validateSettings reports whether a settings store holds what Prism takes.
func validateSettings(s *Settings) error {
	given, err := settings.Given(s)
	if err != nil {
		return err
	}
	for name, value := range given {
		if value != nil && !slices.Contains(supportedSettings, name) {
			return settingsError("Prism does not support setting %s", name)
		}
	}
	//nolint:staticcheck // the store holds the deprecated setting
	if s.Model.Or("") != modelPrism || s.FilterIncompleteUserTurns.Or(false) {
		return settingsError("Prism requires its own model and server turn detection")
	}
	if _, ok := s.InterruptOnSpeech.Value(); !ok {
		return settingsError("interrupt_on_speech must be a boolean")
	}
	if speed, ok := s.Speed.Value(); ok && !(speed >= 0.85 && speed <= 1.3) {
		return settingsError("Prism speed must be a number from 0.85 to 1.3")
	}
	if truncation, ok := s.Truncation.Value(); ok {
		if err := truncation.validate(); err != nil {
			return settingsError("%s", err.Error())
		}
	}
	_, err = interruptionRulesOf(s, interruptionRules)
	return err
}

// interruptionRulesOf is the interruption rules the settings hold under the
// names given.
func interruptionRulesOf(s *Settings, names []string) (TurnDetectionUpdate, error) {
	rules := TurnDetectionUpdate{}
	if slices.Contains(names, keyInterruptMin) {
		rules.InterruptMinWords = s.InterruptMinWords.Ptr()
	}
	if phrases, ok := s.InterruptPhrases.Value(); ok && slices.Contains(names, keyInterruptPhr) {
		rules.InterruptPhrases = nonNil(phrases)
	}
	if phrases, ok := s.AcknowledgmentPhrases.Value(); ok && slices.Contains(names, keyAckPhrases) {
		rules.AcknowledgmentPhrases = nonNil(phrases)
	}
	if err := rules.validate(); err != nil {
		return TurnDetectionUpdate{}, settingsError("%s", err.Error())
	}
	return rules, nil
}

// nonNil is phrases as a list that is given: an empty one clears the server's.
func nonNil(phrases []string) []string {
	if phrases == nil {
		return []string{}
	}
	return phrases
}

// hasValue reports whether the settings hold a value under name.
func hasValue(s *Settings, name string) bool {
	value, _ := settings.Get(s, name)
	return value != nil && !isNilValue(value)
}

func isNilValue(value any) bool {
	rv := reflect.ValueOf(value)
	switch rv.Kind() {
	case reflect.Map, reflect.Slice, reflect.Pointer, reflect.Interface:
		return rv.IsNil()
	default:
		return false
	}
}

// sameHandler reports whether two tools carry the same handler.
func sameHandler(a, b any) bool {
	if a == nil || b == nil {
		return a == nil && b == nil
	}
	av, bv := reflect.ValueOf(a), reflect.ValueOf(b)
	if av.Kind() != reflect.Func || bv.Kind() != reflect.Func {
		return av.Type() == bv.Type()
	}
	return av.Pointer() == bv.Pointer()
}

// sameTools reports whether two toolsets are the same one, handlers included.
func sameTools(a, b frames.ToolsSchema) bool {
	if len(a.Standard) != len(b.Standard) || !reflect.DeepEqual(a.Custom, b.Custom) {
		return false
	}
	for i := range a.Standard {
		x, y := a.Standard[i], b.Standard[i]
		if x.Name != y.Name || x.Description != y.Description || string(x.Parameters) != string(y.Parameters) ||
			!sameHandler(x.Handler, y.Handler) || !reflect.DeepEqual(x.CancelOnInterruption, y.CancelOnInterruption) ||
			!reflect.DeepEqual(x.TimeoutSecs, y.TimeoutSecs) || !reflect.DeepEqual(x.CancellableByLLM, y.CancellableByLLM) {
			return false
		}
	}
	return true
}
