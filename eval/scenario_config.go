package eval

import (
	"fmt"
	"strings"

	"go.yaml.in/yaml/v3"
)

// The modalities a user: or judge: block may name.
const (
	ModalityText  = "text"
	ModalityAudio = "audio"
)

// Keys and values a scenario's blocks share.
const (
	keyName       = "name"
	keyService    = "service"
	keyModel      = "model"
	keyFactory    = "factory"
	keyEval       = "eval"
	keyType       = "type"
	serviceOllama = "ollama"
)

// DefaultJudgeEvalConfig is the judge a scenario uses when its judge.eval: block
// is omitted: the default local Ollama model.
func DefaultJudgeEvalConfig() JudgeEvalConfig {
	return JudgeEvalConfig{
		Service: serviceOllama,
		Model:   DefaultOllamaJudgeModel,
		Extra:   DefaultOllamaJudgeExtra(),
		Raw: map[string]any{
			keyService: serviceOllama,
			keyModel:   DefaultOllamaJudgeModel,
			"extra":    DefaultOllamaJudgeExtra(),
		},
	}
}

// Configured is the modality and service config a scripted scenario and a
// simulation share, read from their user: and judge: blocks.
type Configured struct {
	// UserAudio is whether the user's turns reach the bot as speech
	// (user.modality: audio) rather than as text.
	UserAudio bool
	// UserSpeech is the user.speech: block, or nil in text modality, or in
	// audio modality when every spoken turn plays a recording.
	UserSpeech *SpeechConfig
	// BotAudio is whether the bot speaks (judge.modality: audio). In text
	// modality, the default, the bot skips TTS and the judge reads its LLM
	// text.
	BotAudio bool
	// Transcriber is the judge.transcription: block, or nil in text modality.
	Transcriber *TranscriptionConfig
	// Judge is the judge.eval: block, DefaultJudgeEvalConfig when it is
	// omitted.
	Judge JudgeEvalConfig
}

// checkModality reads a block's modality:, text when it is not given.
func checkModality(block map[string]any, where, path string) (string, error) {
	modality, set := block["modality"]
	if !set {
		return ModalityText, nil
	}
	if modality != ModalityAudio && modality != ModalityText {
		return "", invalidf("%s: '%s.modality:' must be 'audio' or 'text', got %s", path, where, pyRepr(modality))
	}
	return modality.(string), nil //nolint:forcetypeassert // checked above
}

// decodeBlock reads a block, as the file wrote it, into one of the typed
// config blocks.
func decodeBlock(block any, out any, where, path string) error {
	encoded, err := yaml.Marshal(block)
	if err == nil {
		err = yaml.Unmarshal(encoded, out)
	}
	if err != nil {
		return invalidf("%s: '%s:' is invalid: %v", path, where, err)
	}
	return nil
}

// parseUserBlock parses the user: block into (user audio, speech config).
func parseUserBlock(user any, path string) (bool, *SpeechConfig, error) {
	if user == nil {
		return false, nil, nil // default: text modality
	}
	block, ok := user.(map[string]any)
	if !ok {
		return false, nil, invalidf("%s: 'user:' must be a mapping", path)
	}
	modality, err := checkModality(block, "user", path)
	if err != nil || modality == ModalityText {
		return false, nil, err
	}
	speech, isMapping := block["speech"].(map[string]any)
	if block["speech"] != nil && !isMapping {
		return false, nil, invalidf("%s: 'user.speech:' must be a mapping", path)
	}
	var cfg UserConfig
	if err := decodeBlock(block, &cfg, "user", path); err != nil {
		return false, nil, err
	}
	if cfg.Speech != nil {
		// The block as the file wrote it, which is what a factory is given.
		cfg.Speech.Raw = speech
	}
	return true, cfg.Speech, nil
}

// parseJudgeBlock parses the judge: block into (bot audio, transcriber, eval
// config).
func parseJudgeBlock(judge any, path string) (bool, *TranscriptionConfig, JudgeEvalConfig, error) {
	if judge == nil {
		judge = map[string]any{}
	}
	block, ok := judge.(map[string]any)
	if !ok {
		return false, nil, JudgeEvalConfig{}, invalidf("%s: 'judge:' must be a mapping", path)
	}
	modality, err := checkModality(block, "judge", path)
	if err != nil {
		return false, nil, JudgeEvalConfig{}, err
	}
	evalBlock, isMapping := block[keyEval].(map[string]any)
	if truthy(block[keyEval]) && !isMapping {
		return false, nil, JudgeEvalConfig{}, invalidf(
			"%s: 'judge.eval:' must be a mapping (the judge LLM service)", path)
	}
	transcription, hasTranscription := block["transcription"].(map[string]any)
	if modality == ModalityAudio && !hasTranscription {
		return false, nil, JudgeEvalConfig{}, invalidf(
			"%s: 'judge.modality: audio' requires a 'judge.transcription:' block "+
				"(STT service to transcribe the bot's audio)", path)
	}
	var cfg JudgeConfig
	if err := decodeBlock(block, &cfg, "judge", path); err != nil {
		return false, nil, JudgeEvalConfig{}, err
	}

	evalCfg := DefaultJudgeEvalConfig()
	if len(evalBlock) > 0 && cfg.Eval != nil {
		evalCfg = *cfg.Eval
		// The blocks as the file wrote them, which is what a factory is given.
		evalCfg.Raw = evalBlock
		if explainer, ok := evalBlock["explainer"].(map[string]any); ok && evalCfg.Explainer.LLM != nil {
			evalCfg.Explainer.LLM.Raw = explainer
		}
	}
	if modality == ModalityText {
		return false, nil, evalCfg, nil
	}
	cfg.Transcription.Raw = transcription
	return true, cfg.Transcription, evalCfg, nil
}

// parseLLMConfig parses a block naming an LLM service; where is the block's
// name in the file, for error messages.
func parseLLMConfig(block map[string]any, where, path string) (LLMServiceConfig, error) {
	var cfg LLMServiceConfig
	if err := decodeBlock(block, &cfg, where, path); err != nil {
		return LLMServiceConfig{}, err
	}
	cfg.Raw = block
	return cfg, nil
}

// stringField is a mapping key read into a string field.
type stringField struct {
	key string
	dst *string
}

// ANSI codes for the colored config summary (applied only when color is on).
// The section labels are bold, the separators dim, and each segment's keyword
// gets one hue per category so modality, service and judge LLM are easy to tell
// apart; the values are left uncolored.
const (
	cfgLabel    = "1"  // bold: the "user" / "judge" section labels
	cfgSep      = "2"  // dim: the "->" arrow and "|" separators
	cfgModality = "33" // yellow: modality keyword
	cfgService  = "32" // green: speech (TTS) / transcription (STT) keywords
	cfgEval     = "35" // magenta: LLM keywords, the judge's eval and a simulation's persona
	cfgLimit    = "36" // cyan: a simulation's run caps
)

// configSegment is one segment of a config line: keyword, value, and the ANSI
// code the keyword is painted with.
type configSegment struct {
	key, value, code string
}

// configLine is a labeled config line: its label, then its segments, or plain
// text when it has none.
type configLine struct {
	label string
	segs  []configSegment
	text  string
}

// svcModel is what a service config block builds, as the run summary names
// it: factory:<name> for a factory, else service/model, the model being the
// block's or the one the service builds with by default, or the service alone
// when neither is known.
func svcModel(factory, service, model, defaultService string, defaultModels map[string]string) string {
	if factory != "" {
		return "factory:" + factory
	}
	if service == "" {
		service = defaultService
	}
	if model == "" {
		model = defaultModels[strings.ToLower(service)]
	}
	if model != "" {
		return service + "/" + model
	}
	return service
}

// llmIdentity is what a judge.eval: or simulator: block builds, as the run
// summary names it.
func llmIdentity(cfg LLMServiceConfig) string {
	return svcModel(cfg.Factory, cfg.Service, cfg.Model, serviceOllama,
		map[string]string{serviceOllama: DefaultOllamaJudgeModel, "openai": DefaultOpenAIModel})
}

// userSegments are the user line's segments: modality, and the TTS the user's
// turns are spoken with.
func userSegments(c *Configured) []configSegment {
	segs := []configSegment{{"modality", modalityName(c.UserAudio), cfgModality}}
	if c.UserSpeech != nil && len(c.UserSpeech.Raw) > 0 {
		// The TTS voice is the speech config's model-equivalent.
		s := c.UserSpeech
		segs = append(segs, configSegment{"speech", svcModel(s.Factory, s.Service, s.Voice, "?", nil), cfgService})
	}
	return segs
}

// judgeSegments are the judge line's segments: modality, the bot-speech STT,
// and the judge LLM.
func judgeSegments(c *Configured) []configSegment {
	segs := []configSegment{{"modality", modalityName(c.BotAudio), cfgModality}}
	if c.BotAudio {
		t := c.Transcriber
		if t == nil {
			t = &TranscriptionConfig{}
		}
		segs = append(segs, configSegment{
			"transcription", svcModel(t.Factory, t.Service, t.Model, "whisper", nil), cfgService,
		})
	}
	return append(segs, configSegment{keyEval, llmIdentity(c.Judge.LLMServiceConfig), cfgEval})
}

// modalityName is a modality flag as a block names it.
func modalityName(audio bool) string {
	if audio {
		return ModalityAudio
	}
	return ModalityText
}

// renderConfigLines renders labeled config lines, `label -> key: value | key:
// value`. With color, the label is bold, the separators dim, and each keyword
// takes its segment's ANSI code.
func renderConfigLines(lines []configLine, color bool) string {
	paint := func(text, code string) string {
		if !color {
			return text
		}
		return "\033[" + code + "m" + text + "\033[0m"
	}
	arrow := paint(" -> ", cfgSep)
	sep := paint(" | ", cfgSep)
	out := make([]string, 0, len(lines))
	for _, line := range lines {
		body := line.text
		if line.segs != nil {
			parts := make([]string, 0, len(line.segs))
			for _, s := range line.segs {
				parts = append(parts, paint(s.key+":", s.code)+" "+s.value)
			}
			body = strings.Join(parts, sep)
		}
		out = append(out, paint(fmt.Sprintf("%-5s", line.label), cfgLabel)+arrow+body)
	}
	return strings.Join(out, "\n")
}

// DescribeConfig is a two-line summary of a scenario's user and judge config,
// for pre-run logs. Each line is a set of `key: value` segments separated by
// `|`:
//
//	user  -> modality: audio | speech: kokoro/af_heart
//	judge -> modality: audio | transcription: moonshine/small-streaming | eval: ollama/gemma4:12b
//
// With color, each segment's keyword is ANSI-colored by category (modality,
// service, judge LLM) so they are easy to tell apart.
func DescribeConfig(c *Configured, color bool) string {
	return renderConfigLines([]configLine{
		{label: "user", segs: userSegments(c)},
		{label: "judge", segs: judgeSegments(c)},
	}, color)
}
