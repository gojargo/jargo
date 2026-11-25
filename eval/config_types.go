package eval

import (
	"go.yaml.in/yaml/v3"
)

// UserConfig is a scenario's `user:` block: how the user's turns reach the
// bot, as text or as speech a TTS synthesizes.
type UserConfig struct {
	// Modality is "text" (the default) or "audio".
	Modality string `yaml:"modality,omitempty"`
	// Speech is the TTS the user's turns are spoken with, in audio modality.
	Speech *SpeechConfig `yaml:"speech,omitempty"`
}

// JudgeConfig is a scenario's `judge:` block: how the bot's reply is judged,
// from its LLM text or from a transcription of its speech, and by which
// judge.
type JudgeConfig struct {
	// Modality is "text" (the default) or "audio".
	Modality string `yaml:"modality,omitempty"`
	// Transcription is the STT that transcribes the bot's audio, required in
	// audio modality.
	Transcription *TranscriptionConfig `yaml:"transcription,omitempty"`
	// Eval is the judge that decides the verdicts; nil uses the default.
	Eval *JudgeEvalConfig `yaml:"eval,omitempty"`
}

// LLMServiceConfig is a block naming an LLM: the judge's (inside
// `judge.eval:`), its explainer's, or a simulation's.
//
// A Factory builds the service; otherwise Service names a built-in provider,
// "ollama" by default.
type LLMServiceConfig struct {
	// Factory is the name a factory was registered under (see RegisterFactory);
	// it is given the whole block and returns the service.
	Factory string `yaml:"factory,omitempty"`
	// Service is the built-in provider, "ollama" by default.
	Service string `yaml:"service,omitempty"`
	// Model is the model; empty uses the provider's default.
	Model string `yaml:"model,omitempty"`
	// Endpoint is the provider's base URL; empty uses its default.
	Endpoint string `yaml:"endpoint,omitempty"`
	// Extra is forwarded to the model as top-level request parameters. Nil,
	// with no Model either, gives the default model the extras it needs; an
	// empty map sends none.
	Extra map[string]any `yaml:"extra,omitempty"`
	// Raw is the block as written, every key included, which is what a factory
	// is given. Filled when the block is read from YAML; for a block built in
	// code, the fields above are used.
	Raw map[string]any `yaml:"-"`
}

// UnmarshalYAML reads the block and keeps it as written in Raw.
func (c *LLMServiceConfig) UnmarshalYAML(node *yaml.Node) error {
	type plain LLMServiceConfig
	var p plain
	if err := node.Decode(&p); err != nil {
		return err
	}
	if err := node.Decode(&p.Raw); err != nil {
		return err
	}
	*c = LLMServiceConfig(p)
	return nil
}

// empty reports whether the block sets nothing, as `explainer: {}` does.
func (c *LLMServiceConfig) empty() bool {
	return c.Factory == "" && c.Service == "" && c.Model == "" && c.Endpoint == "" && c.Extra == nil &&
		len(c.Raw) == 0
}

// raw is the block as a factory is given it.
func (c *LLMServiceConfig) raw() map[string]any {
	if c.Raw != nil {
		return c.Raw
	}
	out := map[string]any{}
	putString(out, "factory", c.Factory)
	putString(out, "service", c.Service)
	putString(out, "model", c.Model)
	putString(out, "endpoint", c.Endpoint)
	if c.Extra != nil {
		out["extra"] = c.Extra
	}
	return out
}

// JudgeEvalConfig is a scenario's `judge.eval:` block: the classifier that
// decides the verdicts, the LLM that explains them, and how.
//
//	judge:
//	  eval:
//	    service: ollama          # the LLM that classifies (the default), or
//	    model: gemma4:12b        #   factory: a registered factory that takes
//	                             #   this block and returns a classifier
//	    explainer:               # optional; the LLM that gives reasons
//	      service: ollama        #   (the judging LLM if omitted, the default
//	      model: gemma4:12b      #   LLM for a factory's classifier; none if
//	                             #   set to false)
//	    explain_below: 0.75      # optional
//	    allow_continue: true     # optional; false judges a reply yes or no only
type JudgeEvalConfig struct {
	// LLMServiceConfig names the LLM that classifies, or the factory that
	// builds the classifier. Its Raw is the whole `judge.eval:` block.
	LLMServiceConfig `yaml:",inline"`
	// Explainer is the LLM that gives the reasons.
	Explainer ExplainerConfig `yaml:"explainer,omitempty"`
	// ExplainBelow is the confidence under which a yes is explained too; nil
	// uses 0.75.
	ExplainBelow *float64 `yaml:"explain_below,omitempty"`
	// AllowContinue is whether a reply may be judged continue; nil allows it.
	AllowContinue *bool `yaml:"allow_continue,omitempty"`
}

// UnmarshalYAML reads the block and keeps it as written in Raw.
func (c *JudgeEvalConfig) UnmarshalYAML(node *yaml.Node) error {
	type plain struct {
		Factory       string          `yaml:"factory,omitempty"`
		Service       string          `yaml:"service,omitempty"`
		Model         string          `yaml:"model,omitempty"`
		Endpoint      string          `yaml:"endpoint,omitempty"`
		Extra         map[string]any  `yaml:"extra,omitempty"`
		Explainer     ExplainerConfig `yaml:"explainer,omitempty"`
		ExplainBelow  *float64        `yaml:"explain_below,omitempty"`
		AllowContinue *bool           `yaml:"allow_continue,omitempty"`
	}
	var p plain
	if err := node.Decode(&p); err != nil {
		return err
	}
	var raw map[string]any
	if err := node.Decode(&raw); err != nil {
		return err
	}
	*c = JudgeEvalConfig{
		Factory:       p.Factory,
		Service:       p.Service,
		Model:         p.Model,
		Endpoint:      p.Endpoint,
		Extra:         p.Extra,
		Raw:           raw,
		Explainer:     p.Explainer,
		ExplainBelow:  p.ExplainBelow,
		AllowContinue: p.AllowContinue,
	}
	return nil
}

// ExplainerConfig is a `judge.eval.explainer:` value: an LLM block, or false
// for no explainer. Absent, the judging LLM explains, or the default LLM when
// a factory built the classifier.
type ExplainerConfig struct {
	// Disabled is set by `explainer: false`.
	Disabled bool
	// LLM is the explainer's LLM block; nil when none was given.
	LLM *LLMServiceConfig
}

// UnmarshalYAML reads false or an LLM block.
func (c *ExplainerConfig) UnmarshalYAML(node *yaml.Node) error {
	if node.Kind == yaml.ScalarNode && node.Tag == "!!bool" {
		var on bool
		if err := node.Decode(&on); err != nil {
			return err
		}
		*c = ExplainerConfig{Disabled: !on}
		return nil
	}
	var llmCfg LLMServiceConfig
	if err := node.Decode(&llmCfg); err != nil {
		return err
	}
	*c = ExplainerConfig{LLM: &llmCfg}
	return nil
}

// SpeechConfig is a scenario's `user.speech:` block: the TTS the user's turns
// are spoken with.
type SpeechConfig struct {
	// Factory is the name a factory was registered under (see RegisterFactory);
	// it is given the whole block and returns the TTS.
	Factory string `yaml:"factory,omitempty"`
	// Service names a built-in TTS: "kokoro", or the deprecated "cartesia".
	Service string `yaml:"service,omitempty"`
	// Voice is the voice to speak in.
	Voice string `yaml:"voice,omitempty"`
	// Model is the TTS model; empty uses the service's default.
	Model string `yaml:"model,omitempty"`
	// Language is a language code such as "zh"; absent or blank keeps the
	// service's own default. It is read as written, so a value that is not a
	// string (YAML reads a bare `no` as false) is reported rather than passed
	// on.
	Language any `yaml:"language,omitempty"`
	// Speed is the speech rate multiplier; nil keeps the service's default.
	Speed *float64 `yaml:"speed,omitempty"`
	// SampleRate is the rate the user's audio is streamed at; zero uses 16 kHz.
	SampleRate int `yaml:"sample_rate,omitempty"`
	// APIKey is the deprecated Cartesia service's key; it falls back to
	// CARTESIA_API_KEY.
	APIKey string `yaml:"api_key,omitempty"`
	// Raw is the block as written, every key included, which is what a factory
	// is given. Filled when the block is read from YAML.
	Raw map[string]any `yaml:"-"`
}

// UnmarshalYAML reads the block and keeps it as written in Raw.
func (c *SpeechConfig) UnmarshalYAML(node *yaml.Node) error {
	type plain SpeechConfig
	var p plain
	if err := node.Decode(&p); err != nil {
		return err
	}
	if err := node.Decode(&p.Raw); err != nil {
		return err
	}
	*c = SpeechConfig(p)
	return nil
}

// raw is the block as a factory is given it.
func (c *SpeechConfig) raw() map[string]any {
	if c.Raw != nil {
		return c.Raw
	}
	out := map[string]any{}
	putString(out, "factory", c.Factory)
	putString(out, "service", c.Service)
	putString(out, "voice", c.Voice)
	putString(out, "model", c.Model)
	putString(out, "api_key", c.APIKey)
	if c.Language != nil {
		out["language"] = c.Language
	}
	if c.Speed != nil {
		out["speed"] = *c.Speed
	}
	if c.SampleRate != 0 {
		out["sample_rate"] = c.SampleRate
	}
	return out
}

// TranscriptionConfig is a scenario's `judge.transcription:` block: the STT
// that transcribes the bot's audio.
type TranscriptionConfig struct {
	// Factory is the name a factory was registered under (see RegisterFactory);
	// it is given the whole block and returns the STT.
	Factory string `yaml:"factory,omitempty"`
	// Service names a built-in STT: "moonshine" (the default) or "whisper".
	Service string `yaml:"service,omitempty"`
	// Model is the STT model; empty uses the service's default.
	Model string `yaml:"model,omitempty"`
	// Language is a language code such as "es"; absent or blank keeps the
	// service's own default. Read as written, as SpeechConfig.Language is.
	Language any `yaml:"language,omitempty"`
	// Device is where Whisper runs: "cpu" (the default) or "cuda".
	Device string `yaml:"device,omitempty"`
	// ComputeType is Whisper's compute type; empty uses int8 on the CPU.
	ComputeType string `yaml:"compute_type,omitempty"`
	// Raw is the block as written, every key included, which is what a factory
	// is given. Filled when the block is read from YAML.
	Raw map[string]any `yaml:"-"`
}

// UnmarshalYAML reads the block and keeps it as written in Raw.
func (c *TranscriptionConfig) UnmarshalYAML(node *yaml.Node) error {
	type plain TranscriptionConfig
	var p plain
	if err := node.Decode(&p); err != nil {
		return err
	}
	if err := node.Decode(&p.Raw); err != nil {
		return err
	}
	*c = TranscriptionConfig(p)
	return nil
}

// raw is the block as a factory is given it.
func (c *TranscriptionConfig) raw() map[string]any {
	if c.Raw != nil {
		return c.Raw
	}
	out := map[string]any{}
	putString(out, "factory", c.Factory)
	putString(out, "service", c.Service)
	putString(out, "model", c.Model)
	putString(out, "device", c.Device)
	putString(out, "compute_type", c.ComputeType)
	if c.Language != nil {
		out["language"] = c.Language
	}
	return out
}

// putString sets key to value when value is not empty.
func putString(m map[string]any, key, value string) {
	if value != "" {
		m[key] = value
	}
}
