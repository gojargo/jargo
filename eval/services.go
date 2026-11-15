package eval

import (
	"cmp"
	"errors"
	"fmt"
	"log/slog"
	"maps"
	"os"
	"slices"
	"strings"
	"sync"

	"github.com/gojargo/jargo/classifier"
	llmclassifier "github.com/gojargo/jargo/classifier/llm"
	"github.com/gojargo/jargo/language"
	"github.com/gojargo/jargo/provider/cartesia"
	"github.com/gojargo/jargo/provider/moonshine"
	"github.com/gojargo/jargo/provider/ollama"
	"github.com/gojargo/jargo/provider/openai/chat"
	"github.com/gojargo/jargo/service/llm"
	"github.com/gojargo/jargo/service/stt"
	"github.com/gojargo/jargo/service/tts"
)

// Service constructors for the harness: a service from a scenario's config
// block.
//
// The dispatchers pick a provider by its `service:` name, or call a
// `factory:`; the per-provider builders do the rest. The built-in names are the
// local, keyless services: Ollama for the judge and the persona, Kokoro for the
// user's voice, Moonshine and Whisper for the bot's audio. Any other provider
// is a factory: a function registered under a dotted name with
// RegisterFactory, which takes the config block and returns the service, so a
// scenario stays a description of the test while the provider, its settings
// and its key live in code:
//
//	eval.RegisterFactory("my_evals.judge", func(config map[string]any) (any, error) {
//		model, _ := config["model"].(string)
//		return chat.NewLLM(chat.LLMConfig{APIKey: key, Model: model}), nil
//	})
//
//	# in the scenario
//	judge:
//	  eval: {factory: my_evals.judge, model: gpt-4o-mini}
//
// What a factory must return: the judge's LLM an llm.Inferencer, or a
// classifier.Classifier for judge.eval; the user's TTS a local or HTTP service
// (a *tts.Base or a tts.Synthesizer), since the cache around it drives its
// RunTTS directly and a WebSocket-streaming one yields nothing there; the
// bot's STT a *stt.SegmentService, since only its transcripts can be traced
// to when their audio began.
//
// This is harness code, so like the factories the release suite names, it
// reads the API key a deprecated paid provider needs from the environment when
// the block carries none: CARTESIA_API_KEY for `service: cartesia` and
// OPENAI_API_KEY for `service: openai`. It reads where libmoonshine is from
// JARGO_MOONSHINE_LIB for `service: moonshine`.

// Default models and endpoint of the built-in LLM services.
const (
	// DefaultOllamaJudgeModel is the model the judge classifies with when the
	// block names none.
	DefaultOllamaJudgeModel = "gemma4:12b"
	// DefaultOpenAIModel is the model the deprecated `service: openai` builds
	// with when the block names none.
	DefaultOpenAIModel = "gpt-4o"
	// DefaultOllamaEndpoint is where the Ollama service is reached when the
	// block names no endpoint.
	DefaultOllamaEndpoint = "http://localhost:11434/v1"
	// defaultCartesiaModel is the model the deprecated `service: cartesia`
	// builds with when the block names none.
	defaultCartesiaModel = "sonic-2"
)

// DefaultOllamaJudgeExtra is the request parameters the default judge model is
// sent. It is thinking-capable, and only its JSON verdict is ever read, so
// reasoning buys nothing while costing latency and eating into the token
// budget the verdict needs.
func DefaultOllamaJudgeExtra() map[string]any {
	return map[string]any{"reasoning_effort": "none"}
}

// Factory builds a service from the config block that names it. It is given
// the whole block as written, and returns the service: a classifier or an LLM,
// a TTS, or an STT, by the block it is named in.
type Factory func(config map[string]any) (any, error)

// STTServiceBuilder builds the STT a built-in `judge.transcription.service:`
// name stands for.
type STTServiceBuilder func(cfg *TranscriptionConfig) (*stt.SegmentService, error)

// TTSServiceBuilder builds the TTS a built-in `user.speech.service:` name
// stands for. It should synthesize at CacheSampleRate, the rate the cache
// stores, so its audio is cached as it came.
type TTSServiceBuilder func(cfg *SpeechConfig) (*tts.Base, error)

// The built-in local services, named by a block's `service:`.
const (
	sttMoonshine = "moonshine"
	sttWhisper   = "whisper"
	ttsKokoro    = "kokoro"
	ttsCartesia  = "cartesia"
	llmOllama    = "ollama"
	llmOpenAI    = "openai"
)

// registry holds the factories and the builders of the built-in local
// services. The built-in names start out with no builder, until the service
// is available to the harness; a builder is registered for them.
//
//nolint:gochecknoglobals // the process-wide registration point for factories
var registry = struct {
	sync.RWMutex
	factories map[string]Factory
	stt       map[string]STTServiceBuilder
	tts       map[string]TTSServiceBuilder
}{
	factories: map[string]Factory{
		typesafeClassifierFactory: TypesafeClassifier,
	},
	stt: map[string]STTServiceBuilder{sttMoonshine: moonshineService, sttWhisper: nil},
	tts: map[string]TTSServiceBuilder{ttsKokoro: nil},
}

// Errors the service constructors report.
var (
	// ErrUnknownService reports a `service:` name that is not built in.
	ErrUnknownService = errors.New("eval: unknown service")
	// ErrServiceUnavailable reports a built-in service name with no builder
	// registered yet.
	ErrServiceUnavailable = errors.New("eval: service not available")
	// ErrFactory reports a `factory:` that is not a registered dotted name, or
	// that returned something the block cannot use.
	ErrFactory = errors.New("eval: bad factory")
	// errUnknownLanguage reports a `language:` that is not a language code.
	errUnknownLanguage = errors.New("eval: unknown language in eval speech/transcription config; " +
		"expected a language code like 'zh' or a Language value")
	// errSpeechConfig reports a user.speech block naming no service or voice.
	errSpeechConfig = errors.New("eval: user.speech config requires at least 'service' and 'voice'")
)

// RegisterFactory registers a factory under name, the dotted name a block's
// `factory:` gives, replacing any registered before. The names the release
// suite uses come registered, so its scenarios load unchanged:
// "evals.judges.typesafe_classifier" is TypesafeClassifier.
func RegisterFactory(name string, f Factory) {
	registry.Lock()
	defer registry.Unlock()
	registry.factories[name] = f
}

// RegisterSTTService registers the builder of a built-in transcription
// service, such as "moonshine" or "whisper", replacing any registered before.
func RegisterSTTService(name string, build STTServiceBuilder) {
	registry.Lock()
	defer registry.Unlock()
	registry.stt[strings.ToLower(name)] = build
}

// RegisterTTSService registers the builder of a built-in user TTS, such as
// "kokoro", replacing any registered before.
func RegisterTTSService(name string, build TTSServiceBuilder) {
	registry.Lock()
	defer registry.Unlock()
	registry.tts[strings.ToLower(name)] = build
}

// factory is the factory a block's `factory:` names, where being the block's
// name in the file, for the errors.
func factory(where, name string) (Factory, error) {
	if !strings.Contains(name, ".") || strings.HasPrefix(name, ".") || strings.HasSuffix(name, ".") {
		return nil, fmt.Errorf("%w: %s.factory must be a dotted path: %q", ErrFactory, where, name)
	}
	registry.RLock()
	f, ok := registry.factories[name]
	registry.RUnlock()
	if !ok || f == nil {
		return nil, fmt.Errorf("%w: %s.factory %q is not registered; register it with eval.RegisterFactory",
			ErrFactory, where, name)
	}
	return f, nil
}

// known is the names a builder map holds, sorted, as an error lists them.
func known[B any](builders map[string]B) string {
	return strings.Join(slices.Sorted(maps.Keys(builders)), ", ")
}

// cfgLanguage is a config's optional `language` as a Language: absent or blank
// leaves the service's own default, reported as "". A code string or a
// Language is accepted; anything else, a code that names no language included,
// is an error.
func cfgLanguage(value any) (language.Language, error) {
	var code string
	switch v := value.(type) {
	case nil:
		return "", nil
	case language.Language:
		code = string(v)
	case string:
		code = strings.TrimSpace(v)
		if code == "" {
			return "", nil
		}
	default:
		return "", fmt.Errorf("%w: %v", errUnknownLanguage, value)
	}
	if slices.Contains(language.All(), language.Language(code)) {
		return language.Language(code), nil
	}
	return "", fmt.Errorf("%w: %q", errUnknownLanguage, code)
}

// STTServiceFromConfig builds the STT that transcribes the bot's audio, from a
// `judge.transcription:` block.
//
// A factory builds it; otherwise the service name picks one, "moonshine" by
// default. A factory must return a *stt.SegmentService, as the built-in ones
// are. A nil cfg uses the Moonshine default.
func STTServiceFromConfig(cfg *TranscriptionConfig) (*stt.SegmentService, error) {
	if cfg == nil {
		cfg = &TranscriptionConfig{}
	}
	if cfg.Factory != "" {
		f, err := factory("transcription", cfg.Factory)
		if err != nil {
			return nil, err
		}
		built, err := f(cfg.raw())
		if err != nil {
			return nil, err
		}
		service, ok := built.(*stt.SegmentService)
		if !ok {
			return nil, fmt.Errorf("%w: transcription.factory %q returned %T, not a *stt.SegmentService",
				ErrFactory, cfg.Factory, built)
		}
		return service, nil
	}

	name := strings.ToLower(cmp.Or(cfg.Service, sttMoonshine))
	registry.RLock()
	build, ok := registry.stt[name]
	names := known(registry.stt)
	registry.RUnlock()
	if !ok {
		return nil, fmt.Errorf("%w: unknown STT service: %q. Known: %s. For any other, set "+
			"transcription.factory to a registered factory returning a *stt.SegmentService",
			ErrUnknownService, name, names)
	}
	if build == nil {
		return nil, fmt.Errorf("%w: the %s STT service is not available to the harness yet; register it "+
			"with eval.RegisterSTTService, or set transcription.factory", ErrServiceUnavailable, name)
	}
	return build(cfg)
}

// moonshineService is a local Moonshine STT from a `judge.transcription:`
// block: its `model` (default small-streaming) and its optional `language`
// code, English when absent. libmoonshine is loaded from JARGO_MOONSHINE_LIB,
// or by its conventional name when that is unset.
//
// Prefer Whisper for a non-English bot: Moonshine's non-English models
// transcribe synthesized speech unreliably, returning an empty transcript or
// dropping the tail of an utterance, and an empty transcript is
// indistinguishable from a bot that said nothing.
func moonshineService(cfg *TranscriptionConfig) (*stt.SegmentService, error) {
	lang, err := cfgLanguage(cfg.Language)
	if err != nil {
		return nil, err
	}
	return moonshine.NewSTT(moonshine.STTConfig{
		Model:       moonshine.Model(cmp.Or(cfg.Model, string(moonshine.ModelSmallStreaming))),
		Language:    lang,
		LibraryPath: os.Getenv(moonshine.LibPathEnv),
	})
}

// TTSServiceFromConfig builds the user-audio TTS, wrapped in a cache, from a
// `user.speech:` block.
//
// A factory builds the inner service; otherwise the service name picks one,
// "kokoro". The wrapper synthesizes each user utterance once and reuses it
// across runs, from cacheDir (empty uses DefaultCacheDir), and useCache false
// forces fresh synthesis. The wrapper streams at the block's sample rate (see
// TTSSampleRate).
//
// `service: cartesia` still builds a Cartesia HTTP TTS, with a warning.
func TTSServiceFromConfig(cfg *SpeechConfig, cacheDir string, useCache bool) (*CachingTTSService, error) {
	if cfg == nil {
		cfg = &SpeechConfig{}
	}
	var (
		inner any
		err   error
	)
	if cfg.Factory != "" {
		f, ferr := factory("user.speech", cfg.Factory)
		if ferr != nil {
			return nil, ferr
		}
		inner, err = f(cfg.raw())
	} else {
		name := strings.ToLower(cfg.Service)
		if name == "" || cfg.Voice == "" {
			return nil, errSpeechConfig
		}
		inner, err = builtinTTS(name, cfg)
	}
	if err != nil {
		return nil, err
	}
	syn, err := innerSynthesizer(cfg.Factory, inner)
	if err != nil {
		return nil, err
	}
	return NewCachingTTSService(syn, CachingTTSConfig{
		CacheKey:   TTSCacheKey(cfg),
		CacheDir:   cacheDir,
		UseCache:   &useCache,
		SampleRate: TTSSampleRate(cfg),
	})
}

// builtinTTS is the TTS a built-in `user.speech.service:` name builds.
func builtinTTS(name string, cfg *SpeechConfig) (*tts.Base, error) {
	if name == ttsCartesia {
		slog.Warn("eval: `user.speech.service: cartesia` is deprecated and will be removed. Use `factory` instead.")
		return cartesiaService(cfg)
	}
	registry.RLock()
	build, ok := registry.tts[name]
	names := known(registry.tts)
	registry.RUnlock()
	if !ok {
		return nil, fmt.Errorf("%w: unknown TTS service: %q. Known: %s. For any other, set "+
			"user.speech.factory to a registered factory returning a TTS service", ErrUnknownService, name, names)
	}
	if build == nil {
		return nil, fmt.Errorf("%w: the %s TTS service is not available to the harness yet; register it "+
			"with eval.RegisterTTSService, or set user.speech.factory", ErrServiceUnavailable, name)
	}
	return build(cfg)
}

// innerSynthesizer is the synthesizer of what a factory or a builder built: a
// TTS service, or a synthesizer itself.
func innerSynthesizer(factoryName string, built any) (tts.Synthesizer, error) {
	switch v := built.(type) {
	case *tts.Base:
		if v != nil {
			return v.Synthesizer(), nil
		}
	case tts.Synthesizer:
		return v, nil
	}
	return nil, fmt.Errorf("%w: user.speech.factory %q returned %T, not a TTS service", ErrFactory, factoryName, built)
}

// cartesiaService is a Cartesia HTTP TTS from a `user.speech:` block, keyed
// from the block or, so committed scenarios carry no secrets, from
// CARTESIA_API_KEY. It synthesizes at the cache's rate.
//
// Deprecated: use a `user.speech.factory` that builds a Cartesia HTTP TTS.
func cartesiaService(cfg *SpeechConfig) (*tts.Base, error) {
	apiKey := cmp.Or(cfg.APIKey, os.Getenv("CARTESIA_API_KEY"))
	if apiKey == "" {
		return nil, fmt.Errorf("%w: Cartesia API key not found; set $CARTESIA_API_KEY or user.speech.api_key",
			ErrServiceUnavailable)
	}
	lang, err := cfgLanguage(cfg.Language)
	if err != nil {
		return nil, err
	}
	return cartesia.NewHTTPTTS(cartesia.HTTPTTSConfig{
		APIKey:     apiKey,
		VoiceID:    cfg.Voice,
		Model:      cmp.Or(cfg.Model, defaultCartesiaModel),
		Language:   lang,
		SampleRate: CacheSampleRate,
	}), nil
}

// ClassifierFromConfig builds the classifier a `judge.eval:` block names.
//
// A factory builds it, and may return a classifier.Classifier or an LLM
// service (an llm.Inferencer), which a classifier/llm classifier then asks.
// Otherwise the block names an LLM, as LLMServiceFromConfig reads it, and a
// classifier/llm classifier asks that. where is the block's name in the file,
// for the errors; a nil cfg uses all the defaults.
func ClassifierFromConfig(cfg *LLMServiceConfig, where string) (classifier.Classifier, error) {
	if cfg == nil {
		cfg = &LLMServiceConfig{}
	}
	if cfg.Factory != "" {
		f, err := factory(where, cfg.Factory)
		if err != nil {
			return nil, err
		}
		built, err := f(cfg.raw())
		if err != nil {
			return nil, err
		}
		switch v := built.(type) {
		case classifier.Classifier:
			return v, nil
		case llm.Inferencer:
			return llmclassifier.New(llmclassifier.Config{LLM: v})
		}
		return nil, fmt.Errorf("%w: %s.factory %q returned %T, not a classifier or an LLM service",
			ErrFactory, where, cfg.Factory, built)
	}
	service, err := LLMServiceFromConfig(cfg, where)
	if err != nil {
		return nil, err
	}
	return llmclassifier.New(llmclassifier.Config{LLM: service})
}

// LLMServiceFromConfig builds an LLM service from a block naming one: the
// judge's, its explainer's, or a persona's.
//
// A factory builds it; otherwise the service name picks a provider, "ollama"
// by default. where is the block's name in the file ("judge.eval",
// "simulator"), for the errors; a nil cfg uses all the defaults.
//
// `service: openai` still builds an OpenAI service, with a warning, keyed from
// OPENAI_API_KEY.
func LLMServiceFromConfig(cfg *LLMServiceConfig, where string) (llm.Inferencer, error) {
	if cfg == nil {
		cfg = &LLMServiceConfig{}
	}
	if cfg.Factory != "" {
		f, err := factory(where, cfg.Factory)
		if err != nil {
			return nil, err
		}
		built, err := f(cfg.raw())
		if err != nil {
			return nil, err
		}
		service, ok := built.(llm.Inferencer)
		if !ok {
			return nil, fmt.Errorf("%w: %s.factory %q returned %T, not an LLM service",
				ErrFactory, where, cfg.Factory, built)
		}
		return service, nil
	}
	switch name := strings.ToLower(cmp.Or(cfg.Service, llmOllama)); name {
	case llmOllama:
		return OllamaService(cfg), nil
	case llmOpenAI:
		slog.Warn(fmt.Sprintf("eval: `%s.service: openai` is deprecated and will be removed. Use `factory` instead.",
			where))
		return openAIService(cfg), nil
	default:
		return nil, fmt.Errorf("%w: unknown %s service: %q. Known: ollama. For any other, set %s.factory "+
			"to a registered factory returning an LLM service", ErrUnknownService, where, name, where)
	}
}

// OllamaService builds a local Ollama LLM service from a block.
//
// An `extra:` mapping goes to the model as request parameters, which is how
// `reasoning_effort: none` reaches a thinking model. The default model comes
// with the extras it needs (DefaultOllamaJudgeExtra); a named model gets only
// the extras asked for.
func OllamaService(cfg *LLMServiceConfig) *chat.LLMService {
	extra := cfg.Extra
	if extra == nil {
		extra = map[string]any{}
		if cfg.Model == "" {
			extra = DefaultOllamaJudgeExtra()
		}
	}
	return ollama.NewLLM(chat.LLMConfig{
		BaseURL: cmp.Or(cfg.Endpoint, DefaultOllamaEndpoint),
		Model:   cmp.Or(cfg.Model, DefaultOllamaJudgeModel),
		Extra:   extra,
	})
}

// openAIService is an OpenAI LLM from a block, its `extra:` forwarded as
// request parameters, keyed from OPENAI_API_KEY.
//
// Deprecated: use a `judge.eval.factory` that builds an OpenAI service.
func openAIService(cfg *LLMServiceConfig) *chat.LLMService {
	return chat.NewLLM(chat.LLMConfig{
		APIKey: os.Getenv("OPENAI_API_KEY"),
		Model:  cmp.Or(cfg.Model, DefaultOpenAIModel),
		Extra:  cfg.Extra,
	})
}
