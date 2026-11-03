package realtime

import (
	"context"
	"encoding/json"
	"log/slog"
	"reflect"
	"strings"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/provider/openai/realtime/events"
	"github.com/gojargo/jargo/service/settings"
)

// Settings is what the service holds of its configuration while the pipeline
// runs: the general model settings and the session's properties. The model and
// the system instruction are kept in step with the properties' own: whichever
// of the two an update gives, both end up holding it, the top-level field
// winning when an update gives both.
type Settings struct {
	settings.LLM

	// SessionProperties are the session's properties. An update giving them
	// replaces them whole.
	SessionProperties settings.Opt[events.SessionProperties] `settings:"session_properties"`
}

var (
	_ settings.Updater = (*Settings)(nil)
	_ settings.Mapper  = (*Settings)(nil)
)

// syncTopLevelToProperties copies the model and the system instruction into the
// session properties.
func (s *Settings) syncTopLevelToProperties() {
	sp, ok := s.SessionProperties.Value()
	if !ok {
		return
	}
	if model, ok := s.Model.Value(); ok && model != "" {
		sp.Model = model
	}
	if instruction, ok := s.SystemInstruction.Value(); ok {
		sp.Instructions = instruction
	}
	s.SessionProperties = settings.Set(sp)
}

// ApplyUpdate merges an update, keeping the model and the system instruction in
// step with the session properties.
//
// Properties an update gives replace the stored ones whole. When they carry a
// model or an instruction the update did not give at the top level, those are
// taken up to the top level; then the top level is copied back into the
// properties, so the top-level values win whenever the two disagree.
func (s *Settings) ApplyUpdate(delta any) (settings.Changed, error) {
	changed, err := settings.ApplyFields(s, delta)
	if err != nil {
		return nil, err
	}
	if sp, ok := s.SessionProperties.Value(); ok && changed.Has("session_properties") {
		if !changed.Has("model") && sp.Model != "" {
			takeUp(&s.Model, sp.Model, "model", changed)
		}
		if !changed.Has("system_instruction") && sp.Instructions != "" {
			takeUp(&s.SystemInstruction, sp.Instructions, "system_instruction", changed)
		}
	}
	s.syncTopLevelToProperties()
	return changed, nil
}

// takeUp sets a top-level field to a value the session properties carry,
// reporting it changed when it did.
func takeUp(field *settings.Opt[string], value, name string, changed settings.Changed) {
	old := field.Or("")
	*field = settings.Set(value)
	if old != value {
		changed[name] = old
	}
}

// FromMap fills a delta from plain names and values, routing the names of
// session properties into the properties. The model always goes to the top
// level; a name that is neither a setting nor a property lands in Extra.
func (s *Settings) FromMap(values map[string]any) error {
	own := map[string]bool{}
	for _, name := range settingNames(s) {
		own[name] = true
	}
	properties := sessionPropertyNames()

	top := map[string]any{}
	sp := map[string]any{}
	for key, value := range values {
		switch {
		case own[key]:
			top[key] = value
		case properties[key]:
			sp[key] = value
		default:
			top[key] = value // FromMapFields puts an unknown name in Extra.
		}
	}
	if len(sp) > 0 {
		raw, err := json.Marshal(sp)
		if err != nil {
			return err
		}
		var props events.SessionProperties
		if err := json.Unmarshal(raw, &props); err != nil {
			return err
		}
		s.SessionProperties = settings.Set(props)
	}
	return settings.FromMapFields(s, top)
}

// settingNames are the names of the settings a settings value declares.
func settingNames(v any) []string {
	var names []string
	var walk func(t reflect.Type)
	walk = func(t reflect.Type) {
		for f := range t.Fields() {
			if f.Anonymous && f.Type.Kind() == reflect.Struct {
				walk(f.Type)
				continue
			}
			if name := f.Tag.Get("settings"); name != "" && name != "-" {
				names = append(names, name)
			}
		}
	}
	walk(reflect.TypeOf(v).Elem())
	return names
}

// sessionPropertyNames are the wire names of the session properties, the model
// excepted: it is a setting of its own.
func sessionPropertyNames() map[string]bool {
	names := map[string]bool{}
	t := reflect.TypeFor[events.SessionProperties]()
	for field := range t.Fields() {
		name, _, _ := strings.Cut(field.Tag.Get("json"), ",")
		if name != "" && name != "-" && name != "model" {
			names[name] = true
		}
	}
	return names
}

// newSettings is the starting state, taken from what the service was built
// with: the model and instruction kept in step with the properties.
func newSettings(cfg Config) *Settings {
	s := &Settings{}
	s.Model = settings.Set(cfg.Model)
	s.SystemInstruction = settings.Set(cfg.Instructions)
	s.Temperature = settings.Cleared[float64]()
	s.MaxTokens = settings.Cleared[int]()
	s.TopP = settings.Cleared[float64]()
	s.TopK = settings.Cleared[int]()
	s.FrequencyPenalty = settings.Cleared[float64]()
	s.PresencePenalty = settings.Cleared[float64]()
	s.Seed = settings.Cleared[int]()
	s.FilterIncompleteUserTurns = settings.Set(false)
	s.SessionProperties = settings.Set(cfg.SessionProperties.Copy())
	s.syncTopLevelToProperties()
	omitUnsupportedTranscriptionPrompt(s)
	return s
}

// omitUnsupportedTranscriptionPrompt drops the transcription prompt the
// realtime whisper model does not take, reporting whether it did.
func omitUnsupportedTranscriptionPrompt(s *Settings) bool {
	sp, ok := s.SessionProperties.Value()
	if !ok {
		return false
	}
	t := sp.InputTranscription()
	if t == nil || t.Prompt == "" || (t.Model != "" && t.Model != events.WhisperModel) {
		return false
	}
	t.Prompt = ""
	s.SessionProperties = settings.Set(sp)
	slog.Warn(events.WhisperModel + " does not support the prompt parameter; omitting prompt from " +
		"OpenAI Realtime input audio transcription settings")
	return true
}

// reasoningModels are substrings of the models that take a reasoning
// configuration. Matching on a substring lets the dated variants of a model
// match too.
//
//nolint:gochecknoglobals // a lookup table
var reasoningModels = []string{"gpt-realtime-2", "gpt-realtime-2.1"}

// stripUnsupportedReasoning drops the reasoning configuration from outgoing
// properties when the model cannot take it, since the server would otherwise
// reject the whole update and end the session. It reports the stripping once.
func (s *Service) stripUnsupportedReasoning(sp events.SessionProperties) events.SessionProperties {
	if sp.Reasoning == nil || sp.Model == "" {
		return sp
	}
	for _, m := range reasoningModels {
		if strings.Contains(sp.Model, m) {
			return sp
		}
	}
	if s.reasoningWarned.CompareAndSwap(false, true) {
		slog.Warn("stripping reasoning from session.update: the model isn't a known reasoning-capable "+
			"Realtime model", "service", s.Name(), "model", sp.Model)
	}
	sp.Reasoning = nil
	return sp
}

// Settings returns the store a settings update is merged into.
func (s *Service) Settings() any { return s.live }

// UpdateSettings acts on a settings update: changed session properties or a
// changed system instruction reach the session as a session update, and so
// does a transcription prompt the update gave that had to be dropped. Anything
// else is reported as not applied to the running session.
func (s *Service) UpdateSettings(ctx context.Context, changed settings.Changed) error {
	promptOmitted := omitUnsupportedTranscriptionPrompt(s.live)
	if changed.Has("session_properties") || changed.Has("system_instruction") || promptOmitted {
		s.sendSessionUpdate(ctx)
	}
	s.WarnUnhandledUpdatedSettings(ctx, changed.Except("session_properties", "system_instruction"))
	return nil
}

// sessionProperties are the session properties the service holds.
func (s *Service) sessionProperties() events.SessionProperties {
	sp, _ := s.live.SessionProperties.Value()
	return sp
}

// ServiceTools returns the tools configured in the session properties, which
// the base registers the handlers of while the conversation advertises none.
func (s *Service) ServiceTools() frames.ToolsSchema { return s.sessionProperties().Tools }

// model is the model the session runs.
func (s *Service) model() string { return s.live.Model.Or("") }
