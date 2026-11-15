package speech

import (
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"strings"

	"cloud.google.com/go/speech/apiv2/speechpb"
	"google.golang.org/protobuf/encoding/protojson"
	"google.golang.org/protobuf/proto"
)

// Errors reported for an adaptation or denoiser setting Google's API cannot
// represent.
//
//nolint:gochecknoglobals // sentinel errors
var (
	errInvalidPhraseSet = errors.New("invalid Google SpeechAdaptation phrase_set entry")
	errAdaptationType   = errors.New("google stt: adaptation must be a *speechpb.SpeechAdaptation or a map")
	errDenoiserType     = errors.New("google stt: denoiser config must be a *speechpb.DenoiserConfig or a map")
)

// modelSupportsAdaptation reports whether a Speech-to-Text V2 model accepts a
// SpeechAdaptation config.
//
// The telephony model rejects it with "Recognizer does not support feature:
// speech_adaptation_boost". Support elsewhere is a per-model, per-language
// matrix that Google validates on its own, so every other model is passed
// through and left to report its own errors.
func modelSupportsAdaptation(model string) bool {
	return strings.ToLower(model) != "telephony"
}

// normalizeSpeechAdaptation turns an adaptation setting into a SpeechAdaptation
// message.
//
// The setting is a message, returned as it is, or the equivalent map. In the
// map, a "phrase_sets" entry may be the resource name of a phrase set or a bare
// inline phrase set, and a single entry may be given in place of a list.
func normalizeSpeechAdaptation(adaptation any) (*speechpb.SpeechAdaptation, error) {
	var payload map[string]any
	switch a := adaptation.(type) {
	case *speechpb.SpeechAdaptation:
		return a, nil
	case map[string]any:
		payload = a
	default:
		return nil, fmt.Errorf("%w, got %T", errAdaptationType, adaptation)
	}

	normalized := maps.Clone(payload)

	phraseSets, err := phraseSetList(normalized["phrase_sets"])
	if err != nil {
		return nil, err
	}

	converted := make([]any, 0, len(phraseSets))
	for _, phraseSet := range phraseSets {
		entry, err := convertPhraseSet(phraseSet)
		if err != nil {
			return nil, err
		}
		converted = append(converted, entry)
	}

	if len(converted) > 0 {
		normalized["phrase_sets"] = converted
	}

	out := &speechpb.SpeechAdaptation{}
	if err := fromMap(normalized, out); err != nil {
		return nil, fmt.Errorf("google stt: adaptation: %w", err)
	}
	return out, nil
}

// phraseSetList is the "phrase_sets" entries as a list, a single entry given
// in place of one included.
func phraseSetList(raw any) ([]any, error) {
	var phraseSets []any
	switch r := raw.(type) {
	case nil:
	case string, map[string]any:
		phraseSets = []any{r}
	case []any:
		phraseSets = r
	case []string:
		for _, p := range r {
			phraseSets = append(phraseSets, p)
		}
	case []map[string]any:
		for _, p := range r {
			phraseSets = append(phraseSets, p)
		}
	default:
		return nil, fmt.Errorf("%w: expected string, object or list, got %T", errInvalidPhraseSet, raw)
	}
	return phraseSets, nil
}

// convertPhraseSet turns one "phrase_sets" entry into the shape Google's API
// takes: a resource name becomes a reference to the phrase set, and a bare
// phrase set becomes an inline one.
func convertPhraseSet(phraseSet any) (any, error) {
	switch p := phraseSet.(type) {
	case string:
		return map[string]any{"phrase_set": p}, nil
	case map[string]any:
		_, named := p["phrase_set"]
		_, inline := p["inline_phrase_set"]
		if named || inline {
			return p, nil
		}
		return map[string]any{"inline_phrase_set": p}, nil
	default:
		return nil, fmt.Errorf("%w: expected string or object, got %T", errInvalidPhraseSet, phraseSet)
	}
}

// denoiserConfig turns a denoiser setting into a DenoiserConfig message: a
// message is returned as it is, and a map is read as its JSON form.
func denoiserConfig(config any) (*speechpb.DenoiserConfig, error) {
	switch c := config.(type) {
	case *speechpb.DenoiserConfig:
		return c, nil
	case map[string]any:
		out := &speechpb.DenoiserConfig{}
		if err := fromMap(c, out); err != nil {
			return nil, fmt.Errorf("google stt: denoiser config: %w", err)
		}
		return out, nil
	default:
		return nil, fmt.Errorf("%w, got %T", errDenoiserType, config)
	}
}

// fromMap reads a map into a message the way the message's JSON form reads, by
// either the field's own name or its JSON name.
func fromMap(m map[string]any, msg proto.Message) error {
	data, err := json.Marshal(m)
	if err != nil {
		return err
	}
	return protojson.Unmarshal(data, msg)
}
