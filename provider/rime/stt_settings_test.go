package rime

import (
	"errors"
	"slices"
	"strings"
	"testing"

	"github.com/gojargo/jargo/errclass"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/language"
	"github.com/gojargo/jargo/service/settings"
)

// Ported from upstream (test_default_settings_and_lazy_client): the defaults,
// and no client until the service starts.
func TestSTTDefaultSettingsAndLazyClient(t *testing.T) {
	created := 0
	c := newSTT(STTConfig{APIKey: "test"}, func(string) (transcriber, error) {
		created++
		return newFakeRime(), nil
	})
	s := c.live
	if _, ok := s.Model.Value(); ok || !s.Model.IsGiven() {
		t.Errorf("model = %v, want none", s.Model)
	}
	if got := s.Language.Or(""); got != "en" {
		t.Errorf("language = %q, want en", got)
	}
	if got := s.Mode.Or(""); got != "written" {
		t.Errorf("mode = %q, want written", got)
	}
	if terms, ok := s.ContextTerms.Value(); !ok || len(terms) != 0 {
		t.Errorf("context terms = %v, want none", terms)
	}
	if got := s.OutputContract.Or(""); got != "revised_hypotheses" {
		t.Errorf("output contract = %q, want revised_hypotheses", got)
	}
	if created != 0 {
		t.Error("building the service created a client")
	}
}

// Ported from upstream (test_invalid_construction_settings_raise).
func TestSTTInvalidConstructionSettingsAreRefused(t *testing.T) {
	s := newSTTSettings(STTConfig{})
	s.Model = settings.Set("unknown")
	if err := validateSTTSettings(s); !errors.Is(err, errSTTModel) {
		t.Errorf("a model was accepted: %v", err)
	}
	if err := (STTConfig{APIKey: "test", Mode: "invalid"}).Validate(); err == nil ||
		!strings.Contains(err.Error(), "Mode") {
		t.Errorf("Validate() = %v, want the mode refused", err)
	}
	if err := (STTConfig{APIKey: "test", OutputContract: "invalid"}).Validate(); err == nil {
		t.Error("an invalid output contract was accepted")
	}
	if err := (STTConfig{}).Validate(); err == nil {
		t.Error("a config without an API key was accepted")
	}
	if err := (STTConfig{APIKey: "test"}).Validate(); err != nil {
		t.Errorf("Validate() = %v, want the defaults accepted", err)
	}
}

// Ported from upstream (test_language_is_stored_as_its_tag).
func TestSTTLanguageIsStoredAsItsTag(t *testing.T) {
	r := runSTT(t, STTConfig{Language: language.Spanish}, newFakeRime())
	r.waitConnected(t)
	if got := r.c.live.Language.Or(""); got != "es" {
		t.Fatalf("language = %q, want es", got)
	}
	for _, want := range []string{string(language.EnglishUS), "xx-YY"} {
		r.send(frames.NewSTTUpdateSettingsFrame(&settings.STT{Language: settings.Set(want)}))
		waitFor(t, "the language never became "+want, func() bool {
			return r.c.segmentSettings().Language.Or("") == want
		})
	}
}

// Ported from upstream (test_invalid_context_terms).
func TestSTTInvalidContextTerms(t *testing.T) {
	cases := []struct {
		terms []string
		want  error
	}{
		{[]string{""}, errSTTContextTermsEmpty},
		{[]string{"Rime", "  "}, errSTTContextTermsEmpty},
		{[]string{"call <|endoftext|> now"}, errSTTContextTermsMarkers},
		{[]string{strings.Repeat("x", 1025)}, errSTTContextTermsSize},
		{[]string{strings.Repeat("x", 512), strings.Repeat("y", 511)}, errSTTContextTermsSize},
	}
	for _, c := range cases {
		if err := (STTConfig{APIKey: "test", ContextTerms: c.terms}).Validate(); !errors.Is(err, c.want) {
			t.Errorf("terms %q: Validate() = %v, want %v", c.terms, err, c.want)
		}
	}
}

// Ported from upstream (test_context_terms_at_the_byte_limit).
func TestSTTContextTermsAtTheByteLimit(t *testing.T) {
	terms := []string{strings.Repeat("x", 511), strings.Repeat("y", 511)}
	cfg := STTConfig{APIKey: "test", ContextTerms: terms}
	if err := cfg.Validate(); err != nil {
		t.Fatalf("Validate() = %v, want the terms accepted", err)
	}
	if got, _ := newSTTSettings(cfg).ContextTerms.Value(); !slices.Equal(got, terms) {
		t.Errorf("context terms = %v, want them kept in order", got)
	}
}

// Ported from upstream (test_model_and_extra_updates_are_ignored).
func TestSTTModelAndExtraUpdatesAreIgnored(t *testing.T) {
	r := runSTT(t, STTConfig{}, newFakeRime())
	r.waitConnected(t)
	delta := &STTSettings{}
	delta.Language = settings.Set("es")
	delta.Model = settings.Set("unknown")
	delta.Extra = map[string]any{"speed": 2}
	r.send(frames.NewSTTUpdateSettingsFrame(delta))
	waitFor(t, "the language was never updated", func() bool {
		return r.c.segmentSettings().Language.Or("") == "es"
	})
	s := r.c.segmentSettings()
	if _, ok := s.Model.Value(); ok {
		t.Errorf("model = %v, want none", s.Model)
	}
	if len(s.Extra) != 0 {
		t.Errorf("extra = %v, want none", s.Extra)
	}
	if len(r.errors()) != 0 {
		t.Errorf("errors = %v, want none", r.errors())
	}
}

// Ported from upstream (test_invalid_settings_update_is_rejected_whole).
func TestSTTInvalidSettingsUpdateIsRejectedWhole(t *testing.T) {
	r := runSTT(t, STTConfig{}, newFakeRime())
	r.waitConnected(t)
	delta := &STTSettings{Mode: settings.Set("invalid")}
	delta.Language = settings.Set("es-MX")
	r.send(frames.NewSTTUpdateSettingsFrame(delta))

	ef := r.waitErrors(t, 1)[0]
	if ef.Category != errclass.Application || ef.Err == nil || !strings.Contains(ef.Err.Error(), "mode") {
		t.Errorf("error = %v (%v), want an application error naming the mode", ef.Err, ef.Category)
	}
	if got := r.c.live.Language.Or(""); got != "en" {
		t.Errorf("language = %q, want the update refused whole", got)
	}
	if !r.c.svc.Usable() {
		t.Error("a refused update left the service unusable")
	}

	r.send(frames.NewSTTUpdateSettingsFrame(&settings.STT{Language: settings.Set("es-MX")}))
	waitFor(t, "a valid update was not applied", func() bool {
		return r.c.segmentSettings().Language.Or("") == "es-MX"
	})
}

// Ported from upstream (test_client_is_created_on_start_and_closed_on_cleanup).
func TestSTTClientIsCreatedOnStartAndClosedOnCleanup(t *testing.T) {
	client := newFakeRime()
	r := runSTT(t, STTConfig{APIKey: "secret"}, client)
	r.waitConnected(t)
	if !slices.Equal(r.apiKeys(), []string{"secret"}) {
		t.Errorf("clients created with %v, want one with the key", r.apiKeys())
	}
	r.task.Cancel(t.Context(), "done")
	waitFor(t, "the client was not closed", client.closed.Load)
}
