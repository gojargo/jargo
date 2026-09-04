package stt

import (
	"context"
	"testing"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/service/settings"
)

// Tests for the model name a service labels its telemetry with. The usage is
// priced against the model, so a service reporting an empty one puts the cost
// and the latency of a call against nothing.

// describingTranscriber is a segmented transcriber that describes itself.
type describingTranscriber struct{ meta Metadata }

func (t *describingTranscriber) Transcribe(context.Context, []byte, int) (string, error) {
	return "", nil
}
func (t *describingTranscriber) Metadata() Metadata { return t.meta }

// describingConnector is a streaming connector that describes itself.
type describingConnector struct{ meta Metadata }

func (c *describingConnector) Connect(ctx context.Context, _ int) (Stream, error) {
	return &settingsStream{ctx: ctx}, nil
}
func (c *describingConnector) Metadata() Metadata { return c.meta }

// TestSegmentServiceTakesTheModelFromItsTranscriber checks the model a
// segmented service labels with comes from what the provider describes.
func TestSegmentServiceTakesTheModelFromItsTranscriber(t *testing.T) {
	tr := &describingTranscriber{meta: Metadata{Model: "whisper-large-v3-turbo"}}
	if s := NewSegment("TestSTT", tr, 0); s.model != "whisper-large-v3-turbo" {
		t.Errorf("model = %q, want the transcriber's", s.model)
	}
}

// TestStreamServiceTakesTheModelFromItsConnector checks the same for a
// streaming service.
func TestStreamServiceTakesTheModelFromItsConnector(t *testing.T) {
	c := &describingConnector{meta: Metadata{Model: "nova-3"}}
	if s := NewStream("TestSTT", c, 0); s.modelName() != "nova-3" {
		t.Errorf("model = %q, want the connector's", s.modelName())
	}
}

// TestServiceWithoutADescriberHasNoModel checks a provider that describes
// nothing leaves the label empty rather than inventing one.
func TestServiceWithoutADescriberHasNoModel(t *testing.T) {
	if s := NewStream("TestSTT", &plainConnector{}, 0); s.modelName() != "" {
		t.Errorf("model = %q, want it empty for a provider that describes nothing", s.modelName())
	}
}

// settingsProvider is a connector that holds a settings store, the way a
// provider whose model or language can change mid-call does.
type settingsProvider struct {
	describingConnector
	store STTSettings
}

func (c *settingsProvider) Settings() any { return &c.store }

// TestModelFollowsASettingsChange checks the relabeling a mid-call model change
// triggers: what follows is labeled with the model now in force, or its cost
// lands against the one it is no longer using.
func TestModelFollowsASettingsChange(t *testing.T) {
	c := &settingsProvider{meta: Metadata{Model: "old-model"}}
	if err := settings.SetNamed(&c.store, "model", "old-model"); err != nil {
		t.Fatal(err)
	}
	s := NewStream("TestSTT", c, 0)
	if s.modelName() != "old-model" {
		t.Fatalf("model = %q, want the connector's", s.modelName())
	}

	delta := &STTSettings{}
	if err := settings.SetNamed(delta, "model", "new-model"); err != nil {
		t.Fatal(err)
	}
	if _, err := s.set.apply(context.Background(), frames.NewSTTUpdateSettingsFrame(delta)); err != nil {
		t.Fatalf("apply: %v", err)
	}

	if s.modelName() != "new-model" {
		t.Errorf("model = %q after the change, want the one now in force", s.modelName())
	}
	if v, _ := settings.Get(&c.store, "model"); v != "new-model" {
		t.Errorf("the store holds %v, want the new model", v)
	}
}

// TestModelIsLeftAloneWhenSomethingElseChanges checks a change to another
// setting does not disturb the label.
func TestModelIsLeftAloneWhenSomethingElseChanges(t *testing.T) {
	c := &settingsProvider{meta: Metadata{Model: "kept"}}
	s := NewStream("TestSTT", c, 0)

	delta := &STTSettings{}
	if err := settings.SetNamed(delta, "language", "fr"); err != nil {
		t.Fatal(err)
	}
	if _, err := s.set.apply(context.Background(), frames.NewSTTUpdateSettingsFrame(delta)); err != nil {
		t.Fatalf("apply: %v", err)
	}

	if s.modelName() != "kept" {
		t.Errorf("model = %q, want it left alone", s.modelName())
	}
}

// TestApplyIgnoresAProviderWithoutSettings checks an update aimed at a provider
// that holds none is reported rather than panicking.
func TestApplyIgnoresAProviderWithoutSettings(t *testing.T) {
	s := NewStream("TestSTT", &plainConnector{}, 0)

	delta := &STTSettings{}
	if err := settings.SetNamed(delta, "model", "whatever"); err != nil {
		t.Fatal(err)
	}
	reopen, err := s.set.apply(context.Background(), frames.NewSTTUpdateSettingsFrame(delta))
	if err != nil || reopen {
		t.Errorf("apply = (%v, %v), want it to do nothing quietly", reopen, err)
	}
}
