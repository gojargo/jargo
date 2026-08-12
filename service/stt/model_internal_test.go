package stt

import (
	"context"
	"testing"

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

// TestSyncModelRelabelsFromTheSettingsStore checks the relabeling a mid-call
// model change triggers: what follows the change is labeled with the model now
// in force, or its cost lands against the one it is no longer using.
func TestSyncModelRelabelsFromTheSettingsStore(t *testing.T) {
	s := NewStream("TestSTT", &describingConnector{meta: Metadata{Model: "old-model"}}, 0)
	if s.modelName() != "old-model" {
		t.Fatalf("model = %q, want the connector's", s.modelName())
	}

	store := &STTSettings{}
	if err := settings.SetNamed(store, "model", "new-model"); err != nil {
		t.Fatalf("SetNamed: %v", err)
	}
	s.syncModel(store)

	if s.modelName() != "new-model" {
		t.Errorf("model = %q after the change, want the one now in force", s.modelName())
	}
}

// TestSyncModelFollowsAClearedModel checks a model cleared mid-call clears the
// label too. Naming no model is a request in its own right, and the label has to
// follow the store rather than keep reporting a model no longer in use.
func TestSyncModelFollowsAClearedModel(t *testing.T) {
	s := NewStream("TestSTT", &describingConnector{meta: Metadata{Model: "old-model"}}, 0)

	store := &STTSettings{}
	if err := settings.SetNamed(store, "model", nil); err != nil {
		t.Fatalf("SetNamed: %v", err)
	}
	s.syncModel(store)

	if s.modelName() != "" {
		t.Errorf("model = %q after it was cleared, want it empty", s.modelName())
	}
}
