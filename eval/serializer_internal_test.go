package eval

import (
	"encoding/json"
	"strings"
	"testing"
)

// The harness asks for the model's markers under the llm_markers key the bot's
// eval serializer reads.
func TestConfigureMessageAsksForLLMMarkers(t *testing.T) {
	on := true
	sent, err := json.Marshal(configureMessage(nil, nil, &on))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(sent), `"llm_markers":true`) {
		t.Fatalf("configure message %s, want the markers asked for as llm_markers", sent)
	}
}
