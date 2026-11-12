package rtvi

import (
	"encoding/json"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
)

// metricsJSON renders the metrics message a frame carrying data becomes, as the
// client receives it.
func metricsJSON(t *testing.T, data ...frames.MetricsData) map[string]any {
	t.Helper()
	raw, err := json.Marshal(metricsMessage(frames.NewMetricsFrame(data...)).Data)
	if err != nil {
		t.Fatal(err)
	}
	var out map[string]any
	if err := json.Unmarshal(raw, &out); err != nil {
		t.Fatal(err)
	}
	return out
}

// firstEntry returns the first entry under key, failing if there is none.
func firstEntry(t *testing.T, d map[string]any, key string) map[string]any {
	t.Helper()
	list, ok := d[key].([]any)
	if !ok || len(list) == 0 {
		t.Fatalf("%q = %v, want a list with an entry", key, d[key])
	}
	entry, ok := list[0].(map[string]any)
	if !ok {
		t.Fatalf("%q[0] = %v, want an object", key, list[0])
	}
	return entry
}

// TestTTFATForwardedToClients checks the time to first answer token reaches
// the client under "ttfat", with its breakdown.
func TestTTFATForwardedToClients(t *testing.T) {
	base := frames.BaseMetricsData{Processor: "llm", Model: "m"}
	d := metricsJSON(t, frames.TTFATMetricsData{
		BaseMetricsData: base,
		TTFAT:           1400 * time.Millisecond,
		TTFB:            400 * time.Millisecond,
		ThinkingTime:    time.Second,
	})
	e := firstEntry(t, d, "ttfat")
	if e["processor"] != "llm" || e["model"] != "m" || e["ttfat"] != 1.4 || e["ttfb"] != 0.4 || e["thinking_time"] != 1.0 {
		t.Errorf("ttfat entry = %v, want processor llm, model m, ttfat 1.4, ttfb 0.4, thinking_time 1", e)
	}
}
