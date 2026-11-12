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

// Ported from upstream (test_all_metric_types_accumulate_into_one_message).
// Every kind of metric reaches the client in one message; a token-usage entry
// is the usage itself, the counts the service did not report left out, and an
// STT usage entry carries its value as an object.
func TestAllMetricTypesAccumulateIntoOneMessage(t *testing.T) {
	base := func(processor string) frames.BaseMetricsData { return frames.BaseMetricsData{Processor: processor} }
	d := metricsJSON(t,
		frames.TTFBMetricsData{BaseMetricsData: base("tts"), Value: 200 * time.Millisecond},
		frames.TTFAMetricsData{
			BaseMetricsData: base("tts"), TTFA: 600 * time.Millisecond,
			TTFB: 200 * time.Millisecond, LeadingSilence: 400 * time.Millisecond,
		},
		frames.ProcessingMetricsData{BaseMetricsData: base("llm"), Value: 50 * time.Millisecond},
		frames.LLMUsageMetricsData{BaseMetricsData: base("llm"), Value: frames.LLMTokenUsage{
			PromptTokens: 10, CompletionTokens: 20, TotalTokens: 30,
			InputAudioTokens: new(int64(7)), OutputAudioTokens: new(int64(13)),
		}},
		frames.STTUsageMetricsData{Processor: "soniox_stt", Value: frames.STTUsage{AudioSeconds: 1.5}},
		frames.TTSUsageMetricsData{BaseMetricsData: base("tts"), Value: 42},
	)
	if v := firstEntry(t, d, "ttfb")["value"]; v != 0.2 {
		t.Errorf("ttfb value = %v, want 0.2", v)
	}
	if v := firstEntry(t, d, "ttfa")["leading_silence"]; v != 0.4 {
		t.Errorf("ttfa leading_silence = %v, want 0.4", v)
	}
	if v := firstEntry(t, d, "processing")["value"]; v != 0.05 {
		t.Errorf("processing value = %v, want 0.05", v)
	}
	tokens := firstEntry(t, d, "tokens")
	want := map[string]any{
		"prompt_tokens": 10.0, "completion_tokens": 20.0, "total_tokens": 30.0,
		"input_audio_tokens": 7.0, "output_audio_tokens": 13.0,
	}
	if len(tokens) != len(want) {
		t.Errorf("tokens = %v, want exactly %v", tokens, want)
	}
	for k, v := range want {
		if tokens[k] != v {
			t.Errorf("tokens[%q] = %v, want %v", k, tokens[k], v)
		}
	}
	stt := firstEntry(t, d, "stt_usage")
	if value, ok := stt["value"].(map[string]any); !ok || value["audio_seconds"] != 1.5 {
		t.Errorf("stt_usage value = %v, want {audio_seconds: 1.5}", stt["value"])
	}
	if stt["processor"] != "soniox_stt" {
		t.Errorf("stt_usage processor = %v, want soniox_stt", stt["processor"])
	}
	if v := firstEntry(t, d, "characters")["value"]; v != 42.0 {
		t.Errorf("characters value = %v, want 42", v)
	}
}
