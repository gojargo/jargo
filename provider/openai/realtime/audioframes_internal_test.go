package realtime

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"sync"
	"testing"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/provider/openai/realtime/events"
)

// Ports of tests/test_openai_realtime_audio_frames.py: a response whose audio
// spans several output items is one spoken turn, and an interruption truncates
// the item being heard.

// recorder is a processor recording the frames a service pushes to it.
type recorder struct {
	*processor.Base
	mu  sync.Mutex
	got []frames.Frame
}

func newRecorder() *recorder {
	r := &recorder{}
	r.Base = processor.New("Recorder", r, processor.WithDirectMode())
	return r
}

func (r *recorder) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := r.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	r.mu.Lock()
	r.got = append(r.got, f)
	r.mu.Unlock()
	return nil
}

// spoken are the frames of a spoken turn the service pushed, in order, as
// "started", "audio" and "stopped".
func (r *recorder) spoken() []string {
	r.mu.Lock()
	defer r.mu.Unlock()
	var out []string
	for _, f := range r.got {
		switch f.(type) {
		case *frames.TTSStartedFrame:
			out = append(out, "started")
		case *frames.TTSAudioRawFrame:
			out = append(out, "audio")
		case *frames.TTSStoppedFrame:
			out = append(out, "stopped")
		}
	}
	return out
}

// framedService is an offline service whose pushed frames are recorded.
func framedService(t *testing.T) (*Service, *recorder, *sentEvents) {
	t.Helper()
	s, sent := offlineService(t, Config{})
	r := newRecorder()
	s.Link(r)
	return s, r, sent
}

// drive feeds server events through the service's handler.
func drive(t *testing.T, s *Service, evs ...map[string]any) {
	t.Helper()
	for _, e := range evs {
		raw, _ := json.Marshal(e)
		ev, err := events.ParseServerEvent(raw)
		if err != nil || ev == nil {
			t.Fatalf("event %s: %v", raw, err)
		}
		s.handleEvent(context.Background(), ev)
	}
}

func audioDelta(responseID, itemID string, outputIndex int) map[string]any {
	return map[string]any{
		"type": "response.output_audio.delta", "event_id": "evt_delta_" + itemID,
		"response_id": responseID, "item_id": itemID, "output_index": outputIndex, "content_index": 0,
		"delta": base64.StdEncoding.EncodeToString(make([]byte, 480)),
	}
}

func audioDone(responseID, itemID string, outputIndex int) map[string]any {
	return map[string]any{
		"type": "response.output_audio.done", "event_id": "evt_done_" + itemID,
		"response_id": responseID, "item_id": itemID, "output_index": outputIndex, "content_index": 0,
	}
}

func responseDone(responseID string) map[string]any {
	return map[string]any{
		"type": "response.done", "event_id": "evt_response_done_" + responseID,
		"response": map[string]any{
			"id": responseID, "object": "realtime.response", "status": "completed", "output": []any{},
			"usage": map[string]any{
				"total_tokens": 10, "input_tokens": 5, "output_tokens": 5,
				"input_token_details":  map[string]any{"cached_tokens": 0, "text_tokens": 5, "audio_tokens": 0},
				"output_token_details": map[string]any{"text_tokens": 5, "audio_tokens": 0},
			},
		},
	}
}

func equal(got []string, want ...string) bool {
	if len(got) != len(want) {
		return false
	}
	for i := range want {
		if got[i] != want[i] {
			return false
		}
	}
	return true
}

func (s *Service) trackedItem() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.audioResp == nil {
		return ""
	}
	return s.audioResp.itemID
}

// truncates are the truncation events the service sent.
func (r *sentEvents) truncates() []events.ConversationItemTruncateEvent {
	r.mu.Lock()
	defer r.mu.Unlock()
	var out []events.ConversationItemTruncateEvent
	for _, e := range r.events {
		if tr, ok := e.(events.ConversationItemTruncateEvent); ok {
			out = append(out, tr)
		}
	}
	return out
}

func TestSingleItemResponseUnchanged(t *testing.T) {
	s, r, _ := framedService(t)
	drive(t, s,
		audioDelta("resp_single", "item_A", 0), audioDelta("resp_single", "item_A", 0),
		audioDone("resp_single", "item_A", 0), responseDone("resp_single"))
	if got := r.spoken(); !equal(got, "started", "audio", "audio", "stopped") {
		t.Errorf("frames = %v, want one started/stopped pair around the audio", got)
	}
	if item := s.trackedItem(); item != "item_A" {
		t.Errorf("tracked item = %q, want item_A", item)
	}
}

func TestInterruptDuringSingleItemTruncatesAgainstThatItem(t *testing.T) {
	s, _, sent := framedService(t)
	drive(t, s, audioDelta("resp", "item_A", 0), audioDelta("resp", "item_A", 0))
	s.truncateAudioResponse(context.Background())
	tr := sent.truncates()
	if len(tr) != 1 || tr[0].ItemID != "item_A" || tr[0].ContentIndex != 0 || tr[0].AudioEndMs < 0 {
		t.Errorf("truncates = %+v, want one against item_A", tr)
	}
}

func TestMultiItemResponseOverlappingEmitsSingleTTSTurn(t *testing.T) {
	s, r, _ := framedService(t)
	drive(t, s,
		audioDelta("resp", "item_A", 0), audioDelta("resp", "item_A", 0),
		audioDelta("resp", "item_B", 1), audioDelta("resp", "item_B", 1),
		audioDone("resp", "item_A", 0), audioDone("resp", "item_B", 1), responseDone("resp"))
	if got := r.spoken(); !equal(got, "started", "audio", "audio", "audio", "audio", "stopped") {
		t.Errorf("frames = %v, want one turn around all four chunks", got)
	}
	if item := s.trackedItem(); item != "item_B" {
		t.Errorf("tracked item = %q, want item_B", item)
	}
}

func TestMultiItemResponseNonOverlappingEmitsSingleTTSTurn(t *testing.T) {
	s, r, _ := framedService(t)
	drive(t, s,
		audioDelta("resp", "item_A", 0), audioDelta("resp", "item_A", 0), audioDone("resp", "item_A", 0),
		audioDelta("resp", "item_B", 1), audioDelta("resp", "item_B", 1), audioDone("resp", "item_B", 1),
		responseDone("resp"))
	if got := r.spoken(); !equal(got, "started", "audio", "audio", "audio", "audio", "stopped") {
		t.Errorf("frames = %v, want one turn around all four chunks", got)
	}
	if item := s.trackedItem(); item != "item_B" {
		t.Errorf("tracked item = %q, want item_B", item)
	}
}

func TestInterruptDuringSecondItemTruncatesAgainstSecondItem(t *testing.T) {
	s, _, sent := framedService(t)
	drive(t, s,
		audioDelta("resp", "item_A", 0), audioDelta("resp", "item_A", 0),
		audioDelta("resp", "item_B", 1), audioDelta("resp", "item_B", 1))
	s.truncateAudioResponse(context.Background())
	tr := sent.truncates()
	if len(tr) != 1 || tr[0].ItemID != "item_B" || tr[0].ContentIndex != 0 {
		t.Errorf("truncates = %+v, want one against the latest item", tr)
	}
}

// Ports of the service cases of tests/test_openai_realtime_usage_metrics.py.
// (The span-attribute cases are covered where jargo records usage on spans,
// in telemetry/tracing.)

func TestFailedResponseWithoutStatusDetailsPushesError(t *testing.T) {
	s, _, _ := framedService(t)
	src := newRecorder()
	src.Link(s)
	drive(t, s, map[string]any{
		"type": "response.done", "event_id": "ev", "response": map[string]any{"status": "failed"},
	})
	src.mu.Lock()
	defer src.mu.Unlock()
	var msgs []string
	for _, f := range src.got {
		if e, ok := f.(*frames.ErrorFrame); ok {
			msgs = append(msgs, e.Error)
		}
	}
	if len(msgs) != 1 || msgs[0] != "Response failed" {
		t.Errorf("errors = %q, want Response failed", msgs)
	}
}

func TestResponseDoneWithoutUsageReportsNoUsage(t *testing.T) {
	s, r, _ := framedService(t)
	drive(t, s, map[string]any{
		"type": "response.done", "event_id": "ev", "response": map[string]any{"status": "completed"},
	})
	r.mu.Lock()
	defer r.mu.Unlock()
	for _, f := range r.got {
		if m, ok := f.(*frames.MetricsFrame); ok {
			for _, d := range m.Data {
				if _, ok := d.(frames.LLMUsageMetricsData); ok {
					t.Error("usage was reported for a response that carried none")
				}
			}
		}
	}
}

func TestCachedAudioTokensAreReported(t *testing.T) {
	five, three := int64(5), int64(3)
	got := tokenUsage(&events.Usage{
		InputTokens: &five, OutputTokens: &five,
		InputTokenDetails: &events.TokenDetails{
			CachedTokens:        &five,
			CachedTokensDetails: &events.CachedTokensDetails{AudioTokens: &three},
		},
	})
	if got.CacheReadAudioTokens == nil || *got.CacheReadAudioTokens != 3 {
		t.Errorf("cached audio tokens = %v, want 3", got.CacheReadAudioTokens)
	}
}
