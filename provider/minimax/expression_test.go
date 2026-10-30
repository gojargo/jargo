package minimax

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"reflect"
	"sync"
	"testing"
	"time"

	ttsadapter "github.com/gojargo/jargo/adapter/tts"
	ttsminimax "github.com/gojargo/jargo/adapter/tts/minimax"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
)

func TestExpressionAdapterFollowsTheModel(t *testing.T) {
	for _, tt := range []struct {
		model   string
		adapter ttsadapter.ExpressionAdapter
	}{
		{"speech-2.8-turbo", ttsminimax.MiniMaxExpressionAdapter{}},
		{"speech-2.8-hd", ttsminimax.MiniMaxExpressionAdapter{}},
		{"speech-2.6-turbo", ttsadapter.BaseExpressionAdapter{}},
	} {
		s := &synthesizer{cfg: Config{Model: tt.model}}
		if got := s.ExpressionAdapter(); reflect.TypeOf(got) != reflect.TypeOf(tt.adapter) {
			t.Errorf("%s: adapter %T, want %T", tt.model, got, tt.adapter)
		}
	}
}

// recordedRequests runs a reply through a MiniMax service in an expressive
// conversation and returns each request's text and voice settings.
func recordedRequests(t *testing.T, cfg Config, reply string) []map[string]any {
	t.Helper()
	var mu sync.Mutex
	var payloads []map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		_ = json.NewDecoder(r.Body).Decode(&body)
		mu.Lock()
		payloads = append(payloads, body)
		mu.Unlock()
		_, _ = w.Write([]byte("data: {\"data\":{\"audio\":\"00000000\",\"status\":1}}\n\n"))
	}))
	t.Cleanup(srv.Close)
	cfg.APIKey, cfg.GroupID, cfg.BaseURL = "key", "group", srv.URL

	task := pipeline.NewWorker(pipeline.New(NewTTS(cfg)), pipeline.WorkerConfig{
		Params: pipeline.Params{EnableExpressiveConversation: true},
	})
	done := make(chan error, 1)
	go func() { done <- task.Run(context.Background()) }()
	task.QueueFrames([]frames.Frame{
		frames.NewLLMFullResponseStartFrame(),
		frames.NewLLMTextFrame(reply),
		frames.NewLLMFullResponseEndFrame(),
	})
	time.Sleep(200 * time.Millisecond)
	task.StopWhenDone()
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("the pipeline did not finish")
	}
	mu.Lock()
	defer mu.Unlock()
	return payloads
}

func TestEachRequestSendsTheEmotionOfItsDelivery(t *testing.T) {
	payloads := recordedRequests(t, Config{},
		"<<happy>> Great news. [[laughs]] It's back. <<neutral>> Anything else?")

	type request struct {
		text    string
		emotion any
	}
	var got []request
	for _, p := range payloads {
		voice, _ := p["voice_setting"].(map[string]any)
		text, _ := p["text"].(string)
		got = append(got, request{text, voice["emotion"]})
	}
	// The baseline sends no emotion of its own.
	want := []request{{"Great news.", "happy"}, {"(laughs)It's back.", "happy"}, {"Anything else?", nil}}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("requests = %+v, want %+v", got, want)
	}
}

func TestEachRequestSendsTheSpeedOfItsDelivery(t *testing.T) {
	speed := 1.0
	payloads := recordedRequests(t, Config{Speed: &speed}, "<<calm>> It's <<slow: A B 7 4>>. Anything else?")

	type request struct {
		text    string
		emotion any
		speed   any
	}
	var got []request
	for _, p := range payloads {
		voice, _ := p["voice_setting"].(map[string]any)
		text, _ := p["text"].(string)
		got = append(got, request{text, voice["emotion"], voice["speed"]})
	}
	want := []request{{"It's A B 7 4.", "calm", 0.75}, {"Anything else?", "calm", 1.0}}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("requests = %+v, want %+v", got, want)
	}
}
