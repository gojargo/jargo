package inworld

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"reflect"
	"slices"
	"sync"
	"testing"
	"time"

	ttsadapter "github.com/gojargo/jargo/adapter/tts"
	ttsinworld "github.com/gojargo/jargo/adapter/tts/inworld"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
)

func TestExpressionAdapterFollowsTheModel(t *testing.T) {
	for _, tt := range []struct {
		model   string
		adapter ttsadapter.ExpressionAdapter
	}{
		{"inworld-tts-2", ttsinworld.InworldExpressionAdapter{}},
		{"inworld-tts-2-flash", ttsinworld.InworldFlashExpressionAdapter{}},
		{"inworld-tts-1.5-max", ttsadapter.BaseExpressionAdapter{}},
	} {
		s := &synthesizer{cfg: Config{Model: tt.model}}
		if got := s.ExpressionAdapter(); reflect.TypeOf(got) != reflect.TypeOf(tt.adapter) {
			t.Errorf("%s: adapter %T, want %T", tt.model, got, tt.adapter)
		}
	}
}

func TestEachRequestWritesItsInstructions(t *testing.T) {
	var mu sync.Mutex
	var texts []string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		_ = json.NewDecoder(r.Body).Decode(&body)
		mu.Lock()
		text, _ := body["text"].(string)
		texts = append(texts, text)
		mu.Unlock()
		_, _ = w.Write([]byte(`{"result":{"audioContent":"AAAAAA=="}}` + "\n"))
	}))
	t.Cleanup(srv.Close)

	task := pipeline.NewWorker(pipeline.New(NewTTS(Config{APIKey: "key", URL: srv.URL, Model: "inworld-tts-2"})),
		pipeline.WorkerConfig{Params: pipeline.Params{EnableExpressiveConversation: true}})
	done := make(chan error, 1)
	go func() { done <- task.Run(context.Background()) }()
	task.QueueFrames([]frames.Frame{
		frames.NewLLMFullResponseStartFrame(),
		frames.NewLLMTextFrame("<<sympathetic>> I'm sorry. [[sigh]] Your code is <<slow: A B 7>>."),
		frames.NewLLMTextFrame(" <<neutral>> Anything ^^else^^?"),
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
	want := []string{
		"[sympathetic] I'm sorry. [reset]",
		"[sigh] [sympathetic] Your code is [sympathetic and slow] A B 7. [reset]",
		"Anything ELSE?",
	}
	if !slices.Equal(texts, want) {
		t.Fatalf("requests = %q, want %q", texts, want)
	}
}
