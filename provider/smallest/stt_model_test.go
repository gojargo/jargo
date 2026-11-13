package smallest

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/service/stt"
)

// TestSTTMetadataReportsTheModel checks the connector names the model it is
// transcribing with, which is what labels the service's metrics and spans.
func TestSTTMetadataReportsTheModel(t *testing.T) {
	c := &connector{cfg: STTConfig{Model: "pulse-custom"}}
	if got := c.Metadata().Model; got != "pulse-custom" {
		t.Errorf("Model = %q, want the configured model", got)
	}
	def := &connector{cfg: STTConfig{Model: defaultSTTModel}}
	if got := def.Metadata().Model; got != defaultSTTModel {
		t.Errorf("Model = %q, want the default %q", got, defaultSTTModel)
	}
	if *c.Metadata().TTFSP99 != stt.SmallestTTFSP99 {
		t.Errorf("TTFSP99 = %v, want the measured figure", *c.Metadata().TTFSP99)
	}
}

// Ported from upstream (test_default_model_is_pulse). A service built without a
// model opens its session on Pulse.
func TestTheDefaultModelIsPulse(t *testing.T) {
	models := make(chan string, 4)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		select {
		case models <- r.URL.Query().Get("model"):
		default:
		}
		c, err := websocket.Accept(w, r, nil)
		if err != nil {
			return
		}
		defer func() { _ = c.CloseNow() }()
		for {
			if _, _, err := c.Read(r.Context()); err != nil {
				return
			}
		}
	}))
	t.Cleanup(srv.Close)

	svc := NewSTT(STTConfig{APIKey: "k", BaseURL: wsURL(srv.URL), SampleRate: 16000})
	task := pipeline.NewWorker(pipeline.New(svc), pipeline.WorkerConfig{IdleTimeout: -1})
	done := make(chan error, 1)
	go func() { done <- task.Run(context.Background()) }()
	t.Cleanup(func() {
		task.StopWhenDone()
		select {
		case <-done:
		case <-time.After(5 * time.Second):
			t.Error("the pipeline did not shut down")
		}
	})

	select {
	case model := <-models:
		if model != "pulse" {
			t.Errorf("model = %q, want pulse", model)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("no session was opened")
	}
}
