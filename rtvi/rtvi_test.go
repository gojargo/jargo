package rtvi_test

import (
	"context"
	"encoding/json"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/rtvi"
)

func TestBotReadyJSON(t *testing.T) {
	raw, err := json.Marshal(rtvi.BotReady("req-1"))
	if err != nil {
		t.Fatal(err)
	}
	var got map[string]any
	if err := json.Unmarshal(raw, &got); err != nil {
		t.Fatal(err)
	}
	if got["label"] != "rtvi-ai" || got["type"] != "bot-ready" || got["id"] != "req-1" {
		t.Fatalf("unexpected envelope: %s", raw)
	}
	data, _ := got["data"].(map[string]any)
	if data["version"] != "2.0.0" {
		t.Fatalf("version = %v, want 2.0.0: %s", data["version"], raw)
	}
}

func TestUserTranscriptionJSON(t *testing.T) {
	raw, _ := json.Marshal(rtvi.UserTranscription("hello", "user-1", "ts", true))
	var got struct {
		Type string `json:"type"`
		Data struct {
			Text   string `json:"text"`
			UserID string `json:"user_id"`
			Final  bool   `json:"final"`
		} `json:"data"`
	}
	if err := json.Unmarshal(raw, &got); err != nil {
		t.Fatal(err)
	}
	if got.Type != "user-transcription" || got.Data.Text != "hello" || got.Data.UserID != "user-1" || !got.Data.Final {
		t.Fatalf("unexpected user-transcription: %s", raw)
	}
}

// TestProcessorHandshakeAndTranscript drives the RTVI processor in a pipeline:
// a client-ready message must produce a bot-ready, and a TranscriptionFrame must
// produce a user-transcription — both as OutputTransportMessageFrames.
func TestProcessorHandshakeAndTranscript(t *testing.T) {
	out := make(chan rtvi.Message, 8)
	task := pipeline.NewTask(pipeline.New(rtvi.NewProcessor()), pipeline.TaskParams{
		OnReachedDownstream: func(f frames.Frame) {
			if m, ok := f.(*frames.OutputTransportMessageFrame); ok {
				if msg, ok := m.Message.(rtvi.Message); ok {
					out <- msg
				}
			}
		},
	})

	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	// client-ready -> bot-ready
	clientReady, _ := json.Marshal(rtvi.Message{
		Label: rtvi.MessageLabel, Type: rtvi.TypeClientReady, ID: "req-1",
		Data: map[string]any{"version": "2.0.0"},
	})
	task.QueueFrame(frames.NewInputTransportMessageFrame(clientReady))

	got := waitMessage(t, out)
	if got.Type != rtvi.TypeBotReady || got.ID != "req-1" {
		t.Fatalf("expected bot-ready id req-1, got %+v", got)
	}

	// TranscriptionFrame -> user-transcription
	task.QueueFrame(frames.NewTranscriptionFrame("bonjour", "user-1", "ts"))
	got = waitMessage(t, out)
	if got.Type != rtvi.TypeUserTranscription {
		t.Fatalf("expected user-transcription, got %+v", got)
	}
	if d, ok := got.Data.(rtvi.UserTranscriptionData); !ok || d.Text != "bonjour" || !d.Final {
		t.Fatalf("unexpected transcription data: %+v", got.Data)
	}

	task.StopWhenDone()
	<-runDone
}

func waitMessage(t *testing.T, ch <-chan rtvi.Message) rtvi.Message {
	t.Helper()
	select {
	case m := <-ch:
		return m
	case <-time.After(3 * time.Second):
		t.Fatal("timed out waiting for an RTVI message")
		return rtvi.Message{}
	}
}
