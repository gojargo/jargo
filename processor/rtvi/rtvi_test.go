package rtvi_test

import (
	"context"
	"encoding/json"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor/aggregators"
	"github.com/gojargo/jargo/processor/rtvi"
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
// produce a user-transcription — both as OutputTransportMessageUrgentFrames.
func TestProcessorHandshakeAndTranscript(t *testing.T) {
	out := make(chan rtvi.Message, 8)
	proc := rtvi.NewProcessor()
	task := pipeline.NewTask(pipeline.New(proc), pipeline.TaskParams{
		// Events are reported by the observer; the processor only carries them.
		Observers: []pipeline.Observer{rtvi.NewObserver(proc)},
		OnReachedDownstream: func(f frames.Frame) {
			if m, ok := f.(*frames.OutputTransportMessageUrgentFrame); ok {
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

// TestProcessorLifecycleAndFunctionCalls drives the LLM/TTS lifecycle frames and
// function-call frames through the RTVI processor and checks each maps to its
// wire message.
func TestProcessorLifecycleAndFunctionCalls(t *testing.T) {
	out := make(chan rtvi.Message, 16)
	proc := rtvi.NewProcessor()
	task := pipeline.NewTask(pipeline.New(proc), pipeline.TaskParams{
		// Events are reported by the observer; the processor only carries them.
		Observers: []pipeline.Observer{rtvi.NewObserver(proc)},
		OnReachedDownstream: func(f frames.Frame) {
			if m, ok := f.(*frames.OutputTransportMessageUrgentFrame); ok {
				if msg, ok := m.Message.(rtvi.Message); ok {
					out <- msg
				}
			}
		},
	})

	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	task.QueueFrame(frames.NewLLMFullResponseStartFrame())
	if got := waitMessage(t, out); got.Type != rtvi.TypeBotLLMStarted {
		t.Fatalf("expected bot-llm-started, got %+v", got)
	}
	task.QueueFrame(frames.NewLLMFullResponseEndFrame())
	if got := waitMessage(t, out); got.Type != rtvi.TypeBotLLMStopped {
		t.Fatalf("expected bot-llm-stopped, got %+v", got)
	}

	task.QueueFrame(frames.NewTTSStartedFrame())
	if got := waitMessage(t, out); got.Type != rtvi.TypeBotTTSStarted {
		t.Fatalf("expected bot-tts-started, got %+v", got)
	}
	task.QueueFrame(frames.NewTTSStoppedFrame())
	if got := waitMessage(t, out); got.Type != rtvi.TypeBotTTSStopped {
		t.Fatalf("expected bot-tts-stopped, got %+v", got)
	}

	task.QueueFrame(frames.NewFunctionCallInProgressFrame("call-1", "get_weather", nil, true, "g1"))
	got := waitMessage(t, out)
	if got.Type != rtvi.TypeLLMFunctionCall {
		t.Fatalf("expected llm-function-call-in-progress, got %+v", got)
	}
	if d, ok := got.Data.(rtvi.LLMFunctionCallData); !ok || d.FunctionName != "get_weather" || d.ToolCallID != "call-1" {
		t.Fatalf("unexpected function-call data: %+v", got.Data)
	}

	task.QueueFrame(frames.NewFunctionCallResultFrame("call-1", "get_weather", nil, "sunny, 24C"))
	got = waitMessage(t, out)
	if got.Type != rtvi.TypeLLMFunctionCallResult {
		t.Fatalf("expected llm-function-call-result, got %+v", got)
	}
	if d, ok := got.Data.(rtvi.LLMFunctionCallResultData); !ok || d.Result != "sunny, 24C" {
		t.Fatalf("unexpected function-call-result data: %+v", got.Data)
	}

	task.StopWhenDone()
	<-runDone
}

// TestProcessorSendTextInjectsUserTurn verifies an inbound send-text message
// appends a user message to the shared context and triggers an LLM run, even
// though the RTVI processor sits downstream of the aggregator (the injection is
// pushed upstream).
func TestProcessorSendTextInjectsUserTurn(t *testing.T) {
	convo := frames.NewLLMContext("system")
	pair := aggregators.New(convo)

	got := make(chan *frames.LLMContextFrame, 1)
	task := pipeline.NewTask(pipeline.New(pair.User(), rtvi.NewProcessor()), pipeline.TaskParams{
		OnReachedDownstream: func(f frames.Frame) {
			if cf, ok := f.(*frames.LLMContextFrame); ok {
				select {
				case got <- cf:
				default:
				}
			}
		},
	})

	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	sendText, _ := json.Marshal(rtvi.Message{
		Label: rtvi.MessageLabel, Type: rtvi.TypeSendText, ID: "req-1",
		Data: map[string]any{
			"content": "what time is it",
			"options": map[string]any{"run_immediately": true, "audio_response": false},
		},
	})
	task.QueueFrame(frames.NewInputTransportMessageFrame(sendText))

	select {
	case cf := <-got:
		msgs := cf.Context.Messages()
		if len(msgs) == 0 {
			t.Fatal("send-text produced an empty context")
		}
		last := msgs[len(msgs)-1]
		if last.Role != frames.RoleUser || last.Text != "what time is it" {
			t.Fatalf("expected appended user message, got %+v", msgs)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("send-text did not trigger an LLM run")
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
