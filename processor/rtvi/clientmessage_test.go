package rtvi_test

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/processor/rtvi"
	"github.com/gojargo/jargo/utils/events"
)

// sink records every frame that reaches the end of the pipeline, which is where
// what the RTVI processor injects arrives. Recording from a processor rather
// than a pipeline event keeps the order the pipeline actually delivered.
type sink struct {
	*processor.Base
	frames chan frames.Frame
}

func newSink() *sink {
	s := &sink{frames: make(chan frames.Frame, 64)}
	s.Base = processor.New("Sink", s)
	return s
}

func (s *sink) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := s.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	if dir == processor.Downstream {
		select {
		case s.frames <- f:
		default:
		}
	}
	return s.PushFrame(ctx, f, dir)
}

// client drives the RTVI processor the way a connected client does: it feeds
// messages in and collects what comes back, both the wire messages and the
// frames the processor injected into the pipeline.
type client struct {
	task *pipeline.Worker
	proc *rtvi.Processor
	sink *sink
	out  chan rtvi.Message
	done chan error
	// ended records that the pipeline has already finished, so the cleanup does
	// not go on to stop a worker that stopped itself.
	ended bool
}

func newClient(t *testing.T) *client {
	t.Helper()
	proc := rtvi.NewProcessor()
	sk := newSink()
	out := make(chan rtvi.Message, 16)
	task := pipeline.NewWorker(pipeline.New(proc, sk), pipeline.WorkerConfig{
		Observers:               []pipeline.Observer{rtvi.NewObserver(proc)},
		ReachedDownstreamFilter: pipeline.AnyFrame,
	})
	events.On(&task.Registry, pipeline.EventFrameReachedDownstream, func(_ context.Context, f frames.Frame) {
		if m, ok := f.(*frames.OutputTransportMessageUrgentFrame); ok {
			if msg, ok := m.Message.(rtvi.Message); ok {
				out <- msg
			}
		}
	})

	c := &client{task: task, proc: proc, sink: sk, out: out, done: make(chan error, 1)}
	go func() { c.done <- task.Run(context.Background()) }()
	t.Cleanup(func() {
		if c.ended {
			return
		}
		task.StopWhenDone()
		c.waitForEnd(t)
	})
	return c
}

// waitForEnd waits for the pipeline to finish.
func (c *client) waitForEnd(t *testing.T) {
	t.Helper()
	select {
	case err := <-c.done:
		c.ended = true
		if err != nil {
			t.Errorf("pipeline: %v", err)
		}
	case <-time.After(5 * time.Second):
		t.Error("pipeline did not finish")
	}
}

// send queues one RTVI message from the client.
func (c *client) send(t *testing.T, msg rtvi.Message) {
	t.Helper()
	raw, err := json.Marshal(msg)
	if err != nil {
		t.Fatalf("marshal %s: %v", msg.Type, err)
	}
	c.task.QueueFrame(frames.NewInputTransportMessageFrame(raw))
}

// sendRaw queues bytes the client sent as they stand, for a message that is not
// well-formed enough to build.
func (c *client) sendRaw(raw string) {
	c.task.QueueFrame(frames.NewInputTransportMessageFrame([]byte(raw)))
}

// await returns the next frame of type T to reach the end of the pipeline.
func await[T frames.Frame](t *testing.T, c *client) T {
	t.Helper()
	deadline := time.After(3 * time.Second)
	for {
		select {
		case f := <-c.sink.frames:
			if got, ok := f.(T); ok {
				return got
			}
		case <-deadline:
			var zero T
			t.Fatalf("timed out waiting for a %T", zero)
			return zero
		}
	}
}

// TestClientMessageReachesThePipelineAndIsAnswered checks the custom-message
// channel end to end: what the client asks travels the pipeline as a frame, and
// the answer pushed back reaches the client paired with the request it answers.
func TestClientMessageReachesThePipelineAndIsAnswered(t *testing.T) {
	c := newClient(t)

	heard := make(chan *rtvi.ClientMessageFrame, 1)
	events.On(c.proc.Events(), rtvi.EventClientMessage,
		func(_ context.Context, f *rtvi.ClientMessageFrame) { heard <- f })

	c.send(t, rtvi.Message{
		Label: rtvi.MessageLabel, Type: rtvi.TypeClientMessage, ID: "req-7",
		Data: rtvi.RawClientMessageData{T: "set-theme", D: json.RawMessage(`{"theme":"dark"}`)},
	})

	got := await[*rtvi.ClientMessageFrame](t, c)
	if got.MsgID != "req-7" || got.Type != "set-theme" {
		t.Fatalf("client message = id %q type %q, want req-7 / set-theme", got.MsgID, got.Type)
	}
	if string(got.Data) != `{"theme":"dark"}` {
		t.Errorf("client message data = %s, want the object the client sent", got.Data)
	}

	select {
	case f := <-heard:
		if f.MsgID != "req-7" {
			t.Errorf("the event announced id %q, want req-7", f.MsgID)
		}
	case <-time.After(3 * time.Second):
		t.Error("no on_client_message event")
	}

	// A processor answering it pushes the response back through the pipeline.
	c.task.QueueFrame(rtvi.NewServerResponseFrame(got, map[string]any{"ok": true}))
	msg := waitMessageOfType(t, c.out, rtvi.TypeServerResponse)
	if msg.ID != "req-7" {
		t.Errorf("server-response id = %q, want req-7", msg.ID)
	}
	d, ok := msg.Data.(rtvi.RawServerResponseData)
	if !ok || d.T != "set-theme" {
		t.Fatalf("server-response data = %+v, want the request's own type back", msg.Data)
	}
}

// TestClientMessageCanBeRefused checks a request the bot will not carry out
// comes back as an error-response rather than going unanswered.
func TestClientMessageCanBeRefused(t *testing.T) {
	c := newClient(t)
	c.send(t, rtvi.Message{
		Label: rtvi.MessageLabel, Type: rtvi.TypeClientMessage, ID: "req-8",
		Data: rtvi.RawClientMessageData{T: "launch-missiles"},
	})
	got := await[*rtvi.ClientMessageFrame](t, c)

	c.task.QueueFrame(rtvi.NewServerErrorResponseFrame(got, "not allowed"))
	msg := waitMessageOfType(t, c.out, rtvi.TypeErrorResponse)
	if msg.ID != "req-8" {
		t.Errorf("error-response id = %q, want req-8", msg.ID)
	}
	if d, ok := msg.Data.(rtvi.ErrorResponseData); !ok || d.Error != "not allowed" {
		t.Errorf("error-response data = %+v, want the reason given", msg.Data)
	}
}

// TestServerMessageIsSentUnprompted checks the bot can tell the client something
// no request asked for.
func TestServerMessageIsSentUnprompted(t *testing.T) {
	c := newClient(t)
	c.task.QueueFrame(rtvi.NewServerMessageFrame(map[string]any{"stage": "thinking"}))
	msg := waitMessageOfType(t, c.out, rtvi.TypeServerMessage)
	if msg.ID != "" {
		t.Errorf("server-message id = %q, want none: nothing asked for it", msg.ID)
	}
}

// TestUnsupportedMessageIsRefused checks a message type the bot does not know is
// answered rather than dropped, so a client waiting on a reply is not left
// waiting for one that is never coming.
func TestUnsupportedMessageIsRefused(t *testing.T) {
	c := newClient(t)
	c.send(t, rtvi.Message{Label: rtvi.MessageLabel, Type: "no-such-thing", ID: "req-9"})
	msg := waitMessageOfType(t, c.out, rtvi.TypeErrorResponse)
	if msg.ID != "req-9" {
		t.Errorf("error-response id = %q, want req-9", msg.ID)
	}
}

// TestMalformedMessageIsReported checks a message carrying the RTVI label but
// nothing readable is reported. The id is part of what could not be read, so it
// is the general error rather than a response to one request.
func TestMalformedMessageIsReported(t *testing.T) {
	c := newClient(t)
	c.sendRaw(`{"label":"rtvi-ai","type":42}`)
	msg := waitMessageOfType(t, c.out, rtvi.TypeError)
	if d, ok := msg.Data.(rtvi.ErrorData); !ok || d.Error == "" {
		t.Errorf("error data = %+v, want a reason", msg.Data)
	}
}

// TestNonRTVIMessageIsIgnored checks a message on the channel that was never
// meant for this protocol draws no complaint: a transport is free to use the
// same channel for its own signaling.
func TestNonRTVIMessageIsIgnored(t *testing.T) {
	c := newClient(t)
	c.sendRaw(`{"label":"something-else","type":"ping"}`)
	select {
	case msg := <-c.out:
		t.Errorf("a non-RTVI message drew a %s", msg.Type)
	case <-time.After(500 * time.Millisecond):
	}
}

// TestDisconnectBotEndsTheWorker checks the client can hang up: the pipeline is
// ended gracefully, so what is in flight finishes rather than being cut.
func TestDisconnectBotEndsTheWorker(t *testing.T) {
	c := newClient(t)
	c.send(t, rtvi.Message{Label: rtvi.MessageLabel, Type: rtvi.TypeDisconnectBot, ID: "req-10"})
	await[*frames.EndWorkerFrame](t, c)
	// The frame reaching the pipeline is only half of it: the worker has to act
	// on it, which is what makes disconnect-bot a hang-up rather than a frame
	// that travels the pipeline and is ignored at the end of it.
	c.waitForEnd(t)
}

// TestFunctionCallResultReachesTheConversation checks a tool the client ran on
// the bot's behalf lands as an ordinary result frame, so the aggregator fills in
// the placeholder the call left exactly as it would for a result produced here.
func TestFunctionCallResultReachesTheConversation(t *testing.T) {
	c := newClient(t)
	c.send(t, rtvi.Message{
		Label: rtvi.MessageLabel, Type: rtvi.TypeFunctionCallResult, ID: "req-11",
		Data: rtvi.FunctionCallResultData{
			FunctionName: "get_location",
			ToolCallID:   "call-1",
			Arguments:    json.RawMessage(`{"precision":"city"}`),
			Result:       json.RawMessage(`{"city":"Lyon"}`),
		},
	})

	got := await[*frames.FunctionCallResultFrame](t, c)
	if got.ToolCallID != "call-1" || got.ToolName != "get_location" {
		t.Fatalf("result = call %q tool %q, want call-1 / get_location", got.ToolCallID, got.ToolName)
	}
	if string(got.Args) != `{"precision":"city"}` {
		t.Errorf("args = %s, want the arguments the call was made with", got.Args)
	}
	if got.Result != `{"city":"Lyon"}` {
		t.Errorf("result = %q, want the object's JSON text", got.Result)
	}
}

// TestFunctionCallResultUnwrapsAStringResult checks a result the client sent as
// a JSON string reaches the conversation as that string, not as a quoted one.
func TestFunctionCallResultUnwrapsAStringResult(t *testing.T) {
	c := newClient(t)
	c.send(t, rtvi.Message{
		Label: rtvi.MessageLabel, Type: rtvi.TypeFunctionCallResult, ID: "req-12",
		Data: rtvi.FunctionCallResultData{
			FunctionName: "greet", ToolCallID: "call-2",
			Arguments: json.RawMessage(`{}`), Result: json.RawMessage(`"bonjour"`),
		},
	})
	if got := await[*frames.FunctionCallResultFrame](t, c); got.Result != "bonjour" {
		t.Errorf("result = %q, want the string unquoted", got.Result)
	}
}

// TestRawAudioIsHeard checks a client doing its own capture is heard: the PCM it
// sends over the message channel enters the pipeline as input audio, the way a
// media track's would.
func TestRawAudioIsHeard(t *testing.T) {
	c := newClient(t)
	pcm := []byte{1, 2, 3, 4}
	c.send(t, rtvi.Message{
		Label: rtvi.MessageLabel, Type: rtvi.TypeRawAudio, ID: "req-13",
		Data: rtvi.RawAudioData{
			Base64Audio: base64.StdEncoding.EncodeToString(pcm),
			SampleRate:  16000, NumChannels: 1,
		},
	})

	got := await[*frames.InputAudioRawFrame](t, c)
	if string(got.Audio) != string(pcm) {
		t.Errorf("audio = %v, want the PCM the client sent", got.Audio)
	}
	if got.SampleRate != 16000 || got.NumChannels != 1 {
		t.Errorf("audio = %d Hz %d channels, want 16000/1", got.SampleRate, got.NumChannels)
	}
}

// TestRawAudioBatchIsHeardInOrder checks a batch arrives as one frame per chunk,
// in the order the client captured them.
func TestRawAudioBatchIsHeardInOrder(t *testing.T) {
	c := newClient(t)
	first, second := []byte{1, 1}, []byte{2, 2}
	c.send(t, rtvi.Message{
		Label: rtvi.MessageLabel, Type: rtvi.TypeRawAudioBatch, ID: "req-14",
		Data: rtvi.RawAudioData{
			Base64AudioBatch: []string{
				base64.StdEncoding.EncodeToString(first),
				base64.StdEncoding.EncodeToString(second),
			},
			SampleRate: 16000, NumChannels: 1,
		},
	})

	if got := await[*frames.InputAudioRawFrame](t, c); string(got.Audio) != string(first) {
		t.Errorf("first chunk = %v, want %v", got.Audio, first)
	}
	if got := await[*frames.InputAudioRawFrame](t, c); string(got.Audio) != string(second) {
		t.Errorf("second chunk = %v, want %v", got.Audio, second)
	}
}

// waitMessageOfType returns the next message of the given type, skipping the
// events the pipeline reports along the way.
func waitMessageOfType(t *testing.T, ch <-chan rtvi.Message, msgType string) rtvi.Message {
	t.Helper()
	deadline := time.After(3 * time.Second)
	for {
		select {
		case m := <-ch:
			if m.Type == msgType {
				return m
			}
		case <-deadline:
			t.Fatalf("timed out waiting for a %s message", msgType)
			return rtvi.Message{}
		}
	}
}

// TestSendHelpersAnswerFromOutsideThePipeline checks the direct-call API: a
// listener on EventClientMessage sits outside the pipeline and has no frame to
// push, so it answers through the processor itself. The wire result must be the
// same as pushing a response frame.
func TestSendHelpersAnswerFromOutsideThePipeline(t *testing.T) {
	c := newClient(t)

	answered := make(chan error, 1)
	events.On(c.proc.Events(), rtvi.EventClientMessage,
		func(ctx context.Context, f *rtvi.ClientMessageFrame) {
			answered <- c.proc.SendServerResponse(ctx, f, map[string]any{"ok": true})
		})

	c.send(t, rtvi.Message{
		Label: rtvi.MessageLabel, Type: rtvi.TypeClientMessage, ID: "req-20",
		Data: rtvi.RawClientMessageData{T: "ping"},
	})

	select {
	case err := <-answered:
		if err != nil {
			t.Fatalf("SendServerResponse: %v", err)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("the listener never ran")
	}

	msg := waitMessageOfType(t, c.out, rtvi.TypeServerResponse)
	if msg.ID != "req-20" {
		t.Errorf("server-response id = %q, want req-20", msg.ID)
	}
	if d, ok := msg.Data.(rtvi.RawServerResponseData); !ok || d.T != "ping" {
		t.Errorf("server-response data = %+v, want the request's own type back", msg.Data)
	}
}

// TestSendErrorResponseRefusesFromOutsideThePipeline checks the refusal has the
// same direct-call route as the answer.
func TestSendErrorResponseRefusesFromOutsideThePipeline(t *testing.T) {
	c := newClient(t)

	events.On(c.proc.Events(), rtvi.EventClientMessage,
		func(ctx context.Context, f *rtvi.ClientMessageFrame) {
			_ = c.proc.SendErrorResponse(ctx, f, "no")
		})

	c.send(t, rtvi.Message{
		Label: rtvi.MessageLabel, Type: rtvi.TypeClientMessage, ID: "req-21",
		Data: rtvi.RawClientMessageData{T: "ping"},
	})

	msg := waitMessageOfType(t, c.out, rtvi.TypeErrorResponse)
	if msg.ID != "req-21" {
		t.Errorf("error-response id = %q, want req-21", msg.ID)
	}
	if d, ok := msg.Data.(rtvi.ErrorResponseData); !ok || d.Error != "no" {
		t.Errorf("error-response data = %+v, want the reason given", msg.Data)
	}
}

// TestSendServerMessageFromOutsideThePipeline checks the unprompted message has
// the same direct-call route.
func TestSendServerMessageFromOutsideThePipeline(t *testing.T) {
	c := newClient(t)

	if err := c.proc.SendServerMessage(context.Background(), map[string]any{"stage": "ready"}); err != nil {
		t.Fatalf("SendServerMessage: %v", err)
	}
	if msg := waitMessageOfType(t, c.out, rtvi.TypeServerMessage); msg.ID != "" {
		t.Errorf("server-message id = %q, want none", msg.ID)
	}
}
