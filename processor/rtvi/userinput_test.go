package rtvi_test

import (
	"encoding/json"
	"fmt"
	"slices"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/processor/rtvi"
)

// pushedBy records the frames one processor pushes, in the order it pushes
// them, before any queue between processors can reorder them.
type pushedBy struct {
	source processor.Processor

	mu     sync.Mutex
	pushed []frames.Frame
}

func (o *pushedBy) OnPushFrame(data processor.FramePushed) {
	if data.Source != o.source {
		return
	}
	o.mu.Lock()
	o.pushed = append(o.pushed, data.Frame)
	o.mu.Unlock()
}

// sendTextPushes runs a send-text message through a processor and returns the
// frames it pushes for it, ending with the SendTextFrame or the frame after it.
func sendTextPushes(t *testing.T, msg *frames.InputTransportMessageFrame, want int) []frames.Frame {
	t.Helper()
	proc := rtvi.NewProcessor()
	rec := &pushedBy{source: proc}
	off := false
	task := pipeline.NewWorker(pipeline.New(proc), pipeline.WorkerConfig{
		EnableRTVI: &off,
		Observers:  []processor.Observer{rec},
	})
	done := make(chan error, 1)
	go func() { done <- task.Run(t.Context()) }()
	defer func() {
		task.StopWhenDone()
		<-done
	}()
	task.QueueFrame(msg)

	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		if got := relevant(rec); len(got) >= want {
			return got
		}
		time.Sleep(5 * time.Millisecond)
	}
	t.Fatalf("pushed %v, want %d frames for the send-text", relevant(rec), want)
	return nil
}

// relevant is what the processor pushed for the send-text, leaving out the
// frames every pipeline carries.
func relevant(rec *pushedBy) []frames.Frame {
	rec.mu.Lock()
	defer rec.mu.Unlock()
	var out []frames.Frame
	for _, f := range rec.pushed {
		switch f.(type) {
		case *frames.LLMMessagesAppendFrame, *frames.LLMConfigureOutputFrame, *rtvi.SendTextFrame:
			out = append(out, f)
		}
	}
	return out
}

func sendTextMessage(t *testing.T, audio bool) *frames.InputTransportMessageFrame {
	t.Helper()
	runNow := false
	encoded, err := json.Marshal(rtvi.Message{
		Label: rtvi.MessageLabel, Type: rtvi.TypeSendText, ID: "msg-1",
		Data: rtvi.SendTextData{
			Content: "Hello.",
			Options: &rtvi.SendTextOptions{RunImmediately: &runNow, AudioResponse: &audio},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	return frames.NewInputTransportMessageFrame(encoded)
}

func frameNames(fs []frames.Frame) []string {
	out := make([]string, 0, len(fs))
	for _, f := range fs {
		out = append(out, fmt.Sprintf("%T", f))
	}
	return out
}

func TestTheTextIsAcknowledgedOnceItIsAppended(t *testing.T) {
	got := sendTextPushes(t, sendTextMessage(t, true), 2)

	want := []string{"*frames.LLMMessagesAppendFrame", "*rtvi.SendTextFrame"}
	if names := frameNames(got); !slices.Equal(names, want) {
		t.Fatalf("pushed %v, want %v", names, want)
	}
	if f, ok := got[1].(*rtvi.SendTextFrame); !ok || f.MsgID != "msg-1" || f.Text != "Hello." {
		t.Errorf("acknowledged %+v, want msg-1 carrying the text", f)
	}
}

func TestTheTextIsAcknowledgedWithTheSenderTheTransportNames(t *testing.T) {
	msg := sendTextMessage(t, true)
	msg.ParticipantID = "user"
	got := sendTextPushes(t, msg, 2)

	f, ok := got[len(got)-1].(*rtvi.SendTextFrame)
	if !ok || f.MsgID != "msg-1" || f.UserID != "user" {
		t.Errorf("acknowledged %v, want msg-1 from user", got[len(got)-1])
	}
}

func TestTheTextIsAcknowledgedWithoutASenderTheTransportDoesNotName(t *testing.T) {
	got := sendTextPushes(t, sendTextMessage(t, true), 2)

	f, ok := got[len(got)-1].(*rtvi.SendTextFrame)
	if !ok || f.MsgID != "msg-1" || f.UserID != "" {
		t.Errorf("acknowledged %v, want msg-1 from nobody named", got[len(got)-1])
	}
}

func TestTheTextIsAcknowledgedBeforeTheTTSSettingIsRestored(t *testing.T) {
	got := sendTextPushes(t, sendTextMessage(t, false), 4)

	want := []string{
		"*frames.LLMConfigureOutputFrame", "*frames.LLMMessagesAppendFrame",
		"*rtvi.SendTextFrame", "*frames.LLMConfigureOutputFrame",
	}
	if names := frameNames(got); !slices.Equal(names, want) {
		t.Fatalf("pushed %v, want %v", names, want)
	}
}
