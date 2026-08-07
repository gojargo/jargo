package rtvi_test

import (
	"context"
	"encoding/json"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/processor/aggregators"
	"github.com/gojargo/jargo/processor/rtvi"
)

// openEndedLLM answers with text but never ends the response, leaving the
// assistant aggregator holding uncommitted text. That is the state a bot is in
// mid-answer, and what an interruption has to commit.
type openEndedLLM struct {
	*processor.Base
	// answering is closed once the answer's text has been pushed.
	answering chan struct{}
}

func newOpenEndedLLM() *openEndedLLM {
	l := &openEndedLLM{answering: make(chan struct{})}
	l.Base = processor.New("OpenEndedLLM", l)
	return l
}

func (l *openEndedLLM) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := l.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	if _, ok := f.(*frames.LLMContextFrame); !ok {
		return l.PushFrame(ctx, f, dir)
	}

	if err := l.PushFrame(ctx, frames.NewLLMFullResponseStartFrame(), processor.Downstream); err != nil {
		return err
	}
	if err := l.PushFrame(ctx, frames.NewLLMTextFrame("I was still talking"), processor.Downstream); err != nil {
		return err
	}
	select {
	case <-l.answering:
	default:
		close(l.answering)
	}
	return nil // no end frame: the answer stays open
}

// sendText builds the transport frame carrying an RTVI send-text message.
func sendTextFrame(t *testing.T, content string) frames.Frame {
	t.Helper()
	runNow, noAudio := true, false
	msg := rtvi.Message{
		Label: rtvi.MessageLabel, Type: rtvi.TypeSendText, ID: "test",
		Data: rtvi.SendTextData{
			Content: content,
			Options: &rtvi.SendTextOptions{RunImmediately: &runNow, AudioResponse: &noAudio},
		},
	}
	encoded, err := json.Marshal(msg)
	if err != nil {
		t.Fatal(err)
	}
	return frames.NewInputTransportMessageFrame(encoded)
}

// TestSendTextRunImmediatelyCommitsBeforeAppending checks that a turn sent while
// the bot is still answering lands in the context *after* what the bot was
// saying, not before it.
//
// The interruption commits the in-progress answer, and the new user message is
// appended by a different processor, reached the other way round the pipeline.
// Without waiting for the pipeline to settle in between, the append can overtake
// the commit and the model sees the two the wrong way round.
func TestSendTextRunImmediatelyCommitsBeforeAppending(t *testing.T) {
	convo := frames.NewLLMContext("system")
	agg := aggregators.New(convo)
	proc := rtvi.NewProcessor()
	llm := newOpenEndedLLM()

	task := pipeline.NewTask(
		pipeline.New(agg.User(), llm, proc, agg.Assistant()),
		pipeline.TaskParams{},
	)
	done := make(chan error, 1)
	go func() { done <- task.Run(t.Context()) }()

	// The first turn gets the bot talking.
	task.QueueFrame(sendTextFrame(t, "tell me about Paris"))
	select {
	case <-llm.answering:
	case <-time.After(5 * time.Second):
		t.Fatal("the bot never started answering")
	}

	// The second turn arrives mid-answer, so it interrupts.
	task.QueueFrame(sendTextFrame(t, "actually, never mind"))

	// Wait for the interrupted answer and the new turn to both be in.
	deadline := time.Now().Add(5 * time.Second)
	for len(texts(convo)) < 3 && time.Now().Before(deadline) {
		time.Sleep(10 * time.Millisecond)
	}
	task.StopWhenDone()
	if err := <-done; err != nil {
		t.Fatal(err)
	}

	want := []string{"tell me about Paris", "I was still talking", "actually, never mind"}
	got := texts(convo)
	if len(got) != len(want) {
		t.Fatalf("context has %d messages, want %d: %v", len(got), len(want), got)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("the interrupted answer must be committed before the new turn.\ngot  %v\nwant %v", got, want)
		}
	}
}

// texts is the conversation's messages, in order.
func texts(convo *frames.LLMContext) []string {
	msgs := convo.Messages()
	out := make([]string, 0, len(msgs))
	for _, m := range msgs {
		out = append(out, m.Text)
	}
	return out
}
