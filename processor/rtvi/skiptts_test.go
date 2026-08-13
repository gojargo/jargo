package rtvi_test

import (
	"context"
	"encoding/json"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor/aggregators"
	"github.com/gojargo/jargo/processor/rtvi"
	"github.com/gojargo/jargo/service/llm"
	"github.com/gojargo/jargo/service/tts"
)

// replyGen answers every turn with one fixed reply.
type replyGen struct{}

func (replyGen) Generate(_ context.Context, _ *frames.LLMContext, emit llm.Emit) error {
	return emit("the reply")
}

// recordingSynth records the text it was asked to speak, so a test can tell
// whether a turn reached synthesis at all.
type recordingSynth struct {
	mu     sync.Mutex
	spoken []string
}

func (s *recordingSynth) SampleRate() int { return 16000 }

func (s *recordingSynth) RunTTS(_ context.Context, text, _ string, yield func(frames.Frame) error) error {
	s.mu.Lock()
	s.spoken = append(s.spoken, text)
	s.mu.Unlock()
	return yield(frames.NewTTSAudioRawFrame(make([]byte, 640), 16000, 1))
}

func (s *recordingSynth) texts() []string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]string(nil), s.spoken...)
}

// audioResponseFrame builds the transport frame carrying a send-text message
// that asks for, or declines, a spoken reply.
func audioResponseFrame(t *testing.T, content string, audio bool) frames.Frame {
	t.Helper()
	runNow := true
	msg := rtvi.Message{
		Label: rtvi.MessageLabel, Type: rtvi.TypeSendText, ID: "test",
		Data: rtvi.SendTextData{
			Content: content,
			Options: &rtvi.SendTextOptions{RunImmediately: &runNow, AudioResponse: &audio},
		},
	}
	encoded, err := json.Marshal(msg)
	if err != nil {
		t.Fatal(err)
	}
	return frames.NewInputTransportMessageFrame(encoded)
}

// TestSendTextAudioResponse drives send-text turns through a pipeline shaped
// like a bot's and checks the client's audio_response is honored: a turn that
// declines a spoken reply is answered without reaching synthesis, and a turn
// after it is spoken again, because the setting applies to the one turn that
// asked for it.
func TestSendTextAudioResponse(t *testing.T) {
	synth := &recordingSynth{}
	agg := aggregators.New(frames.NewLLMContext("system"))
	proc := rtvi.NewProcessor()

	task := pipeline.NewWorker(pipeline.New(
		proc, agg.User(), llm.New("TestLLM", replyGen{}), tts.New("TestTTS", synth), agg.Assistant(),
	), pipeline.WorkerConfig{})

	done := make(chan error, 1)
	go func() { done <- task.Run(t.Context()) }()

	task.QueueFrame(audioResponseFrame(t, "stay quiet", false))
	time.Sleep(750 * time.Millisecond)
	if spoken := synth.texts(); len(spoken) != 0 {
		t.Errorf("the turn declined a spoken reply but reached synthesis: %q", spoken)
	}

	task.QueueFrame(audioResponseFrame(t, "speak up", true))
	time.Sleep(750 * time.Millisecond)
	if spoken := synth.texts(); len(spoken) == 0 {
		t.Error("the turn asked to be spoken but never reached synthesis")
	}

	task.StopWhenDone()
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("the pipeline did not finish")
	}
}
