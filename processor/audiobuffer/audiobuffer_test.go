package audiobuffer_test

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor/audiobuffer"
)

func TestRecordsAndMergesTracks(t *testing.T) {
	const rate = 24000
	var mu sync.Mutex
	var started, stopped bool
	var merged []byte
	var gotUser, gotBot []byte
	done := make(chan struct{})

	proc := audiobuffer.New(audiobuffer.Config{
		AutoStart: true,
		OnRecordingStarted: func() {
			mu.Lock()
			started = true
			mu.Unlock()
		},
		OnAudioData: func(audio []byte, _, _ int) {
			mu.Lock()
			merged = audio
			mu.Unlock()
		},
		OnTrackAudioData: func(user, bot []byte, _, _ int) {
			mu.Lock()
			gotUser, gotBot = user, bot
			mu.Unlock()
		},
		OnRecordingStopped: func() {
			mu.Lock()
			stopped = true
			mu.Unlock()
			close(done)
		},
	})

	task := pipeline.NewWorker(pipeline.New(proc), pipeline.WorkerConfig{})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	// 400 samples of user audio and 400 of bot audio at the output rate.
	user := make([]byte, 800)
	for i := range user {
		user[i] = byte(i)
	}
	bot := make([]byte, 800)
	for i := range bot {
		bot[i] = byte(255 - i)
	}
	task.QueueFrame(frames.NewInputAudioRawFrame(user, rate, 1))
	task.QueueFrame(frames.NewOutputAudioRawFrame(bot, rate, 1))

	task.StopWhenDone()
	select {
	case <-done:
	case <-time.After(3 * time.Second):
		t.Fatal("recording never stopped")
	}
	<-runDone

	mu.Lock()
	defer mu.Unlock()
	if !started {
		t.Error("OnRecordingStarted was not called")
	}
	if !stopped {
		t.Error("OnRecordingStopped was not called")
	}
	if len(merged) == 0 {
		t.Error("merged audio is empty")
	}
	if len(gotUser) != len(gotBot) {
		t.Errorf("tracks not aligned: user=%d bot=%d", len(gotUser), len(gotBot))
	}
	// Mono mix is one 16-bit stream: same byte length as the aligned tracks.
	if len(merged) != len(gotUser) {
		t.Errorf("mono merge length = %d, want %d", len(merged), len(gotUser))
	}
}
