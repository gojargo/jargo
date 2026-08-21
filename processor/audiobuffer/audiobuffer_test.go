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

// Ported from upstream. Turn audio is reported once per turn, carrying that
// turn's speech and its number. A turn can hold several runs of speech, and the
// per-run events say nothing about which turn they belong to, so a consumer
// working turn by turn had to reconcile them against the turn tracker itself.
func TestTurnAudioIsReportedOncePerTurn(t *testing.T) {
	const rate = 24000

	var mu sync.Mutex
	var userTurns []audiobuffer.TurnAudioData
	var runs int
	reported := make(chan struct{}, 4)

	proc := audiobuffer.New(audiobuffer.Config{
		SampleRate:      rate,
		AutoStart:       true,
		EnableTurnAudio: true,
		OnUserTurnAudio: func(d audiobuffer.TurnAudioData) {
			mu.Lock()
			userTurns = append(userTurns, d)
			mu.Unlock()
			reported <- struct{}{}
		},
		OnUserTurnAudioData: func([]byte, int, int) {
			mu.Lock()
			runs++
			mu.Unlock()
		},
	})

	// The turn ends once the bot has been silent for the tracker's timeout, which
	// defaults to 2.5 seconds.
	task := pipeline.NewWorker(pipeline.New(proc), pipeline.WorkerConfig{})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	audio := make([]byte, 800)
	for i := range audio {
		audio[i] = byte(i)
	}

	// One turn holding two runs of the user's speech.
	for range 2 {
		task.QueueFrame(frames.NewUserStartedSpeakingFrame())
		task.QueueFrame(frames.NewInputAudioRawFrame(audio, rate, 1))
		task.QueueFrame(frames.NewUserStoppedSpeakingFrame())
	}
	// The bot speaking and falling silent is what ends the turn.
	task.QueueFrame(frames.NewBotStartedSpeakingFrame())
	task.QueueFrame(frames.NewBotStoppedSpeakingFrame())

	select {
	case <-reported:
	case <-time.After(10 * time.Second):
		t.Fatal("the turn's audio was never reported")
	}

	task.StopWhenDone()
	<-runDone

	mu.Lock()
	defer mu.Unlock()
	if len(userTurns) != 1 {
		t.Fatalf("the user's audio was reported %d times, want once for the turn", len(userTurns))
	}
	if runs != 2 {
		t.Errorf("the per-run event fired %d times, want once per run of speech", runs)
	}
	turn := userTurns[0]
	if turn.TurnNumber == 0 {
		t.Error("the report carries no turn number")
	}
	if turn.SampleRate != rate || turn.NumChannels != 1 {
		t.Errorf("the report carries %d Hz / %d channels, want %d / 1",
			turn.SampleRate, turn.NumChannels, rate)
	}
	// Both runs of speech are in it, so the turn carries everything the user said.
	if len(turn.Audio) < 2*len(audio) {
		t.Errorf("the turn carries %d bytes, want at least the %d of both runs",
			len(turn.Audio), 2*len(audio))
	}
}
