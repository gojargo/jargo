package openairealtime_test

import (
	"context"
	"encoding/base64"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/provider/openairealtime"
)

func TestConfigValidate(t *testing.T) {
	if err := (openairealtime.Config{}).Validate(); err == nil {
		t.Error("Validate() with empty APIKey: want error, got nil")
	}
	if err := (openairealtime.Config{APIKey: "k"}).Validate(); err != nil {
		t.Errorf("Validate() with APIKey: %v", err)
	}
}

// fakeRealtime is a WebSocket server that accepts a connection, discards client
// messages, and streams a canned sequence of Realtime server events.
func fakeRealtime(t *testing.T, audio []byte) *httptest.Server {
	t.Helper()
	events := []string{
		`{"type":"response.created"}`,
		`{"type":"response.audio.delta","delta":"` + base64.StdEncoding.EncodeToString(audio) + `"}`,
		`{"type":"response.audio_transcript.delta","delta":"hello"}`,
		`{"type":"input_audio_buffer.speech_started"}`,
		`{"type":"conversation.item.input_audio_transcription.completed","transcript":"hi there"}`,
		`{"type":"response.done"}`,
	}
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		c, err := websocket.Accept(w, r, &websocket.AcceptOptions{InsecureSkipVerify: true})
		if err != nil {
			return
		}
		defer func() { _ = c.Close(websocket.StatusNormalClosure, "") }()
		ctx := r.Context()
		go func() {
			for {
				if _, _, err := c.Read(ctx); err != nil {
					return
				}
			}
		}()
		for _, e := range events {
			if err := c.Write(ctx, websocket.MessageText, []byte(e)); err != nil {
				return
			}
		}
		<-ctx.Done()
	}))
}

func TestRealtimeStreamsEvents(t *testing.T) {
	audio := []byte{1, 2, 3, 4, 5, 6}
	srv := fakeRealtime(t, audio)
	defer srv.Close()
	wsURL := "ws" + strings.TrimPrefix(srv.URL, "http")

	svc := openairealtime.New(openairealtime.Config{APIKey: "k", BaseURL: wsURL})

	var mu sync.Mutex
	var got []frames.Frame
	task := pipeline.NewTask(pipeline.New(svc), pipeline.TaskParams{
		AudioInSampleRate:  24000,
		AudioOutSampleRate: 24000,
		OnReachedDownstream: func(f frames.Frame) {
			mu.Lock()
			got = append(got, f)
			mu.Unlock()
		},
	})

	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() { done <- task.Run(ctx) }()

	// Exercise the input path; the fake server discards it.
	task.QueueFrame(frames.NewInputAudioRawFrame([]byte{0, 0}, 24000, 1))

	// Wait until the canned events have propagated downstream.
	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		mu.Lock()
		n := len(got)
		mu.Unlock()
		if n >= 6 {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}

	cancel()
	<-done

	mu.Lock()
	defer mu.Unlock()

	var (
		gotAudio                []byte
		botText, userTranscript string
		interrupted, botStarted bool
		botStopped, userStarted bool
	)
	for _, f := range got {
		switch fr := f.(type) {
		case *frames.TTSAudioRawFrame:
			gotAudio = fr.Audio
			if fr.SampleRate != 24000 {
				t.Errorf("bot audio sample rate = %d, want 24000", fr.SampleRate)
			}
		case *frames.LLMTextFrame:
			botText = fr.Text
		case *frames.TranscriptionFrame:
			userTranscript = fr.Text
		case *frames.InterruptionFrame:
			interrupted = true
		case *frames.UserStartedSpeakingFrame:
			userStarted = true
		case *frames.BotStartedSpeakingFrame:
			botStarted = true
		case *frames.BotStoppedSpeakingFrame:
			botStopped = true
		}
	}

	if string(gotAudio) != string(audio) {
		t.Errorf("bot audio = %v, want %v", gotAudio, audio)
	}
	if botText != "hello" {
		t.Errorf("bot transcript = %q, want %q", botText, "hello")
	}
	if userTranscript != "hi there" {
		t.Errorf("user transcript = %q, want %q", userTranscript, "hi there")
	}
	if !interrupted || !userStarted {
		t.Error("speech_started did not produce barge-in (interruption + user-started)")
	}
	if !botStarted || !botStopped {
		t.Error("response lifecycle did not produce bot started/stopped speaking")
	}
}
