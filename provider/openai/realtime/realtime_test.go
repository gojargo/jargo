package realtime_test

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
	"github.com/gojargo/jargo/provider/openai/realtime"
	"github.com/gojargo/jargo/utils/events"
)

func TestConfigValidate(t *testing.T) {
	if err := (realtime.Config{}).Validate(); err == nil {
		t.Error("Validate() with empty APIKey: want error, got nil")
	}
	if err := (realtime.Config{APIKey: "k"}).Validate(); err != nil {
		t.Errorf("Validate() with APIKey: %v", err)
	}
}

// fakeRealtime is a WebSocket server that accepts a connection, discards client
// messages, and streams a canned sequence of Realtime server events.
func fakeRealtime(t *testing.T, audio []byte) *httptest.Server {
	t.Helper()
	serverEvents := []string{
		`{"type":"response.created"}`,
		`{"type":"response.audio.delta","delta":"` + base64.StdEncoding.EncodeToString(audio) + `"}`,
		`{"type":"response.audio_transcript.delta","delta":"hello"}`,
		`{"type":"input_audio_buffer.speech_started"}`,
		`{"type":"input_audio_buffer.speech_stopped"}`,
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
		for _, e := range serverEvents {
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

	svc := realtime.New(realtime.Config{APIKey: "k", BaseURL: wsURL})

	var mu sync.Mutex
	var got []frames.Frame
	task := pipeline.NewWorker(pipeline.New(svc), pipeline.WorkerConfig{
		ReachedDownstreamFilter: pipeline.AnyFrame,
		// A transcript of what the user said travels upstream, where the user
		// aggregator sits, so both ends of the pipeline are watched.
		ReachedUpstreamFilter: pipeline.AnyFrame,
		Params: pipeline.Params{
			AudioInSampleRate:  24000,
			AudioOutSampleRate: 24000,
		},
	})
	record := func(_ context.Context, f frames.Frame) {
		mu.Lock()
		got = append(got, f)
		mu.Unlock()
	}
	events.On(&task.Registry, pipeline.EventFrameReachedDownstream, record)
	events.On(&task.Registry, pipeline.EventFrameReachedUpstream, record)

	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() { done <- task.Run(ctx) }()

	// Exercise the input path; the fake server discards it.
	task.QueueFrame(frames.NewInputAudioRawFrame([]byte{0, 0}, 24000, 1))

	// Wait for the frame the canned sequence ends with rather than for a count of
	// frames: the events do not map one-to-one onto what reaches the pipeline, so
	// a count can be reached while the end of the response is still in flight.
	// response.done is the last event the fake server sends, and the bot-stopped
	// frame is what it produces.
	deadline := time.Now().Add(5 * time.Second)
	var arrived bool
	for time.Now().Before(deadline) {
		mu.Lock()
		for _, f := range got {
			if _, ok := f.(*frames.BotStoppedSpeakingFrame); ok {
				arrived = true
				break
			}
		}
		mu.Unlock()
		if arrived {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}
	if !arrived {
		t.Error("the canned response never finished reaching the pipeline")
	}

	cancel()
	<-done

	mu.Lock()
	defer mu.Unlock()

	var (
		gotAudio                  []byte
		botText, userTranscript   string
		interrupted, botStarted   bool
		botStopped, announcedTurn bool
		proposedStart             bool
		proposedStop              bool
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
		case *frames.ProposedUserStartedSpeakingFrame:
			proposedStart = true
		case *frames.ProposedUserStoppedSpeakingFrame:
			proposedStop = true
		case *frames.UserStartedSpeakingFrame, *frames.UserStoppedSpeakingFrame:
			announcedTurn = true
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
	if !proposedStart {
		t.Error("speech_started did not propose the start of the user's turn")
	}
	// The stop matters as much as the start. The strategies close the turn on
	// this proposal; a start with no stop would leave everything keyed off the
	// turn believing the user never finished.
	if !proposedStop {
		t.Error("speech_stopped did not propose the end of the user's turn")
	}
	// The strategies the service recommends decide the turn and the barge-in;
	// the session only proposes where the boundary falls.
	if announcedTurn {
		t.Error("the service announced a user turn itself, rather than proposing one")
	}
	if interrupted {
		t.Error("the service broadcast an interruption itself, rather than proposing a turn")
	}
	if !botStarted || !botStopped {
		t.Error("response lifecycle did not produce bot started/stopped speaking")
	}
}
