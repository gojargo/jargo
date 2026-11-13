package deepgram

// How Flux survives its connection dropping, driven end to end through a
// pipeline against a fake Flux server, and how a session is closed. Connections
// that are never confirmed are covered by flux_confirm_test.go.

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"reflect"
	"sync"
	"testing"
	"time"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/service/stt"
	"github.com/gojargo/jargo/service/wsservice"
	"github.com/gojargo/jargo/utils/events"
)

// fakeFlux is a Flux server whose failures a test scripts.
type fakeFlux struct {
	// rejectAfter refuses every handshake after that many; 0 refuses none.
	rejectAfter int
	// dropFirstAfter drops the first connection, with no close frame, after
	// that many audio messages; 0 never drops it.
	dropFirstAfter int
	// dropMidTurn opens a turn just before that drop.
	dropMidTurn bool
	// resumeTurn has the second connection hear the rest of what the user says
	// as a whole turn of its own.
	resumeTurn bool
	// predictMidTurn has Flux predict the end of the turn just before the
	// drop, and the second connection report the user resuming.
	predictMidTurn bool

	mu         sync.Mutex
	handshakes int
	audio      []int
	open       int
	closeCodes []websocket.StatusCode
	srv        *httptest.Server
}

func (f *fakeFlux) start(t *testing.T) string {
	t.Helper()
	f.srv = httptest.NewServer(http.HandlerFunc(f.serve))
	t.Cleanup(f.srv.Close)
	return wsURL(f.srv.URL)
}

// counts is how many handshakes were made, and how many audio messages each
// accepted connection received.
func (f *fakeFlux) counts() (int, []int) {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.handshakes, append([]int(nil), f.audio...)
}

// connections is how many connections are open, and the close code each
// closed one ended with.
func (f *fakeFlux) connections() (int, []websocket.StatusCode) {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.open, append([]websocket.StatusCode(nil), f.closeCodes...)
}

func (f *fakeFlux) serve(w http.ResponseWriter, r *http.Request) {
	f.mu.Lock()
	f.handshakes++
	n := f.handshakes
	f.mu.Unlock()
	if f.rejectAfter > 0 && n > f.rejectAfter {
		http.Error(w, "unavailable", http.StatusServiceUnavailable)
		return
	}
	c, err := websocket.Accept(w, r, nil)
	if err != nil {
		return
	}
	defer func() { _ = c.CloseNow() }()
	ctx := r.Context()
	f.mu.Lock()
	index := len(f.audio)
	f.audio = append(f.audio, 0)
	f.open++
	f.mu.Unlock()
	code := websocket.StatusCode(-1)
	defer func() {
		f.mu.Lock()
		f.open--
		f.closeCodes = append(f.closeCodes, code)
		f.mu.Unlock()
	}()
	send := func(m map[string]any) {
		b, _ := json.Marshal(m)
		_ = c.Write(ctx, websocket.MessageText, b)
	}
	turnInfo := func(event, transcript string) {
		send(map[string]any{"type": fluxMsgTurnInfo, "event": event, "transcript": transcript})
	}
	send(map[string]any{"type": fluxMsgConnected, "request_id": "test"})
	for {
		kind, _, err := c.Read(ctx)
		if err != nil {
			code = websocket.CloseStatus(err)
			return
		}
		if kind != websocket.MessageBinary {
			continue
		}
		f.mu.Lock()
		f.audio[index]++
		count := f.audio[index]
		f.mu.Unlock()
		if index == 1 && f.resumeTurn && count == 3 {
			turnInfo(fluxEventStartOfTurn, "how are you")
			turnInfo(fluxEventEndOfTurn, "how are you")
		}
		if index == 1 && f.predictMidTurn && count == 3 {
			turnInfo(fluxEventTurnResumed, "")
		}
		if index == 0 && count == f.dropFirstAfter {
			if f.dropMidTurn {
				turnInfo(fluxEventStartOfTurn, "hello")
				turnInfo(fluxEventUpdate, "hello there")
				if f.predictMidTurn {
					turnInfo(fluxEventEagerEndOfTurn, "hello there")
				}
				time.Sleep(50 * time.Millisecond)
			}
			// Drop the TCP connection with no close frame.
			_ = c.CloseNow()
			return
		}
	}
}

// fluxRun is what a run captured.
type fluxRun struct {
	down    []frames.Frame
	elapsed time.Duration
}

// runFluxPipeline streams real-time 20 ms frames of silence through a Flux
// service built from cfg for d, and returns every frame that reached the end of
// the pipeline.
func runFluxPipeline(t *testing.T, cfg FluxConfig, d time.Duration) (*stt.StreamService, fluxRun) {
	t.Helper()
	cfg.APIKey = "test-key"
	cfg.Model = defaultFluxModel
	c := newFluxConnector(cfg)
	svc := stt.NewStream("DeepgramFluxSTT", c, 16000)
	c.svc = svc

	var mu sync.Mutex
	var res fluxRun
	task := pipeline.NewWorker(pipeline.New(svc), pipeline.WorkerConfig{
		IdleTimeout:             -1,
		Params:                  pipeline.Params{EnableUsageMetrics: true},
		ReachedDownstreamFilter: pipeline.AnyFrame,
	})
	events.On(&task.Registry, pipeline.EventFrameReachedDownstream, func(_ context.Context, f frames.Frame) {
		mu.Lock()
		res.down = append(res.down, f)
		mu.Unlock()
	})
	start := time.Now()
	done := make(chan error, 1)
	go func() { done <- task.Run(context.Background()) }()
	for range int(d / (20 * time.Millisecond)) {
		task.QueueFrame(frames.NewInputAudioRawFrame(make([]byte, 640), 16000, 1))
		time.Sleep(20 * time.Millisecond)
	}
	task.StopWhenDone()
	select {
	case <-done:
	case <-time.After(10 * time.Second):
		t.Fatal("pipeline did not shut down")
	}
	res.elapsed = time.Since(start)
	mu.Lock()
	defer mu.Unlock()
	return svc, res
}

type fluxTranscript struct {
	text      string
	finalized bool
}

// fluxTurnEnds returns the transcriptions pushed, and the positions of the
// proposed turn stops.
func fluxTurnEnds(down []frames.Frame) ([]fluxTranscript, []int) {
	var ts []fluxTranscript
	var stops []int
	for i, f := range down {
		switch fr := f.(type) {
		case *frames.TranscriptionFrame:
			ts = append(ts, fluxTranscript{fr.Text, fr.Finalized})
		case *frames.ProposedUserStoppedSpeakingFrame:
			stops = append(stops, i)
		}
	}
	return ts, stops
}

// TestFluxATurnCutShortByADropEndsWithItsTranscriptSoFar checks a turn the
// connection drops in the middle of ends at the drop, with what was transcribed
// so far as its transcript, unfinalized since Flux never committed it, followed
// by a proposed stop.
func TestFluxATurnCutShortByADropEndsWithItsTranscriptSoFar(t *testing.T) {
	t.Parallel()
	server := &fakeFlux{dropFirstAfter: 10, dropMidTurn: true}
	_, res := runFluxPipeline(t, FluxConfig{ListenURL: server.start(t)}, time.Second)

	ts, stops := fluxTurnEnds(res.down)
	if want := []fluxTranscript{{"hello there", false}}; !reflect.DeepEqual(ts, want) {
		t.Errorf("transcripts = %v, want %v", ts, want)
	}
	if len(stops) != 1 {
		t.Fatalf("proposed stops = %d, want 1", len(stops))
	}
	first := -1
	usage := false
	for i, f := range res.down {
		if _, ok := f.(*frames.TranscriptionFrame); ok {
			first = i
			break
		}
		if mf, ok := f.(*frames.MetricsFrame); ok {
			for _, d := range mf.Data {
				if _, ok := d.(frames.STTUsageMetricsData); ok {
					usage = true
				}
			}
		}
	}
	if first < 0 || first > stops[0] {
		t.Errorf("the transcript is at %d and the stop at %d, want the transcript first", first, stops[0])
	}
	// The transcript is the billing event, so the audio it covers is reported
	// as usage ahead of it.
	if !usage {
		t.Error("no STT usage was reported ahead of the transcript")
	}
	if _, audio := server.counts(); len(audio) != 2 {
		t.Errorf("connections = %d, want 2", len(audio))
	}
}

// TestFluxSpeechContinuingAfterADropBecomesANewTurn checks speech carrying on
// after a drop is a turn of its own on the new connection.
func TestFluxSpeechContinuingAfterADropBecomesANewTurn(t *testing.T) {
	t.Parallel()
	server := &fakeFlux{dropFirstAfter: 10, dropMidTurn: true, resumeTurn: true}
	_, res := runFluxPipeline(t, FluxConfig{ListenURL: server.start(t)}, time.Second)

	ts, stops := fluxTurnEnds(res.down)
	if want := []fluxTranscript{{"hello there", false}, {"how are you", true}}; !reflect.DeepEqual(ts, want) {
		t.Errorf("transcripts = %v, want %v", ts, want)
	}
	if len(stops) != 2 {
		t.Errorf("proposed stops = %d, want 2", len(stops))
	}
}

// TestFluxATurnCutShortResolvesItsPrediction checks a prediction Flux made on a
// turn the connection then dropped is resolved with the turn, so the user
// resuming on the new connection withdraws nothing.
func TestFluxATurnCutShortResolvesItsPrediction(t *testing.T) {
	t.Parallel()
	server := &fakeFlux{dropFirstAfter: 10, dropMidTurn: true, predictMidTurn: true}
	_, res := runFluxPipeline(t, FluxConfig{ListenURL: server.start(t), EagerEOTThreshold: new(0.5)}, time.Second)

	eager, withdrawn := 0, 0
	for _, f := range res.down {
		switch f.(type) {
		case *frames.EagerTranscriptionFrame:
			eager++
		case *frames.UserTurnResponseCancelFrame:
			withdrawn++
		}
	}
	if eager != 1 {
		t.Errorf("predictions = %d, want 1", eager)
	}
	if withdrawn != 0 {
		t.Errorf("withdrawals = %d, want none: the prediction ended with its turn", withdrawn)
	}
	if _, audio := server.counts(); len(audio) != 2 {
		t.Errorf("connections = %d, want 2", len(audio))
	}
}

// TestFluxATurnEndsWhenReconnectingGivesUp checks a turn cut short ends at the
// drop even when no connection replaces it, and the service is left unusable.
func TestFluxATurnEndsWhenReconnectingGivesUp(t *testing.T) {
	t.Parallel()
	server := &fakeFlux{dropFirstAfter: 10, dropMidTurn: true, rejectAfter: 1}
	svc, res := runFluxPipeline(t, FluxConfig{
		ListenURL: server.start(t),
		Websocket: wsservice.Options{
			ReconnectBackoffMinWait: 10 * time.Millisecond,
			ReconnectBackoffMaxWait: 10 * time.Millisecond,
		},
	}, time.Second)

	ts, stops := fluxTurnEnds(res.down)
	if want := []fluxTranscript{{"hello there", false}}; !reflect.DeepEqual(ts, want) {
		t.Errorf("transcripts = %v, want %v", ts, want)
	}
	if len(stops) != 1 {
		t.Errorf("proposed stops = %d, want 1", len(stops))
	}
	if svc.Usable() {
		t.Error("the service is still usable after reconnecting gave up")
	}
}

// TestFluxAudioKeepsFlowingWhileTheProviderIsDown checks the connection drops
// and every reconnect is refused while frames pass through the service on
// time: reconnecting happens in the background, each attempt failing and
// backing off without holding the audio up.
func TestFluxAudioKeepsFlowingWhileTheProviderIsDown(t *testing.T) {
	t.Parallel()
	server := &fakeFlux{dropFirstAfter: 5, rejectAfter: 1}
	_, res := runFluxPipeline(t, FluxConfig{
		ListenURL: server.start(t),
		Websocket: wsservice.Options{
			ReconnectBackoffMinWait: 30 * time.Second,
			ReconnectBackoffMaxWait: 30 * time.Second,
		},
	}, 500*time.Millisecond)

	if res.elapsed >= 3*time.Second {
		t.Errorf("run took %s, want < 3s", res.elapsed)
	}
	n := 0
	for _, f := range res.down {
		if _, ok := f.(*frames.InputAudioRawFrame); ok {
			n++
		}
	}
	if n != 25 {
		t.Errorf("audio frames downstream = %d, want 25", n)
	}
}

// openFluxSession connects to server directly and waits for the server to
// hold the connection open.
func openFluxSession(t *testing.T, server *fakeFlux, closeWait time.Duration) *fluxStream {
	t.Helper()
	c := newFluxConnector(FluxConfig{APIKey: "test-key", ListenURL: server.start(t), Model: defaultFluxModel})
	if closeWait > 0 {
		c.closeStreamTimeout = closeWait
	}
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)
	stream, err := c.Connect(ctx, 16000)
	if err != nil {
		t.Fatalf("connect: %v", err)
	}
	flux, ok := stream.(*fluxStream)
	if !ok {
		t.Fatalf("connect returned a %T, want a flux session", stream)
	}
	waitForFlux(t, "the server to hold the connection", func() bool {
		open, _ := server.connections()
		return open == 1
	})
	return flux
}

// waitForFlux polls cond until it holds, failing once it is clear it will not.
func waitForFlux(t *testing.T, what string, cond func() bool) {
	t.Helper()
	deadline := time.After(3 * time.Second)
	tick := time.NewTicker(5 * time.Millisecond)
	defer tick.Stop()
	for !cond() {
		select {
		case <-tick.C:
		case <-deadline:
			t.Fatalf("timed out waiting for %s", what)
		}
	}
}

// TestFluxDisconnectingClosesTheConnectionCleanly checks closing a session
// ends the connection with a normal closure.
func TestFluxDisconnectingClosesTheConnectionCleanly(t *testing.T) {
	t.Parallel()
	server := &fakeFlux{}
	stream := openFluxSession(t, server, 0)

	if err := stream.Close(); err != nil {
		t.Errorf("close: %v", err)
	}

	waitForFlux(t, "the connection to close", func() bool {
		open, _ := server.connections()
		return open == 0
	})
	if _, codes := server.connections(); !reflect.DeepEqual(codes, []websocket.StatusCode{websocket.StatusNormalClosure}) {
		t.Errorf("close codes = %v, want one normal closure", codes)
	}
}

// TestFluxADisconnectStuckBeforeCloseStreamStillClosesTheConnection checks a
// teardown that cannot get the CloseStream message out, because a send is
// stuck on a full write buffer, does not leave the connection holding a Flux
// concurrency slot: the socket is dropped once the wait for the message runs
// out.
func TestFluxADisconnectStuckBeforeCloseStreamStillClosesTheConnection(t *testing.T) {
	t.Parallel()
	server := &fakeFlux{}
	stream := openFluxSession(t, server, 100*time.Millisecond)

	// A send waiting on a full write buffer holds the session's writes.
	stream.writeMu.Lock()
	closed := make(chan error, 1)
	go func() { closed <- stream.Close() }()
	select {
	case err := <-closed:
		if err != nil && !errors.Is(err, context.Canceled) {
			t.Errorf("close: %v", err)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("closing the session waited on the stuck send")
	}
	waitForFlux(t, "the connection to close", func() bool {
		open, _ := server.connections()
		return open == 0
	})
	stream.writeMu.Unlock()
}
