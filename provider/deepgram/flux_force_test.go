package deepgram

import (
	"context"
	"encoding/json"
	"errors"
	"maps"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/event"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/service/stt"
)

// Tests for the turn detection modes: Flux deciding where turns end, and the
// pipeline's own strategies deciding with VADForceTurnEndpoint.

// turnInfo is a TurnInfo message as Flux sends it.
func turnInfo(event string, extra map[string]any) map[string]any {
	m := map[string]any{"type": fluxMsgTurnInfo, "event": event, "transcript": "hello there"}
	maps.Copy(m, extra)
	return m
}

// forcingFlux is a fake Flux endpoint. Once it has audio it opens a turn, and
// it answers a ForceEndTurn with the EndOfTurn Flux sends for one. It records
// the text messages it was sent.
type forcingFlux struct {
	*httptest.Server
	// onAudio is what it sends once the first audio arrives.
	onAudio []map[string]any

	mu   sync.Mutex
	sent []string
}

func newForcingFlux(t *testing.T, onAudio ...map[string]any) *forcingFlux {
	t.Helper()
	f := &forcingFlux{onAudio: onAudio}
	f.Server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		c, err := websocket.Accept(w, r, nil)
		if err != nil {
			return
		}
		defer func() { _ = c.CloseNow() }()
		ctx := r.Context()
		write := func(m map[string]any) {
			data, _ := json.Marshal(m)
			_ = c.Write(ctx, websocket.MessageText, data)
		}
		write(map[string]any{"type": fluxMsgConnected})
		heard := false
		for {
			typ, data, err := c.Read(ctx)
			if err != nil {
				return
			}
			if typ == websocket.MessageBinary {
				if !heard {
					heard = true
					for _, m := range f.onAudio {
						write(m)
					}
				}
				continue
			}
			f.mu.Lock()
			f.sent = append(f.sent, string(data))
			f.mu.Unlock()
			if string(data) == fluxForceEndTurn {
				write(turnInfo(fluxEventEndOfTurn, map[string]any{"trigger": fluxTriggerManual}))
			}
		}
	}))
	t.Cleanup(f.Close)
	return f
}

// received is the text messages the endpoint was sent.
func (f *forcingFlux) received() []string {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]string(nil), f.sent...)
}

// openForcingSession opens a session on the endpoint and reads up to the
// update of the turn it opens, so the session knows the user is speaking. With
// the turn edges not proposed, the start of the turn gives no result, so the
// endpoint has to follow it with an update.
func openForcingSession(t *testing.T, f *forcingFlux, cfg FluxConfig) *fluxStream {
	t.Helper()
	cfg.APIKey, cfg.ListenURL, cfg.Model = "k", wsURL(f.URL), defaultFluxModel
	ctx, cancel := context.WithCancel(context.Background())
	stream, err := newFluxConnector(cfg).Connect(ctx, 16000)
	if err != nil {
		cancel()
		t.Fatalf("Connect: %v", err)
	}
	t.Cleanup(func() { cancel(); _ = stream.Close() })
	s, ok := stream.(*fluxStream)
	if !ok {
		t.Fatalf("stream is %T", stream)
	}
	if err := s.Send([]byte{1, 2, 3, 4}); err != nil {
		t.Fatalf("Send: %v", err)
	}
	for !s.turnInProgress() {
		if _, err := s.Recv(); err != nil {
			t.Fatalf("Recv: %v", err)
		}
	}
	return s
}

// Ported from upstream (test_flux_rejects_eager_end_of_turn_with_vad_force_turn_endpoint).
func TestFluxRejectsEagerEndOfTurnWithVADForceTurnEndpoint(t *testing.T) {
	cfg := FluxConfig{APIKey: "k", VADForceTurnEndpoint: true, EagerEOTThreshold: new(0.5)}
	if err := cfg.Validate(); err == nil {
		t.Fatal("a config predicting the end of turn while the pipeline owns the turn was accepted")
	}
	if err := (FluxConfig{APIKey: "k", VADForceTurnEndpoint: true}).Validate(); err != nil {
		t.Fatalf("VADForceTurnEndpoint alone rejected: %v", err)
	}
}

// Ported from upstream (test_vad_force_turn_endpoint_force_ends_turn_on_vad_stop,
// test_vad_force_turn_endpoint_suppresses_turn_proposals and
// test_manual_trigger_confirms_a_requested_finalize). The VAD stop asks Flux to
// finalize the audio sent so far, and the EndOfTurn that answers it reaches the
// pipeline as a finalized transcript, with neither turn edge proposed: the
// configured strategies own the turn.
func TestFluxVADForceTurnEndpointForceEndsTheTurnOnVADStop(t *testing.T) {
	flux := newForcingFlux(t,
		turnInfo(fluxEventStartOfTurn, nil),
		turnInfo(fluxEventUpdate, nil),
	)
	svc := NewFluxSTT(FluxConfig{
		APIKey: "k", ListenURL: wsURL(flux.URL), VADForceTurnEndpoint: true,
	})

	var (
		mu        sync.Mutex
		proposals int
	)
	interim := make(chan struct{}, 1)
	transcripts := make(chan *frames.TranscriptionFrame, 1)
	task := pipeline.NewWorker(pipeline.New(svc), pipeline.WorkerConfig{ReachedDownstreamFilter: pipeline.AnyFrame})
	event.On(&task.Registry, pipeline.EventFrameReachedDownstream, func(_ context.Context, f frames.Frame) {
		switch fr := f.(type) {
		case *frames.ProposedUserStartedSpeakingFrame, *frames.ProposedUserStoppedSpeakingFrame:
			mu.Lock()
			proposals++
			mu.Unlock()
		case *frames.InterimTranscriptionFrame:
			select {
			case interim <- struct{}{}:
			default:
			}
		case *frames.TranscriptionFrame:
			transcripts <- fr
		}
	})
	done := make(chan error, 1)
	go func() { done <- task.Run(context.Background()) }()
	defer func() {
		task.StopWhenDone()
		<-done
	}()

	task.QueueFrame(frames.NewVADUserStartedSpeakingFrame(0.2, time.Now()))
	task.QueueFrame(frames.NewInputAudioRawFrame(make([]byte, 640), 16000, 1))
	select {
	case <-interim:
	case <-time.After(3 * time.Second):
		t.Fatal("Flux never reported the turn in progress")
	}
	task.QueueFrame(frames.NewVADUserStoppedSpeakingFrame(0.2, time.Now()))

	select {
	case tf := <-transcripts:
		if tf.Text != "hello there" || !tf.Finalized {
			t.Errorf("transcript = %q (finalized %v), want the turn finalized", tf.Text, tf.Finalized)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("no transcript arrived after the VAD stop")
	}
	if got := flux.received(); len(got) != 1 || got[0] != fluxForceEndTurn {
		t.Errorf("Flux was sent %q, want one ForceEndTurn", got)
	}
	mu.Lock()
	defer mu.Unlock()
	if proposals != 0 {
		t.Errorf("proposed %d turn edges, want none", proposals)
	}
}

// Ported from upstream (test_flux_turn_detection_ignores_vad_stop). When Flux
// owns the turn, the VAD must not cut it short.
func TestFluxTurnDetectionIgnoresTheVADStop(t *testing.T) {
	flux := newForcingFlux(t, turnInfo(fluxEventStartOfTurn, nil), turnInfo(fluxEventUpdate, nil))
	s := openForcingSession(t, flux, FluxConfig{})

	if err := s.Finalize(); err != nil {
		t.Errorf("Finalize: %v", err)
	}
	if got := flux.received(); len(got) != 0 {
		t.Errorf("Flux was sent %q, want nothing", got)
	}
}

// Ported from upstream (test_force_end_turn_is_inert_with_no_active_turn). With
// no turn in progress there is nothing to finalize, so nothing is asked.
func TestForceEndTurnIsInertWithNoActiveTurn(t *testing.T) {
	flux := newForcingFlux(t, turnInfo(fluxEventStartOfTurn, nil), turnInfo(fluxEventUpdate, nil))
	s := openForcingSession(t, flux, FluxConfig{VADForceTurnEndpoint: true})
	s.stateMu.Lock()
	s.speaking = false
	s.stateMu.Unlock()

	if err := s.Finalize(); err != nil {
		t.Errorf("Finalize: %v", err)
	}
	if got := flux.received(); len(got) != 0 {
		t.Errorf("Flux was sent %q, want nothing", got)
	}
}

// Ported from upstream (test_force_end_turn_is_inert_on_dead_transport). A
// closed connection cannot carry a ForceEndTurn.
func TestForceEndTurnIsInertOnAClosedSession(t *testing.T) {
	flux := newForcingFlux(t, turnInfo(fluxEventStartOfTurn, nil), turnInfo(fluxEventUpdate, nil))
	s := openForcingSession(t, flux, FluxConfig{VADForceTurnEndpoint: true})
	if err := s.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	if err := s.Finalize(); err != nil {
		t.Errorf("Finalize: %v", err)
	}
	for _, m := range flux.received() {
		if m == fluxForceEndTurn {
			t.Error("a closed session sent a ForceEndTurn")
		}
	}
}

// ForceEndTurn is how the application finalizes on a signal of its own. It
// asks nothing of a service with no session open, and refuses a service that
// is not Flux.
func TestForceEndTurnNeedsAFluxSession(t *testing.T) {
	if err := ForceEndTurn(NewFluxSTT(FluxConfig{APIKey: "k"}).StreamService); err != nil {
		t.Errorf("ForceEndTurn with no session = %v, want nothing done", err)
	}
	if err := ForceEndTurn(NewSTT(Config{APIKey: "k"}).StreamService); !errors.Is(err, errNotFlux) {
		t.Errorf("ForceEndTurn on a listen service = %v, want it refused", err)
	}
}

// Ported from upstream (test_vad_force_turn_endpoint_suppresses_turn_proposals).
// The service transcribes only: the configured strategies own the turn.
func TestVADForceTurnEndpointSuppressesTurnProposals(t *testing.T) {
	s := &fluxStream{model: defaultFluxModel, forceTurns: true}

	if res := s.results(fluxMessage{Type: fluxMsgTurnInfo, Event: fluxEventStartOfTurn}); len(res) != 0 {
		t.Errorf("StartOfTurn results = %+v, want nothing proposed", res)
	}
	end := fluxMessage{Type: fluxMsgTurnInfo, Event: fluxEventEndOfTurn, Transcript: "hello there", Trigger: "manual"}
	res := s.results(end)
	// The transcript still reaches the pipeline; only the proposal is dropped.
	if len(res) != 1 || res[0].Text != "hello there" || !res[0].Final || res[0].Speech != stt.SpeechUnknown {
		t.Errorf("EndOfTurn results = %+v, want the finalized transcript alone", res)
	}
	if len(s.EndSession()) != 0 {
		t.Error("a session closed with no turn open reported one")
	}
}

// Ported from upstream (test_flux_turn_detection_proposes_both_turn_edges).
// Flux's turn edges go to the external turn strategies.
func TestFluxTurnDetectionProposesBothTurnEdges(t *testing.T) {
	s := &fluxStream{model: defaultFluxModel}

	start := s.results(fluxMessage{Type: fluxMsgTurnInfo, Event: fluxEventStartOfTurn})
	end := s.results(fluxMessage{Type: fluxMsgTurnInfo, Event: fluxEventEndOfTurn, Transcript: "hi", Trigger: "model"})
	if len(start) != 1 || start[0].Speech != stt.SpeechStarted || len(end) != 1 || end[0].Speech != stt.SpeechStopped {
		t.Errorf("results = %+v then %+v, want the start then the stop proposed", start, end)
	}
}

// An EndOfTurn whose words fall below the confidence floor still answers a
// ForceEndTurn: the confirmation stands with no text, which the service leaves
// to mark the next transcript rather than pushing anything for it.
func TestADroppedManualEndOfTurnStillConfirmsTheFinalize(t *testing.T) {
	m := fluxMessage{Type: fluxMsgTurnInfo, Event: fluxEventEndOfTurn, Transcript: "hi", Trigger: "manual"}
	res := fluxResults(m, defaultFluxModel, new(0.5))
	if len(res) != 1 || res[0].Text != "" || !res[0].Final || !res[0].FromFinalize || res[0].PushFinalized {
		t.Errorf("results = %+v, want an empty confirmation pushing nothing", res)
	}
	forced := (&fluxStream{model: defaultFluxModel, forceTurns: true, minConf: new(0.5)}).results(m)
	if len(forced) != 1 || !forced[0].FromFinalize || forced[0].Speech != stt.SpeechUnknown {
		t.Errorf("results with the turns left to the pipeline = %+v, want the confirmation alone", forced)
	}
}

// Ported from upstream (test_manual_trigger_confirms_a_requested_finalize). An
// EndOfTurn answering a ForceEndTurn confirms the finalize asked for; one Flux
// ended on its own does not.
func TestOnlyAManualTriggerConfirmsTheFinalize(t *testing.T) {
	for trigger, want := range map[string]bool{"timeout": false, "model": false, "manual": true} {
		m := fluxMessage{Type: fluxMsgTurnInfo, Event: fluxEventEndOfTurn, Transcript: "hi", Trigger: trigger}
		res := fluxResults(m, defaultFluxModel, nil)
		if len(res) != 1 || res[0].FromFinalize != want {
			t.Errorf("trigger %q: results = %+v, want confirming %v", trigger, res, want)
		}
	}
}

// Ported from upstream (test_force_end_turn_losing_the_race_is_not_an_error and
// test_unrecognized_warning_is_not_an_error). Flux may have ended the turn
// already, and warnings never interrupt the stream, so none of them is
// reported as an error.
func TestFluxWarningsAreNotErrors(t *testing.T) {
	warning := func(code string) map[string]any {
		return map[string]any{"type": fluxMsgWarning, "code": code, "description": "a warning"}
	}
	flux := newForcingFlux(t,
		warning(fluxWarningNoActiveTurn),
		warning("SOMETHING_NEW"),
		turnInfo(fluxEventUpdate, nil),
	)
	conn := newFluxConnector(FluxConfig{APIKey: "k", ListenURL: wsURL(flux.URL), Model: defaultFluxModel})
	ctx, cancel := context.WithCancel(context.Background())
	stream, err := conn.Connect(ctx, 16000)
	if err != nil {
		cancel()
		t.Fatalf("Connect: %v", err)
	}
	defer func() { cancel(); _ = stream.Close() }()
	s, ok := stream.(*fluxStream)
	if !ok {
		t.Fatalf("stream is %T", stream)
	}
	var reported []string
	s.report = func(_ context.Context, msg string) { reported = append(reported, msg) }
	s.fail = func(_ context.Context, err error) { reported = append(reported, err.Error()) }

	if err := s.Send([]byte{1, 2, 3, 4}); err != nil {
		t.Fatalf("Send: %v", err)
	}
	res, err := s.Recv()
	if err != nil {
		t.Fatalf("Recv: %v", err)
	}
	if len(res) != 1 || res[0].Text != "hello there" {
		t.Errorf("results = %+v, want the update after the warnings", res)
	}
	if len(reported) != 0 {
		t.Errorf("reported %q, want no error", reported)
	}
}

// With VADForceTurnEndpoint the service recommends no strategies, leaving the
// turns to the pipeline's own, and TTFS applies: the VAD marks the end of
// speech, and the transcript trails it.
func TestFluxVADForceTurnEndpointRecommendsNothingAndReportsTTFS(t *testing.T) {
	m := newFluxConnector(FluxConfig{Model: defaultFluxModel, VADForceTurnEndpoint: true}).Metadata()
	if m.UserTurnStrategies != nil {
		t.Errorf("recommended %v, want nothing", m.UserTurnStrategies)
	}
	if m.SupportsTTFS != nil && !*m.SupportsTTFS {
		t.Error("TTFS reported as not applying")
	}
	if m.TTFSP99 == nil || *m.TTFSP99 != stt.DeepgramFluxTTFSP99 {
		t.Errorf("TTFS p99 = %v, want %v", m.TTFSP99, stt.DeepgramFluxTTFSP99)
	}

	plain := newFluxConnector(FluxConfig{Model: defaultFluxModel}).Metadata()
	if plain.SupportsTTFS == nil || *plain.SupportsTTFS {
		t.Error("TTFS reported as applying while Flux marks the end of speech")
	}
}
