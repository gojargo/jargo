package soniox

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/language"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/service/settings"
	"github.com/gojargo/jargo/service/stt"
	errs "github.com/gojargo/jargo/utils/errors"
	"github.com/gojargo/jargo/utils/events"
)

// Tests for the live session. Soniox is dialed and then handed a configuration
// over the socket itself, which carries the key: nothing happens until that
// handshake lands, and none of it had coverage.

// sonioxSession is what the fake endpoint saw and what it will say.
type sonioxSession struct {
	// auth is the Authorization header the session was opened with.
	auth      chan string
	handshake chan string
	// control receives the protocol messages after the handshake, the
	// keepalive and the finalize, which carry a type.
	control chan string
	audio   chan []byte
	send    chan string
	// ended receives once for each session the client has closed.
	ended chan struct{}
}

// sonioxServer stands up a fake Soniox live endpoint.
func sonioxServer(t *testing.T) (endpoint string, got *sonioxSession) {
	t.Helper()
	got = &sonioxSession{
		auth:      make(chan string, 4),
		handshake: make(chan string, 4),
		control:   make(chan string, 8),
		audio:     make(chan []byte, 8),
		send:      make(chan string, 8),
		ended:     make(chan struct{}, 4),
	}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		select {
		case got.auth <- r.Header.Get("Authorization"):
		default:
		}
		c, err := websocket.Accept(w, r, nil)
		if err != nil {
			return
		}
		defer func() { _ = c.CloseNow() }()
		defer func() { got.ended <- struct{}{} }()
		go func() {
			for msg := range got.send {
				if err := c.Write(r.Context(), websocket.MessageText, []byte(msg)); err != nil {
					return
				}
			}
		}()
		for {
			typ, data, err := c.Read(r.Context())
			if err != nil {
				return
			}
			if typ == websocket.MessageBinary {
				select {
				case got.audio <- data:
				default:
				}
				continue
			}
			target := got.handshake
			var typed struct {
				Type string `json:"type"`
			}
			if json.Unmarshal(data, &typed) == nil && typed.Type != "" {
				target = got.control
			}
			select {
			case target <- string(data):
			default:
			}
		}
	}))
	t.Cleanup(srv.Close)
	return "ws" + strings.TrimPrefix(srv.URL, "http"), got
}

// dialSoniox opens a session against the fake endpoint.
func dialSoniox(t *testing.T, cfg Config) (*stream, *sonioxSession) {
	t.Helper()
	endpoint, got := sonioxServer(t)
	cfg.URL = endpoint
	if cfg.APIKey == "" {
		cfg.APIKey = "test-key"
	}
	if cfg.Model == "" {
		cfg.Model = defaultModel
	}
	c := &connector{cfg: cfg, live: newSTTSettings(cfg)}

	session, err := c.Connect(t.Context(), 16000)
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	t.Cleanup(func() { _ = session.Close() })
	s, ok := session.(*stream)
	if !ok {
		t.Fatalf("Connect returned a %T", session)
	}
	return s, got
}

// TestConnectSendsTheHandshake checks the socket is configured before anything
// else happens, and that the key travels as a Bearer header on the WebSocket
// handshake rather than in the configuration message, which Soniox stops
// accepting it in on 2027-01-15.
func TestConnectSendsTheHandshake(t *testing.T) {
	_, got := dialSoniox(t, Config{Model: "stt-rt-v5"})

	var cfg map[string]any
	select {
	case msg := <-got.handshake:
		if err := json.Unmarshal([]byte(msg), &cfg); err != nil {
			t.Fatalf("the handshake was not JSON: %v", err)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("no handshake reached the server")
	}

	if auth := <-got.auth; auth != "Bearer test-key" {
		t.Errorf("Authorization = %q, want the Bearer key", auth)
	}
	if _, ok := cfg["api_key"]; ok {
		t.Errorf("handshake api_key = %v, want the key only in the header", cfg["api_key"])
	}
	if cfg["model"] != "stt-rt-v5" {
		t.Errorf("handshake model = %v, want stt-rt-v5", cfg["model"])
	}
	if cfg["sample_rate"] != float64(16000) {
		t.Errorf("handshake sample_rate = %v, want the pipeline's 16000", cfg["sample_rate"])
	}
	if cfg["audio_format"] != "pcm_s16le" {
		t.Errorf("handshake audio_format = %v, want pcm_s16le", cfg["audio_format"])
	}
	if cfg["enable_endpoint_detection"] != true {
		t.Errorf("handshake enable_endpoint_detection = %v, want it on by default",
			cfg["enable_endpoint_detection"])
	}
}

// TestConnectOmitsUnsetSettings checks a setting nobody chose is left out rather
// than sent as a zero: a zero means something to Soniox, and sending one would
// override its own default with a value the caller never asked for.
func TestConnectOmitsUnsetSettings(t *testing.T) {
	_, got := dialSoniox(t, Config{})

	var cfg map[string]any
	select {
	case msg := <-got.handshake:
		_ = json.Unmarshal([]byte(msg), &cfg)
	case <-time.After(3 * time.Second):
		t.Fatal("no handshake reached the server")
	}

	for _, key := range []string{
		"language_hints", "context", "enable_speaker_diarization",
		"max_endpoint_delay_ms", "endpoint_sensitivity", "client_reference_id",
	} {
		if _, present := cfg[key]; present {
			t.Errorf("handshake carries %q, want it left out when nothing set it", key)
		}
	}
}

// TestSendWritesAudioAsBinary checks audio goes out as binary frames, the only
// shape Soniox reads PCM from.
func TestSendWritesAudioAsBinary(t *testing.T) {
	s, got := dialSoniox(t, Config{})
	if err := s.Send([]byte{9, 8, 7}); err != nil {
		t.Fatalf("Send: %v", err)
	}
	select {
	case audio := <-got.audio:
		if string(audio) != string([]byte{9, 8, 7}) {
			t.Errorf("the server saw %v, want the PCM that was sent", audio)
		}
	case <-time.After(3 * time.Second):
		t.Error("no audio reached the server")
	}
}

// TestRecvAccumulatesFinalTokensUntilTheTurnEnds checks the shape of Soniox's
// output: final tokens accumulate and are only flushed as a finished utterance
// when the end marker arrives, while provisional tokens surface as interim
// results in the meantime.
func TestRecvAccumulatesFinalTokensUntilTheTurnEnds(t *testing.T) {
	s, got := dialSoniox(t, Config{})

	got.send <- `{"tokens":[{"text":"bon","is_final":true},{"text":"soir","is_final":false}]}`
	interim := recvSoniox(t, s)
	if len(interim) != 1 || interim[0].Final || interim[0].Text != "bonsoir" {
		t.Fatalf("first result = %+v, want the interim \"bonsoir\"", interim)
	}

	got.send <- `{"tokens":[{"text":"soir","is_final":true},{"text":"<end>","is_final":true}]}`
	final := recvSoniox(t, s)
	if len(final) != 1 {
		t.Fatalf("second result = %+v, want one", final)
	}
	if !final[0].Final || !final[0].EndOfTurn || final[0].Text != "bonsoir" {
		t.Errorf("second result = %+v, want the finished utterance \"bonsoir\"", final[0])
	}
}

// TestRecvReportsAServerError checks an error message is reported with the
// reason the server gave, once the utterance it cut short has gone out as the
// transcript it was, and that the session reads on: Soniox decides whether the
// error ends it.
func TestRecvReportsAServerError(t *testing.T) {
	s, got := dialSoniox(t, Config{})
	var mu sync.Mutex
	var reported []string
	s.report = func(_ context.Context, code json.RawMessage, message string) {
		mu.Lock()
		defer mu.Unlock()
		reported = append(reported, string(code)+" "+message)
	}
	reports := func() []string {
		mu.Lock()
		defer mu.Unlock()
		return slices.Clone(reported)
	}

	got.send <- `{"tokens":[{"text":"hello","is_final":true}]}`
	got.send <- `{"tokens":[],"error_code":401,"error_message":"bad key"}`
	got.send <- `{"tokens":[{"text":"again","is_final":false}]}`

	if r := recvSoniox(t, s); len(r) != 1 || r[0].Final || r[0].Text != "hello" {
		t.Fatalf("first results = %+v, want the interim \"hello\"", r)
	}
	r := recvSoniox(t, s)
	if len(r) == 0 || !r[len(r)-1].Final || !r[len(r)-1].EndOfTurn || r[len(r)-1].Text != "hello" {
		t.Fatalf("results at the error = %+v, want them to end with the utterance it cut short", r)
	}
	if n := len(reports()); n != 0 {
		t.Fatalf("the error was reported ahead of the transcript it followed: %v", reports())
	}
	if r := recvSoniox(t, s); len(r) != 1 || r[0].Text != "again" {
		t.Errorf("results after the error = %+v, want the session to read on", r)
	}
	if got := reports(); !slices.Equal(got, []string{"401 bad key"}) {
		t.Errorf("reported %v, want the server's own code and reason", got)
	}
}

// Ported from upstream (the endpoint_and_finalize_push_one_transcription case).
// Soniox answers a finalize with "<fin>", alongside the "<end>" of an endpoint
// it detected or in a message of its own after it. Either way the utterance is
// pushed once: the marker behind the one that ended it has nothing left to end.
func TestEndpointAndFinalizePushOneTranscription(t *testing.T) {
	for name, answer := range map[string][]string{
		"end and fin together": {
			`{"tokens":[{"text":"Bye bye.","is_final":true},{"text":"<end>","is_final":true},` +
				`{"text":"<fin>","is_final":true}]}`,
		},
		"end then fin": {
			`{"tokens":[{"text":"Bye bye.","is_final":true},{"text":"<end>","is_final":true}]}`,
			`{"tokens":[{"text":"<fin>","is_final":true}]}`,
		},
	} {
		t.Run(name, func(t *testing.T) {
			s, got := dialSoniox(t, Config{})
			got.send <- `{"tokens":[{"text":"Bye","is_final":false}]}`
			for _, m := range answer {
				got.send <- m
			}
			got.send <- `{"tokens":[],"finished":true}`

			var results []stt.Result
			for {
				r, err := recvSonioxResults(t, s)
				if err != nil {
					break
				}
				results = append(results, r...)
			}
			want := []stt.Result{
				{Text: "Bye"},
				{Text: "Bye bye.", Final: true, EndOfTurn: true},
			}
			if !slices.Equal(results, want) {
				t.Errorf("results = %+v, want one interim and one transcript %+v", results, want)
			}
		})
	}
}

// Ported from upstream (test_stop_completes_teardown_when_the_end_of_audio_send_fails).
// The end-of-audio frame is best effort: when it cannot be sent, the failure is
// handed back without waiting on anything, and closing the session still
// completes.
func TestTeardownCompletesWhenTheEndOfAudioSendFails(t *testing.T) {
	s, got := dialSoniox(t, Config{})
	// The connection is gone under the stream, so the end-of-audio send fails.
	_ = s.conn.CloseNow()
	waitEnded(t, got)

	done := make(chan struct{})
	go func() {
		defer close(done)
		if err := s.Stop(); err == nil {
			t.Error("Stop reported the end-of-audio sent on a connection that is gone")
		}
		_ = s.Close()
	}()
	select {
	case <-done:
	case <-time.After(3 * time.Second):
		t.Fatal("the teardown did not complete after the end-of-audio send failed")
	}
}

// Ported from upstream (test_stop_disconnects_once). Ending the pipeline
// closes the session once: one end-of-audio frame and one close, though both
// the end of the pipeline and the service's cleanup tear it down.
func TestStopDisconnectsOnce(t *testing.T) {
	r := runSonioxSTT(t, Config{})
	r.stop(t)
	waitEnded(t, r.got)

	var endOfAudio int
	for len(r.got.audio) > 0 {
		if len(<-r.got.audio) == 0 {
			endOfAudio++
		}
	}
	if endOfAudio != 1 {
		t.Errorf("end-of-audio frames = %d, want 1", endOfAudio)
	}
	if n := len(r.got.handshake); n != 0 {
		t.Errorf("%d more sessions were opened after the first", n)
	}
	select {
	case <-r.got.ended:
		t.Error("a second session was closed")
	default:
	}
}

// Ported from upstream (test_cancel_does_not_send_end_of_audio). A canceled
// pipeline closes the session without telling Soniox the audio is complete.
func TestCancelDoesNotSendEndOfAudio(t *testing.T) {
	r := runSonioxSTT(t, Config{})
	r.task.Cancel(context.Background(), "test")
	select {
	case <-r.done:
	case <-time.After(5 * time.Second):
		t.Fatal("the pipeline did not finish")
	}
	waitEnded(t, r.got)

	for len(r.got.audio) > 0 {
		if len(<-r.got.audio) == 0 {
			t.Error("the canceled session was sent the end-of-audio frame")
		}
	}
}

// Ported from upstream (test_endpoint_transcript_emits_usage_before_transcription_frame).
// The audio a transcript covers is reported as usage ahead of the transcript,
// so the span the transcript closes carries it, and is not reported again.
func TestEndpointTranscriptReportsUsageBeforeTheTranscript(t *testing.T) {
	r := runSonioxSTT(t, Config{}, pipeline.Params{EnableUsageMetrics: true})
	r.task.QueueFrame(frames.NewInputAudioRawFrame(make([]byte, 16000), 16000, 1))
	select {
	case <-r.got.audio:
	case <-time.After(3 * time.Second):
		t.Fatal("no audio reached the server")
	}

	r.got.send <- `{"tokens":[{"text":"Hello.","is_final":true},{"text":"<end>","is_final":true}]}`
	r.waitFor(t, "the transcript", func(e any) bool { return e == "Hello." })
	r.stop(t)

	if got, want := r.events(), []any{0.5, "Hello."}; !slices.Equal(got, want) {
		t.Errorf("reached the end %v, want the usage ahead of the transcript %v", got, want)
	}
}

// Ported from upstream (test_soniox_error_code_sets_category). The error code
// Soniox reports is an HTTP status, which gives the error its category; an
// error without one is left unknown.
func TestSTTErrorCodeSetsTheCategory(t *testing.T) {
	r := runSonioxSTT(t, Config{})
	defer r.stop(t)

	r.got.send <- `{"tokens":[],"error_code":402,"error_message":"balance exhausted"}`
	r.got.send <- `{"tokens":[],"error_code":401,"error_message":"invalid key"}`
	r.got.send <- `{"tokens":[],"error_message":"no code"}`

	var got []*frames.ErrorFrame
	waitFor(t, "the errors never reached the pipeline", func() bool {
		got = nil
		for _, ef := range r.reportedErrors() {
			if strings.HasPrefix(ef.Error, "Soniox STT error") {
				got = append(got, ef)
			}
		}
		return len(got) == 3
	})
	want := []errs.Category{errs.Quota, errs.Authentication, errs.Unknown}
	for i, ef := range got {
		if ef.Category != want[i] {
			t.Errorf("error %d (%s) category = %q, want %q", i, ef.Error, ef.Category, want[i])
		}
	}
}

// Ported from upstream (test_language_update_reconnects_with_that_language_as_the_hint).
// Soniox reads the language hints when a session opens, not the language, so a
// language update replaces the hints and reopens the session with it.
func TestLanguageUpdateReconnectsWithThatLanguageAsTheHint(t *testing.T) {
	r := runSonioxSTT(t, Config{Language: language.Finnish})
	defer r.stop(t)
	if hints := r.first["language_hints"]; !slices.Equal(anyStrings(hints), []string{"fi"}) {
		t.Fatalf("first session language_hints = %v, want [fi]", hints)
	}

	delta := &settings.STT{Language: settings.Set("sv")}
	r.task.QueueFrame(frames.NewSTTUpdateSettingsFrame(delta))

	if hints := handshakeOf(t, r.got)["language_hints"]; !slices.Equal(anyStrings(hints), []string{"sv"}) {
		t.Errorf("reopened session language_hints = %v, want [sv]", hints)
	}
	// Ported from upstream (test_language_update_leaves_the_callers_delta_unchanged).
	// The language is moved into the hints of the service's own settings, not
	// in the update the caller handed over.
	if got, _ := delta.Language.Value(); got != "sv" {
		t.Errorf("the caller's update now gives language %q, want sv as given", got)
	}
}

// Ported from upstream (test_language_update_outside_the_language_enum_does_not_raise).
// A language the catalog does not name is not made a hint: the hints stay as
// they were, and the session is reopened for the change all the same.
func TestLanguageUpdateOutsideTheCatalogKeepsTheHints(t *testing.T) {
	r := runSonioxSTT(t, Config{Language: language.Finnish})
	defer r.stop(t)

	r.task.QueueFrame(frames.NewSTTUpdateSettingsFrame(&settings.STT{Language: settings.Set("zh-Hant")}))

	if hints := handshakeOf(t, r.got)["language_hints"]; !slices.Equal(anyStrings(hints), []string{"fi"}) {
		t.Errorf("reopened session language_hints = %v, want [fi] unchanged", hints)
	}
}

// Ported from upstream (test_constructor_language_connects_with_that_language_as_the_hint).
// The language the service is built with is the hint of its first session.
func TestConstructorLanguageConnectsWithThatLanguageAsTheHint(t *testing.T) {
	r := runSonioxSTT(t, Config{Language: language.Swedish})
	defer r.stop(t)
	if hints := r.first["language_hints"]; !slices.Equal(anyStrings(hints), []string{"sv"}) {
		t.Errorf("language_hints = %v, want [sv]", hints)
	}
}

// Ported from upstream (test_connect_failure_clears_stale_websocket_without_raising).
// A session that fails to open leaves the service holding none, rather than
// the one it had before, and the failure is returned rather than raised.
func TestConnectFailureLeavesNoSession(t *testing.T) {
	refused := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusServiceUnavailable)
	}))
	defer refused.Close()

	svc := NewSTT(Config{APIKey: "test-key", URL: "ws" + strings.TrimPrefix(refused.URL, "http"), SampleRate: 16000})
	if err := svc.ConnectWebsocket(t.Context()); err == nil {
		t.Error("ConnectWebsocket succeeded against an endpoint that refuses it")
	}
	if svc.Connected() {
		t.Error("the service holds a session after the connect failed")
	}
}

// Ported from upstream (test_connect_enables_endpoint_detection_in_both_modes).
// Endpoint detection is always on: it ends the transcript of speech the VAD
// missed, so a configuration asking for it off is not followed.
func TestConnectEnablesEndpointDetection(t *testing.T) {
	off := false
	for _, cfg := range []Config{{}, {EnableEndpointDetection: &off}} {
		_, got := dialSoniox(t, cfg)
		if on := handshakeOf(t, got)["enable_endpoint_detection"]; on != true {
			t.Errorf("enable_endpoint_detection = %v, want true", on)
		}
	}
}

// The default model is Soniox's current real-time model.
func TestConnectUsesTheCurrentModelByDefault(t *testing.T) {
	r := runSonioxSTT(t, Config{})
	defer r.stop(t)
	if model := r.first["model"]; model != "stt-rt-v6" {
		t.Errorf("model = %v, want stt-rt-v6", model)
	}
}

// Ported from upstream (test_receive_messages_sets_final_transcription_language).
// The transcript is labeled with the language most of its tokens were
// identified as, and its text is the tokens as Soniox spaced them.
func TestFinalTranscriptCarriesTheIdentifiedLanguage(t *testing.T) {
	s, got := dialSoniox(t, Config{})
	got.send <- `{"tokens":[{"text":"Ik","is_final":true,"language":"nl"},` +
		`{"text":" zoek","is_final":true,"language":"nl"},` +
		`{"text":" computer","is_final":true,"language":"en"},{"text":"<end>","is_final":true}]}`

	r := recvSoniox(t, s)
	want := []stt.Result{{Text: "Ik zoek computer", Final: true, EndOfTurn: true, Language: "nl"}}
	if !slices.Equal(r, want) {
		t.Errorf("results = %+v, want %+v", r, want)
	}
}

// Ported from upstream (test_receive_messages_allows_final_transcription_without_language
// and test_transcript_is_not_split_without_diarization). Tokens naming no
// language leave the transcript unlabeled, and without diarization the
// utterance goes out whole whoever spoke it.
func TestFinalTranscriptWithoutALanguageIsUnlabeledAndWhole(t *testing.T) {
	s, got := dialSoniox(t, Config{})
	got.send <- `{"tokens":[{"text":"Tell","is_final":true,"speaker":"1"},` +
		`{"text":" me","is_final":true,"speaker":"1"},{"text":" a","is_final":true,"speaker":"2"},` +
		`{"text":" joke.","is_final":true,"speaker":"2"},{"text":"<end>","is_final":true}]}`

	r := recvSoniox(t, s)
	want := []stt.Result{{Text: "Tell me a joke.", Final: true, EndOfTurn: true}}
	if !slices.Equal(r, want) {
		t.Errorf("results = %+v, want %+v", r, want)
	}
}

// Text is passed on as Soniox spaced it, leading space and all, and a final
// token Soniox has not yet ended an utterance with still shows in the interim.
// A marker that is not final yet is provisional text like any other.
func TestTranscriptTextIsKeptAsSonioxSpacedIt(t *testing.T) {
	s, got := dialSoniox(t, Config{})
	got.send <- `{"tokens":[{"text":" Hello","is_final":true},{"text":" there","is_final":false},` +
		`{"text":"<end>","is_final":false}]}`
	got.send <- `{"tokens":[{"text":" there","is_final":true},{"text":"<end>","is_final":true}]}`

	if r := recvSoniox(t, s); !slices.Equal(r, []stt.Result{{Text: " Hello there<end>"}}) {
		t.Errorf("interim = %+v, want the text as sent", r)
	}
	want := []stt.Result{{Text: " Hello there", Final: true, EndOfTurn: true}}
	if r := recvSoniox(t, s); !slices.Equal(r, want) {
		t.Errorf("transcript = %+v, want %+v", r, want)
	}
}

// Ported from upstream (the default-mode service metadata cases, which leave
// the turn strategies unset). The pipeline's own turn strategies decide when
// the user's turn ends, so the service recommends none of its own.
func TestServiceMetadataRecommendsNoTurnStrategies(t *testing.T) {
	if got := NewSTT(Config{APIKey: "k"}).ServiceMetadataFrame().RecommendedUserTurnStrategies(); got != nil {
		t.Errorf("recommended turn strategies = %v, want none", got)
	}
}

// Ported from upstream (the default-mode cases where neither transcripts nor
// VAD frames emit turn frames). The service proposes no turns: the VAD and the
// transcripts pass through, and nothing else is added.
// The VAD stop asks Soniox to finalize, so the transcript arrives sooner.
func TestServiceProposesNoTurnsAndFinalizesOnTheVADStop(t *testing.T) {
	r := runSonioxSTT(t, Config{})
	var mu sync.Mutex
	var proposed []frames.Frame
	events.On(&r.task.Registry, pipeline.EventFrameReachedDownstream, func(_ context.Context, f frames.Frame) {
		switch f.(type) {
		case *frames.ProposedUserStartedSpeakingFrame, *frames.ProposedUserStoppedSpeakingFrame:
			mu.Lock()
			proposed = append(proposed, f)
			mu.Unlock()
		}
	})

	r.task.QueueFrame(frames.NewVADUserStartedSpeakingFrame(0, time.Now()))
	r.got.send <- `{"tokens":[{"text":"Hel","is_final":false}]}`
	r.got.send <- `{"tokens":[{"text":"Hello.","is_final":true},{"text":"<end>","is_final":true}]}`
	r.waitFor(t, "the transcript", func(e any) bool { return e == "Hello." })
	r.task.QueueFrame(frames.NewVADUserStoppedSpeakingFrame(0, time.Now()))

	select {
	case msg := <-r.got.control:
		if msg != finalizeMessage {
			t.Errorf("Soniox was sent %s on the VAD stop, want the finalize message", msg)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("the VAD stop sent Soniox no finalize")
	}
	r.stop(t)

	mu.Lock()
	defer mu.Unlock()
	if len(proposed) != 0 {
		t.Errorf("proposed turn frames %v, want none", proposed)
	}
}

// An idle session is kept open with Soniox's own keepalive message rather than
// with silence, which Soniox would bill as audio.
func TestKeepaliveIsSonioxsOwnMessage(t *testing.T) {
	c := &connector{cfg: sttConfig(), live: newSTTSettings(sttConfig())}
	if got, want := c.Keepalive(), (stt.KeepaliveOptions{Timeout: time.Second, Interval: 5 * time.Second}); got != want {
		t.Errorf("keepalive = %+v, want %+v", got, want)
	}

	s, got := dialSoniox(t, Config{})
	if err := s.SendKeepalive(make([]byte, 3200)); err != nil {
		t.Fatalf("SendKeepalive: %v", err)
	}
	select {
	case msg := <-got.control:
		if msg != keepaliveMessage {
			t.Errorf("Soniox was sent %s, want the keepalive message", msg)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("no keepalive reached the server")
	}
	if n := len(got.audio); n != 0 {
		t.Errorf("the keepalive sent %d audio frames, want none", n)
	}
}

// sttRun is a Soniox STT service in a pipeline against the fake endpoint,
// recording each usage report as its seconds and each transcript as its text
// as they reach the end of it, and the errors it reports upstream.
type sttRun struct {
	got  *sonioxSession
	task *pipeline.Worker
	// first is the handshake of the session the service opened on start.
	first map[string]any
	done  chan error

	mu     sync.Mutex
	down   []any
	errors []*frames.ErrorFrame
}

// runSonioxSTT starts the service and waits for its first session to open.
func runSonioxSTT(t *testing.T, cfg Config, params ...pipeline.Params) *sttRun {
	t.Helper()
	endpoint, got := sonioxServer(t)
	cfg.URL = endpoint
	cfg.APIKey = "test-key"
	cfg.SampleRate = 16000
	r := &sttRun{got: got, done: make(chan error, 1)}
	wc := pipeline.WorkerConfig{
		IdleTimeout:             -1,
		ReachedDownstreamFilter: pipeline.AnyFrame,
		ReachedUpstreamFilter:   pipeline.AnyFrame,
	}
	if len(params) > 0 {
		wc.Params = params[0]
	}
	svc := NewSTT(cfg)
	r.task = pipeline.NewWorker(pipeline.New(svc), wc)
	events.On(&r.task.Registry, pipeline.EventFrameReachedDownstream, func(_ context.Context, f frames.Frame) {
		r.mu.Lock()
		defer r.mu.Unlock()
		switch f := f.(type) {
		case *frames.MetricsFrame:
			for _, d := range f.Data {
				if u, ok := d.(frames.STTUsageMetricsData); ok {
					r.down = append(r.down, u.Value.AudioSeconds)
				}
			}
		case *frames.TranscriptionFrame:
			r.down = append(r.down, f.Text)
		}
	})
	events.On(&r.task.Registry, pipeline.EventFrameReachedUpstream, func(_ context.Context, f frames.Frame) {
		if ef, ok := f.(*frames.ErrorFrame); ok {
			r.mu.Lock()
			r.errors = append(r.errors, ef)
			r.mu.Unlock()
		}
	})
	go func() { r.done <- r.task.Run(context.Background()) }()
	r.first = handshakeOf(t, got)
	// The handshake goes out before the service holds the session, and audio
	// it has no session for is dropped.
	waitFor(t, "the service never held its session", svc.Connected)
	return r
}

func (r *sttRun) events() []any {
	r.mu.Lock()
	defer r.mu.Unlock()
	return slices.Clone(r.down)
}

func (r *sttRun) reportedErrors() []*frames.ErrorFrame {
	r.mu.Lock()
	defer r.mu.Unlock()
	return slices.Clone(r.errors)
}

func (r *sttRun) waitFor(t *testing.T, what string, match func(any) bool) {
	t.Helper()
	waitFor(t, what+" never reached the end of the pipeline", func() bool {
		return slices.ContainsFunc(r.events(), match)
	})
}

// stop ends the pipeline and waits for it to finish.
func (r *sttRun) stop(t *testing.T) {
	t.Helper()
	r.task.StopWhenDone()
	select {
	case <-r.done:
	case <-time.After(5 * time.Second):
		t.Fatal("the pipeline did not finish")
	}
}

// handshakeOf waits for the next session to open and returns its handshake.
func handshakeOf(t *testing.T, got *sonioxSession) map[string]any {
	t.Helper()
	select {
	case msg := <-got.handshake:
		var cfg map[string]any
		if err := json.Unmarshal([]byte(msg), &cfg); err != nil {
			t.Fatalf("the handshake was not JSON: %v", err)
		}
		return cfg
	case <-time.After(3 * time.Second):
		t.Fatal("no session was opened")
		return nil
	}
}

// waitEnded waits for the server to see a session closed.
func waitEnded(t *testing.T, got *sonioxSession) {
	t.Helper()
	select {
	case <-got.ended:
	case <-time.After(3 * time.Second):
		t.Fatal("the session was never closed")
	}
}

// anyStrings is a decoded JSON list of strings.
func anyStrings(v any) []string {
	list, _ := v.([]any)
	out := make([]string, 0, len(list))
	for _, item := range list {
		s, _ := item.(string)
		out = append(out, s)
	}
	return out
}

// TestRecvEndsOnFinished checks the finished marker closes the stream cleanly,
// which is what tells the base the session is over rather than broken.
func TestRecvEndsOnFinished(t *testing.T) {
	s, got := dialSoniox(t, Config{})
	got.send <- `{"finished":true}`

	if err := recvSonioxError(t, s); !errors.Is(err, io.EOF) {
		t.Errorf("Recv = %v, want io.EOF", err)
	}
}

// TestStopSignalsEndOfAudio checks the session is not simply dropped when the
// pipeline ends: an empty binary frame is what tells Soniox the audio is
// complete, and it goes out while the socket is still open.
func TestStopSignalsEndOfAudio(t *testing.T) {
	s, got := dialSoniox(t, Config{})
	if err := s.Stop(); err != nil {
		t.Fatalf("Stop: %v", err)
	}
	select {
	case audio := <-got.audio:
		if len(audio) != 0 {
			t.Errorf("the server saw %v on stop, want the empty end-of-audio frame", audio)
		}
	case <-time.After(3 * time.Second):
		t.Error("nothing was sent on stop")
	}
}

// TestCloseSendsNoEndOfAudio checks a session closed to reopen it with new
// settings, or because the pipeline was canceled, is just closed: the
// end-of-audio frame belongs to ending the pipeline.
func TestCloseSendsNoEndOfAudio(t *testing.T) {
	s, got := dialSoniox(t, Config{})
	if err := s.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	waitEnded(t, got)
	if n := len(got.audio); n != 0 {
		t.Errorf("the server saw %d frames on close, want none", n)
	}
}

// recvSonioxError reads until Recv fails and returns why. A read that never
// returns is the failure being tested going unnoticed, so it is bounded rather
// than left to hang until the whole suite times out.
func recvSonioxError(t *testing.T, s stt.Stream) error {
	t.Helper()
	done := make(chan error, 1)
	go func() {
		_, err := s.Recv()
		done <- err
	}()
	select {
	case err := <-done:
		return err
	case <-time.After(3 * time.Second):
		t.Fatal("Recv neither returned a result nor failed")
		return nil
	}
}

// recvSoniox reads one batch of results.
func recvSoniox(t *testing.T, s stt.Stream) []stt.Result {
	t.Helper()
	results, err := recvSonioxResults(t, s)
	if err != nil {
		t.Fatalf("Recv: %v", err)
	}
	return results
}

// recvSonioxResults reads one batch of results, or the error that ended the
// session, bounded like recvSonioxError.
func recvSonioxResults(t *testing.T, s stt.Stream) ([]stt.Result, error) {
	t.Helper()
	type outcome struct {
		results []stt.Result
		err     error
	}
	done := make(chan outcome, 1)
	go func() {
		r, err := s.Recv()
		done <- outcome{r, err}
	}()
	select {
	case o := <-done:
		return o.results, o.err
	case <-time.After(3 * time.Second):
		t.Fatal("Recv returned nothing")
		return nil, nil
	}
}
