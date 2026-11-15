package deepgram

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"net/url"
	"slices"
	"sync"
	"testing"
	"time"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/service"
	"github.com/gojargo/jargo/service/settings"
	"github.com/gojargo/jargo/service/tts"
	"github.com/gojargo/jargo/utils/events"
)

// auraAudioChunk is the audio the fake Aura endpoint answers a Flush with.
//
//nolint:gochecknoglobals // test fixture, read-only
var auraAudioChunk = bytes.Repeat([]byte{0x04, 0x05}, 512)

// auraServerOptions shape how the fake Aura endpoint answers.
type auraServerOptions struct {
	// holdFlush withholds the answer to a Flush, leaving the turn open the way
	// it is while Aura is still synthesizing.
	holdFlush bool
	// warning sends a Warning ahead of the audio.
	warning bool
}

// auraServer is a fake Aura WebSocket endpoint: it records what it was dialed
// with and every message it was sent, answers a Flush with audio and Flushed,
// and a Clear with Cleared.
type auraServer struct {
	opts auraServerOptions

	mu       sync.Mutex
	queries  []url.Values
	paths    []string
	auth     string
	messages []map[string]any
	dials    int
}

// newAuraServer starts a fake Aura endpoint and returns it with its base URL.
func newAuraServer(t *testing.T, opts auraServerOptions) (*auraServer, string) {
	t.Helper()
	f := &auraServer{opts: opts}
	srv := httptest.NewServer(http.HandlerFunc(f.serve))
	t.Cleanup(srv.Close)
	return f, wsURL(srv.URL)
}

func (f *auraServer) serve(w http.ResponseWriter, r *http.Request) {
	f.mu.Lock()
	f.dials++
	f.queries = append(f.queries, r.URL.Query())
	f.paths = append(f.paths, r.URL.Path)
	f.auth = r.Header.Get("Authorization")
	f.mu.Unlock()
	c, err := websocket.Accept(w, r, nil)
	if err != nil {
		return
	}
	defer func() { _ = c.CloseNow() }()
	ctx := r.Context()
	send := func(m map[string]any) { _ = c.Write(ctx, websocket.MessageText, mustJSON(m)) }

	for {
		_, data, err := c.Read(ctx)
		if err != nil {
			return
		}
		msg := map[string]any{}
		if json.Unmarshal(data, &msg) != nil {
			return
		}
		f.mu.Lock()
		f.messages = append(f.messages, msg)
		f.mu.Unlock()

		switch msg["type"] {
		case "Speak":
			send(map[string]any{"type": "Metadata", "request_id": "test-request", "model_name": "aura-2-helena-en"})
		case "Flush":
			if f.opts.holdFlush {
				continue
			}
			if f.opts.warning {
				send(map[string]any{"type": "Warning", "warn_msg": "Text was truncated."})
			}
			_ = c.Write(ctx, websocket.MessageBinary, auraAudioChunk)
			send(map[string]any{"type": "Flushed", "sequence_id": 0})
		case "Clear":
			send(map[string]any{"type": "Cleared", "sequence_id": 0})
		case "Close":
			_ = c.Close(websocket.StatusNormalClosure, "")
			return
		}
	}
}

// sent is every message the endpoint was sent, in order.
func (f *auraServer) sent() []map[string]any {
	f.mu.Lock()
	defer f.mu.Unlock()
	return slices.Clone(f.messages)
}

// sentTypes is the type of every message the endpoint was sent, in order.
func (f *auraServer) sentTypes() []string {
	var out []string
	for _, m := range f.sent() {
		typ, _ := m["type"].(string)
		out = append(out, typ)
	}
	return out
}

// dialCount is how many connections were opened.
func (f *auraServer) dialCount() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.dials
}

// lastQuery is the query the latest connection was opened with.
func (f *auraServer) lastQuery(t *testing.T) url.Values {
	t.Helper()
	f.mu.Lock()
	defer f.mu.Unlock()
	if len(f.queries) == 0 {
		t.Fatal("the endpoint was never dialed")
	}
	return f.queries[len(f.queries)-1]
}

// awaitSentType waits for the endpoint to have been sent a message of type typ.
func (f *auraServer) awaitSentType(t *testing.T, typ string) {
	t.Helper()
	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		if slices.Contains(f.sentTypes(), typ) {
			return
		}
		time.Sleep(5 * time.Millisecond)
	}
	t.Fatalf("the endpoint was sent %v, want a %s", f.sentTypes(), typ)
}

// auraTestConfig is a config pointed at baseURL.
func auraTestConfig(baseURL string) WebsocketTTSConfig {
	return WebsocketTTSConfig{APIKey: "test-key", BaseURL: baseURL, SampleRate: 24000}
}

// connectedAuraSynth builds the service against baseURL and connects it, the
// way the pipeline starting does.
func connectedAuraSynth(t *testing.T, cfg WebsocketTTSConfig) (*tts.Base, *wsSynth) {
	t.Helper()
	b, s := newWebsocketTTS(cfg)
	s.Connect(t.Context())
	t.Cleanup(func() { s.Disconnect(context.Background()) })
	if !s.Connected() {
		t.Fatal("the synthesizer did not connect")
	}
	return b, s
}

// runAuraPipeline runs the service in a pipeline with params, queueing each
// frame in order (a time.Duration waits that long instead), ends the pipeline,
// and returns the frames that left it downstream and upstream.
func runAuraPipeline(t *testing.T, svc *tts.Base, params pipeline.Params, steps ...any) (down, up []frames.Frame) {
	t.Helper()
	var mu sync.Mutex
	task := pipeline.NewWorker(pipeline.New(svc), pipeline.WorkerConfig{
		ReachedDownstreamFilter: pipeline.AnyFrame,
		ReachedUpstreamFilter:   pipeline.AnyFrame,
		Params:                  params,
	})
	events.On(&task.Registry, pipeline.EventFrameReachedDownstream, func(_ context.Context, f frames.Frame) {
		mu.Lock()
		down = append(down, f)
		mu.Unlock()
	})
	events.On(&task.Registry, pipeline.EventFrameReachedUpstream, func(_ context.Context, f frames.Frame) {
		mu.Lock()
		up = append(up, f)
		mu.Unlock()
	})
	done := make(chan error, 1)
	go func() { done <- task.Run(context.Background()) }()
	for _, s := range steps {
		switch v := s.(type) {
		case time.Duration:
			time.Sleep(v)
		case frames.Frame:
			task.QueueFrame(v)
		default:
			t.Fatalf("cannot run step %T", s)
		}
	}
	task.StopWhenDone()
	select {
	case <-done:
	case <-time.After(10 * time.Second):
		t.Fatal("the pipeline did not finish")
	}
	mu.Lock()
	defer mu.Unlock()
	return slices.Clone(down), slices.Clone(up)
}

// connectionErrors records the messages of the service's connection-error
// events.
type connectionErrors struct {
	mu   sync.Mutex
	msgs []string
}

func watchConnectionErrors(b *tts.Base) *connectionErrors {
	c := &connectionErrors{}
	events.On(b.Events(), service.EventConnectionError, func(_ context.Context, msg string) {
		c.mu.Lock()
		c.msgs = append(c.msgs, msg)
		c.mu.Unlock()
	})
	return c
}

// await waits for n events, since event handlers run in the background.
func (c *connectionErrors) await(n int) []string {
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		c.mu.Lock()
		got := slices.Clone(c.msgs)
		c.mu.Unlock()
		if len(got) >= n {
			return got
		}
		time.Sleep(5 * time.Millisecond)
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	return slices.Clone(c.msgs)
}

// A rejected API key would be rejected again on a retry, so the service gives
// up on the first refusal.
func TestWebsocketTTSRejectedAPIKeyMakesTheServiceUnusable(t *testing.T) {
	b, s := newWebsocketTTS(auraTestConfig(refusingServer(t, http.StatusUnauthorized, "")))

	_ = s.ConnectWebsocket(t.Context())

	if b.Usable() {
		t.Error("the service is still usable after its API key was rejected")
	}
}

// A server error is worth retrying, so the service stays usable.
func TestWebsocketTTSServerErrorLeavesTheServiceUsable(t *testing.T) {
	b, s := newWebsocketTTS(auraTestConfig(refusingServer(t, http.StatusServiceUnavailable, "")))

	_ = s.ConnectWebsocket(t.Context())

	if !b.Usable() {
		t.Error("a server error made the service unusable")
	}
}

// Deepgram's reason for refusing the handshake reaches the error and the
// event.
func TestWebsocketTTSRefusalReportsDeepgramsReason(t *testing.T) {
	reason := "'speed' must be within documented range."
	b, s := newWebsocketTTS(auraTestConfig(refusingServer(t, http.StatusBadRequest, reason)))
	errs := watchConnectionErrors(b)

	_ = s.ConnectWebsocket(t.Context())

	want := "Deepgram refused the connection (HTTP 400): " + reason
	if got := errs.await(1); !slices.Equal(got, []string{want}) {
		t.Errorf("connection errors = %q, want %q", got, want)
	}
}

// RunTTS sends the text and returns, leaving the audio to the receive loop, so
// the service reports no processing time.
func TestWebsocketTTSDoesNotReportProcessingMetrics(t *testing.T) {
	_, baseURL := newAuraServer(t, auraServerOptions{})
	svc, _ := newWebsocketTTS(auraTestConfig(baseURL))
	noInitial := false

	down, _ := runAuraPipeline(t, svc,
		pipeline.Params{EnableMetrics: true, SendInitialEmptyMetrics: &noInitial},
		frames.NewLLMFullResponseStartFrame(),
		frames.NewLLMTextFrame("One. Two."),
		frames.NewLLMFullResponseEndFrame(),
		300*time.Millisecond,
		frames.NewBotStoppedSpeakingFrame(),
	)

	for _, mf := range framesOf[*frames.MetricsFrame](down) {
		for _, d := range mf.Data {
			if _, ok := d.(frames.ProcessingMetricsData); ok {
				t.Fatalf("a processing metric was reported: %+v", d)
			}
		}
	}
	if len(framesOf[*frames.TTSAudioRawFrame](down)) == 0 {
		t.Error("no audio was played")
	}
}

// The WebSocket service formats a pronunciation exactly as the HTTP one does.
func TestWebsocketTTSFormatsPronunciationsLikeHTTP(t *testing.T) {
	want, _ := (&synthesizer{}).FormatPronunciation("Crete", "kriːt")
	if got, _ := (&wsSynth{}).FormatPronunciation("Crete", "kriːt"); got != want {
		t.Fatalf("the WebSocket service formats %q, want %q", got, want)
	}
}

func TestWebsocketTTSValidate(t *testing.T) {
	if err := (WebsocketTTSConfig{APIKey: "k"}).Validate(); err != nil {
		t.Fatalf("valid config rejected: %v", err)
	}
	if err := (WebsocketTTSConfig{}).Validate(); err == nil {
		t.Fatal("config without APIKey should be rejected")
	}
	for _, enc := range []string{"linear16", "mulaw", "alaw", "MULAW"} {
		if err := (WebsocketTTSConfig{APIKey: "k", Encoding: enc}).Validate(); err != nil {
			t.Errorf("encoding %q rejected: %v", enc, err)
		}
	}
	err := WebsocketTTSConfig{APIKey: "k", Encoding: "mp3"}.Validate()
	if !errors.Is(err, errUnsupportedEncoding) {
		t.Errorf("encoding mp3 = %v, want it rejected", err)
	}
}

func TestWebsocketTTSDefaults(t *testing.T) {
	cfg := WebsocketTTSConfig{APIKey: "k"}.withDefaults()
	if cfg.Voice != "aura-2-helena-en" || cfg.BaseURL != "wss://api.deepgram.com" ||
		cfg.SampleRate != defaultTTSSampleRate || cfg.Encoding != "linear16" {
		t.Fatalf("defaults = %+v", cfg)
	}
	svc, s := newWebsocketTTS(WebsocketTTSConfig{APIKey: "k"})
	if svc.TypeName() != "DeepgramWebsocketTTS" {
		t.Errorf("service type = %q", svc.TypeName())
	}
	if meta := s.Metadata(); meta.Model != "aura-2-helena-en" || meta.VoiceID != "aura-2-helena-en" {
		t.Errorf("Metadata() = %+v, want the voice as the model", meta)
	}
	if !s.RequiresTrailingSpace() {
		t.Error("the service does not ask for a trailing space")
	}
}

// The connection is opened on the speak path, authorized with Deepgram's token
// scheme, with the voice as the model and the options as query parameters.
func TestWebsocketTTSQueryParams(t *testing.T) {
	server, baseURL := newAuraServer(t, auraServerOptions{})
	cfg := auraTestConfig(baseURL)
	cfg.Voice = "aura-2-andromeda-en"
	cfg.SampleRate = 16000
	cfg.Encoding = "mulaw"
	cfg.Speed = new(0.9)
	cfg.MipOptOut = new(true)
	connectedAuraSynth(t, cfg)

	want := url.Values{
		"model":       {"aura-2-andromeda-en"},
		"encoding":    {"mulaw"},
		"sample_rate": {"16000"},
		"speed":       {"0.9"},
		"mip_opt_out": {"true"},
	}
	if q := server.lastQuery(t); !mapsEqual(q, want) {
		t.Errorf("query = %v, want %v", q, want)
	}
	server.mu.Lock()
	defer server.mu.Unlock()
	if server.paths[0] != "/v1/speak" || server.auth != "Token test-key" {
		t.Errorf("dialed %q with Authorization %q", server.paths[0], server.auth)
	}
}

// Speed and the opt-out are left off the query when unset.
func TestWebsocketTTSUnsetOptionsAreOmitted(t *testing.T) {
	server, baseURL := newAuraServer(t, auraServerOptions{})
	connectedAuraSynth(t, auraTestConfig(baseURL))

	q := server.lastQuery(t)
	for _, k := range []string{"speed", "mip_opt_out"} {
		if q.Has(k) {
			t.Errorf("%s = %v sent, want it omitted", k, q[k])
		}
	}
}

// A turn's text is sent as Speak, flushed once the turn ends, and its audio is
// played and closed out by the Flushed that answers the Flush.
func TestWebsocketTTSProtocolRoundtrip(t *testing.T) {
	server, baseURL := newAuraServer(t, auraServerOptions{})
	svc, _ := newWebsocketTTS(auraTestConfig(baseURL))

	down, up := runAuraPipeline(t, svc, pipeline.Params{},
		frames.NewTTSSpeakFrame("Hello from Aura."),
		300*time.Millisecond,
		frames.NewBotStoppedSpeakingFrame(),
	)

	if errs := framesOf[*frames.ErrorFrame](slices.Concat(down, up)); len(errs) > 0 {
		t.Errorf("errors pushed: %v", errs)
	}
	if len(framesOf[*frames.TTSStartedFrame](down)) != 1 || len(framesOf[*frames.TTSStoppedFrame](down)) != 1 {
		t.Errorf("the turn was not bracketed by one started and one stopped frame")
	}
	var pcm []byte
	for _, f := range framesOf[*frames.TTSAudioRawFrame](down) {
		if f.SampleRate != 24000 || f.NumChannels != 1 {
			t.Errorf("audio at %d Hz on %d channels, want 24000 Hz mono", f.SampleRate, f.NumChannels)
		}
		pcm = append(pcm, f.Audio...)
	}
	if !bytes.Equal(pcm, auraAudioChunk) {
		t.Errorf("played %d bytes of audio, want the chunk sent", len(pcm))
	}
	sent := server.sent()
	if len(sent) < 2 || sent[0]["type"] != "Speak" || sent[0]["text"] != "Hello from Aura. " ||
		sent[1]["type"] != "Flush" {
		t.Errorf("sent %v, want the text with a trailing space, then a Flush", sent)
	}
	if got := server.sentTypes(); got[len(got)-1] != "Close" {
		t.Errorf("sent %v, want a Close on disconnecting", got)
	}
	if n := server.dialCount(); n != 1 {
		t.Errorf("dialed %d connections, want the one held for the call", n)
	}
}

// A Warning is logged, not pushed as an error.
func TestWebsocketTTSWarningIsNotFatal(t *testing.T) {
	_, baseURL := newAuraServer(t, auraServerOptions{warning: true})
	svc, _ := newWebsocketTTS(auraTestConfig(baseURL))

	down, up := runAuraPipeline(t, svc, pipeline.Params{},
		frames.NewTTSSpeakFrame("Hello from Aura."),
		300*time.Millisecond,
		frames.NewBotStoppedSpeakingFrame(),
	)

	if errs := framesOf[*frames.ErrorFrame](slices.Concat(down, up)); len(errs) > 0 {
		t.Errorf("errors pushed: %v", errs)
	}
	if len(framesOf[*frames.TTSAudioRawFrame](down)) == 0 {
		t.Error("no audio was played")
	}
}

// An interruption while Aura is still generating sends a Clear and keeps the
// connection.
func TestWebsocketTTSInterruptionSendsClear(t *testing.T) {
	server, baseURL := newAuraServer(t, auraServerOptions{holdFlush: true})
	svc, _ := newWebsocketTTS(auraTestConfig(baseURL))

	runAuraPipeline(t, svc, pipeline.Params{},
		frames.NewTTSSpeakFrame("Hello from Aura."),
		200*time.Millisecond,
		frames.NewInterruptionFrame(),
		200*time.Millisecond,
	)

	server.awaitSentType(t, "Clear")
	if n := server.dialCount(); n != 1 {
		t.Errorf("dialed %d connections, want the connection kept through the interruption", n)
	}
	if ids := svc.AudioContextIDs(); len(ids) != 0 {
		t.Errorf("audio contexts %v left open", ids)
	}
}

// A settings change reconnects with the new settings in the query.
func TestWebsocketTTSSettingsUpdateReconnects(t *testing.T) {
	server, baseURL := newAuraServer(t, auraServerOptions{})
	_, s := connectedAuraSynth(t, auraTestConfig(baseURL))

	changed, err := settings.Apply(s.live, &TTSSettings{Speed: settings.Set(1.2)})
	if err != nil {
		t.Fatalf("applying the settings: %v", err)
	}
	if err := s.UpdateSettings(t.Context(), changed); err != nil {
		t.Fatalf("UpdateSettings: %v", err)
	}

	if server.dialCount() != 2 || !s.Connected() {
		t.Errorf("dialed %d connections (connected %v), want a second one", server.dialCount(), s.Connected())
	}
	if got := server.lastQuery(t).Get("speed"); got != "1.2" {
		t.Errorf("the new connection's speed = %q, want 1.2", got)
	}
	if got := server.sentTypes(); !slices.Equal(got, []string{"Close"}) {
		t.Errorf("sent %v, want the old connection closed with a Close", got)
	}
}

// A voice change reconnects with the new voice, and relabels the model the
// synthesis is reported against: Deepgram takes the voice as the model.
func TestWebsocketTTSVoiceUpdateFollowsTheModel(t *testing.T) {
	server, baseURL := newAuraServer(t, auraServerOptions{})
	_, s := connectedAuraSynth(t, auraTestConfig(baseURL))

	changed, err := settings.Apply(s.live, &TTSSettings{Voice: settings.Set("aura-2-thalia-en")})
	if err != nil {
		t.Fatalf("applying the settings: %v", err)
	}
	if changed["model"] != "aura-2-helena-en" || changed["voice"] != "aura-2-helena-en" {
		t.Errorf("changed = %v, want the voice and the model, from the default", changed)
	}
	if err := s.UpdateSettings(t.Context(), changed); err != nil {
		t.Fatalf("UpdateSettings: %v", err)
	}
	if meta := s.Metadata(); meta.Model != "aura-2-thalia-en" || meta.VoiceID != "aura-2-thalia-en" {
		t.Errorf("Metadata() = %+v, want the new voice as the model", meta)
	}
	if server.dialCount() != 2 || server.lastQuery(t).Get("model") != "aura-2-thalia-en" {
		t.Errorf("dialed %d connections, the last with %v, want one with the new voice",
			server.dialCount(), server.lastQuery(t))
	}
}

// A sentence that finds the connection closed opens a new one and goes out on
// it.
func TestWebsocketTTSRunTTSReconnectsAClosedConnection(t *testing.T) {
	server, baseURL := newAuraServer(t, auraServerOptions{})
	_, s := newWebsocketTTS(auraTestConfig(baseURL))
	t.Cleanup(func() { s.Disconnect(context.Background()) })
	if err := s.ConnectWebsocket(t.Context()); err != nil {
		t.Fatalf("connect: %v", err)
	}
	// Marked closed the way the receive loop marks a connection whose read
	// ended.
	s.mu.Lock()
	s.conn.readEnded.Store(true)
	s.mu.Unlock()

	if err := s.RunTTS(t.Context(), "hello", "c1", nil); err != nil {
		t.Fatalf("RunTTS: %v", err)
	}
	if server.dialCount() != 2 || !s.Connected() || !s.ws.ReceiveTaskRunning() {
		t.Errorf("dialed %d connections (connected %v, receiving %v), want a fresh one with its loop",
			server.dialCount(), s.Connected(), s.ws.ReceiveTaskRunning())
	}
	server.awaitSentType(t, "Speak")
	if sent := server.sent(); sent[0]["text"] != "hello" {
		t.Errorf("sent %v, want the text on the new connection", sent)
	}
}

// A sentence that cannot get a connection yields an error in its stream rather
// than being taken as spoken.
func TestWebsocketTTSRunTTSWithoutAConnectionYieldsAnError(t *testing.T) {
	_, s := newWebsocketTTS(auraTestConfig(refusingServer(t, http.StatusServiceUnavailable, "")))
	t.Cleanup(func() { s.Disconnect(context.Background()) })

	var yielded []frames.Frame
	err := s.RunTTS(t.Context(), "hello", "c1", func(f frames.Frame) error {
		yielded = append(yielded, f)
		return nil
	})
	if err != nil {
		t.Fatalf("RunTTS: %v", err)
	}
	if len(yielded) != 1 {
		t.Fatalf("yielded %v, want one error", yielded)
	}
	ef, ok := yielded[0].(*frames.ErrorFrame)
	if !ok || !errors.Is(ef.Err, errWebsocketNotConnected) {
		t.Errorf("yielded %v, want the service reported not connected", yielded[0])
	}
}

// mapsEqual reports whether two query strings hold the same values.
func mapsEqual(a, b url.Values) bool {
	if len(a) != len(b) {
		return false
	}
	for k, v := range a {
		if !slices.Equal(v, b[k]) {
			return false
		}
	}
	return true
}
