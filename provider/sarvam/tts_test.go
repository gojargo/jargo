package sarvam

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"maps"
	"net/http"
	"net/http/httptest"
	"net/url"
	"slices"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/internal/providertest"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/service"
	"github.com/gojargo/jargo/service/settings"
	"github.com/gojargo/jargo/service/tts"
	"github.com/gojargo/jargo/service/wsservice"
	"github.com/gojargo/jargo/service/wsutil"
	uctx "github.com/gojargo/jargo/utils/context"
	"github.com/gojargo/jargo/utils/events"
)

// Tests for the connection held open for the call. Sarvam pins the model on the
// URL, opens a session with a config message, takes text and a flush, and
// streams base64 PCM that names no request, ending each synthesis with a final
// event.

const v4Flash = "bulbul:v4-flash"

// ttsFakeServer is a fake Sarvam synthesis endpoint. It records every
// connection and every message the client sends, and answers each text with
// what reply returns.
type ttsFakeServer struct {
	url  string
	sent chan map[string]any

	mu      sync.Mutex
	reply   func(msg map[string]any) []map[string]any
	refuse  int
	conns   []*websocket.Conn
	queries []url.Values
	headers []http.Header
}

func newTTSFakeServer(t *testing.T, reply func(msg map[string]any) []map[string]any) *ttsFakeServer {
	t.Helper()
	s := &ttsFakeServer{sent: make(chan map[string]any, 256), reply: reply}
	srv := httptest.NewServer(http.HandlerFunc(s.serve))
	t.Cleanup(srv.Close)
	s.url = "ws" + strings.TrimPrefix(srv.URL, "http")
	return s
}

func (s *ttsFakeServer) serve(w http.ResponseWriter, r *http.Request) {
	s.mu.Lock()
	refuse := s.refuse
	s.mu.Unlock()
	if refuse != 0 {
		http.Error(w, "refused", refuse)
		return
	}
	c, err := websocket.Accept(w, r, nil)
	if err != nil {
		return
	}
	s.mu.Lock()
	s.conns = append(s.conns, c)
	s.queries = append(s.queries, r.URL.Query())
	s.headers = append(s.headers, r.Header.Clone())
	reply := s.reply
	s.mu.Unlock()
	defer func() { _ = c.CloseNow() }()
	ctx := r.Context()
	for {
		_, data, err := c.Read(ctx)
		if err != nil {
			return
		}
		var msg map[string]any
		if json.Unmarshal(data, &msg) != nil {
			return
		}
		s.sent <- msg
		if reply == nil {
			continue
		}
		for _, ev := range reply(msg) {
			b, _ := json.Marshal(ev)
			if c.Write(ctx, websocket.MessageText, b) != nil {
				return
			}
		}
	}
}

func (s *ttsFakeServer) connections() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return len(s.conns)
}

// query is the URL query of the i-th connection.
func (s *ttsFakeServer) query(i int) url.Values {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.queries[i]
}

// drop ends every connection without a closing handshake.
func (s *ttsFakeServer) drop() {
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, c := range s.conns {
		_ = c.CloseNow()
	}
}

func (s *ttsFakeServer) next(t *testing.T) map[string]any {
	t.Helper()
	select {
	case msg := <-s.sent:
		return msg
	case <-time.After(3 * time.Second):
		t.Fatal("the client sent nothing")
		return nil
	}
}

// nextConfig reads the client's messages until a config message, returning its
// data.
func (s *ttsFakeServer) nextConfig(t *testing.T) map[string]any {
	t.Helper()
	for {
		if msg := s.next(t); msg[msgType] == "config" {
			data, _ := msg[msgData].(map[string]any)
			return data
		}
	}
}

// quiet reports whether the client sends nothing for a moment.
func (s *ttsFakeServer) quiet() bool {
	select {
	case <-s.sent:
		return false
	case <-time.After(50 * time.Millisecond):
		return true
	}
}

// onText answers every text sent with events.
func onText(events ...map[string]any) func(map[string]any) []map[string]any {
	return func(msg map[string]any) []map[string]any {
		if msg[msgType] == "text" {
			return events
		}
		return nil
	}
}

// audioMessage is a base64 audio chunk as Sarvam sends it.
func audioMessage(pcm []byte) map[string]any {
	return map[string]any{
		"type": "audio",
		"data": map[string]any{"audio": base64.StdEncoding.EncodeToString(pcm), "request_id": "r1"},
	}
}

// finalEvent is the completion event that ends a synthesis.
func finalEvent() map[string]any {
	return map[string]any{"type": "event", "data": map[string]any{"event_type": "final"}}
}

// fakeTTSSvc stands in for the TTS service the synthesizer asks about its
// audio contexts.
type fakeTTSSvc struct {
	mu     sync.Mutex
	active string
}

func (f *fakeTTSSvc) ActiveAudioContextID() string {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.active
}

func (f *fakeTTSSvc) setActive(id string) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.active = id
}

func (f *fakeTTSSvc) StopAllMetrics(context.Context) {}

// fakeTTSHost stands in for the tts.Base the provider appends its audio to.
// Only the contexts opened on it are available.
type fakeTTSHost struct {
	mu      sync.Mutex
	open    map[string]bool
	frames  map[string][]frames.Frame
	errors  []string
	removed []string
}

func newFakeTTSHost(ids ...string) *fakeTTSHost {
	h := &fakeTTSHost{open: map[string]bool{}, frames: map[string][]frames.Frame{}}
	for _, id := range ids {
		h.open[id] = true
	}
	return h
}

func (h *fakeTTSHost) AppendToAudioContext(id string, f frames.Frame) {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.frames[id] = append(h.frames[id], f)
}

func (h *fakeTTSHost) AppendWordToAudioContext(string, string, float64) {}

func (h *fakeTTSHost) AddWordTimestamps(string, []uctx.WordTiming, tts.WordTimingOptions) {}

func (h *fakeTTSHost) RemoveAudioContext(id string) {
	h.mu.Lock()
	defer h.mu.Unlock()
	delete(h.open, id)
	h.removed = append(h.removed, id)
}

func (h *fakeTTSHost) AudioContextAvailable(id string) bool {
	h.mu.Lock()
	defer h.mu.Unlock()
	return h.open[id]
}

func (h *fakeTTSHost) RotateTurnContext(context.Context) {}

func (h *fakeTTSHost) ReplaceConnection() {}

func (h *fakeTTSHost) PushFrame(context.Context, frames.Frame, processor.Direction) error { return nil }

func (h *fakeTTSHost) PushError(_ context.Context, msg string, _ error, _ bool, _ ...processor.ErrorOption) {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.errors = append(h.errors, msg)
}

func (h *fakeTTSHost) ResetActiveAudioContext() {}

// audioFor is the PCM appended to the context.
func (h *fakeTTSHost) audioFor(id string) []byte {
	h.mu.Lock()
	defer h.mu.Unlock()
	var pcm []byte
	for _, f := range h.frames[id] {
		if a, ok := f.(*frames.TTSAudioRawFrame); ok {
			pcm = append(pcm, a.Audio...)
		}
	}
	return pcm
}

func (h *fakeTTSHost) framesFor(id string) []frames.Frame {
	h.mu.Lock()
	defer h.mu.Unlock()
	return slices.Clone(h.frames[id])
}

func (h *fakeTTSHost) reported() []string {
	h.mu.Lock()
	defer h.mu.Unlock()
	return slices.Clone(h.errors)
}

func ttsWaitUntil(t *testing.T, msg string, cond func() bool) {
	t.Helper()
	deadline := time.After(3 * time.Second)
	for !cond() {
		select {
		case <-deadline:
			t.Fatal(msg)
		case <-time.After(5 * time.Millisecond):
		}
	}
}

// connectedSynth builds a service for server, with the synthesizer's host and
// service replaced by fakes, and starts it. The config message that opens the
// session is read off.
func connectedSynth(t *testing.T, server *ttsFakeServer, cfg TTSConfig) (*synthesizer, *fakeTTSHost, *fakeTTSSvc) {
	t.Helper()
	cfg.APIKey, cfg.URL = "test-key", server.url
	_, s := newTTS(cfg)
	host, svc := newFakeTTSHost(), &fakeTTSSvc{}
	s.host, s.svc = host, svc
	s.Start(context.Background())
	if !s.Connected() {
		t.Fatal("the service did not connect when it started")
	}
	t.Cleanup(func() { s.Disconnect(context.Background()) })
	server.nextConfig(t)
	return s, host, svc
}

// updateSettings applies delta the way the base does, then hands the change to
// the synthesizer, returning what changed.
func updateSettings(t *testing.T, s *synthesizer, delta *TTSSettings) settings.Changed {
	t.Helper()
	changed, err := settings.Apply(s.live, delta)
	if err != nil {
		t.Fatal(err)
	}
	if err := s.UpdateSettings(context.Background(), changed); err != nil {
		t.Fatal(err)
	}
	return changed
}

// sentConfig decodes the config message the synthesizer would send now.
func sentConfig(t *testing.T, s *synthesizer) map[string]any {
	t.Helper()
	var msg map[string]any
	if err := json.Unmarshal(s.configMessage(), &msg); err != nil {
		t.Fatalf("decoding the config message: %v", err)
	}
	data, _ := msg[msgData].(map[string]any)
	return data
}

// TestTTSConfigValidate pins the credential the synthesis API requires.
func TestTTSConfigValidate(t *testing.T) {
	providertest.Configs(t, []providertest.ConfigCase{
		{Name: "missing API key", Cfg: TTSConfig{}, Valid: false},
		{Name: "API key only", Cfg: TTSConfig{APIKey: "k"}, Valid: true},
		{Name: "v4-flash", Cfg: TTSConfig{APIKey: "k", Model: v4Flash}, Valid: true},
	})
}

// TestNewTTS checks the constructor returns a service under the label that
// identifies it in logs, metrics and traces.
func TestNewTTS(t *testing.T) {
	providertest.Service(t, "SarvamTTS", NewTTS(TTSConfig{APIKey: "k"}))
}

// TestStartOpensASession checks the connection opens when the service starts,
// authorized, with the model pinned on the URL and the completion event asked
// for, and that the session is opened with the config message.
func TestStartOpensASession(t *testing.T) {
	server := newTTSFakeServer(t, nil)
	_, s := newTTS(TTSConfig{APIKey: "test-key", URL: server.url})
	s.host, s.svc = newFakeTTSHost(), &fakeTTSSvc{}
	s.Start(context.Background())
	t.Cleanup(func() { s.Disconnect(context.Background()) })

	config := server.next(t)
	if config[msgType] != "config" {
		t.Fatalf("first message = %v, want the config", config)
	}
	server.mu.Lock()
	header := server.headers[0]
	server.mu.Unlock()
	if h := header.Get("api-subscription-key"); h != "test-key" {
		t.Errorf("api-subscription-key = %q, want the configured key", h)
	}
	q := server.query(0)
	if q.Get("send_completion_event") != "true" {
		t.Errorf("send_completion_event = %q, want true so a synthesis is known to be complete",
			q.Get("send_completion_event"))
	}
	if q.Get("model") != defaultTTSModel {
		t.Errorf("model = %q, want %q", q.Get("model"), defaultTTSModel)
	}

	data, _ := config[msgData].(map[string]any)
	// Sarvam takes the rate as a string, not a number.
	wantRate := strconv.Itoa(ttsModelConfigs[defaultTTSModel].defaultSampleRate)
	for key, want := range map[string]any{
		"speech_sample_rate":   wantRate,
		"target_language_code": "en-IN",
		"model":                defaultTTSModel,
		"speaker":              "shubh",
		"enable_preprocessing": true,
		"min_buffer_size":      50.0,
		"max_chunk_length":     150.0,
		"output_audio_codec":   "linear16",
		"output_audio_bitrate": "128k",
		"pace":                 1.0,
	} {
		if data[key] != want {
			t.Errorf("%s = %v, want %v", key, data[key], want)
		}
	}
}

// Every sentence goes out on the one connection, and its audio comes back
// labeled with the context playing.
func TestSentencesShareTheConnection(t *testing.T) {
	server := newTTSFakeServer(t, onText(audioMessage([]byte{1, 2}), audioMessage([]byte{3})))
	s, host, svc := connectedSynth(t, server, TTSConfig{})

	ctx := context.Background()
	for _, id := range []string{"c1", "c2"} {
		svc.setActive(id)
		if err := s.RunTTS(ctx, "Hello "+id+".", id, nil); err != nil {
			t.Fatal(err)
		}
		msg := server.next(t)
		data, _ := msg[msgData].(map[string]any)
		if msg[msgType] != "text" || data["text"] != "Hello "+id+"." {
			t.Errorf("sent %v, want the sentence as text", msg)
		}
		ttsWaitUntil(t, "the audio never arrived", func() bool {
			return bytes.Equal(host.audioFor(id), []byte{1, 2, 3})
		})
	}
	if n := server.connections(); n != 1 {
		t.Errorf("the server saw %d connections, want 1", n)
	}
	for _, f := range host.framesFor("c1") {
		if a, ok := f.(*frames.TTSAudioRawFrame); ok && (a.ContextID != "c1" || a.SampleRate != 24000) {
			t.Errorf("frame labeled %q at %d Hz, want c1 at the model rate", a.ContextID, a.SampleRate)
		}
	}
}

// The final event ends the context playing at once, with a stop frame, rather
// than leaving it to the stop-frame timeout. One for a context no longer open
// does nothing.
func TestFinalEventEndsTheContext(t *testing.T) {
	s := newSynthesizer(TTSConfig{APIKey: "k"})
	host, svc := newFakeTTSHost("c1"), &fakeTTSSvc{active: "c1"}
	s.host, s.svc = host, svc
	final, _ := json.Marshal(finalEvent())

	if err := s.handleMessage(context.Background(), final); err != nil {
		t.Fatal(err)
	}
	got := host.framesFor("c1")
	if len(got) != 1 {
		t.Fatalf("frames = %v, want one stop frame", got)
	}
	if stopped, ok := got[0].(*frames.TTSStoppedFrame); !ok || stopped.ContextID != "c1" {
		t.Errorf("frame = %v, want the context stopped", got[0])
	}
	if host.AudioContextAvailable("c1") {
		t.Error("the context is still open after the final event")
	}

	if err := s.handleMessage(context.Background(), final); err != nil {
		t.Fatal(err)
	}
	if got := host.framesFor("c1"); len(got) != 1 {
		t.Errorf("a final event for a closed context appended %v", got[1:])
	}
}

// An error the server reports is pushed, and handed to the context playing.
func TestServerErrorIsReported(t *testing.T) {
	s := newSynthesizer(TTSConfig{APIKey: "k"})
	host := newFakeTTSHost("c1")
	s.host, s.svc = host, &fakeTTSSvc{active: "c1"}

	msg := []byte(`{"type":"error","data":{"message":"unknown speaker"}}`)
	if err := s.handleMessage(context.Background(), msg); err != nil {
		t.Fatal(err)
	}
	if got := host.reported(); !slices.Equal(got, []string{"TTS Error: unknown speaker"}) {
		t.Errorf("reported %v, want the server's message", got)
	}
	got := host.framesFor("c1")
	if len(got) != 1 {
		t.Fatalf("frames = %v, want one error frame", got)
	}
	if ef, ok := got[0].(*frames.ErrorFrame); !ok || ef.Error != "TTS Error: unknown speaker" {
		t.Errorf("frame = %v, want the error on the context", got[0])
	}
}

// Messages the service does not model are stepped over, and audio that is not
// base64, or a message that is not JSON, fails the connection rather than being
// played as rubbish.
func TestUnmodeledAndUnreadableMessages(t *testing.T) {
	s := newSynthesizer(TTSConfig{APIKey: "k"})
	host := newFakeTTSHost("c1")
	s.host, s.svc = host, &fakeTTSSvc{active: "c1"}
	ctx := context.Background()

	if err := s.handleMessage(ctx, []byte(`{"type":"event","data":{"event_type":"started"}}`)); err != nil {
		t.Errorf("an unmodeled event failed: %v", err)
	}
	if got := host.framesFor("c1"); len(got) != 0 {
		t.Errorf("an unmodeled event appended %v", got)
	}
	if err := s.handleMessage(ctx, []byte(`{"type":"audio","data":{"audio":"not base64!!"}}`)); err == nil {
		t.Error("audio that is not base64 was accepted")
	}
	if err := s.handleMessage(ctx, []byte("not json")); err == nil {
		t.Error("a message that is not JSON was accepted")
	}
}

// The keepalive is a ping.
func TestKeepalive(t *testing.T) {
	server := newTTSFakeServer(t, nil)
	s, _, _ := connectedSynth(t, server, TTSConfig{})

	if err := s.SendKeepalive(context.Background()); err != nil {
		t.Fatal(err)
	}
	if got := server.next(t); !maps.Equal(got, map[string]any{msgType: "ping"}) {
		t.Errorf("keepalive = %v, want a ping", got)
	}
	if !s.ws.KeepaliveRunning() {
		t.Error("the keepalive loop is not running while connected")
	}
}

// A flush has Sarvam generate what it is still buffering.
func TestFlushAudio(t *testing.T) {
	server := newTTSFakeServer(t, nil)
	s, _, _ := connectedSynth(t, server, TTSConfig{})

	s.FlushAudio(context.Background(), "")
	if got := server.next(t); !maps.Equal(got, map[string]any{msgType: "flush"}) {
		t.Errorf("flush = %v, want a flush", got)
	}
}

// A connection the server drops is restored by the receive loop, with a fresh
// session.
func TestReconnectsWhenTheServerDrops(t *testing.T) {
	server := newTTSFakeServer(t, nil)
	s, _, _ := connectedSynth(t, server, TTSConfig{Websocket: wsservice.Options{
		ReconnectBackoffMinWait: time.Millisecond,
		ReconnectBackoffMaxWait: time.Millisecond,
	}})

	server.drop()
	server.nextConfig(t)
	ttsWaitUntil(t, "the service never reconnected", func() bool {
		return server.connections() == 2 && s.Connected()
	})
}

// A sentence after the connection closed opens it again before it is sent.
func TestSentenceReopensAClosedConnection(t *testing.T) {
	server := newTTSFakeServer(t, nil)
	s, _, _ := connectedSynth(t, server, TTSConfig{})
	s.Disconnect(context.Background())

	if err := s.RunTTS(context.Background(), "Hello.", "c1", nil); err != nil {
		t.Fatal(err)
	}
	if got := server.next(t); got[msgType] != "config" {
		t.Fatalf("sent %v, want the session opened again with a config message", got)
	}
	if got := server.next(t); got[msgType] != "text" {
		t.Errorf("sent %v, want the sentence", got)
	}
}

// A send that fails stops the context's speech, is reported, and resets the
// connection.
func TestFailedSendResetsTheConnection(t *testing.T) {
	server := newTTSFakeServer(t, nil)
	_, s := newTTS(TTSConfig{APIKey: "k", URL: server.url})
	s.host, s.svc = newFakeTTSHost(), &fakeTTSSvc{}
	t.Cleanup(func() { s.Disconnect(context.Background()) })

	// A connection that is already closed but not yet known to be: the send is
	// where the failure shows.
	conn, err := s.ws.Dial(context.Background(), server.url, http.Header{}, wsutil.DefaultReadLimit)
	if err != nil {
		t.Fatal(err)
	}
	_ = conn.CloseNow()
	s.conn = &ttsConn{conn: conn, reads: make(chan ttsRead), stop: make(chan struct{})}

	var got []frames.Frame
	err = s.RunTTS(context.Background(), "Hello.", "c1", func(f frames.Frame) error {
		got = append(got, f)
		return nil
	})
	if err == nil {
		t.Fatal("RunTTS accepted a failed send")
	}
	if len(got) != 1 {
		t.Fatalf("yielded %v, want one stop frame", got)
	}
	if stopped, ok := got[0].(*frames.TTSStoppedFrame); !ok || stopped.ContextID != "c1" {
		t.Errorf("yielded %v, want the context's speech stopped", got[0])
	}
	// The server counts a connection once its handshake is done, which can
	// trail the dial returning.
	ttsWaitUntil(t, "the connection was not reset", func() bool { return server.connections() == 2 })
	if !s.Connected() {
		t.Error("not connected after the reset")
	}
}

// TestConfigMessageOmitsUnsupportedControls checks a voice control is sent only
// to a model that understands it, and left off otherwise rather than being sent
// and ignored.
func TestConfigMessageOmitsUnsupportedControls(t *testing.T) {
	pitch, loudness, temperature := 0.0, 1.0, 0.5
	cfg := TTSConfig{APIKey: "k", Pitch: &pitch, Loudness: &loudness, Temperature: &temperature}

	data := sentConfig(t, newSynthesizer(cfg))
	mc := ttsModelConfigs[defaultTTSModel]
	for key, supported := range map[string]bool{
		"pitch":       mc.supportsPitch,
		"loudness":    mc.supportsLoudness,
		"temperature": mc.supportsTemperature,
	} {
		_, present := data[key]
		if present != supported {
			t.Errorf("%s present = %v, want %v for the default model", key, present, supported)
		}
	}
}

// The service has to land on the same speaker and rate the API defaults to.
func TestV4FlashDefaults(t *testing.T) {
	s := newSynthesizer(TTSConfig{APIKey: "k", Model: v4Flash})

	if voice, _ := s.live.Voice.Value(); voice != "shubh_en_narration_gentle" {
		t.Errorf("voice = %q, want shubh_en_narration_gentle", voice)
	}
	if s.SampleRate() != 24000 {
		t.Errorf("sample rate = %d, want 24000", s.SampleRate())
	}
	if on, _ := s.live.EnablePreprocessing.Value(); !on {
		t.Error("preprocessing is off, want it forced on")
	}
}

// v4-flash applies both pitch and loudness; the v3 models ignore them.
func TestV4FlashSendsPitchAndLoudness(t *testing.T) {
	server := newTTSFakeServer(t, nil)
	pitch, loudness, pace := 0.3, 1.5, 1.2
	_, s := newTTS(TTSConfig{
		APIKey: "k", URL: server.url, Model: v4Flash, Pitch: &pitch, Loudness: &loudness, Pace: &pace,
	})
	s.host, s.svc = newFakeTTSHost(), &fakeTTSSvc{}
	s.Start(context.Background())
	t.Cleanup(func() { s.Disconnect(context.Background()) })

	config := server.nextConfig(t)
	if config["pitch"] != 0.3 || config["loudness"] != 1.5 || config["pace"] != 1.2 {
		t.Errorf("config = %v, want pitch 0.3, loudness 1.5 and pace 1.2", config)
	}
}

// The API pins v4-flash's temperature to 0.6, so sending any value would only
// mislead.
func TestV4FlashDropsTemperature(t *testing.T) {
	for _, temperature := range []float64{0.6, 0.9} {
		t.Run(strconv.FormatFloat(temperature, 'g', -1, 64), func(t *testing.T) {
			s := newSynthesizer(TTSConfig{APIKey: "k", Model: v4Flash, Temperature: &temperature})
			if _, present := sentConfig(t, s)["temperature"]; present {
				t.Error("temperature was sent to v4-flash")
			}
		})
	}
}

// A temperature given while the pipeline runs is no more use to v4-flash: the
// session is sent a new config without it.
func TestV4FlashDropsARuntimeTemperatureUpdate(t *testing.T) {
	server := newTTSFakeServer(t, nil)
	s, _, _ := connectedSynth(t, server, TTSConfig{Model: v4Flash})

	updateSettings(t, s, &TTSSettings{Temperature: settings.Set(0.9)})
	if _, present := server.nextConfig(t)["temperature"]; present {
		t.Error("temperature was sent to v4-flash")
	}
}

func TestV4FlashClampsPaceToItsOwnRange(t *testing.T) {
	pace := 2.5
	s := newSynthesizer(TTSConfig{APIKey: "k", Model: v4Flash, Pace: &pace})
	if got, _ := s.live.Pace.Value(); got != 2.0 {
		t.Errorf("pace = %v, want it clamped to 2.0", got)
	}
}

// The model is pinned on the URL, so the switch needs a new connection.
func TestSwitchingToV4FlashReconnectsWithItsCapabilities(t *testing.T) {
	server := newTTSFakeServer(t, nil)
	s, _, _ := connectedSynth(t, server, TTSConfig{Model: "bulbul:v3"})

	updateSettings(t, s, &TTSSettings{Model: settings.Set(v4Flash)})
	config := server.nextConfig(t)

	if n := server.connections(); n != 2 {
		t.Fatalf("the server saw %d connections, want the switch to reconnect once", n)
	}
	if got := server.query(1).Get("model"); got != v4Flash {
		t.Errorf("new connection model = %q, want %q", got, v4Flash)
	}
	if !s.mc.supportsPitch {
		t.Error("the model's capabilities did not move with it")
	}
	// bulbul:v4-flash rejects the v3 default, so the voice moves with the model.
	if voice, _ := s.live.Voice.Value(); voice != "shubh_en_narration_gentle" {
		t.Errorf("voice = %q, want the new model's default", voice)
	}
	if config["model"] != v4Flash || config["speaker"] != "shubh_en_narration_gentle" {
		t.Errorf("new session config = %v, want the new model and its default speaker", config)
	}
}

func TestAVoiceSwitchedWithTheModelIsKept(t *testing.T) {
	server := newTTSFakeServer(t, nil)
	s, _, _ := connectedSynth(t, server, TTSConfig{Model: "bulbul:v3"})

	delta := &TTSSettings{Voice: settings.Set("ritu_hi_edtech")}
	delta.Model = settings.Set(v4Flash)
	updateSettings(t, s, delta)

	if voice, _ := s.live.Voice.Value(); voice != "ritu_hi_edtech" {
		t.Errorf("voice = %q, want the one the update named", voice)
	}
}

func TestAnUnsupportedModelUpdateIsIgnored(t *testing.T) {
	server := newTTSFakeServer(t, nil)
	s, _, _ := connectedSynth(t, server, TTSConfig{Model: "bulbul:v3"})

	delta := &TTSSettings{}
	delta.Model = settings.Set("bulbul:v9")
	changed := updateSettings(t, s, delta)

	if model, _ := s.live.Model.Value(); model != "bulbul:v3" {
		t.Errorf("model = %q, want it left at bulbul:v3", model)
	}
	if changed.Has("model") {
		t.Error("the refused model is reported as changed")
	}
	if got := server.query(0).Get("model"); got != "bulbul:v3" || server.connections() != 1 {
		t.Errorf("model on the URL = %q with %d connections, want bulbul:v3 on the one", got, server.connections())
	}
	if !server.quiet() {
		t.Error("the refused model sent something")
	}
}

// An unaliased key lands in Extra, which never reaches the API.
func TestSettingsDictsAcceptTheSarvamAPINames(t *testing.T) {
	for _, tc := range []struct {
		key, field, value string
	}{
		{"voice_id", "voice", "priya"},
		{"speaker", "voice", "priya"},
		{"target_language_code", "language", "hi-IN"},
	} {
		t.Run(tc.key, func(t *testing.T) {
			delta := &TTSSettings{}
			if err := settings.FromMap(delta, map[string]any{tc.key: tc.value}); err != nil {
				t.Fatal(err)
			}
			if got, _ := settings.Get(delta, tc.field); got != tc.value {
				t.Errorf("%s = %v, want %q", tc.field, got, tc.value)
			}
			if len(delta.Extra) != 0 {
				t.Errorf("extra = %v, want nothing", delta.Extra)
			}
		})
	}
}

func TestAVoiceFromASettingsDictIsSentAsTheSpeaker(t *testing.T) {
	for _, key := range []string{"voice_id", "speaker"} {
		t.Run(key, func(t *testing.T) {
			server := newTTSFakeServer(t, nil)
			s, _, _ := connectedSynth(t, server, TTSConfig{Model: "bulbul:v3"})

			delta := &TTSSettings{}
			if err := settings.FromMap(delta, map[string]any{key: "priya"}); err != nil {
				t.Fatal(err)
			}
			updateSettings(t, s, delta)
			if got := server.nextConfig(t)["speaker"]; got != "priya" {
				t.Errorf("speaker = %v, want priya", got)
			}
			if n := server.connections(); n != 1 {
				t.Errorf("a voice change reconnected: %d connections", n)
			}
		})
	}
}

// run is a running pipeline around the service.
type run struct {
	t    *testing.T
	task *pipeline.Worker
	done chan error

	mu   sync.Mutex
	down []frames.Frame
}

func runPipeline(t *testing.T, b *tts.Base) *run {
	t.Helper()
	r := &run{t: t, done: make(chan error, 1)}
	r.task = pipeline.NewWorker(pipeline.New(b), pipeline.WorkerConfig{ReachedDownstreamFilter: pipeline.AnyFrame})
	events.On(&r.task.Registry, pipeline.EventFrameReachedDownstream, func(_ context.Context, f frames.Frame) {
		r.mu.Lock()
		r.down = append(r.down, f)
		r.mu.Unlock()
	})
	go func() { r.done <- r.task.Run(context.Background()) }()
	r.waitFor(func(f frames.Frame) bool { _, ok := f.(*frames.StartFrame); return ok })
	return r
}

func (r *run) waitFor(match func(frames.Frame) bool) {
	r.t.Helper()
	ttsWaitUntil(r.t, "the frame waited for never came", func() bool {
		r.mu.Lock()
		defer r.mu.Unlock()
		return slices.ContainsFunc(r.down, match)
	})
}

func (r *run) stop() {
	r.t.Helper()
	r.task.StopWhenDone()
	r.wait()
}

// cancel ends the pipeline at once, without waiting for audio still playing.
func (r *run) cancel() {
	r.t.Helper()
	r.task.Cancel(context.Background(), "")
	r.wait()
}

func (r *run) wait() {
	r.t.Helper()
	select {
	case <-r.done:
	case <-time.After(5 * time.Second):
		r.t.Fatal("the pipeline did not finish")
	}
}

// The service connects when the pipeline starts, its audio reaches the
// pipeline over that one connection, and the final event ends the speech.
func TestHoldsOneConnectionForThePipeline(t *testing.T) {
	server := newTTSFakeServer(t, onText(audioMessage(make([]byte, 480)), finalEvent()))
	b := NewTTS(TTSConfig{APIKey: "k", URL: server.url})
	r := runPipeline(t, b)
	ttsWaitUntil(t, "the server saw no connection when the pipeline started", func() bool {
		return server.connections() == 1
	})

	r.task.QueueFrame(frames.NewTTSSpeakFrame("Hello there."))
	r.waitFor(func(f frames.Frame) bool {
		a, ok := f.(*frames.TTSAudioRawFrame)
		return ok && a.ContextID != ""
	})
	r.waitFor(func(f frames.Frame) bool { _, ok := f.(*frames.TTSStoppedFrame); return ok })
	r.cancel()

	if n := server.connections(); n != 1 {
		t.Errorf("the server saw %d connections, want one for the whole pipeline", n)
	}
}

// A connection the server refuses is reported as a connection error.
func TestReportsAFailedConnection(t *testing.T) {
	server := newTTSFakeServer(t, nil)
	server.refuse = http.StatusUnauthorized
	b := NewTTS(TTSConfig{APIKey: "k", URL: server.url})
	var mu sync.Mutex
	var reported []string
	events.On(b.Events(), service.EventConnectionError, func(_ context.Context, msg string) {
		mu.Lock()
		reported = append(reported, msg)
		mu.Unlock()
	})

	r := runPipeline(t, b)
	defer r.stop()
	ttsWaitUntil(t, "the refused connection was never reported", func() bool {
		mu.Lock()
		defer mu.Unlock()
		return len(reported) > 0
	})
}
