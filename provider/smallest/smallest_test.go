package smallest

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"math"
	"net/http"
	"net/http/httptest"
	"reflect"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/language"
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

// Tests for the connection held open for the call. Every Smallest request
// carries its settings and the turn's context, the audio and the word timings
// that come back name no request, and the timings of each request start again
// near zero.

const ctxID = "ctx-1"

func TestValidate(t *testing.T) {
	if err := (Config{APIKey: "k"}).Validate(); err != nil {
		t.Fatalf("valid config rejected: %v", err)
	}
	if err := (Config{}).Validate(); err == nil {
		t.Fatal("config without APIKey should be rejected")
	}
	tooLong := 6000
	if err := (Config{APIKey: "k", MaxBufferDelayMs: &tooLong}).Validate(); err == nil {
		t.Fatal("a buffer delay above 5000 ms should be rejected")
	}
	if err := (Config{APIKey: "k", OutputFormat: "ogg"}).Validate(); err == nil {
		t.Fatal("an unknown output format should be rejected")
	}
}

func wsURL(httpURL string) string { return "ws" + strings.TrimPrefix(httpURL, "http") }

// fakeServer is a fake Smallest endpoint. It records every connection and
// every message the client sends, and answers each with what reply returns.
type fakeServer struct {
	url  string
	sent chan map[string]any

	mu      sync.Mutex
	reply   func(msg map[string]any) []map[string]any
	refuse  int
	conns   []*websocket.Conn
	headers []http.Header
}

func newFakeServer(t *testing.T, reply func(msg map[string]any) []map[string]any) *fakeServer {
	t.Helper()
	s := &fakeServer{sent: make(chan map[string]any, 256), reply: reply}
	srv := httptest.NewServer(http.HandlerFunc(s.serve))
	t.Cleanup(srv.Close)
	s.url = wsURL(srv.URL)
	return s
}

func (s *fakeServer) serve(w http.ResponseWriter, r *http.Request) {
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

func (s *fakeServer) connections() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return len(s.conns)
}

// drop ends every connection without a closing handshake.
func (s *fakeServer) drop() {
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, c := range s.conns {
		_ = c.CloseNow()
	}
}

func (s *fakeServer) next(t *testing.T) map[string]any {
	t.Helper()
	select {
	case msg := <-s.sent:
		return msg
	case <-time.After(3 * time.Second):
		t.Fatal("the client sent nothing")
		return nil
	}
}

// quiet reports whether the client sends nothing for a moment.
func (s *fakeServer) quiet() bool {
	select {
	case <-s.sent:
		return false
	case <-time.After(50 * time.Millisecond):
		return true
	}
}

// speaking answers each fragment with text in it with audio and the timing of
// its first word.
func speaking(msg map[string]any) []map[string]any {
	text, _ := msg["text"].(string)
	if strings.TrimSpace(text) == "" {
		return nil
	}
	return []map[string]any{
		chunkMessage(bytes.Repeat([]byte{0, 1}, 1600)),
		wordMessage(strings.Fields(text)[0], 0.0, 0.1, "r1", 0),
	}
}

func chunkMessage(pcm []byte) map[string]any {
	return map[string]any{"status": "chunk", "data": map[string]any{"audio": base64.StdEncoding.EncodeToString(pcm)}}
}

func wordMessage(word string, start, end float64, requestID string, wordID int) map[string]any {
	return map[string]any{
		"request_id": requestID,
		"status":     "word_timestamp",
		"data":       map[string]any{"id": wordID, "word": word, "start": start, "end": end},
	}
}

// fakeSvc stands in for the TTS service the synthesizer asks about its audio
// contexts.
type fakeSvc struct {
	mu      sync.Mutex
	active  string
	stopped int
}

func (f *fakeSvc) ActiveAudioContextID() string {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.active
}

func (f *fakeSvc) StopAllMetrics(context.Context) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.stopped++
}

// fakeHost stands in for the tts.Base the provider appends its audio and its
// word timings to.
type fakeHost struct {
	mu     sync.Mutex
	audio  map[string][]byte
	words  []uctx.WordTiming
	pushed []frames.Frame
	errors []string
}

func newFakeHost() *fakeHost { return &fakeHost{audio: map[string][]byte{}} }

func (h *fakeHost) AppendToAudioContext(id string, f frames.Frame) {
	h.mu.Lock()
	defer h.mu.Unlock()
	if a, ok := f.(*frames.TTSAudioRawFrame); ok {
		h.audio[id] = append(h.audio[id], a.Audio...)
	}
}

func (h *fakeHost) AppendWordToAudioContext(string, string, float64) {}

func (h *fakeHost) AddWordTimestamps(_ string, words []uctx.WordTiming, _ tts.WordTimingOptions) {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.words = append(h.words, words...)
}

func (h *fakeHost) RemoveAudioContext(string) {}

func (h *fakeHost) AudioContextAvailable(string) bool { return true }

func (h *fakeHost) RotateTurnContext(context.Context) {}

func (h *fakeHost) ReplaceConnection() {}

func (h *fakeHost) PushFrame(_ context.Context, f frames.Frame, _ processor.Direction) error {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.pushed = append(h.pushed, f)
	return nil
}

func (h *fakeHost) PushError(_ context.Context, msg string, _ error, _ bool, _ ...processor.ErrorOption) {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.errors = append(h.errors, msg)
}

func (h *fakeHost) ResetActiveAudioContext() {}

func (h *fakeHost) audioFor(id string) []byte {
	h.mu.Lock()
	defer h.mu.Unlock()
	return slices.Clone(h.audio[id])
}

func waitUntil(t *testing.T, msg string, cond func() bool) {
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
// service replaced by fakes, and starts it.
func connectedSynth(t *testing.T, server *fakeServer, cfg Config) (*synthesizer, *fakeHost, *fakeSvc) {
	t.Helper()
	cfg.APIKey, cfg.URL = "test-key", server.url
	_, s := newTTS(cfg)
	host, svc := newFakeHost(), &fakeSvc{active: ctxID}
	s.host, s.svc = host, svc
	s.Start(context.Background())
	if !s.Connected() {
		t.Fatal("the service did not connect when it started")
	}
	// The server counts a connection once its handshake is done, which can
	// trail the dial returning.
	waitUntil(t, "the server never saw the connection", func() bool { return server.connections() == 1 })
	t.Cleanup(func() { s.Disconnect(context.Background()) })
	return s, host, svc
}

// driveWords hands the word-timing messages to the synthesizer as the receive
// loop does, and returns the timings it reported.
func driveWords(t *testing.T, s *synthesizer, messages ...map[string]any) []uctx.WordTiming {
	t.Helper()
	host := newFakeHost()
	s.host, s.svc = host, &fakeSvc{active: ctxID}
	for _, m := range messages {
		data, _ := json.Marshal(m)
		if err := s.handleMessage(context.Background(), data); err != nil {
			t.Fatal(err)
		}
	}
	return host.words
}

// assertWords checks the timings reported, allowing for float rounding.
func assertWords(t *testing.T, got []uctx.WordTiming, want ...uctx.WordTiming) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("words = %v, want %v", got, want)
	}
	for i := range want {
		if got[i].Word != want[i].Word || math.Abs(got[i].Offset-want[i].Offset) > 1e-9 {
			t.Errorf("word %d = %v, want %v", i, got[i], want[i])
		}
	}
}

// Word timestamps are on by default and drive the text frames, so the base
// does not push the whole text.
func TestWordTimestampsEnabledByDefault(t *testing.T) {
	s := newSynthesizer(Config{APIKey: "k"})
	if !s.wordTimestamps {
		t.Fatal("word timestamps are off by default")
	}
	if _, ok := s.driven().(tts.WordTimestamps); !ok {
		t.Error("the base is not on the word-aligned path")
	}
	if got := s.buildMsg("hi", "", true)["word_timestamps"]; got != true {
		t.Errorf("word_timestamps = %v, want true", got)
	}
}

// Turning word timestamps off moves the service back to whole-text frames.
func TestWordTimestampsDisabledPushesWholeText(t *testing.T) {
	off := false
	s := newSynthesizer(Config{APIKey: "k", WordTimestamps: &off})
	if s.wordTimestamps {
		t.Fatal("word timestamps are on though turned off")
	}
	if _, ok := s.driven().(tts.WordTimestamps); ok {
		t.Error("the base is on the word-aligned path")
	}
	if _, present := s.buildMsg("hi", "", true)["word_timestamps"]; present {
		t.Error("word_timestamps was sent though turned off")
	}
}

// A sentence continues its context: the turn's fragments are joined into one
// generation.
func TestBuildMsgDefaultsToContinuingContext(t *testing.T) {
	server := newFakeServer(t, nil)
	s, _, _ := connectedSynth(t, server, Config{})

	if err := s.RunTTS(context.Background(), "hi", ctxID, nil); err != nil {
		t.Fatal(err)
	}
	msg := server.next(t)
	if msg["context_id"] != ctxID || msg["continue"] != true || msg["text"] != "hi" {
		t.Errorf("sent %v, want the sentence continuing %s", msg, ctxID)
	}
}

// The last fragment closes the context.
func TestBuildMsgCanCloseContext(t *testing.T) {
	s := newSynthesizer(Config{APIKey: "k"})
	if got := s.buildMsg("hi", ctxID, false)["continue"]; got != false {
		t.Errorf("continue = %v, want false", got)
	}
}

// The server's own buffering window applies unless overridden.
func TestBuildMsgBufferDelay(t *testing.T) {
	if _, present := newSynthesizer(Config{APIKey: "k"}).buildMsg("hi", "", true)["max_buffer_delay_ms"]; present {
		t.Error("max_buffer_delay_ms was sent though not set")
	}
	delay := 1500
	s := newSynthesizer(Config{APIKey: "k", MaxBufferDelayMs: &delay})
	if got := s.buildMsg("hi", "", true)["max_buffer_delay_ms"]; got != 1500 {
		t.Errorf("max_buffer_delay_ms = %v, want 1500", got)
	}
}

// The request carries what synthesis needs: the voice, the model, the
// language, the rate and the format.
func TestBuildMsgCarriesTheSettings(t *testing.T) {
	msg := newSynthesizer(Config{APIKey: "k", Language: "fr", SampleRate: 16000}).buildMsg("hi", ctxID, true)
	for key, want := range map[string]any{
		"voice_id":      modelDefaultVoices[defaultModel],
		"model":         defaultModel,
		"language":      "fr",
		"sample_rate":   16000,
		"output_format": "pcm",
	} {
		if msg[key] != want {
			t.Errorf("%s = %v, want %v", key, msg[key], want)
		}
	}
	for _, key := range []string{
		"speed", "math_notation", "content_filter", "date_format", "number_pronunciation_language",
	} {
		if _, present := msg[key]; present {
			t.Errorf("%s was sent though not set", key)
		}
	}
}

// The content filter is enabled with the action it was given, and left out
// when nobody set it.
func TestBuildMsgContentFilter(t *testing.T) {
	for _, action := range []string{"reject", "flag", "redact"} {
		t.Run(action, func(t *testing.T) {
			got := newSynthesizer(Config{APIKey: "k", ContentFilter: action}).buildMsg("hi", "", true)["content_filter"]
			if want := map[string]any{"enabled": true, "action": action}; !reflect.DeepEqual(got, want) {
				t.Errorf("content_filter = %v, want %v", got, want)
			}
		})
	}
}

// Numerals are read in the synthesis language unless a language is set, and
// a code, a neutral one or one of Smallest's own, is sent as given.
func TestBuildMsgNumberPronunciationLanguage(t *testing.T) {
	const key = "number_pronunciation_language"
	for _, code := range []string{string(language.Hindi), "auto"} {
		got := newSynthesizer(Config{APIKey: "k", NumberPronunciationLanguage: code}).buildMsg("hi", "", true)[key]
		if got != code {
			t.Errorf("%s = %v, want %s", key, got, code)
		}
	}
}

// The flags that are left out unless set, and sent as given once they are.
func TestBuildMsgOptionalControls(t *testing.T) {
	on, speed := true, 1.3
	msg := newSynthesizer(Config{APIKey: "k", MathNotation: &on, Speed: &speed, DateFormat: "EU"}).buildMsg("hi", "", true)
	if msg["math_notation"] != true || msg["speed"] != 1.3 || msg["date_format"] != "EU" {
		t.Errorf("sent %v, want math_notation, speed and date_format as set", msg)
	}
}

// The two Smallest models serve different voice catalogs, so the voice a caller
// gets when they name none has to follow the model they chose. A voice the
// caller named is sent as given.
func TestDefaultVoiceFollowsTheModel(t *testing.T) {
	for _, tc := range []struct {
		cfg  Config
		want string
	}{
		{Config{Model: modelLightning}, "sophia"},
		{Config{Model: modelLightningP}, "meher"},
		{Config{}, modelDefaultVoices[defaultModel]},
		{Config{Model: "lightning_next"}, modelDefaultVoices[defaultModel]},
		{Config{Model: modelLightning, Voice: "someone"}, "someone"},
	} {
		tc.cfg.APIKey = "k"
		if got := newSynthesizer(tc.cfg).buildMsg("hi", "", true)["voice_id"]; got != tc.want {
			t.Errorf("model %q sent voice %v, want %q", tc.cfg.Model, got, tc.want)
		}
	}
}

// A flush sends an empty, last fragment for the active context.
func TestFlushAudioClosesTheActiveContext(t *testing.T) {
	server := newFakeServer(t, nil)
	s, _, _ := connectedSynth(t, server, Config{})

	s.FlushAudio(context.Background(), "")
	msg := server.next(t)
	if msg["context_id"] != ctxID || msg["continue"] != false || msg["text"] != "" {
		t.Errorf("flush = %v, want an empty last fragment for %s", msg, ctxID)
	}
}

// With no connection there is nowhere to send a flush, and nothing fails.
func TestFlushAudioIsANoopWithoutAConnection(t *testing.T) {
	s := newSynthesizer(Config{APIKey: "k"})
	s.host, s.svc = newFakeHost(), &fakeSvc{active: ctxID}
	s.FlushAudio(context.Background(), "")
}

// Later requests of a turn are moved onto the turn's timeline: the second
// request's words are offset by the end of the first one's last word.
func TestWordTimestampsOffsetAcrossRequests(t *testing.T) {
	got := driveWords(t, newSynthesizer(Config{APIKey: "k"}),
		wordMessage("Hello", 0.2, 0.5, "a", 0),
		wordMessage("World", 0.1, 0.4, "b", 0),
	)
	assertWords(t, got, uctx.WordTiming{Word: "Hello", Offset: 0.2}, uctx.WordTiming{Word: "World", Offset: 0.6})
}

// The offset compounds across three requests of one turn.
func TestOffsetAccumulatesAcrossMultipleRequests(t *testing.T) {
	got := driveWords(t, newSynthesizer(Config{APIKey: "k"}),
		wordMessage("one", 0.0, 1.0, "a", 0),
		wordMessage("two", 0.0, 2.0, "b", 0),
		wordMessage("three", 0.5, 1.0, "c", 0),
	)
	assertWords(t, got,
		uctx.WordTiming{Word: "one", Offset: 0.0},
		uctx.WordTiming{Word: "two", Offset: 1.0},
		uctx.WordTiming{Word: "three", Offset: 3.5},
	)
}

// The words of one request share its offset; only the end grows.
func TestMultipleWordsInOneRequestShareOffset(t *testing.T) {
	got := driveWords(t, newSynthesizer(Config{APIKey: "k"}),
		wordMessage("a", 0.0, 0.4, "r1", 0),
		wordMessage("b", 0.4, 0.9, "r1", 1),
		wordMessage("c", 0.1, 0.5, "r2", 0),
	)
	assertWords(t, got,
		uctx.WordTiming{Word: "a", Offset: 0.0},
		uctx.WordTiming{Word: "b", Offset: 0.4},
		uctx.WordTiming{Word: "c", Offset: 1.0},
	)
}

// A new turn clears the offset built up over the last one.
func TestWordTimestampOffsetResetsOnNewTurn(t *testing.T) {
	s := newSynthesizer(Config{APIKey: "k"})
	driveWords(t, s, wordMessage("Hello", 0.2, 0.5, "a", 0), wordMessage("World", 0.1, 0.4, "b", 0))
	if math.Abs(s.cumulativeTime-0.5) > 1e-9 {
		t.Fatalf("offset = %v, want 0.5 after two requests", s.cumulativeTime)
	}

	s.OnTurnContextCreated(context.Background(), "ctx-2")
	if s.cumulativeTime != 0 || s.wtRequestID != nil {
		t.Errorf("offset %v and request %v after a new turn, want both cleared", s.cumulativeTime, s.wtRequestID)
	}
	got := driveWords(t, s, wordMessage("Fresh", 0.3, 0.6, "c", 0))
	assertWords(t, got, uctx.WordTiming{Word: "Fresh", Offset: 0.3})
}

// Spoken text produces audio and the text of each timed word, pushed as its
// audio plays.
func TestSpeaksWithWordTimestamps(t *testing.T) {
	server := newFakeServer(t, speaking)
	b := NewTTS(Config{APIKey: "k", URL: server.url, SampleRate: 16000})
	r := runPipeline(t, b)
	defer r.cancel()

	r.task.QueueFrame(frames.NewTTSSpeakFrame("Hello."))
	r.waitFor(func(f frames.Frame) bool { _, ok := f.(*frames.TTSAudioRawFrame); return ok })
	r.waitFor(func(f frames.Frame) bool { _, ok := f.(*frames.TTSTextFrame); return ok })
	r.mu.Lock()
	var texts []string
	for _, f := range r.down {
		if tf, ok := f.(*frames.TTSTextFrame); ok {
			texts = append(texts, tf.Text)
		}
	}
	r.mu.Unlock()
	if !slices.Equal(texts, []string{"Hello."}) {
		t.Errorf("text frames = %v, want [Hello.]", texts)
	}
}

// Audio is labeled with the context playing, and a complete turn ends the
// metrics.
func TestAudioAndCompletion(t *testing.T) {
	s := newSynthesizer(Config{APIKey: "k"})
	host, svc := newFakeHost(), &fakeSvc{active: ctxID}
	s.host, s.svc = host, svc
	ctx := context.Background()

	chunk, _ := json.Marshal(chunkMessage([]byte{1, 2}))
	if err := s.handleMessage(ctx, chunk); err != nil {
		t.Fatal(err)
	}
	if got := host.audioFor(ctxID); !bytes.Equal(got, []byte{1, 2}) {
		t.Errorf("audio = %v, want the chunk on the active context", got)
	}
	if err := s.handleMessage(ctx, []byte(`{"status":"complete"}`)); err != nil {
		t.Fatal(err)
	}
	if svc.stopped != 1 {
		t.Errorf("metrics stopped %d times, want once on completion", svc.stopped)
	}
	if err := s.handleMessage(ctx, []byte(`{"status":"queued"}`)); err != nil {
		t.Errorf("an unknown status failed the connection: %v", err)
	}
	if err := s.handleMessage(ctx, []byte(`{"status":"chunk","data":{"audio":"not base64!!"}}`)); err == nil {
		t.Error("audio that is not base64 was accepted")
	}
}

// An error the server reports stops the context playing, ends the metrics and
// is reported, carrying what the server said.
func TestServerErrorIsReported(t *testing.T) {
	s := newSynthesizer(Config{APIKey: "k"})
	host, svc := newFakeHost(), &fakeSvc{active: ctxID}
	s.host, s.svc = host, svc

	if err := s.handleMessage(context.Background(), []byte(`{"status":"error","error":"voice not found"}`)); err != nil {
		t.Fatal(err)
	}
	if len(host.pushed) != 1 {
		t.Fatalf("pushed %v, want one stop frame", host.pushed)
	}
	if stopped, ok := host.pushed[0].(*frames.TTSStoppedFrame); !ok || stopped.ContextID != ctxID {
		t.Errorf("pushed %v, want the active context stopped", host.pushed[0])
	}
	if svc.stopped != 1 {
		t.Errorf("metrics stopped %d times, want once", svc.stopped)
	}
	if !slices.Equal(host.errors, []string{"Smallest TTS error: voice not found"}) {
		t.Errorf("reported %v, want the server's message", host.errors)
	}
}

// The keepalive is a blank text in the voice, the model and the language the
// service speaks, and follows a change of them.
func TestKeepalive(t *testing.T) {
	server := newFakeServer(t, nil)
	s, _, _ := connectedSynth(t, server, Config{})
	ctx := context.Background()

	if err := s.SendKeepalive(ctx); err != nil {
		t.Fatal(err)
	}
	want := map[string]any{"text": " ", "voice_id": "meher", "model": defaultModel, "language": "en"}
	if got := server.next(t); !reflect.DeepEqual(got, want) {
		t.Errorf("keepalive = %v, want %v", got, want)
	}

	changed, err := settings.Apply(s.live, &TTSSettings{Voice: settings.Set("liam")})
	if err != nil {
		t.Fatal(err)
	}
	if err := s.UpdateSettings(ctx, changed); err != nil {
		t.Fatal(err)
	}
	if err := s.SendKeepalive(ctx); err != nil {
		t.Fatal(err)
	}
	if got := server.next(t)["voice_id"]; got != "liam" {
		t.Errorf("keepalive voice = %v, want the new one", got)
	}
	if !s.ws.KeepaliveRunning() {
		t.Error("the keepalive loop is not running while connected")
	}
}

// Every request carries its settings, so a change reaches the next request
// without reconnecting.
func TestSettingsChangeAppliesToTheNextRequest(t *testing.T) {
	server := newFakeServer(t, nil)
	s, _, _ := connectedSynth(t, server, Config{})
	ctx := context.Background()

	delta := &TTSSettings{Voice: settings.Set("liam"), Speed: settings.Set(1.4), DateFormat: settings.Set("EU")}
	changed, err := settings.Apply(s.live, delta)
	if err != nil {
		t.Fatal(err)
	}
	if err := s.UpdateSettings(ctx, changed); err != nil {
		t.Fatal(err)
	}
	if err := s.RunTTS(ctx, "hi", ctxID, nil); err != nil {
		t.Fatal(err)
	}
	msg := server.next(t)
	if msg["voice_id"] != "liam" || msg["speed"] != 1.4 || msg["date_format"] != "EU" {
		t.Errorf("sent %v, want the new settings", msg)
	}
	if n := server.connections(); n != 1 {
		t.Errorf("the change reconnected: %d connections", n)
	}
}

// The connection is authorized with the key, and names its source.
func TestConnectionHeaders(t *testing.T) {
	server := newFakeServer(t, nil)
	connectedSynth(t, server, Config{})

	server.mu.Lock()
	header := server.headers[0]
	server.mu.Unlock()
	if got := header.Get("Authorization"); got != "Bearer test-key" {
		t.Errorf("Authorization = %q, want the key as a bearer token", got)
	}
	if got := header.Get("X-Source"); got != "jargo" {
		t.Errorf("X-Source = %q, want jargo", got)
	}
	if !server.quiet() {
		t.Error("the service sent something on connecting; every request carries its settings")
	}
}

// A connection the server drops is restored by the receive loop.
func TestReconnectsWhenTheServerDrops(t *testing.T) {
	server := newFakeServer(t, nil)
	s, _, _ := connectedSynth(t, server, Config{Websocket: wsservice.Options{
		ReconnectBackoffMinWait: time.Millisecond,
		ReconnectBackoffMaxWait: time.Millisecond,
	}})

	server.drop()
	waitUntil(t, "the service never reconnected", func() bool {
		return server.connections() == 2 && s.Connected()
	})
}

// A sentence after the connection closed opens it again before it is sent.
func TestSentenceReopensAClosedConnection(t *testing.T) {
	server := newFakeServer(t, nil)
	s, _, _ := connectedSynth(t, server, Config{})
	s.Disconnect(context.Background())

	if err := s.RunTTS(context.Background(), "Hello.", ctxID, nil); err != nil {
		t.Fatal(err)
	}
	if got := server.next(t); got["text"] != "Hello." {
		t.Errorf("sent %v, want the sentence", got)
	}
	if n := server.connections(); n != 2 {
		t.Errorf("the server saw %d connections, want the closed one opened again", n)
	}
}

// A send that fails stops the context's speech, is reported, and resets the
// connection.
func TestFailedSendResetsTheConnection(t *testing.T) {
	server := newFakeServer(t, nil)
	_, s := newTTS(Config{APIKey: "k", URL: server.url})
	s.host, s.svc = newFakeHost(), &fakeSvc{}
	t.Cleanup(func() { s.Disconnect(context.Background()) })

	// A connection that is already closed but not yet known to be: the send is
	// where the failure shows.
	conn, err := s.ws.Dial(context.Background(), server.url, http.Header{}, wsutil.DefaultReadLimit)
	if err != nil {
		t.Fatal(err)
	}
	_ = conn.CloseNow()
	s.conn = &wsConn{conn: conn, reads: make(chan wsRead), stop: make(chan struct{})}

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
	waitUntil(t, "the connection was not reset", func() bool { return server.connections() == 2 })
	if !s.Connected() {
		t.Error("not connected after the reset")
	}
}

// TestSmallestLanguage checks a language is named by its base code.
func TestSmallestLanguage(t *testing.T) {
	for in, want := range map[language.Language]string{
		language.English:   "en",
		language.EnglishUS: "en",
		language.Hindi:     "hi",
		language.French:    "fr",
	} {
		if got := smallestLanguage(in); got != want {
			t.Errorf("smallestLanguage(%q) = %q, want %q", in, got, want)
		}
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
	waitUntil(r.t, "the frame waited for never came", func() bool {
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

// The service connects when the pipeline starts and its audio reaches the
// pipeline over that one connection, labeled with the context it was asked
// for.
func TestHoldsOneConnectionForThePipeline(t *testing.T) {
	server := newFakeServer(t, speaking)
	b := NewTTS(Config{APIKey: "k", URL: server.url})
	r := runPipeline(t, b)
	waitUntil(t, "the server saw no connection when the pipeline started", func() bool {
		return server.connections() == 1
	})

	r.task.QueueFrame(frames.NewTTSSpeakFrame("Hello there."))
	r.waitFor(func(f frames.Frame) bool {
		a, ok := f.(*frames.TTSAudioRawFrame)
		return ok && a.ContextID != ""
	})
	r.cancel()

	if n := server.connections(); n != 1 {
		t.Errorf("the server saw %d connections, want one for the whole pipeline", n)
	}
}

// A connection the server refuses is reported as a connection error.
func TestReportsAFailedConnection(t *testing.T) {
	server := newFakeServer(t, nil)
	server.refuse = http.StatusUnauthorized
	b := NewTTS(Config{APIKey: "k", URL: server.url})
	var mu sync.Mutex
	var reported []string
	events.On(b.Events(), service.EventConnectionError, func(_ context.Context, msg string) {
		mu.Lock()
		reported = append(reported, msg)
		mu.Unlock()
	})

	r := runPipeline(t, b)
	defer r.stop()
	waitUntil(t, "the refused connection was never reported", func() bool {
		mu.Lock()
		defer mu.Unlock()
		return len(reported) > 0
	})
}
