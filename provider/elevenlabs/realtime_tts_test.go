package elevenlabs

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"log/slog"
	"maps"
	"math"
	"net/http"
	"net/http/httptest"
	"net/url"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/internal/providertest"
	"github.com/gojargo/jargo/language"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/service"
	"github.com/gojargo/jargo/service/settings"
	"github.com/gojargo/jargo/service/tts"
	"github.com/gojargo/jargo/service/wsservice"
	uctx "github.com/gojargo/jargo/utils/context"
	"github.com/gojargo/jargo/utils/events"
	"github.com/gojargo/jargo/utils/text"
)

// TestRealtimeTTSConfigValidate pins which fields the service requires.
func TestRealtimeTTSConfigValidate(t *testing.T) {
	providertest.Configs(t, []providertest.ConfigCase{
		{Name: "missing API key", Cfg: RealtimeTTSConfig{}, Valid: false},
		{Name: "API key only", Cfg: RealtimeTTSConfig{APIKey: "k"}, Valid: true},
	})
}

// TestNewRealtimeTTS checks the constructor returns a service under the label
// that identifies it in logs, metrics and traces.
func TestNewRealtimeTTS(t *testing.T) {
	providertest.Service(t, "ElevenLabsRealtimeTTS", NewRealtimeTTS(RealtimeTTSConfig{APIKey: "k"}))
}

// Word timings are only reported when they were asked for, so the base takes the
// word-aligned path solely then.
func TestRealtimeTTSWordTimestampsAreOptional(t *testing.T) {
	plain := newRealtimeSynthesizer(RealtimeTTSConfig{APIKey: "k"})
	if _, ok := any(plain).(tts.WordTimestamps); ok {
		t.Fatal("the plain synthesizer must not implement tts.WordTimestamps")
	}
	timed := &timedRealtimeSynthesizer{realtimeSynthesizer: plain}
	if _, ok := any(timed).(tts.WordTimestamps); !ok {
		t.Fatal("the timed synthesizer must implement tts.WordTimestamps")
	}
}

// The connection is opened once and its fixed parameters live in the URL, since
// only the text varies from one sentence to the next.
func TestRealtimeTTSEndpoint(t *testing.T) {
	s := newRealtimeSynthesizer(RealtimeTTSConfig{
		APIKey:                 "k",
		SampleRate:             24000,
		Language:               language.French,
		ApplyTextNormalization: "off",
	})

	raw := s.endpoint()
	if !strings.HasPrefix(raw, defaultRealtimeTTSHost+"/v1/text-to-speech/"+defaultVoiceID+"/multi-stream-input?") {
		t.Fatalf("unexpected endpoint: %s", raw)
	}
	u, err := url.Parse(raw)
	if err != nil {
		t.Fatal(err)
	}
	for key, want := range map[string]string{
		"model_id":                 defaultModel,
		"output_format":            "pcm_24000",
		"language_code":            "fr",
		"apply_text_normalization": "off",
		// Omitting auto_mode leaves the server buffering text that closing the
		// context then discards, so a synthesis returns no audio whatsoever.
		"auto_mode": "true",
	} {
		if got := u.Query().Get(key); got != want {
			t.Errorf("%s = %q, want %q", key, got, want)
		}
	}
}

// TestRealtimeTTSAutoModeCanBeDisabled checks the default is a default and not a
// constant: a caller driving generation itself with explicit flushes can turn it
// off.
func TestRealtimeTTSAutoModeCanBeDisabled(t *testing.T) {
	s := newRealtimeSynthesizer(RealtimeTTSConfig{APIKey: "k", AutoMode: new(false)})

	u, err := url.Parse(s.endpoint())
	if err != nil {
		t.Fatal(err)
	}
	if got := u.Query().Get("auto_mode"); got != "false" {
		t.Errorf("auto_mode = %q, want %q", got, "false")
	}
}

// realtimeServer is a fake multi-stream-input endpoint. It records every
// message the client sends, answers each with what reply returns (stamping the
// context the message named), and closes the connection when asked to.
type realtimeServer struct {
	host string
	sent chan map[string]any

	mu      sync.Mutex
	reply   func(msg map[string]any) []map[string]any
	refuse  int
	conns   []*websocket.Conn
	queries []url.Values
}

func newRealtimeServer(t *testing.T, reply func(msg map[string]any) []map[string]any) *realtimeServer {
	t.Helper()
	s := &realtimeServer{sent: make(chan map[string]any, 256), reply: reply}
	srv := httptest.NewServer(http.HandlerFunc(s.serve))
	t.Cleanup(srv.Close)
	s.host = "ws" + strings.TrimPrefix(srv.URL, "http")
	return s
}

func (s *realtimeServer) serve(w http.ResponseWriter, r *http.Request) {
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
	c.SetReadLimit(-1)
	s.mu.Lock()
	s.conns = append(s.conns, c)
	s.queries = append(s.queries, r.URL.Query())
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
		if closing, _ := msg[keyCloseSocket].(bool); closing {
			_ = c.Close(websocket.StatusNormalClosure, "")
			return
		}
		if reply == nil {
			continue
		}
		for _, ev := range reply(msg) {
			ev = maps.Clone(ev)
			if _, ok := ev["contextId"]; !ok {
				ev["contextId"] = msg[keyContextID]
			}
			b, _ := json.Marshal(ev)
			if c.Write(ctx, websocket.MessageText, b) != nil {
				return
			}
		}
	}
}

// connections is how many connections the server has accepted.
func (s *realtimeServer) connections() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return len(s.conns)
}

// query is the query of the n-th connection, from zero.
func (s *realtimeServer) query(n int) url.Values {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.queries[n]
}

// drop ends every connection without a closing handshake, the way a network
// failure does.
func (s *realtimeServer) drop() {
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, c := range s.conns {
		_ = c.CloseNow()
	}
}

// next is the next message the client sent.
func (s *realtimeServer) next(t *testing.T) map[string]any {
	t.Helper()
	select {
	case msg := <-s.sent:
		return msg
	case <-time.After(3 * time.Second):
		t.Fatal("the client sent nothing")
		return nil
	}
}

// waitFor reads the client's messages until one matches, returning every one
// read.
func (s *realtimeServer) waitFor(t *testing.T, match func(map[string]any) bool) []map[string]any {
	t.Helper()
	var read []map[string]any
	for {
		msg := s.next(t)
		read = append(read, msg)
		if match(msg) {
			return read
		}
	}
}

// onClose answers the closing of a context with events.
func onClose(events ...map[string]any) func(map[string]any) []map[string]any {
	return func(msg map[string]any) []map[string]any {
		if closing, _ := msg[keyCloseContext].(bool); closing {
			return events
		}
		return nil
	}
}

// onText answers every sentence sent with events.
func onText(events ...map[string]any) func(map[string]any) []map[string]any {
	return func(msg map[string]any) []map[string]any {
		if text, _ := msg[keyText].(string); strings.TrimSpace(text) != "" {
			return events
		}
		return nil
	}
}

func isCloseSocket(msg map[string]any) bool {
	closing, _ := msg[keyCloseSocket].(bool)
	return closing
}

func isCloseContext(msg map[string]any) bool {
	closing, _ := msg[keyCloseContext].(bool)
	return closing
}

func audioMessage(pcm []byte) map[string]any {
	return map[string]any{"audio": base64.StdEncoding.EncodeToString(pcm)}
}

// fakeSvc stands in for the TTS service the synthesizer asks about its audio
// contexts.
type fakeSvc struct {
	mu     sync.Mutex
	active string
	ids    []string
	warned []string
}

func (f *fakeSvc) ActiveAudioContextID() string {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.active
}

func (f *fakeSvc) AudioContextIDs() []string {
	f.mu.Lock()
	defer f.mu.Unlock()
	return slices.Clone(f.ids)
}

func (f *fakeSvc) StopAllMetrics(context.Context) {}

func (f *fakeSvc) WarnUnhandledUpdatedSettings(_ context.Context, unhandled []string) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.warned = append(f.warned, unhandled...)
}

// fakeHost stands in for the tts.Base a provider appends its audio to. Only
// the contexts opened on it take frames, as on the base.
type fakeHost struct {
	mu      sync.Mutex
	open    map[string]bool
	audio   []byte
	words   []string
	frames  []string
	removed []string
	errors  []string
}

func newFakeHost(ids ...string) *fakeHost {
	h := &fakeHost{open: map[string]bool{}}
	for _, id := range ids {
		h.open[id] = true
	}
	return h
}

// openContext opens a context, as the base does before asking for speech.
func (h *fakeHost) openContext(id string) {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.open[id] = true
}

func (h *fakeHost) AppendToAudioContext(id string, f frames.Frame) {
	h.mu.Lock()
	defer h.mu.Unlock()
	if !h.open[id] {
		return
	}
	switch fr := f.(type) {
	case *frames.TTSAudioRawFrame:
		h.audio = append(h.audio, fr.Audio...)
		h.frames = append(h.frames, "audio")
	case *frames.TTSStoppedFrame:
		h.frames = append(h.frames, "stopped")
	}
}

func (h *fakeHost) AppendWordToAudioContext(_, word string, _ float64) {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.words = append(h.words, word)
}

// AddWordTimestamps mirrors the base: normalize as asked, then queue each token.
func (h *fakeHost) AddWordTimestamps(contextID string, words []uctx.WordTiming, opts tts.WordTimingOptions) {
	if opts.PreMergeTokens {
		words = uctx.MergePunctTokens(words)
	}
	for _, w := range words {
		h.AppendWordToAudioContext(contextID, w.Word, w.Offset)
	}
}

func (h *fakeHost) RemoveAudioContext(id string) {
	h.mu.Lock()
	defer h.mu.Unlock()
	delete(h.open, id)
	h.removed = append(h.removed, id)
}

func (h *fakeHost) AudioContextAvailable(id string) bool {
	h.mu.Lock()
	defer h.mu.Unlock()
	return h.open[id]
}

func (h *fakeHost) ReplaceConnection() {}

// RotateTurnContext is not exercised here: this provider applies a settings
// change by reconnecting or by closing its contexts.
func (h *fakeHost) RotateTurnContext(context.Context) {}

func (h *fakeHost) PushError(_ context.Context, msg string, _ error, _ bool, _ ...processor.ErrorOption) {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.errors = append(h.errors, msg)
}

func (h *fakeHost) PushFrame(context.Context, frames.Frame, processor.Direction) error { return nil }

func (h *fakeHost) ResetActiveAudioContext() {}

func (h *fakeHost) spokenWords() []string {
	h.mu.Lock()
	defer h.mu.Unlock()
	return slices.Clone(h.words)
}

func (h *fakeHost) snapshot() (audio []byte, frameKinds []string) {
	h.mu.Lock()
	defer h.mu.Unlock()
	return slices.Clone(h.audio), slices.Clone(h.frames)
}

func (h *fakeHost) pushedErrors() []string {
	h.mu.Lock()
	defer h.mu.Unlock()
	return slices.Clone(h.errors)
}

// waitForClose blocks until the provider has closed a context, or fails.
func (h *fakeHost) waitForClose(t *testing.T) {
	t.Helper()
	waitUntil(t, "the provider never closed the audio context", func() bool {
		h.mu.Lock()
		defer h.mu.Unlock()
		return len(h.removed) > 0
	})
}

// waitUntil polls cond until it holds, or fails with msg.
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
// service replaced by fakes, and connects it.
func connectedSynth(t *testing.T, server *realtimeServer, cfg RealtimeTTSConfig) (
	*realtimeSynthesizer, *fakeHost, *fakeSvc,
) {
	t.Helper()
	cfg.APIKey = "k"
	cfg.Host = server.host
	_, s := newRealtimeTTS(cfg)
	host, svc := newFakeHost(), &fakeSvc{}
	s.host, s.svc = host, svc
	s.Start(context.Background())
	if !s.Connected() {
		t.Fatal("the service did not connect when it started")
	}
	t.Cleanup(func() { s.Disconnect(context.Background()) })
	return s, host, svc
}

// speakOne drives one synthesis on its own context and returns the audio the
// provider delivered to its host.
func speakOne(t *testing.T, s *realtimeSynthesizer, host *fakeHost, text string) []byte {
	t.Helper()
	host.openContext("c1")
	ctx := context.Background()
	if err := s.RunTTS(ctx, text, "c1", nil); err != nil {
		t.Fatalf("synthesis failed: %v", err)
	}
	s.OnTurnContextCompleted(ctx, "c1")
	host.waitForClose(t)
	audio, _ := host.snapshot()
	return audio
}

// deliver hands the synthesizer one server message, as the receive loop does.
func deliver(t *testing.T, s *realtimeSynthesizer, msg *realtimeTTSMessage) {
	t.Helper()
	data, err := json.Marshal(msg)
	if err != nil {
		t.Fatal(err)
	}
	if err := s.handleMessage(data); err != nil {
		t.Fatal(err)
	}
}

// TestRealtimeTTSAcceptsALargeAudioMessage checks a single message carrying a
// long sentence's audio is read rather than rejected. Generating a whole sentence
// at once puts well over a megabyte of base64 in one message, and a read limit
// tight enough to cut that off fails the synthesis part-way through a reply.
func TestRealtimeTTSAcceptsALargeAudioMessage(t *testing.T) {
	// Comfortably past the megabyte that used to be the ceiling.
	big := make([]byte, 2<<20)
	for i := range big {
		big[i] = byte(i)
	}
	server := newRealtimeServer(t, onClose(audioMessage(big), map[string]any{"isFinal": true}))
	s, host, _ := connectedSynth(t, server, RealtimeTTSConfig{SampleRate: 24000})

	if total := len(speakOne(t, s, host, "Une phrase longue.")); total != len(big) {
		t.Errorf("delivered %d bytes, want %d", total, len(big))
	}
}

// TestRealtimeTTSSynthesize checks the messages the service sends and the audio
// it emits.
func TestRealtimeTTSSynthesize(t *testing.T) {
	server := newRealtimeServer(t, onClose(
		audioMessage([]byte{1, 2, 3, 4}),
		audioMessage([]byte{5, 6}),
		map[string]any{"isFinal": true},
	))
	s, host, _ := connectedSynth(t, server, RealtimeTTSConfig{})

	if got, want := speakOne(t, s, host, "Bonjour."), []byte{1, 2, 3, 4, 5, 6}; !bytes.Equal(got, want) {
		t.Fatalf("delivered %v, want %v", got, want)
	}

	open := server.next(t)
	if open[keyText] != " " {
		t.Errorf("the opening message should initialize without committing text, got %q", open[keyText])
	}
	body := server.next(t)
	if body[keyText] != "Bonjour." {
		t.Errorf("sent text = %q, want %q", body[keyText], "Bonjour.")
	}
	// Closing the context is what makes the final marker arrive right after the
	// last audio byte instead of the server waiting for more text.
	closing := server.next(t)
	if !isCloseContext(closing) {
		t.Errorf("the context should be closed after the text, got %v", closing)
	}
	if open[keyContextID] != body[keyContextID] || body[keyContextID] != closing[keyContextID] {
		t.Error("every message of a synthesis must carry the same context id")
	}
}

// A second sentence must reuse the open connection: not paying for a handshake
// per sentence is the whole point of this service over the HTTP one.
func TestRealtimeTTSReusesTheConnection(t *testing.T) {
	server := newRealtimeServer(t, onClose(audioMessage([]byte{1}), map[string]any{"isFinal": true}))
	s, _, _ := connectedSynth(t, server, RealtimeTTSConfig{})

	speakOne(t, s, newHostFor(s), "Un.")
	speakOne(t, s, newHostFor(s), "Deux.")
	if n := server.connections(); n != 1 {
		t.Fatalf("the server saw %d connections, want the one opened at start", n)
	}
}

// newHostFor gives s a fresh host and returns it.
func newHostFor(s *realtimeSynthesizer) *fakeHost {
	h := newFakeHost()
	s.host = h
	return h
}

// Audio the server had already generated for an abandoned synthesis arrives
// after it. Attributing by context id is what keeps it out of the next sentence.
func TestRealtimeTTSIgnoresOtherContexts(t *testing.T) {
	server := newRealtimeServer(t, onClose(
		map[string]any{"contextId": "stale", "audio": base64.StdEncoding.EncodeToString([]byte{9, 9, 9})},
		audioMessage([]byte{7}),
		map[string]any{"isFinal": true},
	))
	s, host, _ := connectedSynth(t, server, RealtimeTTSConfig{})

	if got, want := speakOne(t, s, host, "Bonjour."), []byte{7}; !bytes.Equal(got, want) {
		t.Fatalf("delivered %v, want %v: audio from another context leaked in", got, want)
	}
}

// ElevenLabs times every character; the words spoken are assembled from those.
func TestRealtimeTTSReportsWords(t *testing.T) {
	server := newRealtimeServer(t, onClose(
		map[string]any{
			"audio": base64.StdEncoding.EncodeToString([]byte{1, 2}),
			"alignment": map[string]any{
				"chars":            []string{"S", "a", "l", "u", "t", " ", "t", "o", "i"},
				"charStartTimesMs": []float64{0, 50, 100, 150, 200, 250, 300, 350, 400},
			},
		},
		map[string]any{"isFinal": true},
	))
	s, _, _ := connectedSynth(t, server, RealtimeTTSConfig{WordTimestamps: true})
	timed := &timedRealtimeSynthesizer{realtimeSynthesizer: s}

	// Words are delivered to the host with the audio, not returned by the call.
	rec := newOffsetHost("c1")
	s.host = rec
	ctx := context.Background()
	if err := timed.RunTTSTimed(ctx, "Salut toi", "c1", nil, nil); err != nil {
		t.Fatal(err)
	}
	s.OnTurnContextCompleted(ctx, "c1")
	rec.waitForClose(t)

	if got, want := rec.offsets(), []float64{0, 0.3}; !slices.Equal(got, want) {
		t.Fatalf("word offsets = %v, want %v", got, want)
	}
}

// Every sentence of a turn goes into one context, closed once at the end. A
// context per sentence would make the server generate each one cold, and the
// seam between them is what leaves the transport with nothing to send.
func TestRealtimeTTSTurnSharesOneContext(t *testing.T) {
	server := newRealtimeServer(t, onClose(
		audioMessage([]byte{1, 2}),
		audioMessage([]byte{3, 4}),
		map[string]any{"isFinal": true},
	))
	s, host, _ := connectedSynth(t, server, RealtimeTTSConfig{})
	host.openContext("turn-1")

	ctx := context.Background()
	// Neither call waits for the other's audio: the server sends none until the
	// context closes, so a client that waited would never reach the second.
	if err := s.RunTTS(ctx, "Un.", "turn-1", nil); err != nil {
		t.Fatal(err)
	}
	if err := s.RunTTS(ctx, "Deux.", "turn-1", nil); err != nil {
		t.Fatal(err)
	}
	s.OnTurnContextCompleted(ctx, "turn-1")
	host.waitForClose(t)

	audio, seq := host.snapshot()
	if want := []byte{1, 2, 3, 4}; !bytes.Equal(audio, want) {
		t.Fatalf("appended audio %v, want %v", audio, want)
	}
	if len(seq) == 0 || seq[len(seq)-1] != "stopped" {
		t.Fatalf("frame sequence = %v, want it to end on a stop frame", seq)
	}

	sent := server.waitFor(t, isCloseContext)
	ids := map[any]bool{}
	var texts []string
	for _, msg := range sent {
		ids[msg[keyContextID]] = true
		if text, ok := msg[keyText].(string); ok {
			texts = append(texts, text)
		}
	}
	if len(ids) != 1 {
		t.Errorf("a turn's messages must share one context id, got %d ids", len(ids))
	}
	if want := []string{" ", "Un.", "Deux."}; !slices.Equal(texts, want) {
		t.Fatalf("sent texts = %v, want %v", texts, want)
	}
}

// ElevenLabs times each message from the start of the audio it carries, so the
// offsets are rebased onto the context's timeline. Without that, every message
// after the first reports its words near zero and they are released ahead of the
// audio that speaks them.
func TestRealtimeTTSRebasesWordOffsetsOntoTheContext(t *testing.T) {
	alignment := map[string]any{
		"chars":            []string{"U", "n", " "},
		"charStartTimesMs": []float64{0, 100, 200},
		"charDurationsMs":  []float64{100, 100, 100},
	}
	first := audioMessage([]byte{1, 2})
	first["alignment"] = alignment
	second := audioMessage([]byte{3, 4})
	second["alignment"] = alignment

	server := newRealtimeServer(t, onClose(first, second, map[string]any{"isFinal": true}))
	s, _, _ := connectedSynth(t, server, RealtimeTTSConfig{WordTimestamps: true})
	rec := newOffsetHost("turn-1")
	s.host = rec

	ctx := context.Background()
	if err := s.RunTTS(ctx, "Un un.", "turn-1", nil); err != nil {
		t.Fatal(err)
	}
	s.OnTurnContextCompleted(ctx, "turn-1")
	rec.waitForClose(t)

	// The first message ends at 200ms plus 100ms of duration, so the second
	// message's word starts there rather than back at zero.
	if got, want := rec.offsets(), []float64{0, 0.3}; !slices.Equal(got, want) {
		t.Fatalf("word offsets = %v, want %v", got, want)
	}
}

// offsetHost records the offsets words are reported at.
type offsetHost struct {
	*fakeHost
	mu   sync.Mutex
	offs []float64
}

func newOffsetHost(ids ...string) *offsetHost { return &offsetHost{fakeHost: newFakeHost(ids...)} }

func (h *offsetHost) AddWordTimestamps(_ string, words []uctx.WordTiming, opts tts.WordTimingOptions) {
	if opts.PreMergeTokens {
		words = uctx.MergePunctTokens(words)
	}
	h.mu.Lock()
	defer h.mu.Unlock()
	for _, w := range words {
		h.offs = append(h.offs, w.Offset)
	}
}

func (h *offsetHost) offsets() []float64 {
	h.mu.Lock()
	defer h.mu.Unlock()
	return slices.Clone(h.offs)
}

// An interrupted context must not spill into the next one: audio the server had
// already generated for it arrives after the interruption and is dropped by id.
func TestRealtimeTTSInterruptedContextDropsItsAudio(t *testing.T) {
	s := newRealtimeSynthesizer(RealtimeTTSConfig{APIKey: "k"})
	host := newFakeHost("c1")
	s.host = host

	// The base tells the provider, then drops the context.
	s.OnAudioContextInterrupted(context.Background(), "c1")
	host.RemoveAudioContext("c1")

	deliver(t, s, &realtimeTTSMessage{ContextID: "c1", Audio: base64.StdEncoding.EncodeToString([]byte{1, 2})})
	if audio, _ := host.snapshot(); len(audio) != 0 {
		t.Fatalf("audio from the interrupted context reached the host: %v", audio)
	}
}

// A turn must not open with whitespace, but a leading space on a later message
// is a real word separator: the flash models split a sentence that way, and
// dropping it would leave the word carried over from the message before glued to
// the next one. Only the utterance's own leading space comes off.
func TestRealtimeTTSStripsOnlyUtteranceLeadingSpaces(t *testing.T) {
	server := newRealtimeServer(t, onClose(
		map[string]any{"alignment": map[string]any{
			"chars":            []string{" ", " ", "W", "h", "y"},
			"charStartTimesMs": []float64{0, 50, 100, 150, 200},
		}},
		map[string]any{"alignment": map[string]any{
			"chars":            []string{" ", "n", "o", "t"},
			"charStartTimesMs": []float64{250, 300, 350, 400},
		}},
		map[string]any{"isFinal": true},
	))
	s, host, _ := connectedSynth(t, server, RealtimeTTSConfig{WordTimestamps: true})
	host.openContext("c1")

	ctx := context.Background()
	if err := s.RunTTS(ctx, "Why not", "c1", nil); err != nil {
		t.Fatal(err)
	}
	s.OnTurnContextCompleted(ctx, "c1")
	host.waitForClose(t)

	if words, want := host.spokenWords(), []string{"Why", "not"}; !slices.Equal(words, want) {
		t.Fatalf("words = %v, want %v", words, want)
	}
}

// The strip works on the WebSocket schema's own array names, and leaves a later
// message's separating space in place.
func TestRealtimeCharAlignmentStripLeadingSpaces(t *testing.T) {
	a := &charAlignment{
		Chars:            []string{" ", " ", "H", "i"},
		CharStartTimesMs: []float64{0, 50, 100, 150},
		CharDurationsMs:  []float64{50, 50, 50, 50},
	}

	first := a.stripLeadingSpaces(true)
	if strings.Join(first.Chars, "") != "Hi" {
		t.Errorf("chars = %q, want the leading spaces gone", strings.Join(first.Chars, ""))
	}
	if len(first.CharStartTimesMs) != 2 || len(first.CharDurationsMs) != 2 {
		t.Errorf("times = %d/%d, want them cut alongside the characters",
			len(first.CharStartTimesMs), len(first.CharDurationsMs))
	}
	if first.CharStartTimesMs[0] != 100 {
		t.Errorf("first start = %v, want the surviving character's own time", first.CharStartTimesMs[0])
	}

	later := a.stripLeadingSpaces(false)
	if strings.Join(later.Chars, "") != "  Hi" {
		t.Errorf("chars = %q, want a later message's spacing kept", strings.Join(later.Chars, ""))
	}
}

// realtimeWarnings collects what building a realtime service with model logs at
// warning level.
func realtimeWarnings(t *testing.T, model string) string {
	t.Helper()
	var buf bytes.Buffer
	prev := slog.Default()
	slog.SetDefault(slog.New(slog.NewTextHandler(&buf, &slog.HandlerOptions{Level: slog.LevelWarn})))
	defer slog.SetDefault(prev)
	NewRealtimeTTS(RealtimeTTSConfig{APIKey: "test-key", VoiceID: "test-voice", Model: model})
	return buf.String()
}

// The multi-stream endpoint serves no v3/v4 model; the handshake rejects one
// with a 400 that leaves the service unusable, so a clear warning with a
// pointer to the service that does serve them beats it.
func TestRealtimeTTSWarnsOnV3V4Models(t *testing.T) {
	for _, model := range []string{"eleven_v3", "eleven_v4_turbo"} {
		logged := realtimeWarnings(t, model)
		if !strings.Contains(logged, model) || !strings.Contains(logged, "NewTTS") {
			t.Errorf("%s: logged %q, want a warning naming the model and NewTTS", model, logged)
		}
	}
}

func TestRealtimeTTSAcceptsFlashModelsWithoutAWarning(t *testing.T) {
	if logged := realtimeWarnings(t, "eleven_flash_v2_5"); strings.Contains(logged, "v3/v4") {
		t.Errorf("logged %q, want no v3/v4 warning", logged)
	}
}

// Holding frames per turn is the default; turning it off lets contexts overlap.
func TestRealtimeTTSPauseFrameProcessingIsConfigurable(t *testing.T) {
	if got := (RealtimeTTSConfig{}).withDefaults().PauseFrameProcessing; got == nil || !*got {
		t.Errorf("PauseFrameProcessing = %v, want on by default", got)
	}
	cfg := RealtimeTTSConfig{PauseFrameProcessing: new(false)}.withDefaults()
	if *cfg.PauseFrameProcessing {
		t.Error("PauseFrameProcessing turned off came back on")
	}
}

// alignmentChunk times each character of text 100 ms after the last, each
// lasting 100 ms.
func alignmentChunk(text string) *charAlignment {
	a := &charAlignment{}
	for i, r := range []rune(text) {
		a.Chars = append(a.Chars, string(r))
		a.CharStartTimesMs = append(a.CharStartTimesMs, float64(i*100))
		a.CharDurationsMs = append(a.CharDurationsMs, 100)
	}
	return a
}

// contextWord is a word reported for a context, at an offset.
type contextWord struct {
	context string
	word    string
	offset  float64
}

// wordsHost records the words reported for each context, and how they were
// said to be joined.
type wordsHost struct {
	*fakeHost
	mu        sync.Mutex
	words     []contextWord
	spaceless []bool
}

func newWordsHost(ids ...string) *wordsHost { return &wordsHost{fakeHost: newFakeHost(ids...)} }

func (h *wordsHost) AddWordTimestamps(contextID string, words []uctx.WordTiming, opts tts.WordTimingOptions) {
	h.mu.Lock()
	defer h.mu.Unlock()
	for _, w := range words {
		h.words = append(h.words, contextWord{contextID, w.Word, w.Offset})
		h.spaceless = append(h.spaceless, opts.IncludesInterFrameSpaces)
	}
}

func (h *wordsHost) reported() []contextWord {
	h.mu.Lock()
	defer h.mu.Unlock()
	return slices.Clone(h.words)
}

func (h *wordsHost) wordTexts() []string {
	var out []string
	for _, w := range h.reported() {
		out = append(out, w.word)
	}
	return out
}

// joined concatenates the words reported the way a consumer of them does.
func (h *wordsHost) joined() string {
	h.mu.Lock()
	defer h.mu.Unlock()
	parts := make([]text.Part, len(h.words))
	for i, w := range h.words {
		parts[i] = text.Part{Text: w.word, IncludesInterPartSpaces: h.spaceless[i]}
	}
	return text.Concatenate(parts)
}

// timedSynth builds a synthesizer reporting words to a host with the context
// "ctx-1" open.
func timedSynth(lang language.Language) (*realtimeSynthesizer, *wordsHost) {
	s := newRealtimeSynthesizer(RealtimeTTSConfig{APIKey: "k", WordTimestamps: true, Language: lang})
	host := newWordsHost("ctx-1")
	s.host = host
	return s, host
}

// speakChunks delivers each text as one alignment message on "ctx-1", then the
// final message.
func speakChunks(t *testing.T, s *realtimeSynthesizer, chunks ...string) {
	t.Helper()
	for _, c := range chunks {
		deliver(t, s, &realtimeTTSMessage{ContextID: "ctx-1", Alignment: alignmentChunk(c)})
	}
	deliver(t, s, &realtimeTTSMessage{ContextID: "ctx-1", IsFinal: true})
}

// A flash model splits a sentence across messages with the separating space at
// the head of the next one, which has to end the word carried over.
func TestRealtimeTTSFlashAlignmentPreservesInterWordChunkSpace(t *testing.T) {
	s, host := timedSynth("")
	speakChunks(t, s, " Why did the math book", " look so sad? ", " Because it had too m", "any problems. ")

	want := []string{
		"Why", "did", "the", "math", "book", "look", "so", "sad?",
		"Because", "it", "had", "too", "many", "problems.",
	}
	if got := host.wordTexts(); !slices.Equal(got, want) {
		t.Fatalf("words = %v, want %v", got, want)
	}
}

// Japanese is written without spaces between words, so the tokens are joined
// with none.
func TestRealtimeTTSJapaneseTimestampChunksReassembleWithoutSpaces(t *testing.T) {
	s, host := timedSynth("ja-JP")
	speakChunks(t, s, "どんなことでも気 ", "軽に相談してくださいね。 ")

	if got, want := host.wordTexts(), []string{"どんなことでも気", "軽に相談してくださいね。"}; !slices.Equal(got, want) {
		t.Fatalf("words = %v, want %v", got, want)
	}
	if got, want := host.joined(), "どんなことでも気軽に相談してくださいね。"; got != want {
		t.Errorf("joined = %q, want %q", got, want)
	}
}

// Chinese is written without spaces between words, so the tokens are joined
// with none.
func TestRealtimeTTSChineseTimestampChunksReassembleWithoutSpaces(t *testing.T) {
	s, host := timedSynth("zh-CN")
	speakChunks(t, s, "你好，我是 ", "你的智能助手。 ")

	if got, want := host.wordTexts(), []string{"你好，我是", "你的智能助手。"}; !slices.Equal(got, want) {
		t.Fatalf("words = %v, want %v", got, want)
	}
	if got, want := host.joined(), "你好，我是你的智能助手。"; got != want {
		t.Errorf("joined = %q, want %q", got, want)
	}
}

// English words are joined with spaces.
func TestRealtimeTTSEnglishTimestampChunksReassembleWithSpaces(t *testing.T) {
	s, host := timedSynth(language.English)
	speakChunks(t, s, "Hello world. ")

	if got, want := host.joined(), "Hello world."; got != want {
		t.Errorf("joined = %q, want %q", got, want)
	}
}

func TestRealtimeTTSTimestampSpacingLanguages(t *testing.T) {
	for lang, want := range map[string]bool{"ja": true, "zh-CN": true, "yue": true, "en": false, "": false} {
		if got := wordTimestampsIncludeInterFrameSpaces(lang); got != want {
			t.Errorf("%q: includes inter-frame spaces = %v, want %v", lang, got, want)
		}
	}
}

func TestSelectRealtimeAlignmentDefaultPrefersAlignment(t *testing.T) {
	plain, normalized := alignmentChunk("Hi"), alignmentChunk("hi")
	m := &realtimeTTSMessage{Alignment: plain, NormalizedAlignment: normalized}
	if got := selectRealtimeAlignment(m, false); got != plain {
		t.Errorf("selected %v, want the alignment as written", got.Chars)
	}
}

func TestSelectRealtimeAlignmentDictionaryModePrefersNormalized(t *testing.T) {
	plain, normalized := alignmentChunk("Hi"), alignmentChunk("hi")
	m := &realtimeTTSMessage{Alignment: plain, NormalizedAlignment: normalized}
	if got := selectRealtimeAlignment(m, true); got != normalized {
		t.Errorf("selected %v, want the normalized alignment", got.Chars)
	}
}

// Either alignment may be missing (absent from the message or null, which read
// the same), and the other is used then.
func TestSelectRealtimeAlignmentFallsBackWhenPreferredMissing(t *testing.T) {
	only := alignmentChunk("Hi")
	if got := selectRealtimeAlignment(&realtimeTTSMessage{NormalizedAlignment: only}, false); got != only {
		t.Error("with no alignment, the normalized one should be used")
	}
	if got := selectRealtimeAlignment(&realtimeTTSMessage{Alignment: only}, true); got != only {
		t.Error("with no normalized alignment, the plain one should be used")
	}
}

func TestSelectRealtimeAlignmentReturnsNoneWhenBothMissing(t *testing.T) {
	for _, preferNormalized := range []bool{false, true} {
		if got := selectRealtimeAlignment(&realtimeTTSMessage{}, preferNormalized); got != nil {
			t.Errorf("prefer normalized %v: selected %v, want none", preferNormalized, got.Chars)
		}
	}
}

// Alignment for concurrent contexts arrives interleaved on one connection; a
// word split across one context's chunks must not pick up another context's
// characters or clock.
func TestRealtimeTTSInterleavedContextsAlignWordsIndependently(t *testing.T) {
	s, host := timedSynth("")
	host.openContext("ctx-2")

	for _, m := range []struct{ context, text string }{
		{"ctx-1", "Hello wor"},
		{"ctx-2", "Good mor"},
		{"ctx-1", "ld "},
		{"ctx-2", "ning "},
	} {
		deliver(t, s, &realtimeTTSMessage{ContextID: m.context, Alignment: alignmentChunk(m.text)})
	}

	// "world" starts at "wor" in ctx-1's first chunk; "morning" at "mor" in
	// ctx-2's.
	want := []contextWord{
		{"ctx-1", "Hello", 0},
		{"ctx-2", "Good", 0},
		{"ctx-1", "world", 0.6},
		{"ctx-2", "morning", 0.5},
	}
	got := host.reported()
	if len(got) != len(want) {
		t.Fatalf("words = %v, want %v", got, want)
	}
	for i := range want {
		if got[i].context != want[i].context || got[i].word != want[i].word ||
			math.Abs(got[i].offset-want[i].offset) > 1e-9 {
			t.Fatalf("words = %v, want %v", got, want)
		}
	}
}

// Trailing punctuation can arrive in a message of its own, which completes no
// word; its span must still move the clock on.
func TestRealtimeTTSWordlessAlignmentStillAdvancesTheClock(t *testing.T) {
	s, host := timedSynth("")
	for _, c := range []string{"Hi there", "!", " Bye "} {
		deliver(t, s, &realtimeTTSMessage{ContextID: "ctx-1", Alignment: alignmentChunk(c)})
	}

	got := host.reported()
	if words := host.wordTexts(); !slices.Equal(words, []string{"Hi", "there!", "Bye"}) {
		t.Fatalf("words = %v, want Hi, there!, Bye", words)
	}
	// "Bye" starts after the 8 characters of the first chunk, the "!", and its
	// space.
	if last := got[len(got)-1].offset; math.Abs(last-1.0) > 1e-9 {
		t.Errorf("Bye starts at %v, want 1.0", last)
	}
}

// Alignment the server sends after an interruption must not bring back the
// context's word clock, which nothing would remove again.
func TestRealtimeTTSInterruptedContextLateAlignmentIsIgnored(t *testing.T) {
	s, host := timedSynth("")
	deliver(t, s, &realtimeTTSMessage{ContextID: "ctx-1", Alignment: alignmentChunk("Hello ")})
	if !hasWordClock(s, "ctx-1") {
		t.Fatal("the context's alignment started no word clock")
	}

	// The bot is interrupted: the audio context goes away and its clock is
	// dropped.
	host.RemoveAudioContext("ctx-1")
	s.OnAudioContextInterrupted(context.Background(), "ctx-1")

	for _, m := range []*realtimeTTSMessage{
		{ContextID: "ctx-1", Alignment: alignmentChunk("there wor")},
		{ContextID: "ctx-1", Alignment: alignmentChunk("ld ")},
		{ContextID: "ctx-1", IsFinal: true},
	} {
		deliver(t, s, m)
	}

	if hasWordClock(s, "ctx-1") {
		t.Error("late alignment re-created the word clock")
	}
	if got := host.wordTexts(); !slices.Equal(got, []string{"Hello"}) {
		t.Errorf("words = %v, want only Hello", got)
	}
}

func hasWordClock(s *realtimeSynthesizer, id string) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	_, ok := s.wordClocks[id]
	return ok
}

// The keepalive must not name a context before that context's opening message,
// which carries the voice settings, has gone out. Named earlier, the keepalive
// would be the context's first message, and ElevenLabs would reject the opening
// message that follows.
func TestRealtimeTTSKeepaliveDoesNotStampContextBeforeInit(t *testing.T) {
	server := newRealtimeServer(t, nil)
	s, _, svc := connectedSynth(t, server, RealtimeTTSConfig{})
	// The turn has started, but its opening message has not gone out yet.
	svc.active = "ctx-1"

	if err := s.SendKeepalive(context.Background()); err != nil {
		t.Fatal(err)
	}
	if got := server.next(t); !maps.Equal(got, map[string]any{keyText: ""}) {
		t.Errorf("keepalive = %v, want empty text naming no context", got)
	}
}

// Once the context's opening message has gone out, the keepalive names it.
func TestRealtimeTTSKeepaliveStampsContextAfterInit(t *testing.T) {
	server := newRealtimeServer(t, nil)
	s, _, svc := connectedSynth(t, server, RealtimeTTSConfig{})
	svc.active = "ctx-1"
	ctx := context.Background()
	if err := s.sendContextInit(ctx, "ctx-1"); err != nil {
		t.Fatal(err)
	}
	server.next(t)

	if err := s.SendKeepalive(ctx); err != nil {
		t.Fatal(err)
	}
	if got := server.next(t); !maps.Equal(got, map[string]any{keyText: "", keyContextID: "ctx-1"}) {
		t.Errorf("keepalive = %v, want empty text on ctx-1", got)
	}
}

// With no active context, the keepalive is empty text naming none.
func TestRealtimeTTSKeepaliveWithoutActiveContextSendsEmpty(t *testing.T) {
	server := newRealtimeServer(t, nil)
	s, _, _ := connectedSynth(t, server, RealtimeTTSConfig{})

	if err := s.SendKeepalive(context.Background()); err != nil {
		t.Fatal(err)
	}
	if got := server.next(t); !maps.Equal(got, map[string]any{keyText: ""}) {
		t.Errorf("keepalive = %v, want empty text naming no context", got)
	}
}

// The server closing the connection first is the ordinary end of a session,
// and disconnecting then is not an error: one reported here could make a
// service switcher move off a service that was only shutting down.
func TestRealtimeTTSDisconnectDoesNotPushErrorWhenServerClosedFirst(t *testing.T) {
	server := newRealtimeServer(t, nil)
	_, s := newRealtimeTTS(RealtimeTTSConfig{APIKey: "k", Host: server.host})
	host := newFakeHost()
	s.host, s.svc = host, &fakeSvc{}
	ctx := context.Background()
	if err := s.ConnectWebsocket(ctx); err != nil || !s.Connected() {
		t.Fatalf("connect: %v, connected %v", err, s.Connected())
	}

	server.drop()
	waitUntil(t, "the closed connection still reads as open", func() bool { return !s.Connected() })

	if err := s.DisconnectWebsocket(ctx); err != nil {
		t.Fatal(err)
	}
	if errs := host.pushedErrors(); len(errs) != 0 {
		t.Errorf("errors pushed = %v, want none", errs)
	}
	s.mu.Lock()
	conn := s.conn
	s.mu.Unlock()
	if conn != nil {
		t.Error("the connection is still held after disconnecting")
	}
}

// Disconnecting asks ElevenLabs to close the connection before closing it.
func TestRealtimeTTSDisconnectAsksTheServerToClose(t *testing.T) {
	server := newRealtimeServer(t, nil)
	s, _, _ := connectedSynth(t, server, RealtimeTTSConfig{})

	s.Disconnect(context.Background())
	if got := server.next(t); !isCloseSocket(got) {
		t.Errorf("sent %v on disconnect, want close_socket", got)
	}
	if s.Connected() {
		t.Error("still connected after disconnecting")
	}
}

// The text normalization is a query parameter of the connection, so changing
// it reconnects.
func TestRealtimeTTSTextNormalizationChangeReconnects(t *testing.T) {
	server := newRealtimeServer(t, nil)
	s, _, _ := connectedSynth(t, server, RealtimeTTSConfig{})

	updateSettings(t, s, &RealtimeTTSSettings{ApplyTextNormalization: settings.Set("on")})

	server.waitFor(t, isCloseSocket)
	if n := server.connections(); n != 2 {
		t.Fatalf("the server saw %d connections, want a second one after the change", n)
	}
	if got := server.query(1).Get("apply_text_normalization"); got != "on" {
		t.Errorf("apply_text_normalization = %q on the new connection, want on", got)
	}
}

// The voice and the language are query parameters of the connection too.
func TestRealtimeTTSVoiceChangeReconnects(t *testing.T) {
	server := newRealtimeServer(t, nil)
	s, _, _ := connectedSynth(t, server, RealtimeTTSConfig{})

	delta := &RealtimeTTSSettings{}
	delta.Voice = settings.Set("other-voice")
	updateSettings(t, s, delta)

	if n := server.connections(); n != 2 {
		t.Fatalf("the server saw %d connections, want a second one after the change", n)
	}
	if !strings.Contains(s.endpoint(), "/other-voice/") {
		t.Errorf("endpoint %s does not name the new voice", s.endpoint())
	}
}

// A voice control is sent when a context opens, so a change closes the open
// contexts rather than the connection, and the next context opens with it.
func TestRealtimeTTSVoiceSettingsChangeClosesOpenContexts(t *testing.T) {
	server := newRealtimeServer(t, nil)
	s, _, svc := connectedSynth(t, server, RealtimeTTSConfig{})
	svc.ids = []string{"c1", "c2"}

	updateSettings(t, s, &RealtimeTTSSettings{Stability: settings.Set(0.3)})

	for _, want := range svc.ids {
		if got := server.next(t); !isCloseContext(got) || got[keyContextID] != want {
			t.Errorf("sent %v, want %s closed", got, want)
		}
	}
	if n := server.connections(); n != 1 {
		t.Errorf("the server saw %d connections, want the change applied without reconnecting", n)
	}
	if err := s.sendContextInit(context.Background(), "c3"); err != nil {
		t.Fatal(err)
	}
	vs, _ := server.next(t)["voice_settings"].(map[string]any)
	if vs["stability"] != 0.3 {
		t.Errorf("the next context opened with voice settings %v, want stability 0.3", vs)
	}
	if len(svc.warned) != 0 {
		t.Errorf("warned about %v, want nothing", svc.warned)
	}
}

// A setting neither a reconnect nor a new context applies is reported as not
// taking effect.
func TestRealtimeTTSUnhandledSettingIsWarned(t *testing.T) {
	s := newRealtimeSynthesizer(RealtimeTTSConfig{APIKey: "k"})
	svc := &fakeSvc{}
	s.svc = svc
	if err := s.UpdateSettings(context.Background(), settings.Changed{"other": nil}); err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(svc.warned, []string{"other"}) {
		t.Errorf("warned about %v, want other", svc.warned)
	}
}

// updateSettings applies delta the way the base does, then hands the change to
// the synthesizer.
func updateSettings(t *testing.T, s *realtimeSynthesizer, delta *RealtimeTTSSettings) {
	t.Helper()
	changed, err := settings.Apply(s.live, delta)
	if err != nil {
		t.Fatal(err)
	}
	if err := s.UpdateSettings(context.Background(), changed); err != nil {
		t.Fatal(err)
	}
}

// A connection the server drops is restored by the receive loop, and the next
// sentence goes out on the new one.
func TestRealtimeTTSReconnectsWhenTheServerDrops(t *testing.T) {
	server := newRealtimeServer(t, onClose(audioMessage([]byte{4}), map[string]any{"isFinal": true}))
	s, _, _ := connectedSynth(t, server, RealtimeTTSConfig{Websocket: wsservice.Options{
		ReconnectBackoffMinWait: time.Millisecond,
		ReconnectBackoffMaxWait: time.Millisecond,
	}})

	server.drop()
	waitUntil(t, "the service never reconnected", func() bool {
		return server.connections() == 2 && s.Connected()
	})

	if got := speakOne(t, s, newHostFor(s), "Encore."); !bytes.Equal(got, []byte{4}) {
		t.Errorf("delivered %v after reconnecting, want [4]", got)
	}
}

// realtimeRun is a running pipeline around the service.
type realtimeRun struct {
	t    *testing.T
	task *pipeline.Worker
	done chan error

	mu   sync.Mutex
	down []frames.Frame
}

func runRealtime(t *testing.T, b *tts.Base) *realtimeRun {
	t.Helper()
	r := &realtimeRun{t: t, done: make(chan error, 1)}
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

// waitFor waits until a frame matching match has reached the end of the
// pipeline.
func (r *realtimeRun) waitFor(match func(frames.Frame) bool) {
	r.t.Helper()
	waitUntil(r.t, "the frame waited for never came", func() bool {
		r.mu.Lock()
		defer r.mu.Unlock()
		return slices.ContainsFunc(r.down, match)
	})
}

// stop ends the pipeline gracefully.
func (r *realtimeRun) stop() {
	r.t.Helper()
	r.task.StopWhenDone()
	select {
	case <-r.done:
	case <-time.After(5 * time.Second):
		r.t.Fatal("the pipeline did not finish")
	}
}

// The service connects when the pipeline starts, speaks every sentence over
// that one connection, and asks the server to close it when the pipeline ends.
func TestRealtimeTTSHoldsOneConnectionForThePipeline(t *testing.T) {
	server := newRealtimeServer(t, onText(audioMessage(make([]byte, 320)), map[string]any{"isFinal": true}))
	b := NewRealtimeTTS(RealtimeTTSConfig{APIKey: "k", Host: server.host, SampleRate: 16000})
	r := runRealtime(t, b)
	if n := server.connections(); n != 1 {
		t.Fatalf("the server saw %d connections when the pipeline started, want 1", n)
	}

	for _, sentence := range []string{"Hello there.", "How are you?"} {
		r.task.QueueFrame(frames.NewTTSSpeakFrame(sentence))
	}
	waitUntil(t, "both sentences never played", func() bool {
		r.mu.Lock()
		defer r.mu.Unlock()
		stopped := 0
		for _, f := range r.down {
			if _, ok := f.(*frames.TTSStoppedFrame); ok {
				stopped++
			}
		}
		return stopped == 2
	})
	r.stop()

	server.waitFor(t, isCloseSocket)
	if n := server.connections(); n != 1 {
		t.Errorf("the server saw %d connections, want one for the whole pipeline", n)
	}
}

// A connection the server refuses is reported as a connection error.
func TestRealtimeTTSReportsAFailedConnection(t *testing.T) {
	server := newRealtimeServer(t, nil)
	server.refuse = http.StatusUnauthorized
	b := NewRealtimeTTS(RealtimeTTSConfig{APIKey: "k", Host: server.host})
	var mu sync.Mutex
	var reported []string
	events.On(b.Events(), service.EventConnectionError, func(_ context.Context, msg string) {
		mu.Lock()
		reported = append(reported, msg)
		mu.Unlock()
	})

	r := runRealtime(t, b)
	defer r.stop()
	waitUntil(t, "the refused connection was never reported", func() bool {
		mu.Lock()
		defer mu.Unlock()
		return len(reported) > 0
	})
}
