package heygen

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"math"
	"net/http"
	"net/http/httptest"
	"net/url"
	"reflect"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/errclass"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/internal/providertest"
	"github.com/gojargo/jargo/language"
	"github.com/gojargo/jargo/service/settings"
	"github.com/gojargo/jargo/service/tts"
	"github.com/gojargo/jargo/service/wsservice"
	"github.com/gojargo/jargo/service/wsutil"
)

// TestTTSConfigValidate pins which fields the service requires.
func TestTTSConfigValidate(t *testing.T) {
	providertest.Configs(t, []providertest.ConfigCase{
		{Name: "missing API key", Cfg: TTSConfig{}, Valid: false},
		{Name: "API key only", Cfg: TTSConfig{APIKey: "k"}, Valid: true},
	})
}

// TestNewTTS checks the constructor returns a service under the label that
// identifies it in logs, metrics and traces.
func TestNewTTS(t *testing.T) {
	providertest.Service(t, "HeyGenTTS", NewTTS(TTSConfig{APIKey: "k"}))
}

// alignmentOf times each character of text 50 ms after the one before.
func alignmentOf(text string) map[string]any {
	var chars []string
	var starts []float64
	for i, r := range []rune(text) {
		chars = append(chars, string(r))
		starts = append(starts, float64(i*50))
	}
	return map[string]any{"chars": chars, "charStartTimesMs": starts}
}

// audioMessage is HeyGen's audio message for a context, with the alignment of
// text, or none when text is empty.
func audioMessage(contextID string, pcm []byte, text string) map[string]any {
	var a any
	if text != "" {
		a = alignmentOf(text)
	}
	return map[string]any{
		"audio":     base64.StdEncoding.EncodeToString(pcm),
		"alignment": a,
		"contextId": contextID,
		"isFinal":   nil,
	}
}

// refusal is how a handshake is refused: its status, its Retry-After and its
// body.
type refusal struct {
	status     int
	retryAfter string
	body       string
}

// fakeHeyGen is a fake multi-stream-input socket: audio for each text, and the
// final message once a context is closed.
type fakeHeyGen struct {
	// refusedSockets is how many sockets, in order, close with closeCode at
	// their first text instead of speaking.
	refusedSockets int
	closeCode      websocket.StatusCode
	// finalDelay is how long each final message is held back, as real
	// synthesis takes time.
	finalDelay time.Duration
	// finalAfterText says a context ends itself after its first text, as
	// HeyGen ends one left silent mid-turn.
	finalAfterText bool
	// trimAlignment says the alignment leaves out trailing spaces.
	trimAlignment bool
	// refuse decides, for the n-th handshake from 1, whether it is refused.
	refuse func(n int) (refusal, bool)
	// script, when set, answers the socket instead.
	script func(ctx context.Context, send func(map[string]any), msg map[string]any)

	audio []byte
	url   string

	mu         sync.Mutex
	handshakes []time.Time
	headers    []http.Header
	queries    []url.Values
	paths      []string
	sockets    [][]map[string]any
	peakOpen   int
	stray      []map[string]any
	open       map[string]bool
	accepting  map[string]bool
}

func newFakeHeyGen(t *testing.T, configure func(*fakeHeyGen)) *fakeHeyGen {
	t.Helper()
	f := &fakeHeyGen{
		closeCode: websocket.StatusCode(1013),
		audio:     bytes.Repeat([]byte{0, 1}, 2400),
		open:      map[string]bool{},
		accepting: map[string]bool{},
	}
	if configure != nil {
		configure(f)
	}
	srv := httptest.NewServer(http.HandlerFunc(f.serve))
	t.Cleanup(srv.Close)
	f.url = "ws" + strings.TrimPrefix(srv.URL, "http")
	return f
}

func (f *fakeHeyGen) serve(w http.ResponseWriter, r *http.Request) {
	f.mu.Lock()
	f.handshakes = append(f.handshakes, time.Now())
	n := len(f.handshakes)
	f.mu.Unlock()
	if f.refuse != nil {
		if refused, ok := f.refuse(n); ok {
			if refused.retryAfter != "" {
				w.Header().Set("Retry-After", refused.retryAfter)
			}
			w.WriteHeader(refused.status)
			_, _ = w.Write([]byte(refused.body))
			return
		}
	}
	c, err := websocket.Accept(w, r, nil)
	if err != nil {
		return
	}
	c.SetReadLimit(-1)
	defer func() { _ = c.CloseNow() }()

	f.mu.Lock()
	f.headers = append(f.headers, r.Header.Clone())
	f.queries = append(f.queries, r.URL.Query())
	f.paths = append(f.paths, r.URL.Path)
	refusedSocket := len(f.sockets) < f.refusedSockets
	f.sockets = append(f.sockets, nil)
	socket := len(f.sockets) - 1
	f.mu.Unlock()

	ctx := r.Context()
	var writeMu sync.Mutex
	send := func(msg map[string]any) {
		data, _ := json.Marshal(msg)
		writeMu.Lock()
		defer writeMu.Unlock()
		_ = c.Write(ctx, websocket.MessageText, data)
	}
	for {
		_, data, err := c.Read(ctx)
		if err != nil {
			return
		}
		var msg map[string]any
		if json.Unmarshal(data, &msg) != nil {
			return
		}
		contextID, _ := msg[keyContextID].(string)
		text, _ := msg[keyText].(string)
		closing, _ := msg[keyCloseContext].(bool)
		spoken := strings.TrimSpace(text) != ""

		f.mu.Lock()
		f.sockets[socket] = append(f.sockets[socket], msg)
		if spoken {
			f.open[contextID] = true
			f.accepting[contextID] = true
			f.peakOpen = max(f.peakOpen, len(f.open))
		} else if !f.accepting[contextID] && !closing {
			f.stray = append(f.stray, msg)
		}
		f.mu.Unlock()

		if f.script != nil {
			f.script(ctx, send, msg)
			continue
		}
		if spoken && refusedSocket {
			send(map[string]any{
				"message": "Free Turbo capacity is full.",
				"error": map[string]any{
					"code":       "rate_limit_exceeded",
					"message":    "Free Turbo capacity is full.",
					"request_id": "req-123",
				},
			})
			_ = c.Close(f.closeCode, "Free Turbo capacity is full")
			return
		}
		if spoken {
			said := text
			if f.trimAlignment {
				said = strings.TrimRight(text, " ")
			}
			send(audioMessage(contextID, f.audio, said))
			if f.finalAfterText {
				f.final(send, contextID)
			}
		}
		if closing {
			go f.final(send, contextID)
		}
	}
}

// final sends a context's final message, after the final delay.
func (f *fakeHeyGen) final(send func(map[string]any), contextID string) {
	f.mu.Lock()
	delete(f.accepting, contextID)
	f.mu.Unlock()
	time.Sleep(f.finalDelay)
	f.mu.Lock()
	delete(f.open, contextID)
	f.mu.Unlock()
	send(map[string]any{"isFinal": true, "contextId": contextID})
}

// messages is every message the service sent, over every socket.
func (f *fakeHeyGen) messages() []map[string]any {
	f.mu.Lock()
	defer f.mu.Unlock()
	var out []map[string]any
	for _, socket := range f.sockets {
		out = append(out, socket...)
	}
	return out
}

// spokenContexts is the context of every text message the service sent.
func (f *fakeHeyGen) spokenContexts() []string {
	var out []string
	for _, m := range f.messages() {
		if text, _ := m[keyText].(string); strings.TrimSpace(text) != "" {
			id, _ := m[keyContextID].(string)
			out = append(out, id)
		}
	}
	return out
}

// lastQuery is the query of the last socket opened.
func (f *fakeHeyGen) lastQuery(t *testing.T) url.Values {
	t.Helper()
	f.mu.Lock()
	defer f.mu.Unlock()
	if len(f.queries) == 0 {
		t.Fatal("no socket was opened")
	}
	return f.queries[len(f.queries)-1]
}

func (f *fakeHeyGen) socketCount() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return len(f.sockets)
}

func (f *fakeHeyGen) handshakeTimes() []time.Time {
	f.mu.Lock()
	defer f.mu.Unlock()
	return slices.Clone(f.handshakes)
}

// service builds the WebSocket service against the fake.
func service(f *fakeHeyGen, cfg TTSConfig) (*tts.Base, *synthesizer) {
	cfg.APIKey = "test-key"
	cfg.URL = f.url
	if cfg.SampleRate == 0 {
		cfg.SampleRate = 24000
	}
	return newTTS(cfg)
}

// fastReconnect reconnects with almost no backoff.
func fastReconnect() wsservice.Options {
	return wsservice.Options{
		ReconnectBackoffMinWait: 10 * time.Millisecond,
		ReconnectBackoffMaxWait: 10 * time.Millisecond,
	}
}

// connectionQuery is the query of the connection a service built with cfg
// opens.
func connectionQuery(t *testing.T, cfg TTSConfig) url.Values {
	t.Helper()
	f := newFakeHeyGen(t, nil)
	svc, _ := service(f, cfg)
	runTest(t, svc)
	return f.lastQuery(t)
}

// The service connects with the v3 parameters and turns alignment into word
// frames.
func TestTTSStreamsAudioAndWords(t *testing.T) {
	f := newFakeHeyGen(t, nil)
	svc, _ := service(f, TTSConfig{Language: language.EnglishUS})

	down, _ := runTest(t, svc, speak("Hello from HeyGen."), sleep(500*time.Millisecond))

	f.mu.Lock()
	header, path := f.headers[0], f.paths[0]
	f.mu.Unlock()
	if got := header.Get("X-Api-Key"); got != "test-key" {
		t.Fatalf("x-api-key = %q", got)
	}
	if path != ttsPath {
		t.Fatalf("path = %q", path)
	}
	want := url.Values{
		"voice_id":      {defaultVoiceID},
		"model":         {"heygen-voice-1-turbo"},
		"output_format": {"pcm_24000"},
		"auto_mode":     {"true"},
		"language":      {"en"},
	}
	if got := f.lastQuery(t); !reflect.DeepEqual(got, want) {
		t.Fatalf("query = %v\nwant %v", got, want)
	}

	sent := f.messages()
	if len(sent) == 0 {
		t.Fatal("nothing was sent")
	}
	id := sent[0][keyContextID]
	wantSent := []map[string]any{
		{keyText: "Hello from HeyGen. ", keyContextID: id},
		{keyFlush: true, keyContextID: id},
		{keyCloseContext: true, keyContextID: id},
	}
	if !reflect.DeepEqual(sent, wantSent) {
		t.Fatalf("sent %v\nwant %v", sent, wantSent)
	}

	if count[*frames.TTSStartedFrame](down) == 0 || count[*frames.TTSStoppedFrame](down) == 0 {
		t.Fatal("the synthesis was not bracketed by started and stopped frames")
	}
	var audio []byte
	for _, a := range audioFrames(down) {
		audio = append(audio, a.Audio...)
	}
	if !bytes.Equal(audio, f.audio) {
		t.Fatalf("played %d bytes, not the %d HeyGen sent", len(audio), len(f.audio))
	}
	if words := wordTexts(down); !slices.Equal(words, []string{"Hello", "from", "HeyGen."}) {
		t.Fatalf("words = %q", words)
	}
}

// The service asks HeyGen for its path.
func TestTTSEndpointPath(t *testing.T) {
	_, s := newTTS(TTSConfig{APIKey: "k"})
	u, err := url.Parse(s.endpoint())
	if err != nil {
		t.Fatal(err)
	}
	if u.Scheme+"://"+u.Host+u.Path != "wss://api.heygen.com/v3/models/audio/tts/multi-stream-input" {
		t.Fatalf("endpoint = %s", u)
	}
}

// A last word that no space ends still reaches the assistant when the context
// ends.
func TestTTSSpeaksALastWordWithoutATrailingSpace(t *testing.T) {
	f := newFakeHeyGen(t, func(f *fakeHeyGen) { f.trimAlignment = true })
	svc, _ := service(f, TTSConfig{})

	down, _ := runTest(t, svc, speak("Hello there."), sleep(500*time.Millisecond))

	if words := wordTexts(down); !slices.Equal(words, []string{"Hello", "there."}) {
		t.Fatalf("words = %q", words)
	}
}

// HeyGen's error message becomes a recoverable error and the next turn still
// speaks.
func TestTTSRefusalLosesTheTurnNotTheService(t *testing.T) {
	for _, code := range []websocket.StatusCode{1008, 1013} {
		t.Run(code.String(), func(t *testing.T) {
			f := newFakeHeyGen(t, func(f *fakeHeyGen) { f.refusedSockets, f.closeCode = 1, code })
			svc, _ := service(f, TTSConfig{Model: "heygen-voice-1-turbo-free", Websocket: fastReconnect()})

			down, up := runTest(t, svc,
				speak("Lost."), sleep(500*time.Millisecond),
				speak("Spoken."), sleep(500*time.Millisecond))

			if got := f.lastQuery(t).Get("model"); got != "heygen-voice-1-turbo-free" {
				t.Fatalf("model = %q", got)
			}
			errors := errorFrames(up)
			if len(errors) == 0 || !strings.Contains(errors[0].Error, "rate_limit_exceeded") ||
				!strings.Contains(errors[0].Error, "req-123") || errors[0].Category != errclass.RateLimit {
				t.Fatalf("errors = %v", errors)
			}
			if !svc.Usable() {
				t.Fatal("the service was written off")
			}
			if n := f.socketCount(); n != 2 {
				t.Fatalf("opened %d sockets, want 2", n)
			}
			if words := wordTexts(down); !slices.Equal(words, []string{"Spoken."}) {
				t.Fatalf("words = %q", words)
			}
		})
	}
}

// A handshake refused with Retry-After is retried once, after that wait.
func TestTTSReconnectWaitsAsLongAsARefusedHandshakeAsks(t *testing.T) {
	f := newFakeHeyGen(t, func(f *fakeHeyGen) {
		f.refusedSockets, f.closeCode = 1, websocket.StatusPolicyViolation
		f.refuse = func(n int) (refusal, bool) {
			return refusal{status: http.StatusTooManyRequests, retryAfter: "1", body: envelope}, n == 2
		}
	})
	svc, _ := service(f, TTSConfig{Websocket: fastReconnect()})

	down, _ := runTest(t, svc,
		speak("Lost."), sleep(200*time.Millisecond),
		speak("Not connected."), sleep(1200*time.Millisecond),
		speak("Spoken."), sleep(500*time.Millisecond))

	if !svc.Usable() {
		t.Fatal("the service was written off")
	}
	handshakes := f.handshakeTimes()
	if len(handshakes) != 3 {
		t.Fatalf("made %d handshakes, want 3", len(handshakes))
	}
	if waited := handshakes[2].Sub(handshakes[1]); waited < time.Second {
		t.Fatalf("redialed %v after the refusal, want at least the second it asked for", waited)
	}
	if words := wordTexts(down); !slices.Equal(words, []string{"Spoken."}) {
		t.Fatalf("words = %q", words)
	}
}

// After a refused first connect, turns make no handshake until the wait ends. A
// turn refused during the wait reports that the socket is not connected, and
// its context still ends with a stop frame.
func TestTTSOnDemandConnectWaitsAsLongAsARefusedHandshakeAsks(t *testing.T) {
	f := newFakeHeyGen(t, func(f *fakeHeyGen) {
		f.refuse = func(n int) (refusal, bool) {
			return refusal{status: http.StatusTooManyRequests, retryAfter: "1", body: envelope}, n == 1
		}
	})
	svc, _ := service(f, TTSConfig{})

	down, up := runTest(t, svc,
		speak("Too soon."),
		// Past the refused turn's audio context timeout.
		sleep(3500*time.Millisecond),
		speak("Spoken."), sleep(500*time.Millisecond))

	handshakes := f.handshakeTimes()
	if len(handshakes) != 2 {
		t.Fatalf("made %d handshakes, want 2", len(handshakes))
	}
	if waited := handshakes[1].Sub(handshakes[0]); waited < time.Second {
		t.Fatalf("redialed %v after the refusal, want at least the second it asked for", waited)
	}
	if words := wordTexts(down); !slices.Equal(words, []string{"Spoken."}) {
		t.Fatalf("words = %q", words)
	}
	if !slices.ContainsFunc(errorFrames(up), func(e *frames.ErrorFrame) bool {
		return strings.Contains(e.Error, "not connected")
	}) {
		t.Fatalf("no error said the socket is not connected: %v", errorFrames(up))
	}
	started, stopped := count[*frames.TTSStartedFrame](down), count[*frames.TTSStoppedFrame](down)
	if started != 2 || stopped != 2 {
		t.Fatalf("%d started and %d stopped frames, want 2 of each", started, stopped)
	}
}

// Text the socket fails to send reports an error and ends its context with a
// stop frame.
func TestTTSUnsentTextEndsItsContext(t *testing.T) {
	f := newFakeHeyGen(t, nil)
	svc, s := service(f, TTSConfig{})
	// The service sends on a socket that has closed under it, while the receive
	// loop goes on reading the one it was reading.
	breakSocket := func() {
		dead, err := wsutil.Dial(context.Background(), f.url, nil, 0)
		if err != nil {
			t.Error(err)
			return
		}
		_ = dead.CloseNow()
		s.mu.Lock()
		s.conn = dead
		s.mu.Unlock()
	}

	down, up := runTest(t, svc, sleep(200*time.Millisecond), do(breakSocket),
		speak("Never sent."), sleep(500*time.Millisecond))

	if !slices.ContainsFunc(errorFrames(up), func(e *frames.ErrorFrame) bool {
		return strings.Contains(e.Error, "Unknown error occurred")
	}) {
		t.Fatalf("no error reported: %v", errorFrames(up))
	}
	started, stopped := count[*frames.TTSStartedFrame](down), count[*frames.TTSStoppedFrame](down)
	if started != 1 || stopped != 1 {
		t.Fatalf("%d started and %d stopped frames, want 1 of each", started, stopped)
	}
}

// scripted answers the first text of a socket with these audio messages, each
// given as its length in seconds and the text it aligns, and the closing of a
// context with its final message.
func scripted(parts ...scriptPart) func(context.Context, func(map[string]any), map[string]any) {
	return func(_ context.Context, send func(map[string]any), msg map[string]any) {
		contextID, _ := msg[keyContextID].(string)
		if text, _ := msg[keyText].(string); strings.TrimSpace(text) != "" {
			for _, p := range parts {
				pcm := bytes.Repeat([]byte{0, 1}, int(24000*p.seconds))
				send(audioMessage(contextID, pcm, p.text))
			}
		}
		if closing, _ := msg[keyCloseContext].(bool); closing {
			send(map[string]any{"isFinal": true, "contextId": contextID})
		}
	}
}

type scriptPart struct {
	seconds float64
	text    string
}

// Audio HeyGen could not time still moves later words, so a word is never
// early.
func TestTTSWordTimesCountAudioWithNoAlignment(t *testing.T) {
	f := newFakeHeyGen(t, func(f *fakeHeyGen) {
		f.script = scripted(scriptPart{0.1, "One "}, scriptPart{1.0, ""}, scriptPart{0.1, "two "})
	})
	svc, _ := service(f, TTSConfig{})

	down, _ := runTest(t, svc, speak("One two."), sleep(300*time.Millisecond))

	words := wordFrames(down)
	if len(words) != 2 {
		t.Fatalf("words = %q", wordTexts(down))
	}
	first, okFirst := words[0].PTS()
	second, okSecond := words[1].PTS()
	if !okFirst || !okSecond {
		t.Fatal("the words have no presentation time")
	}
	if gap := float64(second - first); math.Abs(gap-1.1e9)/1.1e9 > 0.01 {
		t.Fatalf("the second word is %v after the first, want 1.1s", time.Duration(second-first))
	}
}

// Chinese has no spaces, so words arrive as they are spoken, not at the
// context's end.
func TestTTSTimesChineseCharacterByCharacter(t *testing.T) {
	f := newFakeHeyGen(t, func(f *fakeHeyGen) { f.script = scripted(scriptPart{0.2, "你好。"}) })
	svc, _ := service(f, TTSConfig{Language: language.Chinese})

	down, _ := runTest(t, svc, speak("你好。"), sleep(300*time.Millisecond))

	words := wordFrames(down)
	if texts := wordTexts(down); !slices.Equal(texts, []string{"你", "好。"}) {
		t.Fatalf("words = %q", texts)
	}
	for _, w := range words {
		if !w.IncludesInterFrameSpaces {
			t.Fatalf("%q does not say it carries its own spacing", w.Text)
		}
	}
}

// Every setting is in the URL, so the next handshake carries the change.
func TestTTSSettingsChangeReconnectsWithTheNewSettings(t *testing.T) {
	f := newFakeHeyGen(t, nil)
	svc, _ := service(f, TTSConfig{})
	delta := &TTSSettings{}
	delta.Voice = settings.Set("other-voice")

	runTest(t, svc, queue(frames.NewTTSUpdateSettingsFrame(delta)), sleep(200*time.Millisecond))

	if n := f.socketCount(); n != 2 {
		t.Fatalf("opened %d sockets, want 2", n)
	}
	if got := f.lastQuery(t).Get("voice_id"); got != "other-voice" {
		t.Fatalf("voice_id = %q", got)
	}
}

// After HeyGen ends a context mid-turn, the turn's next sentence opens a new
// one.
func TestTTSNeverReusesAContextAfterItsFinalMessage(t *testing.T) {
	f := newFakeHeyGen(t, func(f *fakeHeyGen) { f.finalAfterText = true })
	svc, _ := service(f, TTSConfig{})

	runTest(t, svc,
		responseStart(),
		llmText("First sentence. "),
		llmText("Next one."),
		sleep(200*time.Millisecond),
		llmText(" And the last."),
		sleep(200*time.Millisecond),
		responseEnd(),
		sleep(300*time.Millisecond))

	contexts := f.spokenContexts()
	if len(contexts) != 3 || len(slices.Compact(slices.Sorted(slices.Values(contexts)))) != 3 {
		t.Fatalf("spoke on contexts %v, want three different ones", contexts)
	}
}

// On barge-in the service closes the HeyGen context and keeps the socket open.
func TestTTSInterruptionClosesContextAndKeepsSocket(t *testing.T) {
	f := newFakeHeyGen(t, func(f *fakeHeyGen) { f.finalDelay = 5 * time.Second })
	svc, _ := service(f, TTSConfig{TextAggregation: frames.TextTypeToken})

	runTest(t, svc,
		responseStart(),
		llmText("Still speaking"),
		sleep(200*time.Millisecond),
		queue(frames.NewInterruptionFrame()),
		sleep(200*time.Millisecond))

	sent := f.messages()
	if len(sent) == 0 {
		t.Fatal("nothing was sent")
	}
	id := sent[0][keyContextID]
	if !slices.ContainsFunc(sent, func(m map[string]any) bool {
		return reflect.DeepEqual(m, map[string]any{keyCloseContext: true, keyContextID: id})
	}) {
		t.Fatalf("the interrupted context was never closed: %v", sent)
	}
	if n := f.socketCount(); n != 1 {
		t.Fatalf("opened %d sockets, want the one", n)
	}
}

// Keepalives name only an open context; HeyGen opens a context for any other.
func TestTTSKeepaliveNeverNamesAClosedContext(t *testing.T) {
	f := newFakeHeyGen(t, func(f *fakeHeyGen) { f.finalDelay = 300 * time.Millisecond })
	svc, s := service(f, TTSConfig{TextAggregation: frames.TextTypeToken})
	s.keepaliveInterval = 50 * time.Millisecond

	runTest(t, svc,
		sleep(200*time.Millisecond),
		responseStart(),
		llmText("Hello there. "),
		sleep(200*time.Millisecond),
		responseEnd(),
		sleep(500*time.Millisecond))

	contexts := f.spokenContexts()
	if len(contexts) == 0 {
		t.Fatal("nothing was spoken")
	}
	sent := f.messages()
	if !slices.ContainsFunc(sent, func(m map[string]any) bool {
		return reflect.DeepEqual(m, map[string]any{keyText: "", keyContextID: contexts[0]})
	}) {
		t.Fatalf("no keepalive named the open context: %v", sent)
	}
	if !slices.ContainsFunc(sent, func(m map[string]any) bool {
		closing, _ := m[keyCloseContext].(bool)
		return closing && m[keyContextID] != contexts[0]
	}) {
		t.Fatalf("no keepalive closed a context the service never opens: %v", sent)
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	if len(f.stray) != 0 {
		t.Fatalf("sent messages naming contexts that were not open: %v", f.stray)
	}
}

// A turn that produced no speech, such as a tool call alone, opens no HeyGen
// context.
func TestTTSTurnWithoutTextSendsNothing(t *testing.T) {
	f := newFakeHeyGen(t, nil)
	svc, _ := service(f, TTSConfig{})

	runTest(t, svc, responseStart(), responseEnd(), sleep(200*time.Millisecond))

	if sent := f.messages(); len(sent) != 0 {
		t.Fatalf("sent %v, want nothing", sent)
	}
}

// Flushing a context the service never opened would open it at HeyGen.
func TestTTSFlushNamesOnlyAnOpenContext(t *testing.T) {
	f := newFakeHeyGen(t, nil)
	_, s := service(f, TTSConfig{})
	s.Connect(context.Background())
	t.Cleanup(func() { s.Disconnect(context.Background()) })
	if !s.Connected() {
		t.Fatal("the service did not connect")
	}

	s.FlushAudio(context.Background(), "never-opened")
	time.Sleep(100 * time.Millisecond)

	if sent := f.messages(); len(sent) != 0 {
		t.Fatalf("sent %v, want nothing", sent)
	}
}

// At a rate HeyGen doesn't offer, the socket asks for 24 kHz, warns once and
// resamples.
func TestTTSUnsupportedSampleRateIsResampled(t *testing.T) {
	f := newFakeHeyGen(t, nil)
	logs := captureWarnings(t)
	svc, _ := service(f, TTSConfig{SampleRate: 11025})

	down, _ := runTest(t, svc, speak("Hello."), sleep(500*time.Millisecond))

	if got := f.lastQuery(t).Get("output_format"); got != "pcm_24000" {
		t.Fatalf("output_format = %q", got)
	}
	if n := strings.Count(logs(), "11025 Hz"); n != 1 {
		t.Fatalf("warned %d times about 11025 Hz, want once:\n%s", n, logs())
	}
	audio := audioFrames(down)
	if len(audio) == 0 {
		t.Fatal("no audio")
	}
	resampled := 0
	for _, a := range audio {
		if a.SampleRate != 11025 {
			t.Fatalf("a frame at %d Hz, want 11025", a.SampleRate)
		}
		resampled += len(a.Audio)
	}
	// 24 kHz to 11.025 kHz is about 0.46 times the bytes.
	if float64(resampled) >= float64(len(f.audio))*0.6 {
		t.Fatalf("played %d bytes of the %d HeyGen sent, want them resampled", resampled, len(f.audio))
	}
}

// Connection options and settings reach the URL only when given.
func TestTTSURLOptions(t *testing.T) {
	plain := connectionQuery(t, TTSConfig{})
	for _, name := range []string{
		"enable_logging", "enable_ssml_parsing", "end_user_id", "apply_text_normalization", "expressiveness_boost",
	} {
		if plain.Has(name) {
			t.Errorf("%s was sent without being given", name)
		}
	}

	query := connectionQuery(t, TTSConfig{
		EnableLogging:          new(false),
		EnableSSMLParsing:      new(true),
		EndUserID:              "user-42",
		ApplyTextNormalization: "off",
		ExpressivenessBoost:    new(0.5),
	})
	for name, want := range map[string]string{
		"enable_logging":           "false",
		"enable_ssml_parsing":      "true",
		"end_user_id":              "user-42",
		"apply_text_normalization": "off",
		"expressiveness_boost":     "0.5",
	} {
		if got := query.Get(name); got != want {
			t.Errorf("%s = %q, want %q", name, got, want)
		}
	}
}

// A language HeyGen doesn't speak is left out, so HeyGen detects the language.
func TestTTSUnsupportedLanguageIsNotSent(t *testing.T) {
	if query := connectionQuery(t, TTSConfig{Language: language.Norwegian}); query.Has("language") {
		t.Fatalf("sent language %q, want none", query.Get("language"))
	}
}

// A refused handshake reaches the error with HeyGen's reason and leaves the
// service unusable.
func TestTTSHandshakeRefusalReportsReason(t *testing.T) {
	f := newFakeHeyGen(t, func(f *fakeHeyGen) {
		f.refuse = func(int) (refusal, bool) { return refusal{status: http.StatusBadRequest, body: envelope}, true }
	})
	svc, _ := service(f, TTSConfig{})

	_, up := runTest(t, svc, speak("Hello."), sleep(300*time.Millisecond))

	errors := errorFrames(up)
	if len(errors) == 0 || !strings.Contains(errors[0].Error, "Voice not found") ||
		!strings.Contains(errors[0].Error, "req-7") || errors[0].Category != errclass.InvalidRequest {
		t.Fatalf("errors = %v", errors)
	}
	if svc.Usable() {
		t.Fatal("the service is still usable")
	}
}

// Seven queued sentences never hold more than HeyGen's 5 contexts open at once.
func TestTTSBackToBackSpeechStaysUnderContextLimit(t *testing.T) {
	f := newFakeHeyGen(t, func(f *fakeHeyGen) { f.finalDelay = 200 * time.Millisecond })
	svc, _ := service(f, TTSConfig{})

	steps := make([]step, 0, 8)
	for i := range 7 {
		steps = append(steps, speak("Sentence "+string(rune('0'+i))+"."))
	}
	runTest(t, svc, append(steps, sleep(2500*time.Millisecond))...)

	contexts := f.spokenContexts()
	if len(slices.Compact(slices.Sorted(slices.Values(contexts)))) != 7 {
		t.Fatalf("spoke on contexts %v, want seven different ones", contexts)
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.peakOpen > 5 {
		t.Fatalf("held %d contexts open at once, want at most 5", f.peakOpen)
	}
}
