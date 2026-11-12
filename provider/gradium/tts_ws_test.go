package gradium

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"maps"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
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

// Tests for the synthesis connection. Gradium takes, per request, a setup
// pinning the voice and the output format, then the text, then an
// end_of_stream that makes it flush; the audio comes back base64, with the
// time each word is spoken, until it says the request has ended. Every message
// names its request by the client_req_id it was set up with.

// ttsEndpoint is a fake Gradium TTS endpoint: it records what it was dialed
// with and every message it was sent, answers each text with the scripted
// messages and each end_of_stream with an end_of_stream of its own, all
// stamped with the request they belong to.
type ttsEndpoint struct {
	reply []map[string]any

	mu       sync.Mutex
	header   http.Header
	messages []map[string]any
	dials    int
}

// newTTSEndpoint starts the fake endpoint and returns it with its URL.
func newTTSEndpoint(t *testing.T, reply ...map[string]any) (*ttsEndpoint, string) {
	t.Helper()
	e := &ttsEndpoint{reply: reply}
	srv := httptest.NewServer(http.HandlerFunc(e.serve))
	t.Cleanup(srv.Close)
	return e, "ws" + strings.TrimPrefix(srv.URL, "http")
}

func (e *ttsEndpoint) serve(w http.ResponseWriter, r *http.Request) {
	// Counted before the handshake is answered, so a dial that has returned
	// is always counted.
	e.mu.Lock()
	e.dials++
	e.header = r.Header.Clone()
	e.mu.Unlock()
	c, err := websocket.Accept(w, r, nil)
	if err != nil {
		return
	}
	defer func() { _ = c.CloseNow() }()
	ctx := r.Context()
	write := func(m map[string]any, id any) bool {
		out := map[string]any{keyClientReqID: id}
		maps.Copy(out, m)
		b, _ := json.Marshal(out)
		return c.Write(ctx, websocket.MessageText, b) == nil
	}
	for {
		_, data, err := c.Read(ctx)
		if err != nil {
			return
		}
		msg := map[string]any{}
		if json.Unmarshal(data, &msg) != nil {
			return
		}
		e.mu.Lock()
		e.messages = append(e.messages, msg)
		e.mu.Unlock()
		id := msg[keyClientReqID]
		switch msg[msgType] {
		case msgSetup:
			write(map[string]any{msgType: msgReady}, id)
		case msgText:
			for _, m := range e.reply {
				if !write(m, id) {
					return
				}
			}
		case msgEndStream:
			write(map[string]any{msgType: msgEndStream}, id)
		}
	}
}

// sent is every message the endpoint was sent, in order.
func (e *ttsEndpoint) sent() []map[string]any {
	e.mu.Lock()
	defer e.mu.Unlock()
	return slices.Clone(e.messages)
}

// dialCount is how many connections were opened.
func (e *ttsEndpoint) dialCount() int {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.dials
}

// awaitSent waits for the endpoint to have been sent n messages.
func (e *ttsEndpoint) awaitSent(t *testing.T, n int) []map[string]any {
	t.Helper()
	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		if got := e.sent(); len(got) >= n {
			return got
		}
		time.Sleep(5 * time.Millisecond)
	}
	t.Fatalf("the endpoint was sent %v, want %d messages", e.sent(), n)
	return nil
}

// ttsAudio is one audio chunk as Gradium sends it.
func ttsAudio(pcm []byte) map[string]any {
	return map[string]any{msgType: msgAudio, "audio": base64.StdEncoding.EncodeToString(pcm)}
}

// ttsWord is one word's timing as Gradium sends it.
func ttsWord(word string, start float64) map[string]any {
	return map[string]any{msgType: msgText, msgText: word, "start_s": start}
}

// testTTS builds the service against endpoint.
func testTTS(endpoint string, opts ...func(*TTSConfig)) (*tts.Base, *ttsSynthesizer) {
	cfg := TTSConfig{APIKey: "test-key", URL: endpoint}
	for _, o := range opts {
		o(&cfg)
	}
	return newTTS(cfg)
}

// connectedTTS builds the service against endpoint and connects it, the way
// the pipeline starting does.
func connectedTTS(t *testing.T, endpoint string) *ttsSynthesizer {
	t.Helper()
	_, s := testTTS(endpoint)
	s.Connect(t.Context())
	t.Cleanup(func() { s.Disconnect(context.Background()) })
	if !s.Connected() {
		t.Fatal("the synthesizer did not connect")
	}
	return s
}

// runTTSPipeline runs the service in a pipeline, queueing each frame in order
// (a time.Duration waits that long instead), ends the pipeline, and returns the
// frames that left it downstream and upstream.
func runTTSPipeline(t *testing.T, svc *tts.Base, steps ...any) (down, up []frames.Frame) {
	t.Helper()
	var mu sync.Mutex
	task := pipeline.NewWorker(pipeline.New(svc), pipeline.WorkerConfig{
		ReachedDownstreamFilter: pipeline.AnyFrame,
		ReachedUpstreamFilter:   pipeline.AnyFrame,
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

// framesOf is the frames of type T, in order.
func framesOf[T frames.Frame](fs []frames.Frame) []T {
	var out []T
	for _, f := range fs {
		if v, ok := f.(T); ok {
			out = append(out, v)
		}
	}
	return out
}

// playedAudio is the PCM of every audio frame, joined.
func playedAudio(fs []frames.Frame) []byte {
	var pcm []byte
	for _, f := range framesOf[*frames.TTSAudioRawFrame](fs) {
		pcm = append(pcm, f.Audio...)
	}
	return pcm
}

// TestTTSStreamsTheAudio checks a turn is set up, sent and ended on one
// request, all named by the turn's context, and that its audio is played at
// Gradium's rate.
func TestTTSStreamsTheAudio(t *testing.T) {
	want := []byte{1, 2, 3, 4, 5, 6}
	endpoint, url := newTTSEndpoint(t, ttsAudio(want[:4]), ttsAudio(want[4:]))
	svc, _ := testTTS(url)

	down, up := runTTSPipeline(t, svc, frames.NewTTSSpeakFrame("hello there"), 300*time.Millisecond)

	if errs := framesOf[*frames.ErrorFrame](slices.Concat(down, up)); len(errs) > 0 {
		t.Errorf("errors pushed: %v", errs)
	}
	if got := playedAudio(down); string(got) != string(want) {
		t.Errorf("audio = %v, want the chunks joined up: %v", got, want)
	}
	for _, f := range framesOf[*frames.TTSAudioRawFrame](down) {
		if f.SampleRate != ttsSampleRate {
			t.Errorf("audio at %d Hz, want %d", f.SampleRate, ttsSampleRate)
		}
	}
	if len(framesOf[*frames.TTSStoppedFrame](down)) != 1 {
		t.Errorf("the turn did not end with one stopped frame")
	}

	endpoint.mu.Lock()
	header := endpoint.header
	endpoint.mu.Unlock()
	if header.Get("x-api-key") != "test-key" || header.Get("x-api-source") != "jargo" {
		t.Errorf("headers = %v, want the configured key and the integration named", header)
	}

	sent := endpoint.sent()
	if len(sent) != 3 {
		t.Fatalf("sent %v, want the setup, the text and the end of stream", sent)
	}
	setup := sent[0]
	if setup[msgType] != msgSetup || setup["output_format"] != encPCM || setup["voice_id"] != defaultVoiceID {
		t.Errorf("first message = %v, want the setup with the default voice and PCM", setup)
	}
	// The connection is reused for the next turn, so the provider must not
	// close it when this one ends.
	if setup["close_ws_on_eos"] != false {
		t.Errorf("close_ws_on_eos = %v, want false", setup["close_ws_on_eos"])
	}
	if _, ok := setup["json_config"]; ok {
		t.Error("json_config was sent though none was configured")
	}
	if sent[1][msgType] != msgText || sent[1][msgText] != "hello there" {
		t.Errorf("second message = %v, want the text", sent[1])
	}
	// Without this the provider waits for more text and never flushes.
	if sent[2][msgType] != msgEndStream {
		t.Errorf("third message = %v, want the end of stream", sent[2])
	}
	id, _ := sent[0][keyClientReqID].(string)
	if id == "" {
		t.Fatal("the setup names no request")
	}
	for i, m := range sent {
		if m[keyClientReqID] != id {
			t.Errorf("message %d names request %v, want the turn's %q", i, m[keyClientReqID], id)
		}
	}
}

// TestTTSSendsTheJSONConfig checks the optional model configuration is
// forwarded on the setup when it is given.
func TestTTSSendsTheJSONConfig(t *testing.T) {
	endpoint, url := newTTSEndpoint(t)
	_, s := testTTS(url, func(c *TTSConfig) { c.JSONConfig = `{"speed":1.2}` })
	s.Connect(t.Context())
	t.Cleanup(func() { s.Disconnect(context.Background()) })

	if err := s.RunTTS(t.Context(), "hello", "c1", nil); err != nil {
		t.Fatalf("RunTTS: %v", err)
	}
	if got := endpoint.awaitSent(t, 1)[0]["json_config"]; got != `{"speed":1.2}` {
		t.Errorf("json_config = %v, want the configured string", got)
	}
}

// TestTTSSharesOneConnection checks the turns of a call are spoken on one
// connection, each as a request of its own.
func TestTTSSharesOneConnection(t *testing.T) {
	endpoint, url := newTTSEndpoint(t, ttsAudio([]byte{1, 2}))
	svc, _ := testTTS(url)

	down, _ := runTTSPipeline(t, svc,
		frames.NewTTSSpeakFrame("one."), 200*time.Millisecond,
		frames.NewTTSSpeakFrame("two."), 200*time.Millisecond,
	)

	if n := endpoint.dialCount(); n != 1 {
		t.Errorf("dialed %d connections, want the one held for the call", n)
	}
	var setups []any
	for _, m := range endpoint.sent() {
		if m[msgType] == msgSetup {
			setups = append(setups, m[keyClientReqID])
		}
	}
	if len(setups) != 2 || setups[0] == setups[1] {
		t.Errorf("set up requests %v, want one per turn", setups)
	}
	if got := playedAudio(down); len(got) != 4 {
		t.Errorf("played %v, want the audio of both turns", got)
	}
}

// TestTTSSetsUpARequestOnce checks a request is set up before its first text
// only, and again once Gradium has ended it.
func TestTTSSetsUpARequestOnce(t *testing.T) {
	endpoint, url := newTTSEndpoint(t)
	s := connectedTTS(t, url)

	for _, text := range []string{"one.", "two."} {
		if err := s.RunTTS(t.Context(), text, "c1", nil); err != nil {
			t.Fatalf("RunTTS: %v", err)
		}
	}
	got := endpoint.awaitSent(t, 3)
	var types []any
	for _, m := range got {
		types = append(types, m[msgType])
	}
	if !slices.Equal(types, []any{msgSetup, msgText, msgText}) {
		t.Errorf("sent %v, want one setup ahead of both texts", types)
	}

	// Gradium ends the request on its end_of_stream, so the next text on the
	// same context sets it up again.
	s.FlushAudio(t.Context(), "c1")
	endpoint.awaitSent(t, 4)
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		s.mu.Lock()
		setUp := s.setupContexts["c1"]
		s.mu.Unlock()
		if !setUp {
			break
		}
		time.Sleep(5 * time.Millisecond)
	}
	if err := s.RunTTS(t.Context(), "three.", "c1", nil); err != nil {
		t.Fatalf("RunTTS: %v", err)
	}
	if got := endpoint.awaitSent(t, 6); got[4][msgType] != msgSetup || got[5][msgText] != "three." {
		t.Errorf("sent %v after the end of stream, want the request set up again", got[4:])
	}
}

// TestTTSReportsWhenEachWordIsSpoken checks the word timings Gradium sends
// become the words spoken, so the conversation keeps only what was heard.
func TestTTSReportsWhenEachWordIsSpoken(t *testing.T) {
	_, url := newTTSEndpoint(t, ttsAudio([]byte{1, 2}), ttsWord("hello", 0), ttsWord("there", 0.4))
	svc, _ := testTTS(url)

	down, _ := runTTSPipeline(t, svc, frames.NewTTSSpeakFrame("hello there"), 300*time.Millisecond)

	var words []string
	for _, f := range framesOf[*frames.TTSTextFrame](down) {
		words = append(words, f.Text)
	}
	if !slices.Equal(words, []string{"hello", "there"}) {
		t.Errorf("words spoken = %q, want each word Gradium timed", words)
	}
}

// TestTTSReportsAnError checks an error message from Gradium stops the turn's
// speech at once and is pushed, not fatal, carrying what the provider said.
func TestTTSReportsAnError(t *testing.T) {
	_, url := newTTSEndpoint(t, map[string]any{msgType: msgError, "message": "unknown voice"})
	svc, _ := testTTS(url)

	down, up := runTTSPipeline(t, svc, frames.NewTTSSpeakFrame("hello"), 300*time.Millisecond)

	// The base also reports the turn ending with no audio.
	errs := framesOf[*frames.ErrorFrame](slices.Concat(down, up))
	if len(errs) == 0 || errs[0].Error != "Error: unknown voice" || errs[0].Fatal {
		t.Errorf("errors = %v, want the provider's message first, not fatal", errs)
	}
	if len(framesOf[*frames.TTSStoppedFrame](down)) == 0 {
		t.Error("the turn's speech was not stopped")
	}
}

// TestTTSSkipsWhatItDoesNotActOn checks a ready message, and a message for a
// request whose context is gone, are stepped over rather than ending the
// connection.
func TestTTSSkipsWhatItDoesNotActOn(t *testing.T) {
	endpoint, url := newTTSEndpoint(t, map[string]any{msgType: msgReady}, ttsAudio([]byte{9, 9}))
	svc, _ := testTTS(url)

	down, _ := runTTSPipeline(t, svc, frames.NewTTSSpeakFrame("hello"), 300*time.Millisecond)

	if got := playedAudio(down); len(got) != 2 {
		t.Errorf("audio = %v, want the one real chunk", got)
	}
	if n := endpoint.dialCount(); n != 1 {
		t.Errorf("dialed %d connections, want the first one kept", n)
	}
}

// TestTTSUnreadableAudioReplacesTheConnection checks audio that is not base64
// is not played as rubbish: the connection carrying it is dropped and
// replaced.
func TestTTSUnreadableAudioReplacesTheConnection(t *testing.T) {
	endpoint, url := newTTSEndpoint(t, map[string]any{msgType: msgAudio, "audio": "not base64!!"})
	b, s := testTTS(url, func(c *TTSConfig) { c.Websocket.ReconnectBackoffMinWait = time.Millisecond })
	s.SetAudioContextHost(openHost{b})
	s.Connect(t.Context())
	t.Cleanup(func() { s.Disconnect(context.Background()) })

	if err := s.RunTTS(t.Context(), "hello", "c1", nil); err != nil {
		t.Fatalf("RunTTS: %v", err)
	}
	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) && endpoint.dialCount() < 2 {
		time.Sleep(5 * time.Millisecond)
	}
	if n := endpoint.dialCount(); n != 2 {
		t.Errorf("dialed %d connections, want the one carrying unreadable audio replaced", n)
	}
}

// openHost is the service with every context taken as open, for a test that
// drives the synthesizer without a pipeline.
type openHost struct{ *tts.Base }

func (openHost) AudioContextAvailable(string) bool { return true }

// TestTTSReportsAnUnreachableEndpoint checks a dial that fails is reported as
// a connection error, and that a sentence finding no connection fails.
func TestTTSReportsAnUnreachableEndpoint(t *testing.T) {
	b, s := testTTS("ws://127.0.0.1:1")
	var mu sync.Mutex
	var reported []string
	events.On(b.Events(), service.EventConnectionError, func(_ context.Context, msg string) {
		mu.Lock()
		reported = append(reported, msg)
		mu.Unlock()
	})
	t.Cleanup(func() { s.Disconnect(context.Background()) })

	if err := s.RunTTS(t.Context(), "hello", "c1", nil); !errors.Is(err, errTTSNotConnected) {
		t.Errorf("RunTTS = %v, want it failing for want of a connection", err)
	}
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		mu.Lock()
		n := len(reported)
		mu.Unlock()
		if n > 0 {
			break
		}
		time.Sleep(5 * time.Millisecond)
	}
	mu.Lock()
	defer mu.Unlock()
	if len(reported) == 0 || !strings.HasPrefix(reported[0], "Unknown error occurred") {
		t.Errorf("connection errors = %q, want the failed dial", reported)
	}
	if s.ws.ReceiveTaskRunning() {
		t.Error("a receive loop runs with no connection to read")
	}
}

// TestTTSFailedSendReconnects checks a sentence that cannot be sent stops its
// speech, reports the failure, and has the connection dropped and dialed
// again.
func TestTTSFailedSendReconnects(t *testing.T) {
	endpoint, url := newTTSEndpoint(t)
	_, s := testTTS(url)
	t.Cleanup(func() { s.Disconnect(context.Background()) })
	// Dialed with no receive loop, so nothing notices the close before the
	// send: the send is what finds the connection dead.
	if err := s.ConnectWebsocket(t.Context()); err != nil {
		t.Fatalf("connect: %v", err)
	}
	s.mu.Lock()
	conn := s.conn
	s.mu.Unlock()
	_ = conn.CloseNow()

	var yielded []frames.Frame
	yield := func(f frames.Frame) error {
		yielded = append(yielded, f)
		return nil
	}
	if err := s.RunTTS(t.Context(), "hi", "c1", yield); err == nil {
		t.Fatal("RunTTS on a dead connection = nil, want the send error")
	}
	if len(yielded) != 1 || len(framesOf[*frames.TTSStoppedFrame](yielded)) != 1 {
		t.Errorf("yielded %v, want the speech stopped", yielded)
	}
	if !s.Connected() || !s.ws.ReceiveTaskRunning() || endpoint.dialCount() != 2 {
		t.Error("the connection was not dialed again after the failed send")
	}
	// The connection works again, and the request is set up afresh on it.
	if err := s.RunTTS(t.Context(), "again", "c1", yield); err != nil {
		t.Fatalf("RunTTS after the redial: %v", err)
	}
	if got := endpoint.awaitSent(t, 2); got[0][msgType] != msgSetup || got[1][msgText] != "again" {
		t.Errorf("sent %v, want the request set up and the next sentence", got)
	}
}

// TestTTSVoiceChangeReconnects checks a new voice reconnects and sets up the
// next request with it, while a setting Gradium is never sent leaves the
// connection alone.
func TestTTSVoiceChangeReconnects(t *testing.T) {
	endpoint, url := newTTSEndpoint(t)
	s := connectedTTS(t, url)

	changed, err := settings.Apply(s.live, &TTSSettings{Language: settings.Set("fr")})
	if err != nil {
		t.Fatalf("applying the settings: %v", err)
	}
	if err := s.UpdateSettings(t.Context(), changed); err != nil {
		t.Fatalf("UpdateSettings: %v", err)
	}
	if n := endpoint.dialCount(); n != 1 {
		t.Errorf("dialed %d connections after a language change, want the open one kept", n)
	}

	changed, err = settings.Apply(s.live, &TTSSettings{Voice: settings.Set("other-voice")})
	if err != nil {
		t.Fatalf("applying the settings: %v", err)
	}
	if err := s.UpdateSettings(t.Context(), changed); err != nil {
		t.Fatalf("UpdateSettings: %v", err)
	}
	if n := endpoint.dialCount(); n != 2 || !s.Connected() {
		t.Errorf("dialed %d connections after a voice change, want a second one", n)
	}
	if err := s.RunTTS(t.Context(), "hello", "c1", nil); err != nil {
		t.Fatalf("RunTTS: %v", err)
	}
	if got := endpoint.awaitSent(t, 1)[0]; got["voice_id"] != "other-voice" {
		t.Errorf("setup = %v, want the new voice", got)
	}
	if meta := s.Metadata(); meta.VoiceID != "other-voice" || meta.Model != defaultTTSModel {
		t.Errorf("Metadata() = %+v, want the new voice", meta)
	}
}

// TestTTSInterruptionSendsNothing checks an interruption tells Gradium
// nothing: the audio still coming for the request is dropped once its context
// is gone.
func TestTTSInterruptionSendsNothing(t *testing.T) {
	endpoint, url := newTTSEndpoint(t)
	s := connectedTTS(t, url)

	s.OnAudioContextInterrupted(t.Context(), "c1")

	time.Sleep(100 * time.Millisecond)
	if sent := endpoint.sent(); len(sent) != 0 {
		t.Errorf("sent %v on an interruption, want nothing", sent)
	}
}

// TestTTSSampleRate checks the fixed rate Gradium streams at, which is what the
// emitted frames are stamped with.
func TestTTSSampleRate(t *testing.T) {
	if got := (&ttsSynthesizer{}).SampleRate(); got != ttsSampleRate {
		t.Errorf("SampleRate() = %d, want %d", got, ttsSampleRate)
	}
}
