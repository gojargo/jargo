package mistral

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"
	"time"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/service/settings"
	"github.com/gojargo/jargo/service/stt"
)

// Tests for the realtime transcription session. Mistral takes the model on the
// query and the audio format in a session update, then base64 audio; the text
// arrives as deltas that accumulate into an interim, and the done event carries
// the finalized transcript.

// sttCapture is what the fake endpoint saw.
type sttCapture struct {
	query  url.Values
	header http.Header
	got    chan map[string]any
}

// sttServer starts a fake Mistral realtime endpoint that records what it is
// sent and replies with the scripted events.
func sttServer(t *testing.T, reply []map[string]any) (endpoint string, seen *sttCapture) {
	t.Helper()
	c := &sttCapture{got: make(chan map[string]any, 16)}

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		c.query, c.header = r.URL.Query(), r.Header.Clone()
		conn, err := websocket.Accept(w, r, nil)
		if err != nil {
			return
		}
		defer func() { _ = conn.Close(websocket.StatusNormalClosure, "") }()
		ctx := r.Context()

		// The session update arrives first, before any audio.
		_, data, err := conn.Read(ctx)
		if err != nil {
			return
		}
		m := map[string]any{}
		if err := json.Unmarshal(data, &m); err != nil {
			t.Errorf("decoding the session update: %v", err)
			return
		}
		c.got <- m

		for _, e := range reply {
			b, err := json.Marshal(e)
			if err != nil {
				t.Errorf("encoding a reply: %v", err)
				return
			}
			if conn.Write(ctx, websocket.MessageText, b) != nil {
				return
			}
		}

		for {
			_, data, err := conn.Read(ctx)
			if err != nil {
				return
			}
			m := map[string]any{}
			if json.Unmarshal(data, &m) == nil {
				select {
				case c.got <- m:
				default:
				}
			}
		}
	}))
	t.Cleanup(srv.Close)

	return "ws" + strings.TrimPrefix(srv.URL, "http"), c
}

// await returns the next message the endpoint was sent.
func (c *sttCapture) await(t *testing.T) map[string]any {
	t.Helper()
	select {
	case m := <-c.got:
		return m
	case <-time.After(2 * time.Second):
		t.Fatal("no message arrived at the endpoint")
		return nil
	}
}

// sttConn builds a connector pointed at endpoint with the defaults NewSTT fills
// in, so the session under test is the real one.
func sttConn(endpoint string, opts ...func(*STTConfig)) *sttConnector {
	cfg := STTConfig{APIKey: "test-key", URL: endpoint, Model: sttDefaultModel}
	for _, o := range opts {
		o(&cfg)
	}
	return newSTTConnector(cfg)
}

func TestSTTConnectSendsTheSessionUpdate(t *testing.T) {
	endpoint, seen := sttServer(t, nil)

	s, err := sttConn(endpoint).Connect(context.Background(), 16000)
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	defer func() { _ = s.Close() }()

	if seen.header.Get("Authorization") != "Bearer test-key" {
		t.Errorf("Authorization = %q, want the key as a bearer token", seen.header.Get("Authorization"))
	}
	if got := seen.query.Get("model"); got != sttDefaultModel {
		t.Errorf("model = %q, want %q", got, sttDefaultModel)
	}

	m := seen.await(t)
	if m[msgType] != sttMsgSessionUpdate {
		t.Fatalf("first message = %v, want the session update", m)
	}
	session, _ := m["session"].(map[string]any)
	format, _ := session["audio_format"].(map[string]any)
	if format["encoding"] != "pcm_s16le" {
		t.Errorf("encoding = %v, want pcm_s16le", format["encoding"])
	}
	if format["sample_rate"] != float64(16000) {
		t.Errorf("sample_rate = %v, want the rate the pipeline runs at", format["sample_rate"])
	}
	// Unset, so the server's own default stands.
	if _, ok := session["target_streaming_delay_ms"]; ok {
		t.Error("target_streaming_delay_ms was sent though none was configured")
	}
}

// TestSTTConnectSendsTheStreamingDelay checks the accuracy-for-latency trade is
// forwarded when it is configured.
func TestSTTConnectSendsTheStreamingDelay(t *testing.T) {
	endpoint, seen := sttServer(t, nil)

	s, err := sttConn(endpoint, func(c *STTConfig) { c.TargetStreamingDelayMs = 250 }).
		Connect(context.Background(), 16000)
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	defer func() { _ = s.Close() }()

	session, _ := seen.await(t)["session"].(map[string]any)
	if session["target_streaming_delay_ms"] != float64(250) {
		t.Errorf("target_streaming_delay_ms = %v, want 250", session["target_streaming_delay_ms"])
	}
}

// TestSTTSendAppendsBase64Audio checks the audio goes as base64 on a text
// message, which is what the realtime API reads.
func TestSTTSendAppendsBase64Audio(t *testing.T) {
	endpoint, seen := sttServer(t, nil)

	s, err := sttConn(endpoint).Connect(context.Background(), 16000)
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	defer func() { _ = s.Close() }()
	seen.await(t) // the session update

	pcm := []byte{1, 2, 3, 4}
	if err := s.Send(pcm); err != nil {
		t.Fatalf("Send: %v", err)
	}

	m := seen.await(t)
	if m[msgType] != sttMsgAudioAppend {
		t.Fatalf("message = %v, want an audio append", m)
	}
	if m["audio"] != base64.StdEncoding.EncodeToString(pcm) {
		t.Errorf("audio = %v, want the PCM base64 encoded", m["audio"])
	}
}

// TestSTTRecvAccumulatesDeltas checks each delta extends the interim, so the
// pipeline sees the sentence growing rather than one word at a time.
func TestSTTRecvAccumulatesDeltas(t *testing.T) {
	endpoint, _ := sttServer(t, []map[string]any{
		{msgType: sttEventTextDelta, "text": "hello"},
		{msgType: sttEventTextDelta, "text": " there"},
	})

	s, err := sttConn(endpoint).Connect(context.Background(), 16000)
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	defer func() { _ = s.Close() }()

	first, err := s.Recv()
	if err != nil {
		t.Fatalf("Recv: %v", err)
	}
	if first[0].Text != "hello" || first[0].Final {
		t.Errorf("result = %+v, want the interim hello", first[0])
	}

	second, err := s.Recv()
	if err != nil {
		t.Fatalf("Recv: %v", err)
	}
	if second[0].Text != "hello there" {
		t.Errorf("text = %q, want the deltas accumulated", second[0].Text)
	}
}

// TestSTTRecvFinalizesOnDone checks the done event carries the finalized text
// and ends the turn, and that the language reported earlier travels with it.
func TestSTTRecvFinalizesOnDone(t *testing.T) {
	endpoint, _ := sttServer(t, []map[string]any{
		{msgType: sttEventLanguage, "audio_language": "fr"},
		{msgType: sttEventTextDelta, "text": "bonjour"},
		{msgType: sttEventDone, "text": "bonjour tout le monde"},
	})

	s, err := sttConn(endpoint).Connect(context.Background(), 16000)
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	defer func() { _ = s.Close() }()

	if _, err := s.Recv(); err != nil { // the interim
		t.Fatalf("Recv: %v", err)
	}
	final, err := s.Recv()
	if err != nil {
		t.Fatalf("Recv: %v", err)
	}
	if final[0].Text != "bonjour tout le monde" {
		t.Errorf("text = %q, want what the done event carried", final[0].Text)
	}
	if !final[0].Final {
		t.Errorf("result = %+v, want it final", final[0])
	}
	// The done event closes the segment the flush asked about; it says nothing
	// about whether the user has finished speaking, which is what the flag means
	// to the turn strategies.
	if final[0].EndOfTurn {
		t.Errorf("result = %+v, want it not marked as ending the turn", final[0])
	}
	if final[0].Language != "fr" {
		t.Errorf("language = %q, want the one reported for the audio", final[0].Language)
	}
}

// TestSTTRecvClearsTheAccumulatorOnDone checks the next utterance starts empty
// rather than carrying the one before it.
func TestSTTRecvClearsTheAccumulatorOnDone(t *testing.T) {
	endpoint, _ := sttServer(t, []map[string]any{
		{msgType: sttEventTextDelta, "text": "first"},
		{msgType: sttEventDone, "text": "first"},
		{msgType: sttEventTextDelta, "text": "second"},
	})

	s, err := sttConn(endpoint).Connect(context.Background(), 16000)
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	defer func() { _ = s.Close() }()

	for range 2 {
		if _, err := s.Recv(); err != nil {
			t.Fatalf("Recv: %v", err)
		}
	}
	next, err := s.Recv()
	if err != nil {
		t.Fatalf("Recv: %v", err)
	}
	if next[0].Text != "second" {
		t.Errorf("text = %q, want the new utterance alone", next[0].Text)
	}
}

// TestSTTRecvIgnoresAnEmptyDone checks a done carrying no text reports no turn,
// so silence does not close one.
func TestSTTRecvIgnoresAnEmptyDone(t *testing.T) {
	endpoint, _ := sttServer(t, []map[string]any{
		{msgType: sttEventDone, "text": ""},
		{msgType: sttEventTextDelta, "text": "after"},
	})

	s, err := sttConn(endpoint).Connect(context.Background(), 16000)
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	defer func() { _ = s.Close() }()

	got, err := s.Recv()
	if err != nil {
		t.Fatalf("Recv: %v", err)
	}
	if got[0].Text != "after" {
		t.Errorf("text = %q, want the empty done stepped over", got[0].Text)
	}
}

// TestSTTRecvReportsAServerError checks an error is surfaced carrying what the
// provider said.
func TestSTTRecvReportsAServerError(t *testing.T) {
	endpoint, _ := sttServer(t, []map[string]any{
		{msgType: sttEventError, "error": map[string]any{"message": "model unavailable"}},
	})

	s, err := sttConn(endpoint).Connect(context.Background(), 16000)
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	defer func() { _ = s.Close() }()

	if _, err := s.Recv(); err == nil {
		t.Fatal("Recv accepted an error event")
	} else if !errors.Is(err, errSTTServer) || !strings.Contains(err.Error(), "model unavailable") {
		t.Errorf("error = %v, want the server error carrying its reason", err)
	}
}

// TestSTTFinalizeFlushesTheSegment checks the session is asked to close the
// segment when the speech ends, which is what makes the provider answer with the
// transcript for the utterance instead of holding it until the call is over.
func TestSTTFinalizeFlushesTheSegment(t *testing.T) {
	endpoint, seen := sttServer(t, nil)

	s, err := sttConn(endpoint).Connect(context.Background(), 16000)
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	defer func() { _ = s.Close() }()
	seen.await(t) // the session update

	fin, ok := s.(stt.Finalizer)
	if !ok {
		t.Fatal("the session cannot be told the speech ended")
	}
	if err := fin.Finalize(); err != nil {
		t.Fatalf("Finalize: %v", err)
	}

	if m := seen.await(t); m[msgType] != sttMsgAudioFlush {
		t.Errorf("message = %v, want the audio flushed", m)
	}
}

// TestSTTSpeechStartedClearsTheAccumulator checks an utterance the provider
// never closed is dropped when the next one begins, rather than being read as
// its opening words.
func TestSTTSpeechStartedClearsTheAccumulator(t *testing.T) {
	endpoint, _ := sttServer(t, []map[string]any{
		{msgType: sttEventTextDelta, "text": "abandoned"},
		{msgType: sttEventTextDelta, "text": "hello"},
	})

	s, err := sttConn(endpoint).Connect(context.Background(), 16000)
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	defer func() { _ = s.Close() }()

	if _, err := s.Recv(); err != nil { // the utterance the provider never closed
		t.Fatalf("Recv: %v", err)
	}

	starter, ok := s.(stt.SpeechStarter)
	if !ok {
		t.Fatal("the session cannot be told the speech started")
	}
	starter.SpeechStarted()

	next, err := s.Recv()
	if err != nil {
		t.Fatalf("Recv: %v", err)
	}
	if next[0].Text != "hello" {
		t.Errorf("text = %q, want the new utterance alone", next[0].Text)
	}
}

// TestSTTCloseEndsTheSessionOnly checks nothing is asked of the server on the
// way out: the transcripts were flushed as each utterance ended, and an answer
// to a request made here would arrive after the reader has gone.
func TestSTTCloseEndsTheSessionOnly(t *testing.T) {
	endpoint, seen := sttServer(t, nil)

	s, err := sttConn(endpoint).Connect(context.Background(), 16000)
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	seen.await(t) // the session update

	if err := s.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	select {
	case m := <-seen.got:
		t.Errorf("Close sent %v, want the session closed and nothing more", m)
	case <-time.After(100 * time.Millisecond):
	}
}

// TestSTTConnectReportsAnUnreachableEndpoint checks a dial that fails is
// reported rather than handing back a session that transcribes nothing.
func TestSTTConnectReportsAnUnreachableEndpoint(t *testing.T) {
	if _, err := sttConn("ws://127.0.0.1:1").Connect(context.Background(), 16000); err == nil {
		t.Fatal("Connect accepted an unreachable endpoint")
	}
}

// TestSTTMetadataReportsTheMeasuredLatency checks the latency the turn
// strategies size their wait by, and that it can be overridden.
func TestSTTMetadataReportsTheMeasuredLatency(t *testing.T) {
	if got := newSTTConnector(STTConfig{}).Metadata().TTFSP99; got != stt.MistralTTFSP99 {
		t.Errorf("TTFSP99 = %v, want the measured figure", got)
	}
	c := newSTTConnector(STTConfig{TTFSP99: 400 * time.Millisecond})
	if got := c.Metadata().TTFSP99; got != 400*time.Millisecond {
		t.Errorf("TTFSP99 = %v, want the override", got)
	}
}

// TestSTTSettingsReopenTheSession checks a settings change asks for a new
// session. Mistral takes the transcription configuration when the session opens,
// so a change reaches it no other way.
func TestSTTSettingsReopenTheSession(t *testing.T) {
	c := newSTTConnector(STTConfig{Model: sttDefaultModel})

	holder, ok := any(c).(stt.SettingsHolder)
	if !ok {
		t.Fatal("the connector holds no settings, so nothing about it can change mid-call")
	}
	live, ok := holder.Settings().(*STTSettings)
	if !ok {
		t.Fatalf("Settings() = %T, want the provider's own store", holder.Settings())
	}
	if got := live.Model.Or(""); got != sttDefaultModel {
		t.Errorf("the store opens on model %q, want %q", got, sttDefaultModel)
	}

	updater, ok := any(c).(stt.SettingsUpdater)
	if !ok {
		t.Fatal("the connector does not act on a settings change")
	}
	reopen, err := updater.UpdateSettings(t.Context(), settings.Changed{"model": nil})
	if err != nil {
		t.Fatalf("UpdateSettings: %v", err)
	}
	if !reopen {
		t.Error("a changed model did not ask for a new session, so it never reaches Mistral")
	}
}

// TestSTTConnectsWithTheUpdatedModel checks the session opens on the model the
// store holds rather than the one the service was built with.
func TestSTTConnectsWithTheUpdatedModel(t *testing.T) {
	endpoint, got := sttServer(t, nil)
	c := newSTTConnector(STTConfig{URL: endpoint, Model: sttDefaultModel})
	c.live.Model = settings.Set("voxtral-other")

	stream, err := c.Connect(t.Context(), 16000)
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	defer func() { _ = stream.Close() }()

	if q := got.query.Get("model"); q != "voxtral-other" {
		t.Errorf("model = %q, want the updated one", q)
	}
	if m := c.Metadata().Model; m != "voxtral-other" {
		t.Errorf("Metadata model = %q, want the updated one", m)
	}
}
