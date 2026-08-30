package kyutai

import (
	"context"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"
	"time"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/frames"
	"github.com/vmihailenco/msgpack/v5"
)

// Tests for the sessions moshi-server speaks. Both directions are MessagePack
// over a socket: synthesis takes the sentence a word at a time and streams
// float32 audio back, and transcription takes fixed-size float32 frames and
// returns words and pause predictions.

// wsSession is what the fake moshi endpoint saw.
type wsSession struct {
	path   string
	query  url.Values
	header http.Header
	got    chan map[string]any
}

// moshiServer starts a fake moshi endpoint. It records every message it is
// sent, replies with the scripted ones, and closes the way moshi does when
// closeNormally is set.
func moshiServer(t *testing.T, reply []map[string]any, closeNormally bool) (endpoint string, seen *wsSession) {
	t.Helper()
	s := &wsSession{got: make(chan map[string]any, 32)}

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		s.path, s.query, s.header = r.URL.Path, r.URL.Query(), r.Header.Clone()
		c, err := websocket.Accept(w, r, nil)
		if err != nil {
			return
		}
		ctx := r.Context()

		for _, m := range reply {
			b, err := msgpack.Marshal(m)
			if err != nil {
				t.Errorf("encoding a reply: %v", err)
				return
			}
			if c.Write(ctx, websocket.MessageBinary, b) != nil {
				return
			}
		}
		if closeNormally {
			_ = c.Close(websocket.StatusNormalClosure, "")
			return
		}

		for {
			_, data, err := c.Read(ctx)
			if err != nil {
				return
			}
			m := map[string]any{}
			if msgpack.Unmarshal(data, &m) == nil {
				select {
				case s.got <- m:
				default:
				}
			}
		}
	}))
	t.Cleanup(srv.Close)

	return "ws" + strings.TrimPrefix(srv.URL, "http"), s
}

// await returns the next message the endpoint was sent.
func (s *wsSession) await(t *testing.T) map[string]any {
	t.Helper()
	select {
	case m := <-s.got:
		return m
	case <-time.After(2 * time.Second):
		t.Fatal("no message arrived at the endpoint")
		return nil
	}
}

// TestSynthesizeStreamsWordsThenFlushes checks the sentence goes one word per
// message and is flushed with an end-of-stream, which is what makes moshi
// synthesize rather than wait for more.
func TestSynthesizeStreamsWordsThenFlushes(t *testing.T) {
	endpoint, seen := moshiServer(t, nil, false)
	s := &synthesizer{cfg: TTSConfig{
		URL: endpoint, APIKey: defaultToken, Voice: "cc0/alice", SampleRate: defaultSampleRate,
	}}

	go func() {
		_ = s.RunTTS(context.Background(), "hello there world", "", func(frames.Frame) error {
			return nil
		})
	}()

	for _, want := range []string{"hello", "there", "world"} {
		m := seen.await(t)
		if m[msgTypeKey] != "Text" || m["text"] != want {
			t.Fatalf("message = %v, want the word %q", m, want)
		}
	}
	if m := seen.await(t); m[msgTypeKey] != "Eos" {
		t.Errorf("last message = %v, want the end of stream", m)
	}

	if seen.header.Get("kyutai-api-key") != defaultToken {
		t.Errorf("kyutai-api-key = %q, want the token", seen.header.Get("kyutai-api-key"))
	}
	if seen.query.Get("format") != "PcmMessagePack" {
		t.Errorf("format = %q, want PcmMessagePack", seen.query.Get("format"))
	}
	if seen.query.Get("voice") != "cc0/alice" {
		t.Errorf("voice = %q, want the configured voice", seen.query.Get("voice"))
	}
}

// TestSynthesizeEmitsAudioAndEndsOnClose checks the float32 audio moshi returns
// is converted to the 16-bit PCM the pipeline carries, and that moshi closing
// the socket normally ends the synthesis rather than reporting a fault.
func TestSynthesizeEmitsAudioAndEndsOnClose(t *testing.T) {
	endpoint, _ := moshiServer(t, []map[string]any{
		{msgTypeKey: "Audio", "pcm": []float32{0, 0.5, -0.5}},
	}, true)
	s := &synthesizer{cfg: TTSConfig{
		URL: endpoint, APIKey: defaultToken, Voice: "cc0/alice", SampleRate: defaultSampleRate,
	}}

	var pcm []byte
	err := s.RunTTS(context.Background(), "hi", "", func(f frames.Frame) error {
		if af, ok := f.(frames.OutputAudioFrame); ok {
			pcm = append(pcm, af.AudioData().Audio...)
		}
		return nil
	})
	if err != nil {
		t.Fatalf("RunTTS: %v", err)
	}
	if want := float32ToInt16Bytes([]float32{0, 0.5, -0.5}); string(pcm) != string(want) {
		t.Errorf("audio = %v, want the converted samples %v", pcm, want)
	}
}

// TestSynthesizeReportsAnUnreachableEndpoint checks a dial that fails is
// reported.
func TestSynthesizeReportsAnUnreachableEndpoint(t *testing.T) {
	s := &synthesizer{cfg: TTSConfig{URL: "ws://127.0.0.1:1", Voice: "v"}}
	err := s.RunTTS(context.Background(), "hi", "", func(frames.Frame) error { return nil })
	if err == nil {
		t.Fatal("RunTTS accepted an unreachable endpoint")
	}
}

// TestSampleRate checks the rate the emitted frames are stamped with.
func TestSampleRate(t *testing.T) {
	if got := (&synthesizer{cfg: TTSConfig{SampleRate: 16000}}).SampleRate(); got != 16000 {
		t.Errorf("SampleRate() = %d, want the configured rate", got)
	}
}

// TestSTTSendsFixedSizeFrames checks the audio is resampled to moshi's rate and
// forwarded in frames of the size it expects, with the remainder held back
// rather than sent short.
func TestSTTSendsFixedSizeFrames(t *testing.T) {
	endpoint, seen := moshiServer(t, nil, false)
	c := &connector{cfg: Config{URL: endpoint, APIKey: defaultToken}}

	st, err := c.Connect(context.Background(), defaultSampleRate)
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	defer func() { _ = st.Close() }()

	// Two frames' worth at moshi's own rate, so no resampling is needed, plus a
	// remainder that must not be sent on its own.
	if err := st.Send(make([]byte, (sttFrameSamples*2+100)*2)); err != nil {
		t.Fatalf("Send: %v", err)
	}

	for range 2 {
		m := seen.await(t)
		if m[msgTypeKey] != "Audio" {
			t.Fatalf("message = %v, want an Audio frame", m)
		}
		pcm, ok := m["pcm"].([]any)
		if !ok || len(pcm) != sttFrameSamples {
			t.Fatalf("frame carried %d samples, want %d", len(pcm), sttFrameSamples)
		}
	}
	select {
	case m := <-seen.got:
		t.Errorf("a third message %v was sent, want the remainder held back", m)
	case <-time.After(200 * time.Millisecond):
	}

	if seen.header.Get("kyutai-api-key") != defaultToken {
		t.Errorf("kyutai-api-key = %q, want the token", seen.header.Get("kyutai-api-key"))
	}
}

// TestSTTRecvReadsWordsAndPauses checks the results moshi returns are decoded:
// a word extends the utterance, and a pause prediction past the threshold ends
// the turn.
func TestSTTRecvReadsWordsAndPauses(t *testing.T) {
	endpoint, _ := moshiServer(t, []map[string]any{
		{msgTypeKey: "Word", "text": "hello"},
		{msgTypeKey: "Word", "text": "there"},
		{msgTypeKey: "Step", "prs": []float64{0, 0, 0.9}},
	}, false)
	c := &connector{cfg: Config{URL: endpoint, APIKey: defaultToken}}

	st, err := c.Connect(context.Background(), defaultSampleRate)
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	defer func() { _ = st.Close() }()

	first, err := st.Recv()
	if err != nil {
		t.Fatalf("Recv: %v", err)
	}
	if len(first) != 1 || first[0].Text != "hello" || first[0].Final {
		t.Errorf("results = %+v, want the interim hello", first)
	}

	second, err := st.Recv()
	if err != nil {
		t.Fatalf("Recv: %v", err)
	}
	if second[0].Text != "hello there" {
		t.Errorf("text = %q, want the utterance so far", second[0].Text)
	}

	final, err := st.Recv()
	if err != nil {
		t.Fatalf("Recv: %v", err)
	}
	if !final[0].Final || !final[0].EndOfTurn || final[0].Text != "hello there" {
		t.Errorf("result = %+v, want the finalized utterance ending the turn", final[0])
	}
}

// TestSTTConnectReportsAnUnreachableEndpoint checks a dial that fails is
// reported rather than handing back a session that transcribes nothing.
func TestSTTConnectReportsAnUnreachableEndpoint(t *testing.T) {
	c := &connector{cfg: Config{URL: "ws://127.0.0.1:1"}}
	if _, err := c.Connect(context.Background(), 16000); err == nil {
		t.Fatal("Connect accepted an unreachable endpoint")
	}
}
