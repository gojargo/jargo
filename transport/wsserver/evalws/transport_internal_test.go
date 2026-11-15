package evalws

import (
	"context"
	"net/http"
	"net/http/httptest"
	"net/url"
	"slices"
	"sync"
	"testing"
	"time"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/transport/wsserver"
)

// request is a connection request for target, as the server sees it.
func request(t *testing.T, target string) *http.Request {
	t.Helper()
	return httptest.NewRequestWithContext(t.Context(), http.MethodGet, target, nil)
}

func TestQueryFlagTrueViaURL(t *testing.T) {
	if !QueryFlag(request(t, "/?skip_tts=true"), SkipTTSQueryParam) {
		t.Fatal("skip_tts=true read as off")
	}
}

// A request whose URL was not parsed is read from its raw target.
func TestQueryFlagTrueViaRequestURI(t *testing.T) {
	r := &http.Request{RequestURI: "/?skip_tts=1"}
	if !QueryFlag(r, SkipTTSQueryParam) {
		t.Fatal("skip_tts=1 read as off")
	}
}

func TestQueryFlagAcceptsYesAndMixedCase(t *testing.T) {
	if !QueryFlag(request(t, "/?skip_tts=YES"), SkipTTSQueryParam) {
		t.Fatal("skip_tts=YES read as off")
	}
}

func TestQueryFlagCaptureAudio(t *testing.T) {
	if !QueryFlag(request(t, "/?capture_bot_audio=true"), CaptureAudioQueryParam) {
		t.Fatal("capture_bot_audio=true read as off")
	}
	if QueryFlag(request(t, "/?skip_tts=true"), CaptureAudioQueryParam) {
		t.Fatal("capture_bot_audio read as on from another flag")
	}
}

func TestQueryFlagCaptureImages(t *testing.T) {
	if !QueryFlag(request(t, "/?capture_bot_images=true"), CaptureImagesQueryParam) {
		t.Fatal("capture_bot_images=true read as off")
	}
	if QueryFlag(request(t, "/?capture_bot_audio=true"), CaptureImagesQueryParam) {
		t.Fatal("capture_bot_images read as on from another flag")
	}
}

func TestQueryFlagFalseWhenAbsent(t *testing.T) {
	if QueryFlag(request(t, "/"), SkipTTSQueryParam) {
		t.Fatal("an absent flag read as on")
	}
}

func TestQueryFlagFalseWhenFalsey(t *testing.T) {
	if QueryFlag(request(t, "/?skip_tts=false"), SkipTTSQueryParam) {
		t.Fatal("skip_tts=false read as on")
	}
}

func TestQueryFlagFalseWhenNoPathAtAll(t *testing.T) {
	if QueryFlag(&http.Request{}, SkipTTSQueryParam) || QueryFlag(nil, SkipTTSQueryParam) {
		t.Fatal("a request with no target read as on")
	}
}

// acceptedConn is the server end of a WebSocket a client opened, as a bot
// handed an accepted socket has it.
func acceptedConn(t *testing.T) *websocket.Conn {
	t.Helper()
	accepted := make(chan *websocket.Conn, 1)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		c, err := websocket.Accept(w, r, nil)
		if err != nil {
			t.Errorf("accept: %v", err)
			return
		}
		accepted <- c
		<-r.Context().Done()
	}))
	t.Cleanup(srv.Close)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	u, _ := url.Parse(srv.URL)
	u.Scheme = "ws"
	client, resp, err := websocket.Dial(ctx, u.String(), nil)
	if resp != nil && resp.Body != nil {
		_ = resp.Body.Close()
	}
	if err != nil {
		t.Fatalf("dial: %v", err)
	}
	t.Cleanup(func() { _ = client.CloseNow() })

	server := <-accepted
	t.Cleanup(func() { _ = server.CloseNow() })
	return server
}

// newTestTransport is the eval transport with the eval serializer, for flags.
func newTestTransport(t *testing.T, flags ConnectionFlags) *Transport {
	t.Helper()
	return NewTransport(acceptedConn(t), TransportParams{
		Params:     wsserver.DefaultParams(),
		Serializer: NewSerializer(),
	}, flags)
}

func TestCaptureFlagsReachTheSerializer(t *testing.T) {
	tr := newTestTransport(t, ConnectionFlags{CaptureBotAudio: true, CaptureBotImages: true})
	s, ok := tr.Params().Serializer.(*Serializer)
	if !ok {
		t.Fatalf("serializer = %T, want the eval serializer", tr.Params().Serializer)
	}
	if !s.captureAudio.Load() || !s.captureImages.Load() {
		t.Fatalf("capture audio %v, images %v, want both on", s.captureAudio.Load(), s.captureImages.Load())
	}

	// Captured, the bot's audio goes out as eval-bot-audio.
	out, err := s.Serialize(frames.NewOutputAudioRawFrame([]byte{1, 2}, 24000, 1))
	if err != nil {
		t.Fatal(err)
	}
	want := `{"label":"rtvi-ai","type":"eval-bot-audio","data":{"audio":"AQI=","sampleRate":24000}}`
	if string(out.Data) != want {
		t.Fatalf("bot audio = %s, want %s", out.Data, want)
	}
}

func TestSkipTTSIsConfiguredBeforeOnClientConnected(t *testing.T) {
	for _, skipTTS := range []bool{true, false} {
		if got := greetingSkipTTS(t, skipTTS); !slices.Equal(got, []bool{skipTTS}) {
			t.Errorf("skip_tts %v: settings by the time the bot greets = %v, want [%v]", skipTTS, got, skipTTS)
		}
	}
}

// greetingSkipTTS returns the skip_tts settings pushed by the time the bot's
// own client connected handler runs.
func greetingSkipTTS(t *testing.T, skipTTS bool) []bool {
	t.Helper()
	tr := newTestTransport(t, ConnectionFlags{SkipTTS: skipTTS})
	var (
		mu     sync.Mutex
		pushed []bool
	)
	tr.pushFrame = func(_ context.Context, f frames.Frame) error {
		if c, ok := f.(*frames.LLMConfigureOutputFrame); ok {
			mu.Lock()
			pushed = append(pushed, c.SkipTTS)
			mu.Unlock()
		}
		return nil
	}
	greeting := make(chan []bool, 1)
	tr.Events().Add(wsserver.EventClientConnected, func(context.Context, any, ...any) {
		mu.Lock()
		greeting <- slices.Clone(pushed)
		mu.Unlock()
	})

	ctx := context.Background()
	tr.Events().Call(ctx, wsserver.EventClientConnected, tr.Input(), struct{}{})
	defer tr.Events().Cleanup(ctx)
	select {
	case got := <-greeting:
		return got
	case <-time.After(time.Second):
		t.Fatal("the bot's handler never ran")
		return nil
	}
}

// The serializer a transport is built without is the eval serializer.
func TestNilSerializerDefaultsToTheEvalSerializer(t *testing.T) {
	tr := NewTransport(acceptedConn(t), TransportParams{Params: wsserver.DefaultParams()}, ConnectionFlags{
		CaptureBotAudio: true,
	})
	s, ok := tr.Params().Serializer.(*Serializer)
	if !ok || !s.captureAudio.Load() {
		t.Fatalf("serializer = %T, want the eval serializer capturing audio", tr.Params().Serializer)
	}
}
