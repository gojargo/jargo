package deepgram

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/internal/providertest"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/service/settings"
	"github.com/gojargo/jargo/service/tts"
)

// speakTripper answers every request from a canned body, recording what it was
// asked. The Aura endpoint is a package constant rather than configuration, so
// intercepting the transport is the only way to see the request it builds.
type speakTripper struct {
	status int
	body   []byte

	url    *url.URL
	header http.Header
	sent   map[string]string
}

func (rt *speakTripper) RoundTrip(req *http.Request) (*http.Response, error) {
	rt.url = req.URL
	rt.header = req.Header.Clone()
	if req.Body != nil {
		raw, err := io.ReadAll(req.Body)
		if err != nil {
			return nil, err
		}
		rt.sent = map[string]string{}
		if err := json.Unmarshal(raw, &rt.sent); err != nil {
			return nil, err
		}
	}
	status := rt.status
	if status == 0 {
		status = http.StatusOK
	}
	return &http.Response{
		StatusCode: status,
		Body:       io.NopCloser(bytes.NewReader(rt.body)),
		Header:     http.Header{},
		Request:    req,
	}, nil
}

// synthesize runs one synthesis through syn and returns the PCM the frames
// carried, along with the rate they declared.
func synthesize(t *testing.T, syn *synthesizer, text string) ([]byte, int) {
	t.Helper()
	var pcm bytes.Buffer
	rate := 0
	err := syn.RunTTS(t.Context(), text, "", func(f frames.Frame) error {
		audio, ok := f.(*frames.TTSAudioRawFrame)
		if !ok {
			t.Errorf("yielded %T, want a TTSAudioRawFrame", f)
			return nil
		}
		rate = audio.SampleRate
		pcm.Write(audio.Audio)
		return nil
	})
	if err != nil {
		t.Fatalf("RunTTS: %v", err)
	}
	return pcm.Bytes(), rate
}

// TestTTSConfigValidate pins the credential the Aura API requires.
func TestTTSConfigValidate(t *testing.T) {
	providertest.Configs(t, []providertest.ConfigCase{
		{Name: "missing API key", Cfg: TTSConfig{}, Valid: false},
		{Name: "API key only", Cfg: TTSConfig{APIKey: "k"}, Valid: true},
		{Name: "WebSocket missing API key", Cfg: WebsocketTTSConfig{}, Valid: false},
		{Name: "WebSocket API key only", Cfg: WebsocketTTSConfig{APIKey: "k"}, Valid: true},
		{Name: "missing STT API key", Cfg: Config{}, Valid: false},
		{Name: "STT API key only", Cfg: Config{APIKey: "k"}, Valid: true},
	})
}

// TestNewServices checks each constructor returns a service under the label that
// identifies it in logs, metrics and traces.
func TestNewServices(t *testing.T) {
	providertest.Service(t, "DeepgramTTS", NewTTS(TTSConfig{APIKey: "k"}))
	providertest.Service(t, "DeepgramWebsocketTTS", NewWebsocketTTS(WebsocketTTSConfig{APIKey: "k"}))
	providertest.Service(t, "DeepgramSTT", NewSTT(Config{APIKey: "k"}))
}

// TestTTSMetadata checks the service reports the voice model synthesis is billed
// against.
func TestTTSMetadata(t *testing.T) {
	s := newSynthesizer(TTSConfig{APIKey: "k", SampleRate: 24000})
	if got := s.Metadata().Model; got != defaultTTSVoice {
		t.Errorf("Metadata().Model = %q, want %q", got, defaultTTSVoice)
	}
	if got := s.SampleRate(); got != 24000 {
		t.Errorf("SampleRate() = %d, want the configured rate", got)
	}
}

// TestTTSDefaults pins the voice spoken when none is set, the same one the
// WebSocket service uses.
func TestTTSDefaults(t *testing.T) {
	s := newSynthesizer(TTSConfig{APIKey: "k"})
	if s.cfg.Voice != "aura-2-helena-en" {
		t.Errorf("voice = %q, want aura-2-helena-en", s.cfg.Voice)
	}
	if s.cfg.SampleRate != 24000 || s.cfg.Encoding != "linear16" || s.cfg.BaseURL != "https://api.deepgram.com" {
		t.Errorf("defaults = %+v", s.cfg)
	}
	if _, ok := any(s).(tts.TrailingSpaceRequirer); ok {
		t.Error("the HTTP service asks for a trailing space, want the text sent as it is")
	}
}

// TestRunTTSRequestShape checks the synthesis is addressed at the speak endpoint
// asking for headerless PCM, authorized with Deepgram's own token scheme, and
// that the audio comes back tagged with the rate it was asked for.
func TestRunTTSRequestShape(t *testing.T) {
	want := bytes.Repeat([]byte{0x11, 0x22}, 64)
	rt := &speakTripper{body: want}
	s := newSynthesizer(TTSConfig{APIKey: "test-key", SampleRate: 24000})
	s.http = &http.Client{Transport: rt}

	pcm, rate := synthesize(t, s, "hello there")
	if !bytes.Equal(pcm, want) {
		t.Errorf("PCM = % x, want the response body verbatim", pcm)
	}
	if rate != 24000 {
		t.Errorf("frame rate = %d, want the configured 24000", rate)
	}

	if !strings.HasPrefix(rt.url.String(), "https://"+defaultHost+speakPath+"?") {
		t.Errorf("URL = %q, want the speak endpoint", rt.url)
	}
	q := rt.url.Query()
	if q.Get("model") != defaultTTSVoice {
		t.Errorf("model = %q, want %q", q.Get("model"), defaultTTSVoice)
	}
	if q.Get("encoding") != defaultTTSEncoding {
		t.Errorf("encoding = %q, want %q", q.Get("encoding"), defaultTTSEncoding)
	}
	if q.Get("sample_rate") != "24000" {
		t.Errorf("sample_rate = %q, want the configured rate", q.Get("sample_rate"))
	}
	// Without this Aura returns a WAV stream, and its header would be played as
	// though it were samples.
	if q.Get("container") != "none" {
		t.Errorf("container = %q, want none so the PCM arrives headerless", q.Get("container"))
	}
	if got := rt.header.Get("Authorization"); got != "Token test-key" {
		t.Errorf("Authorization = %q, want Deepgram's Token scheme", got)
	}
	if got := rt.header.Get("Content-Type"); got != "application/json" {
		t.Errorf("Content-Type = %q, want application/json", got)
	}
	if rt.sent["text"] != "hello there" {
		t.Errorf("text = %q, want the text to speak", rt.sent["text"])
	}
	// Speed and the opt-out are left off the query when unset.
	if q.Has("speed") || q.Has("mip_opt_out") {
		t.Errorf("query = %v, want no speed or mip_opt_out", q)
	}
}

// TestRunTTSSendsSpeedAndOptOut checks the speed and the Model Improvement
// Program opt-out reach the query when set.
func TestRunTTSSendsSpeedAndOptOut(t *testing.T) {
	rt := &speakTripper{body: []byte{0, 0}}
	s := newSynthesizer(TTSConfig{APIKey: "k", Voice: "aura-2-andromeda-en", Speed: new(0.9), MipOptOut: new(true)})
	s.http = &http.Client{Transport: rt}

	synthesize(t, s, "hi")

	q := rt.url.Query()
	if q.Get("model") != "aura-2-andromeda-en" || q.Get("speed") != "0.9" || q.Get("mip_opt_out") != "true" {
		t.Errorf("query = %v, want the voice, speed 0.9 and mip_opt_out=true", q)
	}
}

// TestRunTTSUsesTheSettingsAsTheyStand checks a voice or speed changed while
// the pipeline runs goes out on the next request, and relabels the synthesis.
func TestRunTTSUsesTheSettingsAsTheyStand(t *testing.T) {
	rt := &speakTripper{body: []byte{0, 0}}
	s := newSynthesizer(TTSConfig{APIKey: "k"})
	s.http = &http.Client{Transport: rt}

	update := &TTSSettings{Speed: settings.Set(1.2)}
	update.Voice = settings.Set("aura-2-thalia-en")
	changed, err := settings.Apply(s.Settings(), update)
	if err != nil {
		t.Fatalf("applying the settings: %v", err)
	}
	if !changed.Has("voice") || !changed.Has("model") || !changed.Has("speed") {
		t.Errorf("changed = %v, want voice, model and speed", changed)
	}

	synthesize(t, s, "hi")

	q := rt.url.Query()
	if q.Get("model") != "aura-2-thalia-en" || q.Get("speed") != "1.2" {
		t.Errorf("query = %v, want the updated voice and speed", q)
	}
	if m := s.Metadata(); m.Model != "aura-2-thalia-en" || m.VoiceID != "aura-2-thalia-en" {
		t.Errorf("Metadata() = %+v, want the updated voice", m)
	}
}

// TestHTTPTTSFramesStaySampleAligned checks the audio frames hold whole 16-bit
// samples however the HTTP stream is split, with metrics on, where the time to
// first audio is found by reading the audio as 16-bit samples.
func TestHTTPTTSFramesStaySampleAligned(t *testing.T) {
	pcm := bytes.Repeat([]byte{0x00, 0x01, 0x02, 0x03}, 1024)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != speakPath {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Type", "audio/l16")
		_, _ = w.Write(pcm[:2047])
		if f, ok := w.(http.Flusher); ok {
			f.Flush()
		}
		time.Sleep(10 * time.Millisecond)
		_, _ = w.Write(pcm[2047:])
	}))
	t.Cleanup(srv.Close)

	svc := NewTTS(TTSConfig{APIKey: "test-key", BaseURL: srv.URL, SampleRate: 24000})
	down, _ := runAuraPipeline(t, svc.Base, pipeline.Params{EnableMetrics: true},
		frames.NewTTSSpeakFrame("Hello from Deepgram."), 300*time.Millisecond)

	audio := framesOf[*frames.TTSAudioRawFrame](down)
	if len(audio) == 0 {
		t.Fatal("no audio frames")
	}
	var joined []byte
	for _, f := range audio {
		if len(f.Audio)%2 != 0 {
			t.Errorf("a frame holds %d bytes, want whole 16-bit samples", len(f.Audio))
		}
		joined = append(joined, f.Audio...)
	}
	if !bytes.Equal(joined, pcm) {
		t.Errorf("audio = %d bytes, want the %d bytes sent, in order", len(joined), len(pcm))
	}
}

// TestRunTTSStatusError checks a non-200 is reported rather than streamed on as
// though the body were audio.
func TestRunTTSStatusError(t *testing.T) {
	rt := &speakTripper{status: http.StatusUnauthorized, body: []byte("bad credentials")}
	s := newSynthesizer(TTSConfig{APIKey: "k", SampleRate: 24000})
	s.http = &http.Client{Transport: rt}

	err := s.RunTTS(t.Context(), "hi", "", func(frames.Frame) error {
		t.Error("a frame was yielded for a failed request")
		return nil
	})
	if err == nil {
		t.Fatal("RunTTS on a 401 = nil, want an error")
	}
}

// TestAuthToken pins the scheme Deepgram authorizes with. It is not a Bearer
// token, and sending one is rejected.
func TestAuthToken(t *testing.T) {
	if got := authToken("abc"); got != "Token abc" {
		t.Errorf("authToken() = %q, want the Token scheme", got)
	}
}
