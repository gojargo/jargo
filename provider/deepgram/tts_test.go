package deepgram

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/url"
	"strings"
	"testing"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/internal/providertest"
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
		{Name: "missing STT API key", Cfg: Config{}, Valid: false},
		{Name: "STT API key only", Cfg: Config{APIKey: "k"}, Valid: true},
	})
}

// TestNewServices checks each constructor returns a service under the label that
// identifies it in logs, metrics and traces.
func TestNewServices(t *testing.T) {
	providertest.Service(t, "DeepgramTTS", NewTTS(TTSConfig{APIKey: "k"}))
	providertest.Service(t, "DeepgramSTT", NewSTT(Config{APIKey: "k"}))
}

// TestTTSMetadata checks the service reports the voice model synthesis is billed
// against.
func TestTTSMetadata(t *testing.T) {
	s := &synthesizer{cfg: TTSConfig{Model: defaultTTSModel, SampleRate: 24000}}
	if got := s.Metadata().Model; got != defaultTTSModel {
		t.Errorf("Metadata().Model = %q, want %q", got, defaultTTSModel)
	}
	if got := s.SampleRate(); got != 24000 {
		t.Errorf("SampleRate() = %d, want the configured rate", got)
	}
}

// TestRunTTSRequestShape checks the synthesis is addressed at the speak endpoint
// asking for headerless PCM, authorized with Deepgram's own token scheme, and
// that the audio comes back tagged with the rate it was asked for.
func TestRunTTSRequestShape(t *testing.T) {
	want := bytes.Repeat([]byte{0x11, 0x22}, 64)
	rt := &speakTripper{body: want}
	s := &synthesizer{
		cfg: TTSConfig{
			APIKey:     "test-key",
			BaseURL:    "https://" + defaultHost,
			Model:      defaultTTSModel,
			SampleRate: 24000,
			Encoding:   defaultTTSEncoding,
		},
		http: &http.Client{Transport: rt},
	}

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
	if q.Get("model") != defaultTTSModel {
		t.Errorf("model = %q, want %q", q.Get("model"), defaultTTSModel)
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
}

// TestRunTTSStatusError checks a non-200 is reported rather than streamed on as
// though the body were audio.
func TestRunTTSStatusError(t *testing.T) {
	rt := &speakTripper{status: http.StatusUnauthorized, body: []byte("bad credentials")}
	s := &synthesizer{
		cfg:  TTSConfig{APIKey: "k", BaseURL: "https://" + defaultHost, Model: defaultTTSModel, SampleRate: 24000},
		http: &http.Client{Transport: rt},
	}

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
