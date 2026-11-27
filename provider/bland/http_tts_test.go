package bland

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"sync"
	"testing"

	"github.com/gojargo/jargo/frames"
)

// httpBland is a fake /v2/tts endpoint recording each request and answering
// it with respond.
type httpBland struct {
	base    string
	respond func(w http.ResponseWriter)

	mu       sync.Mutex
	auth     []string
	payloads []map[string]any
}

func newHTTPBland(t *testing.T, respond func(w http.ResponseWriter)) *httpBland {
	t.Helper()
	f := &httpBland{respond: respond}
	mux := http.NewServeMux()
	mux.HandleFunc("POST /v2/tts", func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		payload := map[string]any{}
		_ = json.Unmarshal(raw, &payload)
		f.mu.Lock()
		f.auth = append(f.auth, r.Header.Get("Authorization"))
		f.payloads = append(f.payloads, payload)
		f.mu.Unlock()
		f.respond(w)
	})
	srv := httptest.NewServer(mux)
	t.Cleanup(srv.Close)
	f.base = srv.URL + "/v2"
	return f
}

// pcmResponse answers with body as raw PCM.
func pcmResponse(body []byte) func(w http.ResponseWriter) {
	return func(w http.ResponseWriter) {
		w.Header().Set("Content-Type", "audio/pcm")
		_, _ = w.Write(body)
	}
}

func (f *httpBland) firstPayload(t *testing.T) map[string]any {
	t.Helper()
	f.mu.Lock()
	defer f.mu.Unlock()
	if len(f.payloads) == 0 {
		t.Fatal("the endpoint was never asked")
	}
	return f.payloads[0]
}

// newHTTPService builds the HTTP service against the fake endpoint at 24 kHz.
func newHTTPService(f *httpBland, cfg HTTPTTSConfig) *httpSynthesizer {
	cfg.APIKey = "test-key"
	if cfg.URL == "" {
		cfg.URL = f.base
	}
	if cfg.SampleRate == 0 {
		cfg.SampleRate = 24000
	}
	_, s := newHTTPTTS(cfg)
	return s
}

// The documented request is sent, and the PCM of the response is emitted.
func TestRunHTTPTTSSuccess(t *testing.T) {
	payload := pcmBytes(4096)
	f := newHTTPBland(t, pcmResponse(payload))
	b := NewHTTPTTS(HTTPTTSConfig{APIKey: "test-key", URL: f.base + "/", SampleRate: 24000})

	down, _ := runTest(t, b.Base, queue(frames.NewTTSSpeakFrame("Hello from Bland.")))

	if !hasFrame[*frames.TTSStartedFrame](down) || !hasFrame[*frames.TTSStoppedFrame](down) {
		t.Errorf("frames = %v, want started and stopped", down)
	}
	if f.auth[0] != "Bearer test-key" {
		t.Errorf("Authorization = %q", f.auth[0])
	}
	body := f.firstPayload(t)
	if body["text"] != "Hello from Bland." || body["voice"] != defaultVoiceID {
		t.Errorf("body = %v", body)
	}
	// 24000 is a rate Bland renders directly, so it is requested as is.
	want := map[string]any{"encoding": "pcm_s16le", "sample_rate": float64(24000), "container": "raw"}
	if !reflect.DeepEqual(body["audio"], want) {
		t.Errorf("audio = %v, want %v", body["audio"], want)
	}
	// Nothing the request shape does not define.
	for _, key := range []string{"controls", "auto_formatting", "language", "output_format", "voice_id"} {
		if _, ok := body[key]; ok {
			t.Errorf("%s was sent", key)
		}
	}
	audio := audioOf(down)
	if !bytes.Equal(audio, payload) || bytes.HasPrefix(audio, []byte("RIFF")) {
		t.Errorf("audio = %d bytes, want the payload as sent", len(audio))
	}
	for _, fr := range down {
		if a, ok := fr.(*frames.TTSAudioRawFrame); ok && a.SampleRate != 24000 {
			t.Errorf("audio at %d Hz, want 24000", a.SampleRate)
		}
	}
}

// auto_formatting reaches the /v2/tts body when set.
func TestHTTPTTSSendsAutoFormatting(t *testing.T) {
	f := newHTTPBland(t, pcmResponse(pcmBytes(4096)))
	runTest(t, NewHTTPTTS(HTTPTTSConfig{APIKey: "k", URL: f.base, SampleRate: 24000, AutoFormatting: new(true)}).Base,
		queue(frames.NewTTSSpeakFrame("Call 7327412065.")))

	if got := f.firstPayload(t)["auto_formatting"]; got != true {
		t.Errorf("auto_formatting = %v, want true", got)
	}
}

// A service rate Bland cannot emit falls back to 48 kHz and is resampled down.
func TestHTTPTTSResamplesUnsupportedPipelineRate(t *testing.T) {
	payload := pcmBytes(4800)
	f := newHTTPBland(t, pcmResponse(payload))
	down, _ := runTest(t, NewHTTPTTS(HTTPTTSConfig{APIKey: "k", URL: f.base, SampleRate: 22050}).Base,
		queue(frames.NewTTSSpeakFrame("Hi.")))

	if got := sampleRateOf(f.firstPayload(t)); got != float64(48000) {
		t.Errorf("sample_rate = %v, want 48000", got)
	}
	audio := audioOf(down)
	if len(audio) == 0 || bytes.Equal(audio, payload) {
		t.Errorf("audio = %d bytes, want it resampled", len(audio))
	}
	for _, fr := range down {
		if a, ok := fr.(*frames.TTSAudioRawFrame); ok && a.SampleRate != 22050 {
			t.Errorf("audio at %d Hz, want 22050", a.SampleRate)
		}
	}
}

// A split at an odd byte lands midway through a sample; nothing may be dropped
// or reordered.
func TestHTTPTTSReassemblesAudioSplitAcrossChunks(t *testing.T) {
	payload := pcmBytes(4096)
	splits := []int{1, 3, 1000, 2001, len(payload)}
	f := newHTTPBland(t, func(w http.ResponseWriter) {
		w.Header().Set("Content-Type", "audio/pcm")
		flusher, _ := w.(http.Flusher)
		start := 0
		for _, end := range splits {
			_, _ = w.Write(payload[start:end])
			flusher.Flush()
			start = end
		}
	})
	down, _ := runTest(t, NewHTTPTTS(HTTPTTSConfig{APIKey: "k", URL: f.base, SampleRate: 24000}).Base,
		queue(frames.NewTTSSpeakFrame("Hi.")))

	if got := audioOf(down); !bytes.Equal(got, payload) {
		t.Errorf("audio = %d bytes, want the %d sent, in order", len(got), len(payload))
	}
}

// The settings map into the request body, and only the controls the caller set
// are sent, so unset ones keep Bland's defaults.
func TestHTTPTTSSettingsPayload(t *testing.T) {
	expressiveness, stability := 0.9, 0.4
	for name, tc := range map[string]struct {
		cfg   HTTPTTSConfig
		voice string
		want  map[string]any
	}{
		"all": {
			HTTPTTSConfig{Voice: otherVoiceID, Expressiveness: &expressiveness, Stability: &stability},
			otherVoiceID,
			map[string]any{"expressiveness": 0.9, "stability": 0.4},
		},
		"partial": {HTTPTTSConfig{Stability: &stability}, defaultVoiceID, map[string]any{"stability": 0.4}},
	} {
		t.Run(name, func(t *testing.T) {
			f := newHTTPBland(t, pcmResponse(pcmBytes(4096)))
			s := newHTTPService(f, tc.cfg)
			if err := s.RunTTS(t.Context(), "Hi.", "c1", func(frames.Frame) error { return nil }); err != nil {
				t.Fatal(err)
			}
			body := f.firstPayload(t)
			if body["voice"] != tc.voice || !reflect.DeepEqual(body["controls"], tc.want) {
				t.Errorf("voice = %v, controls = %v, want %s and %v", body["voice"], body["controls"], tc.voice, tc.want)
			}
		})
	}
}

// A refused request yields an error carrying the v2 error code and message,
// and a gateway error with an HTML body still surfaces as one.
func TestHTTPTTSErrorResponse(t *testing.T) {
	for name, tc := range map[string]struct {
		respond func(w http.ResponseWriter)
		want    []string
	}{
		"v2 envelope": {
			func(w http.ResponseWriter) {
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(http.StatusNotFound)
				_, _ = io.WriteString(w, `{"error": {"code": "voice_not_found", "message": "Voice was not found."}}`)
			},
			[]string{"voice_not_found", "Voice was not found."},
		},
		"not json": {
			func(w http.ResponseWriter) {
				w.Header().Set("Content-Type", "text/html")
				w.WriteHeader(http.StatusBadGateway)
				_, _ = io.WriteString(w, "<html>gateway</html>")
			},
			[]string{"502"},
		},
	} {
		t.Run(name, func(t *testing.T) {
			f := newHTTPBland(t, tc.respond)
			_, up := runTest(t, NewHTTPTTS(HTTPTTSConfig{APIKey: "k", URL: f.base, SampleRate: 24000}).Base,
				queue(frames.NewTTSSpeakFrame("Hi.")))

			errs := errorsOf(up)
			if len(errs) == 0 {
				t.Fatal("no error reached the pipeline")
			}
			for _, want := range tc.want {
				if !strings.Contains(errs[0], want) {
					t.Errorf("error = %q, want it to carry %q", errs[0], want)
				}
			}
		})
	}
}
