package sarvam

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/service/settings"
)

// httpTTSServer is a fake Sarvam endpoint recording the JSON body of each POST
// and answering it with reply.
type httpTTSServer struct {
	url    string
	status int
	reply  string

	mu       sync.Mutex
	payloads []map[string]any
	headers  []http.Header
	paths    []string
}

func newHTTPTTSServer(t *testing.T) *httpTTSServer {
	t.Helper()
	f := &httpTTSServer{status: http.StatusOK, reply: `{"audios": [""], "request_id": "test"}`}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		payload := map[string]any{}
		_ = json.Unmarshal(raw, &payload)
		f.mu.Lock()
		f.payloads = append(f.payloads, payload)
		f.headers = append(f.headers, r.Header.Clone())
		f.paths = append(f.paths, r.URL.Path)
		f.mu.Unlock()
		w.WriteHeader(f.status)
		_, _ = io.WriteString(w, f.reply)
	}))
	t.Cleanup(srv.Close)
	f.url = srv.URL
	return f
}

// httpService builds the HTTP service against the fake endpoint, on v4-flash
// unless cfg names a model.
func httpService(f *httpTTSServer, cfg HTTPTTSConfig) *httpSynthesizer {
	cfg.APIKey = "test-key"
	cfg.URL = f.url
	if cfg.Model == "" {
		cfg.Model = v4Flash
	}
	_, s := newHTTPTTS(cfg)
	return s
}

// synthesizeHTTP runs one synthesis and returns what was yielded.
func synthesizeHTTP(t *testing.T, s *httpSynthesizer) []frames.Frame {
	t.Helper()
	var out []frames.Frame
	if err := s.RunTTS(t.Context(), "hello", "ctx", func(f frames.Frame) error {
		out = append(out, f)
		return nil
	}); err != nil {
		t.Fatalf("RunTTS: %v", err)
	}
	return out
}

// lastPayload is the body of the last request the endpoint saw.
func (f *httpTTSServer) lastPayload(t *testing.T) map[string]any {
	t.Helper()
	f.mu.Lock()
	defer f.mu.Unlock()
	if len(f.payloads) == 0 {
		t.Fatal("the endpoint was never asked")
	}
	return f.payloads[len(f.payloads)-1]
}

// updateHTTPSettings applies delta as the base does, then has the synthesizer
// act on it.
func updateHTTPSettings(t *testing.T, s *httpSynthesizer, delta any) settings.Changed {
	t.Helper()
	changed, err := settings.Apply(s.live, delta)
	if err != nil {
		t.Fatal(err)
	}
	if err := s.UpdateSettings(context.Background(), changed); err != nil {
		t.Fatal(err)
	}
	return changed
}

func TestHTTPPayloadCarriesOnlyTheParametersV4FlashSupports(t *testing.T) {
	f := newHTTPTTSServer(t)
	pitch, loudness, temperature := 0.3, 1.5, 0.9
	s := httpService(f, HTTPTTSConfig{Pitch: &pitch, Loudness: &loudness, Temperature: &temperature})

	synthesizeHTTP(t, s)
	payload := f.lastPayload(t)

	if payload["model"] != v4Flash {
		t.Errorf("model = %v, want %s", payload["model"], v4Flash)
	}
	if payload["speaker"] != "shubh_en_narration_gentle" {
		t.Errorf("speaker = %v, want the v4-flash default", payload["speaker"])
	}
	if payload["pitch"] != 0.3 || payload["loudness"] != 1.5 {
		t.Errorf("pitch = %v, loudness = %v, want 0.3 and 1.5", payload["pitch"], payload["loudness"])
	}
	if _, present := payload["temperature"]; present {
		t.Error("temperature was sent to v4-flash")
	}
}

// The payload is built from the model's capabilities, so the switch has to swap
// them.
func TestSwitchingTheHTTPServiceToV4FlashTakesItsCapabilities(t *testing.T) {
	f := newHTTPTTSServer(t)
	s := httpService(f, HTTPTTSConfig{Model: "bulbul:v3"})

	delta := &HTTPTTSSettings{Pitch: settings.Set(0.3), Loudness: settings.Set(1.5)}
	delta.Model = settings.Set(v4Flash)
	updateHTTPSettings(t, s, delta)
	synthesizeHTTP(t, s)
	payload := f.lastPayload(t)

	if payload["model"] != v4Flash {
		t.Errorf("model = %v, want %s", payload["model"], v4Flash)
	}
	if payload["speaker"] != "shubh_en_narration_gentle" {
		t.Errorf("speaker = %v, want the v4-flash default", payload["speaker"])
	}
	if payload["pitch"] != 0.3 {
		t.Errorf("pitch = %v, want 0.3", payload["pitch"])
	}
	if _, present := payload["temperature"]; present {
		t.Error("temperature was sent to v4-flash")
	}
}

// An unaliased key lands in Extra, which never reaches the API.
func TestHTTPSettingsDictsAcceptTheSarvamAPINames(t *testing.T) {
	for _, tc := range []struct {
		key, field, value string
	}{
		{"voice_id", "voice", "priya"},
		{"speaker", "voice", "priya"},
		{"target_language_code", "language", "hi-IN"},
	} {
		t.Run(tc.key, func(t *testing.T) {
			delta := &HTTPTTSSettings{}
			if err := settings.FromMap(delta, map[string]any{tc.key: tc.value}); err != nil {
				t.Fatal(err)
			}
			if got, _ := settings.Get(delta, tc.field); got != tc.value {
				t.Errorf("%s = %v, want %q", tc.field, got, tc.value)
			}
			if len(delta.Extra) != 0 {
				t.Errorf("extra = %v, want nothing", delta.Extra)
			}
		})
	}
}

func TestAVoiceFromASettingsDictReachesTheHTTPPayload(t *testing.T) {
	for _, key := range []string{"voice_id", "speaker"} {
		t.Run(key, func(t *testing.T) {
			f := newHTTPTTSServer(t)
			s := httpService(f, HTTPTTSConfig{Model: "bulbul:v3"})

			delta := &HTTPTTSSettings{}
			if err := settings.FromMap(delta, map[string]any{key: "priya"}); err != nil {
				t.Fatal(err)
			}
			updateHTTPSettings(t, s, delta)
			synthesizeHTTP(t, s)

			if got := f.lastPayload(t)["speaker"]; got != "priya" {
				t.Errorf("speaker = %v, want priya", got)
			}
		})
	}
}

func TestAnUnsupportedModelUpdateIsIgnoredByTheHTTPService(t *testing.T) {
	f := newHTTPTTSServer(t)
	s := httpService(f, HTTPTTSConfig{Model: "bulbul:v3"})

	delta := &HTTPTTSSettings{}
	delta.Model = settings.Set("bulbul:v9")
	changed := updateHTTPSettings(t, s, delta)

	if model, _ := s.live.Model.Value(); model != "bulbul:v3" {
		t.Errorf("model = %q, want it left at bulbul:v3", model)
	}
	if changed.Has("model") {
		t.Error("the refused model is reported as changed")
	}
}

// The request goes to the text-to-speech path, authorized with the
// subscription key, and the audio is yielded with its WAV header stripped.
func TestHTTPRequestAndAudio(t *testing.T) {
	f := newHTTPTTSServer(t)
	pcm := make([]byte, 64)
	for i := range pcm {
		pcm[i] = byte(i)
	}
	wav := append([]byte("RIFF"), make([]byte, wavHeaderSize-4)...)
	f.reply = `{"audios": ["` + base64.StdEncoding.EncodeToString(append(wav, pcm...)) + `"]}`
	s := httpService(f, HTTPTTSConfig{Model: "bulbul:v3"})

	out := synthesizeHTTP(t, s)

	if f.paths[0] != "/text-to-speech" {
		t.Errorf("path = %q, want /text-to-speech", f.paths[0])
	}
	if got := f.headers[0].Get("api-subscription-key"); got != "test-key" {
		t.Errorf("api-subscription-key = %q, want the key", got)
	}
	payload := f.lastPayload(t)
	if payload["speech_sample_rate"] != float64(24000) || payload["target_language_code"] != "en-IN" ||
		payload["pace"] != 1.0 || payload["temperature"] != 0.6 || payload["enable_preprocessing"] != true {
		t.Errorf("payload = %v, want the bulbul:v3 defaults", payload)
	}
	if len(out) != 1 {
		t.Fatalf("yielded %d frames, want one", len(out))
	}
	audio, ok := out[0].(*frames.TTSAudioRawFrame)
	if !ok || string(audio.Audio) != string(pcm) || audio.SampleRate != 24000 {
		t.Errorf("yielded %v, want the audio without its header at 24 kHz", out[0])
	}
}

// A refused request and an answer with no audio are yielded as errors.
func TestHTTPErrorsAreYielded(t *testing.T) {
	for name, tc := range map[string]struct {
		status int
		reply  string
		want   string
	}{
		"refused":  {http.StatusBadRequest, "bad text", "Sarvam API error: bad text"},
		"no audio": {http.StatusOK, `{"audios": []}`, "No audio data received"},
	} {
		t.Run(name, func(t *testing.T) {
			f := newHTTPTTSServer(t)
			f.status, f.reply = tc.status, tc.reply
			out := synthesizeHTTP(t, httpService(f, HTTPTTSConfig{}))
			if len(out) != 1 {
				t.Fatalf("yielded %d frames, want one", len(out))
			}
			if ef, ok := out[0].(*frames.ErrorFrame); !ok || ef.Error != tc.want {
				t.Errorf("yielded %v, want the error %q", out[0], tc.want)
			}
		})
	}
}
