package piper

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"io"
	"maps"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/service/settings"
)

// Tests for the HTTP service, against a fake Piper HTTP server.

// piperServer is a fake Piper HTTP server recording each request body and
// answering with respond.
type piperServer struct {
	url string

	mu     sync.Mutex
	bodies []map[string]any
	header []http.Header
}

func newPiperServer(t *testing.T, path string, respond func(w http.ResponseWriter)) *piperServer {
	t.Helper()
	p := &piperServer{}
	mux := http.NewServeMux()
	mux.HandleFunc("POST "+path, func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		body := map[string]any{}
		_ = json.Unmarshal(raw, &body)
		p.mu.Lock()
		p.bodies = append(p.bodies, body)
		p.header = append(p.header, r.Header.Clone())
		p.mu.Unlock()
		respond(w)
	})
	srv := httptest.NewServer(mux)
	t.Cleanup(srv.Close)
	p.url = srv.URL + path
	return p
}

// A successful request streams its chunked audio between the started and
// stopped frames, with only audio and text between them, at the service rate.
func TestRunPiperTTSSuccess(t *testing.T) {
	const chunkSize = 24000
	srv := newPiperServer(t, "/", func(w http.ResponseWriter) {
		w.Header().Set("Content-Type", "audio/raw")
		_, _ = w.Write(bytes.Repeat([]byte{0x00, 0x01, 0x02, 0x03}, chunkSize))
		if f, ok := w.(http.Flusher); ok {
			f.Flush()
		}
		time.Sleep(10 * time.Millisecond) // simulate an async chunk delay
		_, _ = w.Write(bytes.Repeat([]byte{0x04, 0x05, 0x06, 0x07}, chunkSize))
	})
	// The fake listens on "/", which is posted to with the slash removed.
	b := NewHTTPTTS(HTTPConfig{BaseURL: strings.TrimSuffix(srv.url, "/"), SampleRate: 24000})

	down, _ := runTest(t, b, speakFrame("Hello world.", true))

	for _, name := range []string{"AggregatedTextFrame", "TTSStartedFrame", "TTSStoppedFrame", "TTSTextFrame"} {
		if !hasName(down, name) {
			t.Errorf("no %s in %v", name, frameNames(down))
		}
	}
	started := indexOf[*frames.TTSStartedFrame](down)
	stopped := indexOf[*frames.TTSStoppedFrame](down)
	text := indexOf[*frames.TTSTextFrame](down)
	if started >= text || text >= stopped {
		t.Errorf("frames = %v, want TTSStartedFrame < TTSTextFrame < TTSStoppedFrame", frameNames(down))
	}
	// TTSSpeakFrame appends to the context, so the service commits the spoken
	// text with an LLMAssistantPushAggregationFrame within the utterance.
	for i := started + 1; i < stopped; i++ {
		switch down[i].(type) {
		case *frames.TTSAudioRawFrame, *frames.TTSTextFrame, *frames.LLMAssistantPushAggregationFrame:
		default:
			t.Errorf("unexpected frame between started and stopped: %s", down[i].Name())
		}
	}
	audio := audioFrames(down)
	if len(audio) == 0 {
		t.Fatal("expected at least one audio frame")
	}
	var total int
	for _, a := range audio {
		total += len(a.Audio)
		if a.SampleRate != 24000 {
			t.Errorf("audio at %d Hz, want the service's 24000", a.SampleRate)
		}
	}
	if total != 8*chunkSize {
		t.Errorf("streamed %d bytes, want %d", total, 8*chunkSize)
	}
}

// A non-200 answer is reported upstream with its status, and the utterance
// still closes with no audio.
func TestRunPiperTTSError(t *testing.T) {
	srv := newPiperServer(t, "/", func(w http.ResponseWriter) {
		http.Error(w, "Not found", http.StatusNotFound)
	})
	b := NewHTTPTTS(HTTPConfig{BaseURL: strings.TrimSuffix(srv.url, "/"), SampleRate: 24000})

	down, up := runTest(t, b, speakFrame("Error case.", false))

	var downNames []string
	for _, f := range down {
		switch f.(type) {
		case *frames.AggregatedTextFrame, *frames.TTSStartedFrame, *frames.TTSStoppedFrame:
			downNames = append(downNames, strings.Split(f.Name(), "#")[0])
		}
	}
	if !reflect.DeepEqual(downNames, []string{"AggregatedTextFrame", "TTSStartedFrame", "TTSStoppedFrame"}) {
		t.Errorf("down = %v, want aggregated text, started and stopped", frameNames(down))
	}
	if len(audioFrames(down)) != 0 {
		t.Error("audio was pushed for a failed request")
	}
	// The 404, then the context completing with no audio.
	errs := errorsOf(up)
	if len(errs) != 2 {
		t.Fatalf("errors = %q, want two", errs)
	}
	if !strings.Contains(errs[0], "status: 404") {
		t.Errorf("error = %q, want the 404 reported", errs[0])
	}
}

// The synthesis settings are sent with the text and the voice, and only when
// they are set.
func TestHTTPPiperSendsSynthesisSettingsOnlyWhenSet(t *testing.T) {
	speaker, length, noise, noiseW := 2, 1.3, 0.5, 0.6
	for name, tc := range map[string]struct {
		cfg  HTTPConfig
		want map[string]any
	}{
		"set": {
			HTTPConfig{SpeakerID: &speaker, LengthScale: &length, NoiseScale: &noise, NoiseWScale: &noiseW},
			map[string]any{"speaker_id": 2.0, "length_scale": 1.3, "noise_scale": 0.5, "noise_w_scale": 0.6},
		},
		"unset": {HTTPConfig{}, map[string]any{}},
	} {
		t.Run(name, func(t *testing.T) {
			srv := newPiperServer(t, "/synthesize", func(w http.ResponseWriter) {
				_, _ = w.Write(make([]byte, 4800))
			})
			cfg := tc.cfg
			cfg.BaseURL, cfg.SampleRate, cfg.Voice = srv.url, 24000, "en_US-ryan-high"

			runTest(t, NewHTTPTTS(cfg), speakFrame("Hello.", false))

			want := map[string]any{"text": "Hello.", "voice": "en_US-ryan-high"}
			maps.Copy(want, tc.want)
			if !reflect.DeepEqual(srv.bodies, []map[string]any{want}) {
				t.Errorf("bodies = %v, want [%v]", srv.bodies, want)
			}
			if ct := srv.header[0].Get("Content-Type"); ct != "application/json" {
				t.Errorf("Content-Type = %q", ct)
			}
		})
	}
}

// With no voice configured, the request leaves the choice to the server.
func TestHTTPPiperSendsANullVoiceWhenNoneIsSet(t *testing.T) {
	srv := newPiperServer(t, "/synthesize", func(w http.ResponseWriter) { _, _ = w.Write(make([]byte, 480)) })
	runTest(t, NewHTTPTTS(HTTPConfig{BaseURL: srv.url}), speakFrame("Hi.", false))

	if len(srv.bodies) != 1 {
		t.Fatalf("requests = %d, want 1", len(srv.bodies))
	}
	if v, ok := srv.bodies[0]["voice"]; !ok || v != nil {
		t.Errorf("voice = %v (sent %v), want null", v, ok)
	}
}

// A voice changed while running is sent with the next request.
func TestHTTPPiperSendsAnUpdatedVoice(t *testing.T) {
	srv := newPiperServer(t, "/synthesize", func(w http.ResponseWriter) { _, _ = w.Write(make([]byte, 480)) })
	b := NewHTTPTTS(HTTPConfig{BaseURL: srv.url, Voice: "en_US-ryan-high"})
	delta := &HTTPTTSSettings{}
	delta.Voice = settings.Set("en_US-lessac-medium")

	runTest(t, b,
		speakFrame("One.", false),
		frames.NewTTSUpdateSettingsFrame(delta),
		speakFrame("Two.", false),
	)

	var voices []any
	for _, body := range srv.bodies {
		voices = append(voices, body["voice"])
	}
	if !reflect.DeepEqual(voices, []any{"en_US-ryan-high", "en_US-lessac-medium"}) {
		t.Errorf("voices = %v, want the update applied to the second request", voices)
	}
}

// The WAV header the server opens its answer with is removed, and the audio is
// labeled with the rate it gives.
func TestHTTPPiperStripsTheWAVHeader(t *testing.T) {
	pcm := bytes.Repeat([]byte{0x10, 0x20}, 3000)
	srv := newPiperServer(t, "/synthesize", func(w http.ResponseWriter) {
		w.Header().Set("Content-Type", "audio/wav")
		_, _ = w.Write(wav(pcm, 22050))
	})
	down, _ := runTest(t, NewHTTPTTS(HTTPConfig{BaseURL: srv.url, SampleRate: 22050}), speakFrame("Hi.", false))

	var got []byte
	for _, a := range audioFrames(down) {
		got = append(got, a.Audio...)
		if a.SampleRate != 22050 {
			t.Errorf("audio at %d Hz, want the header's 22050", a.SampleRate)
		}
	}
	if !bytes.Equal(got, pcm) {
		t.Errorf("audio = %d bytes, want the %d PCM bytes without the header", len(got), len(pcm))
	}
}

// streamWAV labels audio with the header's rate, carries an odd byte into the
// next chunk, and pads a last odd byte.
func TestStreamWAV(t *testing.T) {
	var out []*frames.TTSAudioRawFrame
	yield := func(f frames.Frame) error {
		if a, ok := f.(*frames.TTSAudioRawFrame); ok {
			out = append(out, a)
		}
		return nil
	}
	body := append(wav(nil, 16000), 1, 2, 3)
	if err := streamWAV(&slowReader{data: body, step: 45}, 4800, 24000, yield); err != nil {
		t.Fatalf("streamWAV: %v", err)
	}
	var got []byte
	for _, f := range out {
		if len(f.Audio)%2 != 0 {
			t.Errorf("frame of %d bytes, want whole samples", len(f.Audio))
		}
		if f.SampleRate != 16000 {
			t.Errorf("rate = %d, want the header's 16000", f.SampleRate)
		}
		got = append(got, f.Audio...)
	}
	if !bytes.Equal(got, []byte{1, 2, 3, 0}) {
		t.Errorf("audio = %v, want the samples padded", got)
	}
}

// A trailing slash on the base URL is removed.
func TestHTTPPiperRemovesATrailingSlash(t *testing.T) {
	_, s := newHTTPTTS(HTTPConfig{BaseURL: "http://localhost:5000/synthesize/"})
	if s.cfg.BaseURL != "http://localhost:5000/synthesize" {
		t.Errorf("base URL = %q", s.cfg.BaseURL)
	}
	if s.SampleRate() != defaultHTTPSampleRate {
		t.Errorf("sample rate = %d, want %d", s.SampleRate(), defaultHTTPSampleRate)
	}
}

func TestHTTPConfigValidate(t *testing.T) {
	if err := (HTTPConfig{}).Validate(); err == nil {
		t.Error("Validate() with no base URL: want an error")
	}
	if err := (HTTPConfig{BaseURL: "http://localhost:5000/synthesize"}).Validate(); err != nil {
		t.Errorf("Validate(): %v", err)
	}
}

// wav is pcm in a canonical 44-byte 16-bit mono WAV file at rate.
func wav(pcm []byte, rate int) []byte {
	var b bytes.Buffer
	put32 := func(v uint32) { _ = binary.Write(&b, binary.LittleEndian, v) }
	put16 := func(v uint16) { _ = binary.Write(&b, binary.LittleEndian, v) }
	b.WriteString("RIFF")
	put32(uint32(36 + len(pcm))) //nolint:gosec // a test file
	b.WriteString("WAVEfmt ")
	put32(16)
	put16(1)
	put16(1)
	put32(uint32(rate))     //nolint:gosec // a test rate
	put32(uint32(rate * 2)) //nolint:gosec // a test rate
	put16(2)
	put16(16)
	b.WriteString("data")
	put32(uint32(len(pcm))) //nolint:gosec // a test file
	b.Write(pcm)
	return b.Bytes()
}

// slowReader hands its data over step bytes at a time.
type slowReader struct {
	data []byte
	step int
}

func (r *slowReader) Read(p []byte) (int, error) {
	if len(r.data) == 0 {
		return 0, io.EOF
	}
	n := copy(p[:min(len(p), r.step)], r.data)
	r.data = r.data[n:]
	return n, nil
}

// hasName reports whether a frame of the named type is among fs.
func hasName(fs []frames.Frame, name string) bool {
	for _, f := range fs {
		if strings.HasPrefix(f.Name(), name+"#") {
			return true
		}
	}
	return false
}
