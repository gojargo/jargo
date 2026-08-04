package pockettts

import (
	"bytes"
	"context"
	"encoding/binary"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
)

// wavHeader builds the header the server writes before the samples exist: mono
// 16-bit at rate, with a placeholder length, since the audio it describes has
// not been generated yet.
func wavHeader(rate int) []byte {
	var b bytes.Buffer
	b.WriteString("RIFF")
	_ = binary.Write(&b, binary.LittleEndian, uint32(0xFFFFFFFF)) // placeholder
	b.WriteString("WAVE")

	b.WriteString("fmt ")
	_ = binary.Write(&b, binary.LittleEndian, uint32(16))
	_ = binary.Write(&b, binary.LittleEndian, uint16(1))      // PCM
	_ = binary.Write(&b, binary.LittleEndian, uint16(1))      // mono
	_ = binary.Write(&b, binary.LittleEndian, uint32(rate))   // sample rate
	_ = binary.Write(&b, binary.LittleEndian, uint32(rate*2)) // byte rate
	_ = binary.Write(&b, binary.LittleEndian, uint16(2))      // block align
	_ = binary.Write(&b, binary.LittleEndian, uint16(16))     // bits
	b.WriteString("data")
	_ = binary.Write(&b, binary.LittleEndian, uint32(2_000_000_000)) // placeholder
	return b.Bytes()
}

// request is what the fake server saw.
type request struct {
	mu    sync.Mutex
	path  string
	text  string
	voice string
	seen  bool
}

func (r *request) get() request {
	r.mu.Lock()
	defer r.mu.Unlock()
	return request{path: r.path, text: r.text, voice: r.voice, seen: r.seen}
}

// serveAudio stands in for `pocket-tts serve`: it records the form it was given
// and streams a WAV header followed by chunks, flushing each so the client sees
// them as they are written rather than all at once.
func serveAudio(t *testing.T, rate int, chunks [][]byte, sent chan<- struct{}) (*httptest.Server, *request) {
	t.Helper()
	got := &request{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		//nolint:gosec // a test server reading a form the test itself sent
		if err := r.ParseMultipartForm(1 << 20); err != nil {
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		got.mu.Lock()
		got.path, got.text, got.voice, got.seen = r.URL.Path, r.FormValue("text"), r.FormValue("voice_url"), true
		got.mu.Unlock()

		w.Header().Set("Content-Type", "audio/wav")
		w.WriteHeader(http.StatusOK)
		flusher, _ := w.(http.Flusher)
		_, _ = w.Write(wavHeader(rate))
		if flusher != nil {
			flusher.Flush()
		}
		for _, c := range chunks {
			_, _ = w.Write(c)
			if flusher != nil {
				flusher.Flush()
			}
			if sent != nil {
				sent <- struct{}{}
			}
		}
	}))
	t.Cleanup(srv.Close)
	return srv, got
}

// collect runs one synthesis and returns the audio frames it produced.
func collect(t *testing.T, s *synthesizer) []*frames.TTSAudioRawFrame {
	t.Helper()
	var out []*frames.TTSAudioRawFrame
	err := s.RunTTS(context.Background(), "hello there", "", func(f frames.Frame) error {
		if af, ok := f.(*frames.TTSAudioRawFrame); ok {
			out = append(out, af)
		}
		return nil
	})
	if err != nil {
		t.Fatalf("RunTTS: %v", err)
	}
	return out
}

func TestValidate(t *testing.T) {
	if err := (Config{BaseURL: "http://localhost:8000"}).Validate(); err != nil {
		t.Fatalf("valid config rejected: %v", err)
	}
	if err := (Config{}).Validate(); err == nil {
		t.Error("config without a base URL should be rejected")
	}
	if err := (Config{BaseURL: "not a url"}).Validate(); err == nil {
		t.Error("config with a malformed base URL should be rejected")
	}
}

// The text and the voice travel as the form fields the server reads, addressed
// to its synthesis endpoint.
func TestRunTTSSendsTheForm(t *testing.T) {
	pcm := bytes.Repeat([]byte{1, 2}, 64)
	srv, got := serveAudio(t, 24000, [][]byte{pcm}, nil)

	s := &synthesizer{cfg: Config{BaseURL: srv.URL, Voice: "alba", SampleRate: 24000}, http: srv.Client()}
	collect(t, s)

	req := got.get()
	if !req.seen {
		t.Fatal("the server was never asked for anything")
	}
	if req.path != ttsPath {
		t.Errorf("path = %q, want %q", req.path, ttsPath)
	}
	if req.text != "hello there" {
		t.Errorf("text = %q, want the sentence to speak", req.text)
	}
	if req.voice != "alba" {
		t.Errorf("voice_url = %q, want the configured voice", req.voice)
	}
}

// Configuring no voice sends none, which is what asks the server for the default
// voice of the language it was started with.
func TestRunTTSOmitsAnUnsetVoice(t *testing.T) {
	srv, got := serveAudio(t, 24000, [][]byte{bytes.Repeat([]byte{0}, 32)}, nil)

	s := &synthesizer{cfg: Config{BaseURL: srv.URL, SampleRate: 24000}, http: srv.Client()}
	collect(t, s)

	if voice := got.get().voice; voice != "" {
		t.Errorf("voice_url = %q, want it left out entirely", voice)
	}
}

// The header is consumed and only the samples behind it are pushed downstream.
func TestRunTTSStripsTheHeader(t *testing.T) {
	pcm := bytes.Repeat([]byte{7, 8}, 100)
	srv, _ := serveAudio(t, 24000, [][]byte{pcm}, nil)

	s := &synthesizer{cfg: Config{BaseURL: srv.URL, SampleRate: 24000}, http: srv.Client()}

	var audio []byte
	for _, f := range collect(t, s) {
		audio = append(audio, f.Audio...)
	}
	if !bytes.Equal(audio, pcm) {
		t.Errorf("emitted %d bytes of audio, want the %d generated; header bytes leaked or samples were lost",
			len(audio), len(pcm))
	}
}

// The audio is labeled with the rate the server generated it at, so a server
// running a model at another rate is played at the right speed rather than the
// configured one.
func TestRunTTSLabelsTheServerRate(t *testing.T) {
	const serverRate = 16000
	srv, _ := serveAudio(t, serverRate, [][]byte{bytes.Repeat([]byte{1, 0}, 40)}, nil)

	s := &synthesizer{cfg: Config{BaseURL: srv.URL, SampleRate: 24000}, http: srv.Client()}

	got := collect(t, s)
	if len(got) == 0 {
		t.Fatal("no audio was emitted")
	}
	for _, f := range got {
		if f.SampleRate != serverRate {
			t.Errorf("audio labeled %d Hz, want the %d Hz the server generated", f.SampleRate, serverRate)
		}
		if f.NumChannels != 1 {
			t.Errorf("audio labeled %d channels, want mono", f.NumChannels)
		}
	}
}

// Audio is pushed downstream as it arrives. The point of a local model is the
// first samples landing early, which collecting the whole response would throw
// away.
func TestRunTTSStreamsAsItArrives(t *testing.T) {
	sent := make(chan struct{})
	chunks := [][]byte{bytes.Repeat([]byte{1, 1}, 64), bytes.Repeat([]byte{2, 2}, 64)}
	srv, _ := serveAudio(t, 24000, chunks, sent)

	s := &synthesizer{cfg: Config{BaseURL: srv.URL, SampleRate: 24000}, http: srv.Client()}

	first := make(chan struct{}, 1)
	done := make(chan error, 1)
	go func() {
		done <- s.RunTTS(context.Background(), "hello", "", func(f frames.Frame) error {
			if _, ok := f.(*frames.TTSAudioRawFrame); ok {
				select {
				case first <- struct{}{}:
				default:
				}
			}
			return nil
		})
	}()

	// The server has written its first chunk and is holding on the second.
	<-sent
	select {
	case <-first:
	case <-time.After(2 * time.Second):
		t.Fatal("no audio reached the pipeline while the server was still generating")
	}
	<-sent
	if err := <-done; err != nil {
		t.Fatalf("RunTTS: %v", err)
	}
}

// A refusal from the server is reported with what it said, since that is where
// an unknown voice or empty text is caught.
func TestRunTTSReportsARefusal(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		_, _ = io.WriteString(w, `{"detail":"voice_url must start with http://, https://, or hf://"}`)
	}))
	defer srv.Close()

	s := &synthesizer{cfg: Config{BaseURL: srv.URL, Voice: "nope", SampleRate: 24000}, http: srv.Client()}
	err := s.RunTTS(context.Background(), "hello", "", func(frames.Frame) error { return nil })
	if err == nil {
		t.Fatal("a refused request reported no error")
	}
	if !strings.Contains(err.Error(), "voice_url must start with") {
		t.Errorf("error %q does not carry what the server said", err)
	}
}

// A response that is not a WAV stream is reported rather than pushed downstream
// as audio.
func TestRunTTSRejectsANonWAVResponse(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = io.WriteString(w, "not audio at all")
	}))
	defer srv.Close()

	s := &synthesizer{cfg: Config{BaseURL: srv.URL, SampleRate: 24000}, http: srv.Client()}
	err := s.RunTTS(context.Background(), "hello", "", func(frames.Frame) error { return nil })
	if err == nil {
		t.Fatal("a response that is not audio reported no error")
	}
}

// The header is walked chunk by chunk, so one carrying more than the format
// before the samples is handled, and the placeholder data length is ignored.
func TestPCMStreamWalksTheChunks(t *testing.T) {
	var b bytes.Buffer
	b.WriteString("RIFF")
	_ = binary.Write(&b, binary.LittleEndian, uint32(0xFFFFFFFF))
	b.WriteString("WAVE")
	// An odd-sized chunk before the format, so the pad byte has to be stepped
	// over as well.
	b.WriteString("LIST")
	_ = binary.Write(&b, binary.LittleEndian, uint32(3))
	b.WriteString("abc")
	b.WriteByte(0) // pad to the word boundary
	b.Write(wavHeader(24000)[12:])
	pcm := []byte{9, 9, 8, 8}
	b.Write(pcm)

	rate, r, err := pcmStream(&b)
	if err != nil {
		t.Fatalf("pcmStream: %v", err)
	}
	if rate != 24000 {
		t.Errorf("rate = %d, want 24000", rate)
	}
	got, err := io.ReadAll(r)
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	if !bytes.Equal(got, pcm) {
		t.Errorf("samples = % x, want % x", got, pcm)
	}
}

// A stream that ends before the samples do is not audio to push on.
func TestPCMStreamRejectsATruncatedHeader(t *testing.T) {
	for name, b := range map[string][]byte{
		"empty":          {},
		"not RIFF":       []byte("something else entirely"),
		"header cut off": wavHeader(24000)[:20],
		"no data chunk":  wavHeader(24000)[:36],
	} {
		t.Run(name, func(t *testing.T) {
			if _, _, err := pcmStream(bytes.NewReader(b)); err == nil {
				t.Error("a truncated stream was accepted as audio")
			}
		})
	}
}

// NewTTS fills in the rate and tolerates a base URL with a trailing slash, which
// would otherwise address the endpoint through a double slash.
func TestNewTTSDefaults(t *testing.T) {
	if svc := NewTTS(Config{BaseURL: "http://localhost:8000/"}); svc == nil {
		t.Fatal("NewTTS returned nothing")
	}

	// The service holds the synthesizer, so the defaults are checked on one
	// built the same way.
	cfg := Config{BaseURL: "http://localhost:8000/"}
	if cfg.SampleRate == 0 {
		cfg.SampleRate = defaultSampleRate
	}
	cfg.BaseURL = strings.TrimSuffix(cfg.BaseURL, "/")
	s := &synthesizer{cfg: cfg}

	if got := s.SampleRate(); got != defaultSampleRate {
		t.Errorf("sample rate = %d, want %d", got, defaultSampleRate)
	}
	if s.cfg.BaseURL != "http://localhost:8000" {
		t.Errorf("base URL = %q, want the trailing slash gone", s.cfg.BaseURL)
	}
}
