package nvidia

import (
	"context"
	"errors"
	"io"
	"net"
	"strings"
	"testing"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/language"
	"github.com/gojargo/jargo/provider/nvidia/internal/rivapb"
	"github.com/gojargo/jargo/service/tts"
	"google.golang.org/grpc"
	"google.golang.org/grpc/metadata"
)

func TestTTSWithDefaults(t *testing.T) {
	got := TTSConfig{}.withTTSDefaults()
	if got.Server != defaultTTSServer {
		t.Errorf("Server = %q, want %q", got.Server, defaultTTSServer)
	}
	if got.FunctionID != defaultTTSFunctionID {
		t.Errorf("FunctionID = %q, want the hosted Magpie function", got.FunctionID)
	}
	if got.Model != defaultTTSModel {
		t.Errorf("Model = %q, want %q", got.Model, defaultTTSModel)
	}
	if got.Voice != defaultTTSVoice {
		t.Errorf("Voice = %q, want %q", got.Voice, defaultTTSVoice)
	}
	if got.Language != language.EnglishUS {
		t.Errorf("Language = %q, want %q", got.Language, language.EnglishUS)
	}
	if got.SampleRate != defaultTTSSampleRate {
		t.Errorf("SampleRate = %d, want %d", got.SampleRate, defaultTTSSampleRate)
	}
}

// TestTTSSynthesizerDescription checks the rate it emits and the model and
// voice the synthesis is billed against.
func TestTTSSynthesizerDescription(t *testing.T) {
	s := &ttsSynthesizer{cfg: TTSConfig{Voice: "Magpie-Multilingual.FR-FR.Louise"}.withTTSDefaults()}
	if got := s.SampleRate(); got != defaultTTSSampleRate {
		t.Errorf("SampleRate() = %d, want %d", got, defaultTTSSampleRate)
	}
	meta := s.Metadata()
	if meta.Model != defaultTTSModel {
		t.Errorf("Metadata().Model = %q, want %q", meta.Model, defaultTTSModel)
	}
	if meta.VoiceID != "Magpie-Multilingual.FR-FR.Louise" {
		t.Errorf("Metadata().VoiceID = %q, want the configured voice", meta.VoiceID)
	}
}

func TestNewTTSName(t *testing.T) {
	// processor.New appends a per-instance counter (e.g. "NvidiaTTS#1").
	if name := NewTTS(TTSConfig{}).Name(); !strings.HasPrefix(name, "NvidiaTTS") {
		t.Errorf("service name = %q, want NvidiaTTS prefix", name)
	}
}

func TestTTSConfigValidate(t *testing.T) {
	cases := []struct {
		name  string
		cfg   TTSConfig
		valid bool
	}{
		{"empty is usable for a local NIM", TTSConfig{}, true},
		{"zero shot with a prompt", TTSConfig{ZeroShot: &ZeroShot{AudioPrompt: []byte{1}}}, true},
		{"zero shot without a prompt", TTSConfig{ZeroShot: &ZeroShot{}}, false},
		{
			"zero shot with a supported encoding",
			TTSConfig{ZeroShot: &ZeroShot{AudioPrompt: []byte{1}, Encoding: "oggopus"}},
			true,
		},
		{
			"zero shot with an unsupported encoding",
			TTSConfig{ZeroShot: &ZeroShot{AudioPrompt: []byte{1}, Encoding: "flac"}},
			false,
		},
		{
			"zero shot quality above range",
			TTSConfig{ZeroShot: &ZeroShot{AudioPrompt: []byte{1}, Quality: 41}},
			false,
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			err := c.cfg.Validate()
			if c.valid && err != nil {
				t.Errorf("Validate() = %v, want nil", err)
			}
			if !c.valid && err == nil {
				t.Error("Validate() = nil, want an error")
			}
		})
	}
}

// TestCustomDictionary checks Riva's pronunciation format: comma-separated
// pairs, grapheme and phoneme separated by two spaces, ordered stably.
func TestCustomDictionary(t *testing.T) {
	cfg := TTSConfig{CustomDictionary: map[string]string{"NVIDIA": "ɛn.vɪ.diː.ʌ", "AI": "eɪ.aɪ"}}
	if got, want := cfg.customDictionary(), "AI  eɪ.aɪ,NVIDIA  ɛn.vɪ.diː.ʌ"; got != want {
		t.Errorf("customDictionary() = %q, want %q", got, want)
	}
	if got := (TTSConfig{}).customDictionary(); got != "" {
		t.Errorf("customDictionary() with no entries = %q, want empty", got)
	}
}

// TestZeroShotData checks the prompt defaults and encoding mapping.
func TestZeroShotData(t *testing.T) {
	if got := (TTSConfig{}).zeroShotData(); got != nil {
		t.Errorf("zeroShotData() with no prompt = %v, want nil", got)
	}

	cfg := TTSConfig{ZeroShot: &ZeroShot{AudioPrompt: []byte{1, 2}, Transcript: "hi"}}
	got := cfg.zeroShotData()
	if got.GetQuality() != defaultZeroShotQuality {
		t.Errorf("quality = %d, want the default %d", got.GetQuality(), defaultZeroShotQuality)
	}
	if got.GetEncoding() != rivapb.AudioEncoding_LINEAR_PCM {
		t.Errorf("encoding = %v, want LINEAR_PCM by default", got.GetEncoding())
	}
	if got.GetTranscript() != "hi" {
		t.Errorf("transcript = %q, want hi", got.GetTranscript())
	}

	cfg.ZeroShot.Encoding = "oggopus"
	cfg.ZeroShot.Quality = 5
	got = cfg.zeroShotData()
	if got.GetEncoding() != rivapb.AudioEncoding_OGGOPUS {
		t.Errorf("encoding = %v, want OGGOPUS", got.GetEncoding())
	}
	if got.GetQuality() != 5 {
		t.Errorf("quality = %d, want 5", got.GetQuality())
	}
}

// TestChunkText checks the request-length splitting: short text passes through,
// long text breaks on whitespace, and a single oversized word breaks mid-word.
func TestChunkText(t *testing.T) {
	t.Run("short text is one chunk", func(t *testing.T) {
		got := chunkText("hello there", 200)
		if len(got) != 1 || got[0] != "hello there" {
			t.Errorf("chunkText = %q, want the whole sentence", got)
		}
	})

	t.Run("empty text yields nothing", func(t *testing.T) {
		if got := chunkText("   ", 200); got != nil {
			t.Errorf("chunkText = %q, want nil", got)
		}
	})

	t.Run("breaks on whitespace", func(t *testing.T) {
		got := chunkText("aaaa bbbb cccc dddd", 10)
		for _, c := range got {
			if len(c) > 10 {
				t.Errorf("chunk %q is %d bytes, want at most 10", c, len(c))
			}
			if strings.TrimSpace(c) != c {
				t.Errorf("chunk %q has surrounding whitespace", c)
			}
		}
		if strings.Join(got, " ") != "aaaa bbbb cccc dddd" {
			t.Errorf("chunks rejoin to %q, want the original text", strings.Join(got, " "))
		}
	})

	t.Run("breaks a word longer than the limit", func(t *testing.T) {
		got := chunkText("supercalifragilistic", 6)
		for _, c := range got {
			if len(c) > 6 {
				t.Errorf("chunk %q is %d bytes, want at most 6", c, len(c))
			}
		}
		if strings.Join(got, "") != "supercalifragilistic" {
			t.Errorf("chunks rejoin to %q, want the original word", strings.Join(got, ""))
		}
	})
}

// fakeTTS is an in-process Riva TTS server: it records the requests it receives
// and replies with one audio chunk per request.
type fakeTTS struct {
	rivapb.UnimplementedRivaSpeechSynthesisServer
	requests chan *rivapb.SynthesizeSpeechRequest
	md       chan metadata.MD
}

func (f *fakeTTS) SynthesizeOnline(stream rivapb.RivaSpeechSynthesis_SynthesizeOnlineServer) error {
	if md, ok := metadata.FromIncomingContext(stream.Context()); ok {
		select {
		case f.md <- md:
		default:
		}
	}
	for {
		req, err := stream.Recv()
		if errors.Is(err, io.EOF) {
			// The client half-closed: the sentence is complete.
			return nil
		}
		if err != nil {
			return err
		}
		f.requests <- req
		if err := stream.Send(&rivapb.SynthesizeSpeechResponse{Audio: []byte{9, 9}}); err != nil {
			return err
		}
	}
}

// startFakeTTS serves fakeTTS on a loopback port.
func startFakeTTS(t *testing.T) (*fakeTTS, string) {
	t.Helper()
	lis, err := (&net.ListenConfig{}).Listen(context.Background(), "tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	srv := grpc.NewServer()
	fake := &fakeTTS{
		requests: make(chan *rivapb.SynthesizeSpeechRequest, 8),
		md:       make(chan metadata.MD, 1),
	}
	rivapb.RegisterRivaSpeechSynthesisServer(srv, fake)
	go func() { _ = srv.Serve(lis) }()
	t.Cleanup(srv.Stop)
	return fake, lis.Addr().String()
}

// TestTTSSynthesize drives one sentence end to end and checks the request the
// server receives and the audio streamed back.
func TestTTSSynthesize(t *testing.T) {
	fake, addr := startFakeTTS(t)
	no := false
	s := &ttsSynthesizer{cfg: TTSConfig{
		Server:     addr,
		UseSSL:     &no,
		APIKey:     "test-key",
		Language:   language.French,
		SampleRate: 16000,
	}.withTTSDefaults()}
	t.Cleanup(func() { _ = s.Close() })

	var got []byte
	if err := runPCM(s, context.Background(), "hello there", func(pcm []byte) error {
		got = append(got, pcm...)
		return nil
	}); err != nil {
		t.Fatalf("Synthesize: %v", err)
	}

	if string(got) != "\x09\x09" {
		t.Errorf("emitted %q, want the server's audio", got)
	}

	req := <-fake.requests
	if req.GetText() != "hello there" {
		t.Errorf("text = %q, want the sentence", req.GetText())
	}
	if req.GetEncoding() != rivapb.AudioEncoding_LINEAR_PCM {
		t.Errorf("encoding = %v, want LINEAR_PCM", req.GetEncoding())
	}
	if req.GetSampleRateHz() != 16000 {
		t.Errorf("sample rate = %d, want 16000", req.GetSampleRateHz())
	}
	if req.GetLanguageCode() != "fr-FR" {
		t.Errorf("language = %q, want fr-FR", req.GetLanguageCode())
	}
	if req.GetVoiceName() != defaultTTSVoice {
		t.Errorf("voice = %q, want the default", req.GetVoiceName())
	}
	if req.GetZeroShotData() != nil {
		t.Errorf("zero_shot_data = %v, want it omitted", req.GetZeroShotData())
	}

	md := <-fake.md
	if got := md.Get("authorization"); len(got) != 1 || got[0] != "Bearer test-key" {
		t.Errorf("authorization = %v, want the Bearer key", got)
	}
	if got := md.Get("function-id"); len(got) != 1 || got[0] != defaultTTSFunctionID {
		t.Errorf("function-id = %v, want the default function", got)
	}
}

// TestTTSSynthesizeChunksLongText checks a sentence past the request-length
// limit is sent as several requests on the one stream, so the model can still
// generate it as a single utterance.
func TestTTSSynthesizeChunksLongText(t *testing.T) {
	fake, addr := startFakeTTS(t)
	no := false
	s := &ttsSynthesizer{cfg: TTSConfig{Server: addr, UseSSL: &no}.withTTSDefaults()}
	t.Cleanup(func() { _ = s.Close() })

	long := strings.TrimSpace(strings.Repeat("word ", 80)) // ~400 bytes
	var chunks int
	if err := runPCM(s, context.Background(), long, func([]byte) error {
		chunks++
		return nil
	}); err != nil {
		t.Fatalf("Synthesize: %v", err)
	}

	var sent []string
	for range chunks {
		req := <-fake.requests
		sent = append(sent, req.GetText())
		if len(req.GetText()) > maxTTSChunkLen {
			t.Errorf("request text is %d bytes, want at most %d", len(req.GetText()), maxTTSChunkLen)
		}
	}
	if len(sent) < 2 {
		t.Fatalf("sent %d requests, want the text split across several", len(sent))
	}
	if strings.Join(sent, " ") != long {
		t.Error("the requests do not rejoin to the original sentence")
	}
}

// TestTTSLocalNIMOmitsCredentials checks an unauthenticated local deployment
// sends neither header, including when the function id is explicitly cleared.
func TestTTSLocalNIMOmitsCredentials(t *testing.T) {
	fake, addr := startFakeTTS(t)
	no := false
	s := &ttsSynthesizer{cfg: TTSConfig{
		Server:     addr,
		UseSSL:     &no,
		FunctionID: "-",
	}.withTTSDefaults()}
	t.Cleanup(func() { _ = s.Close() })

	if err := runPCM(s, context.Background(), "hi", func([]byte) error { return nil }); err != nil {
		t.Fatalf("Synthesize: %v", err)
	}
	<-fake.requests

	md := <-fake.md
	if got := md.Get("function-id"); len(got) != 0 {
		t.Errorf("function-id = %v, want it omitted for a local NIM", got)
	}
	if got := md.Get("authorization"); len(got) != 0 {
		t.Errorf("authorization = %v, want it omitted with no API key", got)
	}
}

// TestTTSCloseIsIdempotent checks tearing the service down twice is safe, since
// Cleanup can run after an explicit close.
func TestTTSCloseIsIdempotent(t *testing.T) {
	_, addr := startFakeTTS(t)
	no := false
	s := &ttsSynthesizer{cfg: TTSConfig{Server: addr, UseSSL: &no}.withTTSDefaults()}

	if _, err := s.client(); err != nil {
		t.Fatalf("client: %v", err)
	}
	if err := s.Close(); err != nil {
		t.Fatalf("first Close: %v", err)
	}
	if err := s.Close(); err != nil {
		t.Fatalf("second Close: %v", err)
	}
}

// runPCM drives a synthesizer the way the base does, handing back the raw audio
// it yields.
func runPCM(s tts.Synthesizer, ctx context.Context, text string, emit func(pcm []byte) error) error {
	return s.RunTTS(ctx, text, "", func(f frames.Frame) error {
		if af, ok := f.(*frames.TTSAudioRawFrame); ok {
			return emit(af.Audio)
		}
		return nil
	})
}
