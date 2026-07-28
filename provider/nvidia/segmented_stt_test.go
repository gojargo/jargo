package nvidia

import (
	"context"
	"net"
	"strings"
	"testing"

	"github.com/gojargo/jargo/language"
	"github.com/gojargo/jargo/provider/nvidia/internal/rivapb"
	"google.golang.org/grpc"
	"google.golang.org/grpc/metadata"
)

func TestSegmentedWithDefaults(t *testing.T) {
	got := SegmentedSTTConfig{}.withSegmentedDefaults()
	if got.Server != defaultSTTServer {
		t.Errorf("Server = %q, want %q", got.Server, defaultSTTServer)
	}
	if got.Language != language.EnglishUS {
		t.Errorf("Language = %q, want %q", got.Language, language.EnglishUS)
	}
	if got.MaxAlternatives != 1 || got.AudioChannelCount != 1 {
		t.Errorf("alternatives/channels = %d/%d, want 1/1", got.MaxAlternatives, got.AudioChannelCount)
	}
}

func TestNewSegmentedSTTName(t *testing.T) {
	// processor.New appends a per-instance counter.
	if name := NewSegmentedSTT(SegmentedSTTConfig{}).Name(); !strings.HasPrefix(name, "NvidiaSegmentedSTT") {
		t.Errorf("service name = %q, want the NvidiaSegmentedSTT label", name)
	}
}

// fakeBatchASR is an in-process Riva server answering the batch RPC. It records
// the request and returns a transcript split across two results, as Riva does
// for longer audio.
type fakeBatchASR struct {
	rivapb.UnimplementedRivaSpeechRecognitionServer
	requests chan *rivapb.RecognizeRequest
	md       chan metadata.MD
}

func (f *fakeBatchASR) Recognize(
	ctx context.Context, req *rivapb.RecognizeRequest,
) (*rivapb.RecognizeResponse, error) {
	if md, ok := metadata.FromIncomingContext(ctx); ok {
		select {
		case f.md <- md:
		default:
		}
	}
	f.requests <- req
	return &rivapb.RecognizeResponse{Results: []*rivapb.SpeechRecognitionResult{
		{Alternatives: []*rivapb.SpeechRecognitionAlternative{{Transcript: "hello"}}},
		{Alternatives: []*rivapb.SpeechRecognitionAlternative{{Transcript: "there"}}},
		{}, // a result with no alternatives is skipped
	}}, nil
}

// startFakeBatchASR serves fakeBatchASR on a loopback port.
func startFakeBatchASR(t *testing.T) (*fakeBatchASR, string) {
	t.Helper()
	lis, err := (&net.ListenConfig{}).Listen(context.Background(), "tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	srv := grpc.NewServer()
	fake := &fakeBatchASR{
		requests: make(chan *rivapb.RecognizeRequest, 4),
		md:       make(chan metadata.MD, 1),
	}
	rivapb.RegisterRivaSpeechRecognitionServer(srv, fake)
	go func() { _ = srv.Serve(lis) }()
	t.Cleanup(srv.Stop)
	return fake, lis.Addr().String()
}

// TestSegmentedTranscribe checks one utterance goes out as a single batch
// request and its results are joined back into the transcript.
func TestSegmentedTranscribe(t *testing.T) {
	fake, addr := startFakeBatchASR(t)
	no := false
	tr := &segmentedTranscriber{cfg: SegmentedSTTConfig{
		Server:   addr,
		UseSSL:   &no,
		APIKey:   "test-key",
		Language: language.German,
		Model:    "canary",
	}.withSegmentedDefaults()}
	t.Cleanup(func() { _ = tr.Close() })

	text, err := tr.Transcribe(context.Background(), []byte{1, 2, 3, 4}, 16000)
	if err != nil {
		t.Fatalf("Transcribe: %v", err)
	}
	if text != "hello there" {
		t.Errorf("transcript = %q, want the results joined", text)
	}

	req := <-fake.requests
	if string(req.GetAudio()) != "\x01\x02\x03\x04" {
		t.Errorf("audio = %q, want the whole segment", req.GetAudio())
	}
	cfg := req.GetConfig()
	if cfg.GetEncoding() != rivapb.AudioEncoding_LINEAR_PCM {
		t.Errorf("encoding = %v, want LINEAR_PCM", cfg.GetEncoding())
	}
	if cfg.GetSampleRateHertz() != 16000 {
		t.Errorf("sample rate = %d, want 16000", cfg.GetSampleRateHertz())
	}
	if cfg.GetLanguageCode() != "de-DE" {
		t.Errorf("language = %q, want de-DE", cfg.GetLanguageCode())
	}
	if cfg.GetModel() != "canary" {
		t.Errorf("model = %q, want canary", cfg.GetModel())
	}
	if !cfg.GetEnableAutomaticPunctuation() || !cfg.GetVerbatimTranscripts() {
		t.Errorf("punctuation/verbatim defaults not applied: %+v", cfg)
	}

	md := <-fake.md
	if got := md.Get("authorization"); len(got) != 1 || got[0] != "Bearer test-key" {
		t.Errorf("authorization = %v, want the Bearer key", got)
	}
}

// TestSegmentedLocalNIMOmitsCredentials checks an unauthenticated local
// deployment sends neither credential header.
func TestSegmentedLocalNIMOmitsCredentials(t *testing.T) {
	fake, addr := startFakeBatchASR(t)
	no := false
	tr := &segmentedTranscriber{cfg: SegmentedSTTConfig{
		Server: addr,
		UseSSL: &no,
	}.withSegmentedDefaults()}
	t.Cleanup(func() { _ = tr.Close() })

	if _, err := tr.Transcribe(context.Background(), []byte{1}, 16000); err != nil {
		t.Fatalf("Transcribe: %v", err)
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

// TestSegmentedCloseIsIdempotent checks tearing down twice is safe, since
// cleanup can run after an explicit close.
func TestSegmentedCloseIsIdempotent(t *testing.T) {
	_, addr := startFakeBatchASR(t)
	no := false
	tr := &segmentedTranscriber{cfg: SegmentedSTTConfig{Server: addr, UseSSL: &no}.withSegmentedDefaults()}

	if _, err := tr.client(); err != nil {
		t.Fatalf("client: %v", err)
	}
	if err := tr.Close(); err != nil {
		t.Fatalf("first Close: %v", err)
	}
	if err := tr.Close(); err != nil {
		t.Fatalf("second Close: %v", err)
	}
}
