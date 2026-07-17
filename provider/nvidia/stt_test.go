package nvidia

import (
	"context"
	"errors"
	"io"
	"net"
	"strings"
	"testing"

	"github.com/gojargo/jargo/language"
	"github.com/gojargo/jargo/provider/nvidia/internal/rivapb"
	"google.golang.org/grpc"
)

func TestNvidiaLanguage(t *testing.T) {
	cases := map[language.Language]string{
		language.French:         "fr-FR", // base promoted to region
		language.German:         "de-DE",
		language.EnglishUS:      "en-US", // already qualified, passes through
		language.FrenchCA:       "fr-CA", // already qualified, passes through
		language.Language("xx"): "xx",    // unmapped, passes through
	}
	for in, want := range cases {
		if got := nvidiaLanguage(in); got != want {
			t.Errorf("nvidiaLanguage(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestWithDefaults(t *testing.T) {
	got := STTConfig{}.withDefaults()
	if got.Server != defaultSTTServer {
		t.Errorf("Server = %q, want %q", got.Server, defaultSTTServer)
	}
	if got.Language != language.EnglishUS {
		t.Errorf("Language = %q, want %q", got.Language, language.EnglishUS)
	}
	if got.MaxAlternatives != 1 {
		t.Errorf("MaxAlternatives = %d, want 1", got.MaxAlternatives)
	}
	if got.AudioChannelCount != 1 {
		t.Errorf("AudioChannelCount = %d, want 1", got.AudioChannelCount)
	}
}

func TestNewSTTName(t *testing.T) {
	// processor.New appends a per-instance counter (e.g. "NvidiaSTT#1").
	if name := NewSTT(STTConfig{}).Name(); !strings.HasPrefix(name, "NvidiaSTT") {
		t.Errorf("service name = %q, want NvidiaSTT prefix", name)
	}
}

// fakeASR is an in-process Riva ASR server: it records the opening config and,
// on the first audio chunk, replies with one interim then one final transcript.
type fakeASR struct {
	rivapb.UnimplementedRivaSpeechRecognitionServer
	gotConfig chan *rivapb.RecognitionConfig
}

func (f *fakeASR) StreamingRecognize(stream rivapb.RivaSpeechRecognition_StreamingRecognizeServer) error {
	req, err := stream.Recv()
	if err != nil {
		return err
	}
	f.gotConfig <- req.GetStreamingConfig().GetConfig()

	for {
		m, err := stream.Recv()
		if errors.Is(err, io.EOF) {
			return nil
		}
		if err != nil {
			return err
		}
		if m.GetAudioContent() == nil {
			continue
		}
		_ = stream.Send(&rivapb.StreamingRecognizeResponse{Results: []*rivapb.StreamingRecognitionResult{{
			Alternatives: []*rivapb.SpeechRecognitionAlternative{{Transcript: "hel"}},
			IsFinal:      false,
		}}})
		_ = stream.Send(&rivapb.StreamingRecognizeResponse{Results: []*rivapb.StreamingRecognitionResult{{
			Alternatives: []*rivapb.SpeechRecognitionAlternative{{Transcript: "hello", LanguageCode: []string{"en-US"}}},
			IsFinal:      true,
		}}})
	}
}

func TestConnectSendRecv(t *testing.T) {
	lis, err := (&net.ListenConfig{}).Listen(context.Background(), "tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	srv := grpc.NewServer()
	fake := &fakeASR{gotConfig: make(chan *rivapb.RecognitionConfig, 1)}
	rivapb.RegisterRivaSpeechRecognitionServer(srv, fake)
	go func() { _ = srv.Serve(lis) }()
	defer srv.Stop()

	no := false
	conn := &sttConnector{cfg: STTConfig{
		Server:   lis.Addr().String(),
		UseSSL:   &no,
		Language: language.French,
	}.withDefaults()}

	stream, err := conn.Connect(context.Background(), 16000)
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	defer func() { _ = stream.Close() }()

	// The opening config message reflects the sample rate, language mapping,
	// and defaulted fields.
	cfg := <-fake.gotConfig
	if cfg.GetEncoding() != rivapb.AudioEncoding_LINEAR_PCM {
		t.Errorf("encoding = %v, want LINEAR_PCM", cfg.GetEncoding())
	}
	if cfg.GetSampleRateHertz() != 16000 {
		t.Errorf("sample rate = %d, want 16000", cfg.GetSampleRateHertz())
	}
	if cfg.GetLanguageCode() != "fr-FR" {
		t.Errorf("language = %q, want fr-FR", cfg.GetLanguageCode())
	}
	if cfg.GetMaxAlternatives() != 1 {
		t.Errorf("max alternatives = %d, want 1", cfg.GetMaxAlternatives())
	}
	if !cfg.GetEnableAutomaticPunctuation() || !cfg.GetVerbatimTranscripts() {
		t.Errorf("punctuation/verbatim defaults not applied: %+v", cfg)
	}

	if err := stream.Send([]byte{1, 2, 3, 4}); err != nil {
		t.Fatalf("Send: %v", err)
	}

	res, err := stream.Recv()
	if err != nil {
		t.Fatalf("Recv interim: %v", err)
	}
	if len(res) != 1 || res[0].Text != "hel" || res[0].Final {
		t.Fatalf("interim result = %+v", res)
	}

	res, err = stream.Recv()
	if err != nil {
		t.Fatalf("Recv final: %v", err)
	}
	if len(res) != 1 || res[0].Text != "hello" || !res[0].Final || !res[0].EndOfTurn || res[0].Language != "en-US" {
		t.Fatalf("final result = %+v", res)
	}
}
