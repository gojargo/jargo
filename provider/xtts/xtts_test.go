package xtts

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/service/tts"
)

func TestValidate(t *testing.T) {
	if err := (Config{BaseURL: "http://x", Voice: "Claribel"}).Validate(); err != nil {
		t.Fatalf("valid config rejected: %v", err)
	}
	if err := (Config{Voice: "Claribel"}).Validate(); err == nil {
		t.Fatal("config without BaseURL should be rejected")
	}
	if err := (Config{BaseURL: "http://x"}).Validate(); err == nil {
		t.Fatal("config without Voice should be rejected")
	}
}

func TestSynthesizeStreamsPCM(t *testing.T) {
	want := []byte{0x01, 0x02, 0x03, 0x04}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/studio_speakers":
			_, _ = w.Write([]byte(`{"Claribel":{"speaker_embedding":[0.1],"gpt_cond_latent":[[0.2]]}}`))
		case "/tts_stream":
			_, _ = w.Write(want)
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()

	syn := &synthesizer{
		cfg:  Config{BaseURL: srv.URL, Voice: "Claribel", Language: "en", SampleRate: defaultSampleRate},
		http: srv.Client(),
	}
	if syn.SampleRate() != defaultSampleRate {
		t.Fatalf("SampleRate = %d, want %d", syn.SampleRate(), defaultSampleRate)
	}

	var got []byte
	err := runPCM(syn, context.Background(), "Hello.", func(pcm []byte) error {
		got = append(got, pcm...)
		return nil
	})
	if err != nil {
		t.Fatalf("Synthesize: %v", err)
	}
	if string(got) != string(want) {
		t.Fatalf("streamed PCM = %v, want %v", got, want)
	}
}

func TestUnknownSpeaker(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"Claribel":{"speaker_embedding":[0.1],"gpt_cond_latent":[[0.2]]}}`))
	}))
	defer srv.Close()

	syn := &synthesizer{cfg: Config{BaseURL: srv.URL, Voice: "Nobody"}, http: srv.Client()}
	if err := runPCM(syn, context.Background(), "hi", func([]byte) error { return nil }); err == nil {
		t.Fatal("expected error for unknown studio speaker")
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
