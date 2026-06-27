package piper

import (
	"bytes"
	"context"
	"encoding/binary"
	"net/http"
	"net/http/httptest"
	"testing"
)

// buildWAV builds a minimal 16-bit mono PCM WAV around data.
func buildWAV(data []byte, rate int, extraChunks bool) []byte {
	var b bytes.Buffer
	put32 := func(v uint32) { _ = binary.Write(&b, binary.LittleEndian, v) }
	put16 := func(v uint16) { _ = binary.Write(&b, binary.LittleEndian, v) }

	b.WriteString("RIFF")
	put32(0) // size placeholder; the parser does not rely on it
	b.WriteString("WAVE")

	b.WriteString("fmt ")
	put32(16)
	put16(1) // PCM
	put16(1) // mono
	put32(uint32(rate))
	put32(uint32(rate * 2)) // byte rate
	put16(2)                // block align
	put16(16)               // bits per sample

	if extraChunks {
		b.WriteString("LIST")
		put32(4)
		b.WriteString("INFO")
	}

	b.WriteString("data")
	put32(uint32(len(data)))
	b.Write(data)
	return b.Bytes()
}

func TestPCMFromWAV(t *testing.T) {
	data := []byte{1, 2, 3, 4, 5, 6, 7, 8}

	got, err := pcmFromWAV(buildWAV(data, 22050, false))
	if err != nil {
		t.Fatalf("pcmFromWAV: %v", err)
	}
	if !bytes.Equal(got, data) {
		t.Errorf("pcm = %v, want %v", got, data)
	}

	// A header carrying an extra chunk before "data" must still be parsed.
	got, err = pcmFromWAV(buildWAV(data, 22050, true))
	if err != nil {
		t.Fatalf("pcmFromWAV with extra chunk: %v", err)
	}
	if !bytes.Equal(got, data) {
		t.Errorf("pcm with extra chunk = %v, want %v", got, data)
	}
}

func TestPCMFromWAVRejectsNonWAV(t *testing.T) {
	if _, err := pcmFromWAV([]byte("not a wav")); err == nil {
		t.Error("pcmFromWAV(non-wav): want error, got nil")
	}
}

func TestSynthesizeStreamsPCM(t *testing.T) {
	data := bytes.Repeat([]byte{0xAB, 0xCD}, 5000) // larger than one emit chunk
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			t.Errorf("method = %s, want POST", r.Method)
		}
		body, _ := readAll(r)
		if string(body) != "hello" {
			t.Errorf("body = %q, want %q", body, "hello")
		}
		w.Header().Set("Content-Type", "audio/wav")
		_, _ = w.Write(buildWAV(data, defaultSampleRate, false))
	}))
	defer srv.Close()

	s := &synthesizer{cfg: Config{BaseURL: srv.URL, SampleRate: defaultSampleRate}, http: srv.Client()}

	var got bytes.Buffer
	err := s.Synthesize(context.Background(), "hello", func(pcm []byte) error {
		got.Write(pcm)
		return nil
	})
	if err != nil {
		t.Fatalf("Synthesize: %v", err)
	}
	if !bytes.Equal(got.Bytes(), data) {
		t.Errorf("streamed %d bytes, want %d", got.Len(), len(data))
	}
}

func TestConfigValidate(t *testing.T) {
	if err := (Config{}).Validate(); err == nil {
		t.Error("Validate() with empty BaseURL: want error, got nil")
	}
	if err := (Config{BaseURL: "http://localhost:5000"}).Validate(); err != nil {
		t.Errorf("Validate() with valid BaseURL: %v", err)
	}
}

func readAll(r *http.Request) ([]byte, error) {
	var b bytes.Buffer
	_, err := b.ReadFrom(r.Body)
	return b.Bytes(), err
}
