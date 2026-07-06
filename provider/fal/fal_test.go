package fal

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestValidate(t *testing.T) {
	if err := (Config{APIKey: "k"}).Validate(); err != nil {
		t.Fatalf("valid config rejected: %v", err)
	}
	if err := (Config{}).Validate(); err == nil {
		t.Fatal("config without APIKey should be rejected")
	}
}

func TestTranscribe(t *testing.T) {
	var gotAuth, gotBody string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuth = r.Header.Get("Authorization")
		b, _ := io.ReadAll(r.Body)
		gotBody = string(b)
		_, _ = w.Write([]byte(`{"text":"  hello world  "}`))
	}))
	defer srv.Close()

	tr := &transcriber{cfg: Config{APIKey: "k", Endpoint: srv.URL}, http: srv.Client()}
	text, err := tr.Transcribe(context.Background(), []byte{0, 0, 1, 0, 2, 0}, 16000)
	if err != nil {
		t.Fatalf("Transcribe: %v", err)
	}
	if text != "hello world" {
		t.Fatalf("text = %q, want %q", text, "hello world")
	}
	if gotAuth != "Key k" {
		t.Fatalf("Authorization = %q, want %q", gotAuth, "Key k")
	}
	if !strings.Contains(gotBody, "data:audio/x-wav;base64,") {
		t.Fatalf("body missing WAV data URI: %s", gotBody)
	}
	// The uploaded payload must be valid JSON with an audio_url.
	var payload map[string]any
	if err := json.Unmarshal([]byte(gotBody), &payload); err != nil {
		t.Fatalf("request body is not valid JSON: %v", err)
	}
	if _, ok := payload["audio_url"]; !ok {
		t.Fatal("request body missing audio_url")
	}
}

func TestTranscribeErrorStatus(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		http.Error(w, "nope", http.StatusUnauthorized)
	}))
	defer srv.Close()

	tr := &transcriber{cfg: Config{APIKey: "k", Endpoint: srv.URL}, http: srv.Client()}
	if _, err := tr.Transcribe(context.Background(), []byte{0, 0}, 16000); err == nil {
		t.Fatal("expected error on non-200 status")
	}
}
