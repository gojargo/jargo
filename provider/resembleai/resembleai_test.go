package resembleai

import (
	"testing"
	"time"

	"github.com/gojargo/jargo/service/wsservice"
)

func TestValidate(t *testing.T) {
	if err := (Config{APIKey: "k", Voice: "v"}).Validate(); err != nil {
		t.Fatalf("valid config rejected: %v", err)
	}
	if err := (Config{Voice: "v"}).Validate(); err == nil {
		t.Fatal("config without APIKey should be rejected")
	}
	if err := (Config{APIKey: "k"}).Validate(); err == nil {
		t.Fatal("config without Voice should be rejected")
	}
}

// What the caller leaves unset takes Resemble AI's defaults, and the close
// timeout defaults to not waiting, since Resemble AI never acknowledges a
// close. One the caller sets is kept.
func TestDefaults(t *testing.T) {
	s := newSynthesizer(Config{APIKey: "k", Voice: "v"})
	if s.cfg.URL != defaultURL || s.cfg.SampleRate != 22050 || s.cfg.Precision != "PCM_16" ||
		s.cfg.OutputFormat != "wav" {
		t.Errorf("defaults = %+v", s.cfg)
	}
	if ct := s.cfg.Websocket.CloseTimeout; ct == nil || *ct != 0 {
		t.Errorf("close timeout = %v, want zero", ct)
	}

	wait := time.Second
	s = newSynthesizer(Config{APIKey: "k", Voice: "v", Websocket: wsservice.Options{CloseTimeout: &wait}})
	if ct := s.cfg.Websocket.CloseTimeout; ct == nil || *ct != time.Second {
		t.Errorf("close timeout = %v, want the one configured", ct)
	}
}
