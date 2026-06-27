package kokoro_test

import (
	"testing"

	"github.com/gojargo/jargo/provider/kokoro"
)

func TestConfigValidate(t *testing.T) {
	if err := (kokoro.Config{}).Validate(); err == nil {
		t.Error("Validate() with empty BaseURL: want error, got nil")
	}
	if err := (kokoro.Config{BaseURL: "not a url"}).Validate(); err == nil {
		t.Error("Validate() with invalid URL: want error, got nil")
	}
	if err := (kokoro.Config{BaseURL: "http://localhost:8880/v1"}).Validate(); err != nil {
		t.Errorf("Validate() with valid BaseURL: %v", err)
	}
}

func TestNewTTS(t *testing.T) {
	if kokoro.NewTTS(kokoro.Config{BaseURL: "http://localhost:8880/v1"}) == nil {
		t.Error("NewTTS returned nil")
	}
}
