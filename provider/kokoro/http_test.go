package kokoro_test

import (
	"testing"

	"github.com/gojargo/jargo/provider/kokoro"
)

func TestHTTPConfigValidate(t *testing.T) {
	if err := (kokoro.HTTPConfig{}).Validate(); err == nil {
		t.Error("Validate() with empty BaseURL: want error, got nil")
	}
	if err := (kokoro.HTTPConfig{BaseURL: "not a url"}).Validate(); err == nil {
		t.Error("Validate() with invalid URL: want error, got nil")
	}
	if err := (kokoro.HTTPConfig{BaseURL: "http://localhost:8880/v1"}).Validate(); err != nil {
		t.Errorf("Validate() with valid BaseURL: %v", err)
	}
}

func TestNewHTTPTTS(t *testing.T) {
	if kokoro.NewHTTPTTS(kokoro.HTTPConfig{BaseURL: "http://localhost:8880/v1"}) == nil {
		t.Error("NewHTTPTTS returned nil")
	}
}
