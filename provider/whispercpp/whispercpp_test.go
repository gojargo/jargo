package whispercpp_test

import (
	"testing"

	"github.com/gojargo/jargo/provider/whispercpp"
)

func TestConfigValidate(t *testing.T) {
	if err := (whispercpp.Config{}).Validate(); err == nil {
		t.Error("Validate() with empty BaseURL: want error, got nil")
	}
	if err := (whispercpp.Config{BaseURL: "not a url"}).Validate(); err == nil {
		t.Error("Validate() with invalid URL: want error, got nil")
	}
	if err := (whispercpp.Config{BaseURL: "http://localhost:8080/v1"}).Validate(); err != nil {
		t.Errorf("Validate() with valid BaseURL: %v", err)
	}
}

func TestNewSTT(t *testing.T) {
	if whispercpp.NewSTT(whispercpp.Config{BaseURL: "http://localhost:8080/v1"}) == nil {
		t.Error("NewSTT returned nil")
	}
}
