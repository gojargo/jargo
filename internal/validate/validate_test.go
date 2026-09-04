package validate_test

import (
	"errors"
	"strings"
	"testing"

	"github.com/go-playground/validator/v10"
	"github.com/gojargo/jargo/internal/validate"
)

// credentials stands in for the shape every provider Config has: a handful of
// fields that must be present, and some that only have to be sane when set.
type credentials struct {
	APIKey  string `validate:"required"`
	BaseURL string `validate:"omitempty,url"`
	Voice   string `validate:"omitempty,oneof=alpha beta"`
	Timeout int    `validate:"omitempty,min=1"`
}

func validCredentials() credentials {
	return credentials{APIKey: "key", BaseURL: "https://example.test", Voice: "alpha", Timeout: 5}
}

// TestStructAcceptsAValidConfig covers the path every pipeline takes when it
// starts: a configuration that satisfies its tags passes without complaint.
func TestStructAcceptsAValidConfig(t *testing.T) {
	if err := validate.Struct(validCredentials()); err != nil {
		t.Errorf("a valid config was rejected: %v", err)
	}
}

// TestStructRejectsEachBrokenField covers the point of validating at all: a
// configuration that cannot work has to fail when the pipeline starts, naming
// the field at fault, rather than at first use somewhere inside a provider.
func TestStructRejectsEachBrokenField(t *testing.T) {
	tests := []struct {
		name   string
		field  string
		damage func(*credentials)
	}{
		{"a missing key", "APIKey", func(c *credentials) { c.APIKey = "" }},
		{"a URL that is not one", "BaseURL", func(c *credentials) { c.BaseURL = "not a url" }},
		{"a voice off the list", "Voice", func(c *credentials) { c.Voice = "gamma" }},
		{"a timeout below the minimum", "Timeout", func(c *credentials) { c.Timeout = -1 }},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := validCredentials()
			tt.damage(&c)

			err := validate.Struct(c)
			if err == nil {
				t.Fatalf("%s was accepted", tt.name)
			}
			if !strings.Contains(err.Error(), tt.field) {
				t.Errorf("the error does not name %s: %v", tt.field, err)
			}
		})
	}
}

// nested holds a struct that must itself be filled in, which only fails when the
// validator is built with required-struct checking enabled.
type nested struct {
	Creds credentials `validate:"required"`
}

// TestStructChecksNestedStructs covers the option the shared validator is built
// with. Without it a nested configuration left entirely at its zero value passes
// as though it had been filled in, and the provider fails later instead.
func TestStructChecksNestedStructs(t *testing.T) {
	if err := validate.Struct(nested{Creds: validCredentials()}); err != nil {
		t.Errorf("a filled-in nested config was rejected: %v", err)
	}

	err := validate.Struct(nested{})
	if err == nil {
		t.Fatal("a nested config left at its zero value was accepted: " +
			"the validator was built without required-struct checking")
	}
	if !strings.Contains(err.Error(), "Creds") || !strings.Contains(err.Error(), "required") {
		t.Errorf("the error does not name the nested config and the tag it failed: %v", err)
	}
}

// TestStructRejectsANonStruct covers being handed something that has no tags to
// check. It is a programming mistake rather than a bad configuration, and the
// validator reports it as its own kind of error.
func TestStructRejectsANonStruct(t *testing.T) {
	err := validate.Struct("not a struct")
	if err == nil {
		t.Fatal("a non-struct was accepted")
	}
	if _, ok := errors.AsType[*validator.InvalidValidationError](err); !ok {
		t.Errorf("got %T, want an InvalidValidationError: %v", err, err)
	}
}
