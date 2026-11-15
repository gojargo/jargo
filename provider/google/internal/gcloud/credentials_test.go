package gcloud

import (
	"errors"
	"os"
	"path/filepath"
	"testing"

	"github.com/gojargo/jargo/language"
	"golang.org/x/oauth2"
)

// TestValidate checks credentials naming no way to authorize are refused, and
// each way alone is accepted.
func TestValidate(t *testing.T) {
	if err := (Credentials{}).Validate(); !errors.Is(err, ErrNoCredentials) {
		t.Errorf("Validate() = %v, want ErrNoCredentials", err)
	}
	for _, c := range []Credentials{
		{JSON: []byte("{}")},
		{Path: "/key.json"},
		{TokenSource: oauth2.StaticTokenSource(&oauth2.Token{AccessToken: "t"})},
	} {
		if err := c.Validate(); err != nil {
			t.Errorf("Validate(%+v) = %v, want nil", c, err)
		}
	}
}

// TestProjectID checks the project is read from the key, given as JSON or as
// the path of its file, and that a token source names none.
func TestProjectID(t *testing.T) {
	key := []byte(`{"type":"service_account","project_id":"my-project"}`)
	path := filepath.Join(t.TempDir(), "key.json")
	if err := os.WriteFile(path, key, 0o600); err != nil {
		t.Fatal(err)
	}

	for name, c := range map[string]Credentials{"json": {JSON: key}, "path": {Path: path}} {
		if got, err := c.ProjectID(); err != nil || got != "my-project" {
			t.Errorf("%s: ProjectID() = %q, %v, want my-project", name, got, err)
		}
	}
	ts := Credentials{TokenSource: oauth2.StaticTokenSource(&oauth2.Token{AccessToken: "t"})}
	if got, err := ts.ProjectID(); err != nil || got != "" {
		t.Errorf("token source: ProjectID() = %q, %v, want none", got, err)
	}
	if _, err := (Credentials{Path: filepath.Join(t.TempDir(), "missing.json")}).ProjectID(); err == nil {
		t.Error("ProjectID() read a key file that does not exist")
	}
}

// TestResolve checks the JSON key is used ahead of the path and the token
// source, a token source is used as it is, and a key that is not a
// service-account key is refused.
func TestResolve(t *testing.T) {
	ctx := t.Context()
	source := oauth2.StaticTokenSource(&oauth2.Token{AccessToken: "explicit"})

	resolved, err := Credentials{TokenSource: source}.Resolve(ctx)
	if err != nil {
		t.Fatalf("Resolve: %v", err)
	}
	if tok, _ := resolved.TokenSource.Token(); tok.AccessToken != "explicit" {
		t.Errorf("token = %q, want the token source's", tok.AccessToken)
	}

	user := []byte(`{"type":"authorized_user","client_id":"c","client_secret":"s","refresh_token":"r"}`)
	if _, err := (Credentials{JSON: user, TokenSource: source}).Resolve(ctx); err == nil {
		t.Error("Resolve accepted a key that is not a service-account key")
	}
	if _, err := (Credentials{}).Resolve(ctx); !errors.Is(err, ErrNoCredentials) {
		t.Errorf("Resolve() = %v, want ErrNoCredentials", err)
	}
}

// TestTTSLanguage checks languages are named the way Cloud Text-to-Speech
// names them, and an unlisted one passes through as its full code.
func TestTTSLanguage(t *testing.T) {
	for in, want := range map[language.Language]string{
		language.English:    "en-US",
		language.EnglishGB:  "en-GB",
		language.Arabic:     "ar-XA",
		language.Norwegian:  "nb-NO",
		language.Portuguese: "pt-BR",
		language.Chinese:    "cmn-CN",
		language.Icelandic:  "is",
	} {
		if got := TTSLanguage(in); got != want {
			t.Errorf("TTSLanguage(%q) = %q, want %q", in, got, want)
		}
	}
}
