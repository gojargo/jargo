// Package gcloud holds what the Google Cloud speech services share: the
// credentials they authorize with, and the codes Cloud Text-to-Speech names its
// languages by.
package gcloud

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"

	"golang.org/x/oauth2"
	"golang.org/x/oauth2/google"
)

// Scope is the OAuth scope the Cloud speech APIs are authorized under. An
// application building its own token source should request it.
const Scope = "https://www.googleapis.com/auth/cloud-platform"

// ErrNoCredentials is returned when none of the ways to authorize was given.
//
//nolint:gochecknoglobals // sentinel error
var ErrNoCredentials = errors.New("google: no valid credentials provided")

// Credentials says how to authorize against Google Cloud. One field is needed.
// When more than one is set, JSON is used first, then Path, then TokenSource.
type Credentials struct {
	// JSON is a service-account key: the contents of the key file Google Cloud
	// issues.
	JSON []byte
	// Path is the path of a service-account key file, read when JSON is empty.
	Path string
	// TokenSource mints and refreshes the access token, used when neither key
	// is given. It is how an application authorizes with anything beyond a
	// service-account key: Application Default Credentials, workload identity,
	// an impersonated service account. jargo reads no environment, so
	// Application Default Credentials are opt-in this way:
	//
	//	creds, err := google.FindDefaultCredentials(ctx, speech.Scope) // x/oauth2/google
	//	... Credentials{TokenSource: creds.TokenSource}
	TokenSource oauth2.TokenSource
}

// Given reports whether any way to authorize was set.
func (c Credentials) Given() bool {
	return len(c.JSON) > 0 || c.Path != "" || c.TokenSource != nil
}

// Validate reports whether the credentials name a way to authorize.
func (c Credentials) Validate() error {
	if !c.Given() {
		return ErrNoCredentials
	}
	return nil
}

// Resolved is a set of credentials turned into what a client authorizes with.
type Resolved struct {
	// TokenSource mints the access tokens.
	TokenSource oauth2.TokenSource
	// ProjectID is the project a service-account key belongs to, or "" when
	// the credentials were a token source, which names none.
	ProjectID string
}

// Resolve turns the credentials into a token source scoped to Cloud. A key is
// accepted only as a service-account key: another kind of credentials file can
// make the client read files or call URLs it names, which a key handed over as
// configuration should never do. ctx is held by the token source for its
// refreshes, so it must outlive the client using it.
func (c Credentials) Resolve(ctx context.Context) (Resolved, error) {
	key, err := c.key()
	if err != nil {
		return Resolved{}, err
	}
	if key == nil {
		if c.TokenSource == nil {
			return Resolved{}, ErrNoCredentials
		}
		return Resolved{TokenSource: c.TokenSource}, nil
	}
	creds, err := google.CredentialsFromJSONWithType(ctx, key, google.ServiceAccount, Scope)
	if err != nil {
		return Resolved{}, fmt.Errorf("google: service account key: %w", err)
	}
	return Resolved{TokenSource: creds.TokenSource, ProjectID: creds.ProjectID}, nil
}

// ProjectID is the project_id a service-account key names, read without
// exchanging the key. It is "" for a token source, and for a key naming none.
func (c Credentials) ProjectID() (string, error) {
	key, err := c.key()
	if err != nil || key == nil {
		return "", err
	}
	var info struct {
		ProjectID string `json:"project_id"` //nolint:tagliatelle // Google key-file field
	}
	if err := json.Unmarshal(key, &info); err != nil {
		return "", fmt.Errorf("google: service account key: %w", err)
	}
	return info.ProjectID, nil
}

// key is the service-account key the credentials carry, read from Path when
// JSON is empty, or nil when they carry none.
func (c Credentials) key() ([]byte, error) {
	if len(c.JSON) > 0 {
		return c.JSON, nil
	}
	if c.Path == "" {
		return nil, nil
	}
	data, err := os.ReadFile(c.Path)
	if err != nil {
		return nil, fmt.Errorf("google: service account key file: %w", err)
	}
	return data, nil
}
