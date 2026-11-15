// Package speech provides Google Cloud's streaming speech services over gRPC:
// Speech-to-Text V2 streaming recognition (NewSTT) and Text-to-Speech streaming
// synthesis (NewTTS).
//
// They live apart from the gemini package, which holds the REST services that
// need only an HTTP client, because they are built on Google's gRPC clients for
// the two APIs: an application that imports gemini for its LLM does not link
// them.
//
// Both authorize with Google Cloud credentials rather than an API key: a
// service-account key, given as its JSON or the path of its file, or a token
// source the application built itself. jargo reads no environment variables,
// so Application Default Credentials are opt-in through a token source:
//
//	creds, err := google.FindDefaultCredentials(ctx, speech.Scope) // x/oauth2/google
//	stt := speech.NewSTT(speech.STTConfig{
//		ProjectID:   creds.ProjectID,
//		Credentials: speech.Credentials{TokenSource: creds.TokenSource},
//	})
package speech

import (
	"github.com/gojargo/jargo/provider/google/internal/gcloud"
)

// Scope is the OAuth scope the speech APIs are authorized under. An
// application building its own token source should request it.
const Scope = gcloud.Scope

// Credentials says how to authorize against Google Cloud: a service-account
// key as JSON, the path of a key file, or a token source. It is the same type
// the gemini package's Text-to-Speech service takes, so one value serves both.
type Credentials = gcloud.Credentials

// ErrNoCredentials is returned when none of the ways to authorize was given.
//
//nolint:gochecknoglobals // sentinel error
var ErrNoCredentials = gcloud.ErrNoCredentials

// defaultLocation is the Google Cloud location used when none is configured.
const defaultLocation = "global"
