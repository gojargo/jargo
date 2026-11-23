package eval

import (
	"cmp"
	"errors"
	"os"
	"strings"
	"time"

	"github.com/gojargo/jargo/classifier/google/decisions"
	"github.com/gojargo/jargo/classifier/jev"
	zoomdecisions "github.com/gojargo/jargo/classifier/zoom/decisions"
)

// The classifiers the eval suites judge with, named by `judge.eval.factory:`.
// A factory takes the `judge.eval:` block and returns the classifier that
// decides the verdicts. They are registered under the names the release suite
// gives them, so its scenarios load unchanged.

// typesafeClassifierFactory is the name TypesafeClassifier is registered under.
const typesafeClassifierFactory = "evals.judges.typesafe_classifier"

// googleClassifierFactory is the name GoogleClassifier is registered under.
const googleClassifierFactory = "evals.judges.google_classifier"

// zoomClassifierFactory is the name ZoomClassifier is registered under.
const zoomClassifierFactory = "evals.judges.zoom_classifier"

// SuiteClassifierTimeout is how long a suite classifier has to answer a
// question. Jev, and Google's and Zoom's Decisions APIs answer in a few hundred
// milliseconds, so a question still waiting this long is one to ask again.
const SuiteClassifierTimeout = 2500 * time.Millisecond

// errNoTypesafeKey reports judging with Jev without its API key.
var errNoTypesafeKey = errors.New("eval: judging with Jev needs an API key: set TYPESAFE_API_KEY")

// errNoGoogleKey reports judging with Google's Decisions API without its API
// key.
var errNoGoogleKey = errors.New("eval: judging with Google Decisions needs an API key: set GOOGLE_API_KEY")

// errNoZoomKey reports judging with Zoom's Decisions API without its API key.
var errNoZoomKey = errors.New("eval: judging with Zoom Decisions needs an API key: set ZOOM_AI_API_KEY")

// TypesafeClassifier is TypeSafe's Jev, a hosted classifier that answers in a
// few hundred milliseconds. It is the factory registered as
// "evals.judges.typesafe_classifier".
//
// config is the `judge.eval:` block, with an optional `model` and `endpoint`;
// the client's defaults otherwise. This is harness code, the factory the
// release suite names, so the API key comes from TYPESAFE_API_KEY, and its
// absence is an error.
func TypesafeClassifier(config map[string]any) (any, error) {
	apiKey := os.Getenv("TYPESAFE_API_KEY")
	if apiKey == "" {
		return nil, errNoTypesafeKey
	}
	model, _ := config["model"].(string)
	endpoint, _ := config["endpoint"].(string)
	return jev.New(jev.Config{
		APIKey:  apiKey,
		Model:   cmp.Or(model, jev.DefaultModel),
		BaseURL: strings.TrimRight(cmp.Or(endpoint, jev.DefaultBaseURL), "/"),
		Timeout: SuiteClassifierTimeout,
	})
}

// GoogleClassifier is Google's Decisions API, a classifier that answers in a
// few hundred milliseconds. It is the factory registered as
// "evals.judges.google_classifier".
//
// config is the `judge.eval:` block, with an optional `model`; the client's
// default otherwise. This is harness code, the factory the release suite names,
// so the API key comes from GOOGLE_API_KEY, and its absence is an error.
func GoogleClassifier(config map[string]any) (any, error) {
	apiKey := os.Getenv("GOOGLE_API_KEY")
	if apiKey == "" {
		return nil, errNoGoogleKey
	}
	model, _ := config["model"].(string)
	return decisions.New(decisions.Config{
		APIKey:  apiKey,
		Model:   cmp.Or(model, decisions.DefaultModel),
		Timeout: SuiteClassifierTimeout,
	})
}

// ZoomClassifier is Zoom's Decisions API, a classifier that answers in a few
// hundred milliseconds. It is the factory registered as
// "evals.judges.zoom_classifier".
//
// config is the `judge.eval:` block, with an optional `model` ("zoom-decisions"
// or "zoom-decisions-pro"); the client's default otherwise. This is harness
// code, the factory the release suite names, so the API key comes from
// ZOOM_AI_API_KEY, and its absence is an error.
func ZoomClassifier(config map[string]any) (any, error) {
	apiKey := os.Getenv("ZOOM_AI_API_KEY")
	if apiKey == "" {
		return nil, errNoZoomKey
	}
	model, _ := config["model"].(string)
	return zoomdecisions.New(zoomdecisions.Config{
		APIKey:  apiKey,
		Model:   cmp.Or(model, zoomdecisions.DefaultModel),
		Timeout: SuiteClassifierTimeout,
	})
}
