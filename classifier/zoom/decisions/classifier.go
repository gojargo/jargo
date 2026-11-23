package decisions

import (
	"context"
	"encoding/base64"
	"errors"
	"fmt"
	"log/slog"
	"time"

	"github.com/gojargo/jargo/classifier"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/internal/ordered"
)

const (
	// MaxQuestions is the most questions the Decisions API takes in one
	// request.
	MaxQuestions = 10
	// MaxChoiceOptions is the most options the Decisions API takes in one
	// choice question.
	MaxChoiceOptions = 255
	// MaxScoreLevels is the most levels the Decisions API takes in one score
	// question.
	MaxScoreLevels = 10
	// MaxImages is the most images the Decisions API takes in one request.
	MaxImages = 4
)

// errNoKeyOrClient reports a classifier given neither an API key nor a client.
var errNoKeyOrClient = errors.New("decisions: a classifier needs an API key or a client")

// Config configures a Zoom Decisions classifier.
type Config struct {
	// APIKey is the Zoom AI Services API key, when the classifier should have a
	// client of its own. One of APIKey and Client is required.
	APIKey string
	// Client is a client to share between several classifiers.
	Client *Client
	// Name names the classifier in its metrics; empty uses
	// "ZoomDecisionsClassifier#<n>".
	Name string
	// Model is the decision model a client of its own asks, "zoom-decisions"
	// or "zoom-decisions-pro"; empty uses DefaultModel.
	Model string
	// URL is where a client of its own sends its questions; empty uses
	// DefaultURL.
	URL string
	// Timeout is how long a client of its own waits for an answer, retries
	// included; zero uses DefaultTimeout.
	Timeout time.Duration
}

// Classifier answers questions by asking the Zoom Decisions API.
//
// Structured state, instructions and descriptions are sent as JSON. Build one
// with an API key to get a client of its own, or pass a Client to share one
// between several classifiers. A client the classifier created is closed in
// Cleanup; a shared one is left to whoever made it.
//
// One request takes up to MaxQuestions questions, a choice question up to
// MaxChoiceOptions options, and a score question up to MaxScoreLevels levels.
// The Decisions API can see images: up to MaxImages per call.
//
//	c, err := decisions.New(decisions.Config{APIKey: os.Getenv("ZOOM_AI_API_KEY")})
type Classifier struct {
	*classifier.Base
	client     *Client
	ownsClient bool
}

// New builds a Zoom Decisions classifier.
func New(cfg Config) (*Classifier, error) {
	c := &Classifier{client: cfg.Client, ownsClient: cfg.Client == nil}
	if c.client == nil {
		if cfg.APIKey == "" {
			return nil, errNoKeyOrClient
		}
		client, err := NewClient(ClientConfig{
			APIKey: cfg.APIKey, Model: cfg.Model, URL: cfg.URL, Timeout: cfg.Timeout,
		})
		if err != nil {
			return nil, err
		}
		c.client = client
	}
	c.Base = classifier.NewBase("ZoomDecisionsClassifier", cfg.Name, c)
	return c, nil
}

// Client is the client this classifier asks through.
func (c *Classifier) Client() *Client { return c.client }

// Model is the decision model the questions go to.
func (c *Classifier) Model() string { return c.client.Model() }

// SupportsImages reports that the Decisions API can see images.
func (c *Classifier) SupportsImages() bool { return true }

// Setup opens the connection to Zoom ahead of the first question. A connection
// that cannot be opened now is only a warning: the first question opens it
// itself.
func (c *Classifier) Setup(ctx context.Context) error {
	if err := c.Base.Setup(ctx); err != nil {
		return err
	}
	if err := c.client.Connect(ctx); err != nil {
		slog.WarnContext(ctx, "decisions: could not connect", "classifier", c.Name(), "error", err)
	}
	return nil
}

// Cleanup closes the client if this classifier created it.
func (c *Classifier) Cleanup(ctx context.Context) error {
	err := c.Base.Cleanup(ctx)
	if c.ownsClient {
		c.client.Close()
	}
	return err
}

// AskQuestions implements classifier.Asker, answering the questions in one
// request.
func (c *Classifier) AskQuestions(
	ctx context.Context, state any, questions []classifier.Named[classifier.Question], inputs []classifier.Input,
) (map[string]classifier.Result, *frames.LLMTokenUsage, error) {
	images := make([]classifier.Image, 0, len(inputs))
	for _, input := range inputs {
		if image, ok := input.(classifier.Image); ok {
			images = append(images, image)
		}
	}
	if err := checkLimits(questions, images); err != nil {
		return nil, nil, err
	}
	// Zoom reads images only when the request asks it to.
	var options map[string]any
	if len(images) > 0 {
		options = map[string]any{"enable_image_input": true}
	}
	input, err := requestInput(state, images)
	if err != nil {
		return nil, nil, err
	}
	wire, err := questionsToWire(questions)
	if err != nil {
		return nil, nil, err
	}
	answers, usage, err := c.client.Ask(ctx, input, wire, options)
	if err != nil {
		return nil, nil, err
	}
	results := make(map[string]classifier.Result, len(questions))
	for _, q := range questions {
		r, err := resultFromWire(q.Name, q.Question, answers[q.Name])
		if err != nil {
			return nil, nil, err
		}
		results[q.Name] = r
	}
	tokens := &frames.LLMTokenUsage{
		PromptTokens:     int64(usage.InputTokens),
		CompletionTokens: int64(usage.OutputTokens),
		TotalTokens:      int64(usage.InputTokens + usage.OutputTokens),
	}
	if usage.CachedInputTokens != 0 {
		cached := int64(usage.CachedInputTokens)
		tokens.CacheReadTokens = &cached
	}
	return results, tokens, nil
}

// checkLimits reports a request that is more than Zoom takes in one.
func checkLimits(questions []classifier.Named[classifier.Question], images []classifier.Image) error {
	if len(images) > MaxImages {
		return fmt.Errorf("%w: Zoom Decisions takes at most %d images, got %d",
			classifier.ErrClassifier, MaxImages, len(images))
	}
	if len(questions) > MaxQuestions {
		return fmt.Errorf("%w: Zoom Decisions takes at most %d questions per request, got %d",
			classifier.ErrClassifier, MaxQuestions, len(questions))
	}
	for _, q := range questions {
		switch question := q.Question.(type) {
		case classifier.ChoiceQuestion:
			if len(question.Options) > MaxChoiceOptions {
				return fmt.Errorf("%w: Zoom Decisions takes at most %d options, got %d for %q",
					classifier.ErrClassifier, MaxChoiceOptions, len(question.Options), q.Name)
			}
		case classifier.ScoreQuestion:
			if len(question.Levels) > MaxScoreLevels {
				return fmt.Errorf("%w: Zoom Decisions takes at most %d levels, got %d for %q",
					classifier.ErrClassifier, MaxScoreLevels, len(question.Levels), q.Name)
			}
		}
	}
	return nil
}

// requestInput is the request's input: the state as text, or a user message
// with any images first.
func requestInput(state any, images []classifier.Image) (any, error) {
	stateText, err := text(state)
	if err != nil {
		return nil, err
	}
	if len(images) == 0 {
		return stateText, nil
	}
	content := make([]ordered.Object, 0, len(images)+1)
	for _, image := range images {
		content = append(content, ordered.Object{
			{Key: keyType, Value: "input_image"},
			{Key: "image_url", Value: "data:" + image.ContentType + ";base64," +
				base64.StdEncoding.EncodeToString(image.Data)},
		})
	}
	content = append(content, ordered.Object{{Key: keyType, Value: "input_text"}, {Key: "text", Value: stateText}})
	return []ordered.Object{{{Key: "role", Value: "user"}, {Key: "content", Value: content}}}, nil
}

var _ classifier.Classifier = (*Classifier)(nil)
