package jev

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"math"
	"strconv"
	"strings"
	"time"

	"github.com/gojargo/jargo/classifier"
	"github.com/gojargo/jargo/classifier/internal/ordered"
	"github.com/gojargo/jargo/frames"
)

// MaxChoiceOptions is the most options Jev takes in one choice question.
const MaxChoiceOptions = 255

// errNoKeyOrClient reports a classifier given neither an API key nor a client.
var errNoKeyOrClient = errors.New("jev: a classifier needs an API key or a client")

// Config configures a Jev classifier.
type Config struct {
	// APIKey is the Jev API key, when the classifier should have a client of
	// its own. One of APIKey and Client is required.
	APIKey string
	// Client is a client to share between several classifiers.
	Client *Client
	// Name names the classifier in its metrics; empty uses "JevClassifier#<n>".
	Name string
	// Model is the Jev model a client of its own asks; empty uses DefaultModel.
	Model string
	// BaseURL is where a client of its own sends its questions; empty uses
	// DefaultBaseURL.
	BaseURL string
	// Timeout is how long a client of its own waits for an answer; zero uses
	// DefaultTimeout.
	Timeout time.Duration
}

// Classifier answers questions by asking Jev.
//
// Jev's probabilities are calibrated. Build one with an API key to get a client
// of its own, or pass a Client to share one between several classifiers. A
// client the classifier created is closed in Cleanup; a shared one is left to
// whoever made it.
//
//	client, err := jev.NewClient(jev.ClientConfig{APIKey: key})
//	turn, err := jev.New(jev.Config{Client: client})
//	voicemail, err := jev.New(jev.Config{Client: client})
type Classifier struct {
	*classifier.Base
	client     *Client
	ownsClient bool
}

// New builds a Jev classifier.
func New(cfg Config) (*Classifier, error) {
	c := &Classifier{client: cfg.Client, ownsClient: cfg.Client == nil}
	if c.client == nil {
		if cfg.APIKey == "" {
			return nil, errNoKeyOrClient
		}
		client, err := NewClient(ClientConfig{
			APIKey: cfg.APIKey, Model: cfg.Model, BaseURL: cfg.BaseURL, Timeout: cfg.Timeout,
		})
		if err != nil {
			return nil, err
		}
		c.client = client
	}
	c.Base = classifier.NewBase("JevClassifier", cfg.Name, c)
	return c, nil
}

// Client is the client this classifier asks through.
func (c *Classifier) Client() *Client { return c.client }

// Model is the Jev model the questions go to.
func (c *Classifier) Model() string { return c.client.Model() }

// Setup opens the connection to Jev ahead of the first question. A connection
// that cannot be opened now is only a warning: the first question opens it
// itself.
func (c *Classifier) Setup(ctx context.Context) error {
	if err := c.Base.Setup(ctx); err != nil {
		return err
	}
	if err := c.client.Connect(ctx); err != nil {
		slog.WarnContext(ctx, "jev: could not connect", "classifier", c.Name(), "error", err)
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
	ctx context.Context, state any, questions []classifier.Named[classifier.Question],
) (map[string]classifier.Result, *frames.LLMTokenUsage, error) {
	jevQuestions := make([]NamedQuestion, 0, len(questions))
	for _, q := range questions {
		jq, err := toJev(q.Question)
		if err != nil {
			return nil, nil, err
		}
		jevQuestions = append(jevQuestions, NamedQuestion{Name: q.Name, Question: jq})
	}
	answers, usage, err := c.client.Ask(ctx, state, jevQuestions)
	if err != nil {
		return nil, nil, err
	}
	results := make(map[string]classifier.Result, len(questions))
	for _, q := range questions {
		r, err := fromJev(q.Question, answers[q.Name])
		if err != nil {
			return nil, nil, err
		}
		results[q.Name] = r
	}
	return results, &frames.LLMTokenUsage{
		PromptTokens:     int64(usage.InputTokens),
		CompletionTokens: int64(usage.OutputTokens),
		TotalTokens:      int64(usage.InputTokens + usage.OutputTokens),
	}, nil
}

// toJev writes a question in Jev's own format.
func toJev(question classifier.Question) (Question, error) {
	switch q := question.(type) {
	case classifier.YesNoQuestion:
		jq := Question{Type: "noul", Instructions: q.Instructions}
		if q.Yes != nil || q.No != nil {
			jq.Criteria = ordered.Object{{Key: "true", Value: orEmpty(q.Yes)}, {Key: "false", Value: orEmpty(q.No)}}
		}
		return jq, nil
	case classifier.ChoiceQuestion:
		if len(q.Options) > MaxChoiceOptions {
			return Question{}, fmt.Errorf("%w: Jev takes at most %d options, got %d",
				classifier.ErrClassifier, MaxChoiceOptions, len(q.Options))
		}
		criteria := ordered.Object{}
		for _, o := range q.Options {
			criteria.Set(o.Name, o.Description)
		}
		return Question{Type: "choice", Instructions: q.Instructions, Criteria: criteria}, nil
	case classifier.ScoreQuestion:
		return Question{Type: "score", Instructions: q.Instructions, Criteria: q.Levels}, nil
	default:
		return Question{}, fmt.Errorf("%w: unknown question type %T", classifier.ErrClassifier, question)
	}
}

// fromJev builds a result from Jev's answer to the question.
func fromJev(question classifier.Question, answer map[string]any) (classifier.Result, error) {
	switch q := question.(type) {
	case classifier.YesNoQuestion:
		p, err := number(answer, "noul")
		if err != nil {
			return nil, err
		}
		return classifier.YesNoResult{Probability: p}, nil
	case classifier.ChoiceQuestion:
		choice := text(answer["choice"])
		if !q.Has(choice) {
			return nil, fmt.Errorf("%w: Jev chose %q, which is not an option", classifier.ErrClassifier, choice)
		}
		probabilities, err := probabilitiesOf(answer, q.OptionNames())
		if err != nil {
			return nil, err
		}
		confidence, err := number(answer, "confidence")
		if err != nil {
			return nil, err
		}
		return classifier.ChoiceResult{Choice: choice, Probabilities: probabilities, Confidence: confidence}, nil
	case classifier.ScoreQuestion:
		// Jev keys level probabilities by position.
		positions := make([]string, len(q.Levels))
		for i := range q.Levels {
			positions[i] = strconv.Itoa(i)
		}
		probabilities, err := probabilitiesOf(answer, positions)
		if err != nil {
			return nil, err
		}
		score, err := number(answer, "score")
		if err != nil {
			return nil, err
		}
		confidence, err := number(answer, "confidence")
		if err != nil {
			return nil, err
		}
		levels := make([]classifier.ScoreLevel, len(q.Levels))
		for i, level := range q.Levels {
			levels[i] = classifier.ScoreLevel{Level: level, Probability: probabilities[positions[i]]}
		}
		return classifier.ScoreResult{Score: score, Levels: levels, Confidence: confidence}, nil
	default:
		return nil, fmt.Errorf("%w: unknown question type %T", classifier.ErrClassifier, question)
	}
}

// number reads the number Jev wrote under key.
func number(answer map[string]any, key string) (float64, error) {
	v, ok := toFloat(answer[key])
	if !ok {
		return 0, fmt.Errorf("%w: Jev reply has no usable '%s'", classifier.ErrClassifier, key)
	}
	return v, nil
}

// probabilitiesOf reads the probability Jev wrote for each key, 0 for the ones
// it left out.
func probabilitiesOf(answer map[string]any, keys []string) (map[string]float64, error) {
	given, _ := answer["probabilities"].(map[string]any)
	out := make(map[string]float64, len(keys))
	for _, key := range keys {
		v, present := given[key]
		if !present {
			out[key] = 0
			continue
		}
		f, ok := toFloat(v)
		if !ok {
			return nil, fmt.Errorf("%w: Jev reply has an unusable probability: %v", classifier.ErrClassifier, v)
		}
		out[key] = f
	}
	return out, nil
}

// toFloat reads a number as Jev may have written it.
func toFloat(value any) (float64, bool) {
	switch v := value.(type) {
	case float64:
		return v, !math.IsNaN(v)
	case string:
		f, err := strconv.ParseFloat(strings.TrimSpace(v), 64)
		return f, err == nil
	case bool:
		if v {
			return 1, true
		}
		return 0, true
	default:
		return 0, false
	}
}

// text reads the choice Jev wrote.
func text(value any) string {
	switch v := value.(type) {
	case nil:
		return ""
	case string:
		return v
	default:
		return fmt.Sprint(v)
	}
}

// orEmpty is what counts as an answer, or "" when the question leaves it open.
func orEmpty(v any) any {
	if v == nil {
		return ""
	}
	if s, ok := v.(string); ok && s == "" {
		return ""
	}
	return v
}

var _ classifier.Classifier = (*Classifier)(nil)
