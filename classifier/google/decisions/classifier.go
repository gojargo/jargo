package decisions

import (
	"context"
	"encoding/base64"
	"errors"
	"fmt"
	"log/slog"
	"maps"
	"math"
	"slices"
	"strconv"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/gojargo/jargo/audio"
	"github.com/gojargo/jargo/classifier"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/internal/ordered"
)

// MaxChoiceOptions is the most options the Decisions API takes in one choice
// question.
const MaxChoiceOptions = 255

// audioSampleRate is the sample rate clips are sent at, which keeps requests
// small.
const audioSampleRate = 16000

// errNoKeyOrClient reports a classifier given neither an API key nor a client.
var errNoKeyOrClient = errors.New("decisions: a classifier needs an API key or a client")

// FileTypes is every file type the Decisions API reads, by MIME type, and the
// content block it goes in.
//
//nolint:gochecknoglobals // a fixed table of what the API reads
var FileTypes = fileTypes()

func fileTypes() map[string]string {
	types := map[string]string{"application/pdf": "document"}
	for _, kind := range []string{"png", "jpeg", "webp", "heic", "heif", "gif", "bmp", "tiff"} {
		types["image/"+kind] = "image"
	}
	for _, kind := range []string{
		"wav", "mp3", "mpeg", "aiff", "aac", "ogg", "flac", "m4a", "opus", "webm", "l16", "alaw", "mulaw",
	} {
		types["audio/"+kind] = "audio"
	}
	for _, kind := range []string{"mp4", "mpeg", "mov", "avi", "webm", "wmv", "x-flv", "3gpp"} {
		types["video/"+kind] = "video"
	}
	return types
}

// Config configures a Google Decisions classifier.
type Config struct {
	// APIKey is the Gemini API key, when the classifier should have a client
	// of its own. One of APIKey and Client is required.
	APIKey string
	// Client is a client to share between several classifiers.
	Client *Client
	// Name names the classifier in its metrics; empty uses
	// "GoogleDecisionsClassifier#<n>".
	Name string
	// Model is the decision model a client of its own asks; empty uses
	// DefaultModel.
	Model string
	// Timeout is how long a client of its own waits for an answer; zero uses
	// DefaultTimeout.
	Timeout time.Duration
}

// Classifier answers questions by asking Google's Decisions API.
//
// Structured state, instructions and descriptions are sent as JSON text. Build
// one with a Gemini API key to get a client of its own, or pass a Client to
// share one between several classifiers. A client the classifier created is
// closed in Cleanup; a shared one is left to whoever made it.
//
// The Decisions API sees images, hears clips of audio, sent as 16 kHz WAV, and
// reads files of the types in FileTypes: PDFs, videos, and images and audio in
// other formats, as bytes or by URL. The inputs follow the state in the order
// they're given.
//
//	c, err := decisions.New(decisions.Config{APIKey: key})
type Classifier struct {
	*classifier.Base
	client     *Client
	ownsClient bool
}

// New builds a Google Decisions classifier.
func New(cfg Config) (*Classifier, error) {
	c := &Classifier{client: cfg.Client, ownsClient: cfg.Client == nil}
	if c.client == nil {
		if cfg.APIKey == "" {
			return nil, errNoKeyOrClient
		}
		client, err := NewClient(ClientConfig{APIKey: cfg.APIKey, Model: cfg.Model, Timeout: cfg.Timeout})
		if err != nil {
			return nil, err
		}
		c.client = client
	}
	c.Base = classifier.NewBase("GoogleDecisionsClassifier", cfg.Name, c)
	return c, nil
}

// Client is the client this classifier asks through.
func (c *Classifier) Client() *Client { return c.client }

// Model is the decision model the questions go to.
func (c *Classifier) Model() string { return c.client.Model() }

// SupportsImages reports that the Decisions API can see images.
func (c *Classifier) SupportsImages() bool { return true }

// SupportsAudio reports that the Decisions API can hear audio.
func (c *Classifier) SupportsAudio() bool { return true }

// AudioSampleRate is 16 kHz: clips are sent at it, which keeps requests small.
func (c *Classifier) AudioSampleRate() int { return audioSampleRate }

// FileTypes is the file types the Decisions API reads: PDFs, videos, and
// images and audio in many formats.
func (c *Classifier) FileTypes() []string { return slices.Sorted(maps.Keys(FileTypes)) }

// Setup opens the connection to Google ahead of the first question. A
// connection that cannot be opened now is only a warning: the first question
// opens it itself.
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
	googleQuestions := make([]NamedQuestion, 0, len(questions))
	for _, q := range questions {
		gq, err := toGoogle(q.Question)
		if err != nil {
			return nil, nil, err
		}
		googleQuestions = append(googleQuestions, NamedQuestion{Name: q.Name, Question: gq})
	}
	input, err := requestInput(state, inputs)
	if err != nil {
		return nil, nil, err
	}
	answers, usage, err := c.client.Ask(ctx, input, googleQuestions)
	if err != nil {
		return nil, nil, err
	}
	results := make(map[string]classifier.Result, len(questions))
	for _, q := range questions {
		r, err := fromGoogle(q.Question, answers[q.Name])
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

// requestInput is the request's input: the state as text, then a block for
// each input, in order.
func requestInput(state any, inputs []classifier.Input) (any, error) {
	stateText, err := text(state)
	if err != nil {
		return nil, err
	}
	if len(inputs) == 0 {
		return stateText, nil
	}
	blocks := []ordered.Object{{{Key: keyType, Value: "text"}, {Key: "text", Value: stateText}}}
	for _, input := range inputs {
		block, err := contentBlock(input)
		if err != nil {
			return nil, err
		}
		blocks = append(blocks, block)
	}
	return blocks, nil
}

// contentBlock is an input as one of the Decisions API's content blocks.
func contentBlock(input classifier.Input) (ordered.Object, error) {
	switch in := input.(type) {
	case classifier.Image:
		return ordered.Object{
			{Key: keyType, Value: "image"},
			{Key: keyData, Value: base64.StdEncoding.EncodeToString(in.Data)},
			{Key: keyMimeType, Value: in.ContentType},
		}, nil
	case classifier.Audio:
		wav := audio.PCMToWAV(in.Audio, in.SampleRate, 1)
		return ordered.Object{
			{Key: keyType, Value: "audio"},
			{Key: keyData, Value: base64.StdEncoding.EncodeToString(wav)},
			{Key: keyMimeType, Value: "audio/wav"},
		}, nil
	case classifier.File:
		block := ordered.Object{{Key: keyType, Value: FileTypes[in.ContentType]}, {Key: keyMimeType, Value: in.ContentType}}
		if in.Data != nil {
			block.Set(keyData, base64.StdEncoding.EncodeToString(in.Data))
		} else {
			block.Set("uri", in.URL)
		}
		return block, nil
	default:
		return nil, fmt.Errorf("%w: unknown input type %T", classifier.ErrClassifier, input)
	}
}

// toGoogle writes a question in the Decisions API's own format.
func toGoogle(question classifier.Question) (Question, error) {
	switch q := question.(type) {
	case classifier.YesNoQuestion:
		return yesNoQuestion(q)
	case classifier.ChoiceQuestion:
		return choiceQuestion(q)
	case classifier.ScoreQuestion:
		instructions, err := text(q.Instructions)
		if err != nil {
			return Question{}, err
		}
		criteria := make([]string, 0, len(q.Levels))
		for _, level := range q.Levels {
			t, err := text(level)
			if err != nil {
				return Question{}, err
			}
			criteria = append(criteria, t)
		}
		return Question{Type: "score", Instructions: instructions, Criteria: criteria}, nil
	default:
		return Question{}, fmt.Errorf("%w: unknown question type %T", classifier.ErrClassifier, question)
	}
}

// yesNoQuestion writes a yes or no question, saying in its criteria what
// counts as each answer when the question does.
func yesNoQuestion(q classifier.YesNoQuestion) (Question, error) {
	instructions, err := text(q.Instructions)
	if err != nil {
		return Question{}, err
	}
	gq := Question{Type: "noul", Instructions: instructions}
	criteria := ordered.Object{}
	for _, c := range []ordered.Field{{Key: "true_description", Value: q.Yes}, {Key: "false_description", Value: q.No}} {
		if c.Value == nil {
			continue
		}
		t, err := text(c.Value)
		if err != nil {
			return Question{}, err
		}
		criteria.Set(c.Key, t)
	}
	if len(criteria) > 0 {
		gq.Criteria = criteria
	}
	return gq, nil
}

// choiceQuestion writes a choice question, its options as criteria in order.
func choiceQuestion(q classifier.ChoiceQuestion) (Question, error) {
	if len(q.Options) > MaxChoiceOptions {
		return Question{}, fmt.Errorf("%w: Google Decisions takes at most %d options, got %d",
			classifier.ErrClassifier, MaxChoiceOptions, len(q.Options))
	}
	instructions, err := text(q.Instructions)
	if err != nil {
		return Question{}, err
	}
	criteria := ordered.Object{}
	for _, o := range q.Options {
		description := ""
		if o.Description != nil {
			if description, err = text(o.Description); err != nil {
				return Question{}, err
			}
		}
		criteria.Set(o.Name, description)
	}
	return Question{Type: "choice", Instructions: instructions, Criteria: criteria}, nil
}

// fromGoogle builds a result from the Decisions API's answer to the question.
func fromGoogle(question classifier.Question, answer map[string]any) (classifier.Result, error) {
	switch q := question.(type) {
	case classifier.YesNoQuestion:
		p, err := number(answer, "noul")
		if err != nil {
			return nil, err
		}
		return classifier.YesNoResult{Probability: p}, nil
	case classifier.ChoiceQuestion:
		return choiceResult(q, answer)
	case classifier.ScoreQuestion:
		return scoreResult(q, answer)
	default:
		return nil, fmt.Errorf("%w: unknown question type %T", classifier.ErrClassifier, question)
	}
}

// choiceResult builds a choice result, checking the choice is an option.
func choiceResult(q classifier.ChoiceQuestion, answer map[string]any) (classifier.Result, error) {
	choice := choiceText(answer["choice"])
	if !q.Has(choice) {
		return nil, fmt.Errorf("%w: Google Decisions chose %q, which is not an option", classifier.ErrClassifier, choice)
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
}

// scoreResult builds a score result. The Decisions API keys level
// probabilities by position.
func scoreResult(q classifier.ScoreQuestion, answer map[string]any) (classifier.Result, error) {
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
}

// number reads the number the API gave as key, or an error if it gave none.
func number(answer map[string]any, key string) (float64, error) {
	v, ok := toFloat(answer[key])
	if !ok {
		return 0, fmt.Errorf("%w: Google Decisions reply has no usable '%s'", classifier.ErrClassifier, key)
	}
	return v, nil
}

// probabilitiesOf reads the probability the API gave for each key, 0 for the
// ones it left out.
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
			return nil, fmt.Errorf("%w: Google Decisions reply has an unusable probability: %v",
				classifier.ErrClassifier, v)
		}
		out[key] = f
	}
	return out, nil
}

// toFloat reads a number as the API may have written it.
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

// choiceText reads the choice the API gave.
func choiceText(value any) string {
	switch v := value.(type) {
	case nil:
		return ""
	case string:
		return v
	default:
		return fmt.Sprint(v)
	}
}

// text is a value as text for the Decisions API: a string as is, structured
// data as JSON with a space after each comma and colon, and everything outside
// ASCII escaped.
func text(value any) (string, error) {
	if s, ok := value.(string); ok {
		return s, nil
	}
	data, err := ordered.Marshal(value)
	if err != nil {
		return "", fmt.Errorf("%w: cannot write %T as JSON: %w", classifier.ErrClassifier, value, err)
	}
	var b strings.Builder
	inString, escaped := false, false
	for len(data) > 0 {
		r, size := utf8.DecodeRune(data)
		data = data[size:]
		switch {
		case inString && escaped:
			escaped = false
		case inString && r == '\\':
			escaped = true
		case r == '"':
			inString = !inString
		case !inString && (r == ',' || r == ':'):
			b.WriteRune(r)
			b.WriteByte(' ')
			continue
		}
		if r < 0x7f {
			b.WriteRune(r)
			continue
		}
		// Outside ASCII: escaped as UTF-16 code units.
		if r > 0xffff {
			r -= 0x10000
			fmt.Fprintf(&b, `\u%04x\u%04x`, 0xd800+(r>>10), 0xdc00+(r&0x3ff))
		} else {
			fmt.Fprintf(&b, `\u%04x`, r)
		}
	}
	return b.String(), nil
}

var _ classifier.Classifier = (*Classifier)(nil)
