// Package llm is a classifier backed by any LLM service that runs a one-shot
// inference.
//
// Every question about a state goes to the LLM in one out-of-pipeline call,
// through the service's RunInference. The LLM is asked for one JSON object with
// an answer per question, and the object is parsed into results.
package llm

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"regexp"
	"strconv"
	"strings"
	"time"

	"github.com/gojargo/jargo/classifier"
	"github.com/gojargo/jargo/classifier/internal/ordered"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/internal/validate"
	llmservice "github.com/gojargo/jargo/service/llm"
)

// DefaultInstructions is the system instruction the LLM answers under unless
// the classifier is given its own.
const DefaultInstructions = "You are a classifier. Each message describes some input and asks one or " +
	"more questions about it. Every question has a name and says the shape its " +
	"answer must have. Reply with a single JSON object, and nothing else: one " +
	"key per question name, each holding that question's answer in the shape " +
	"asked for. Probabilities and confidences are numbers from 0 to 1. For " +
	`example, a yes or no question named "voicemail" is answered ` +
	`{"voicemail": {"probability": 0.97}}. A choice question named "department" ` +
	"with the options billing, support and sales is answered " +
	`{"department": {"choice": "billing", "probabilities": {"billing": 0.92, ` +
	`"support": 0.06, "sales": 0.02}}}. A score question named "mood" on a ` +
	"scale of three levels is answered with a probability per level, by position, " +
	`{"mood": {"probabilities": {"0": 0.1, "1": 0.7, "2": 0.2}}}. ` +
	"Several questions about one input get one object with an answer for each " +
	"of them."

// defaultTimeout is how long the LLM is given to answer.
const defaultTimeout = 10 * time.Second

// Config configures an LLM classifier.
type Config struct {
	// LLM is the service that answers the questions. Required.
	LLM llmservice.Inferencer `validate:"required"`
	// Name names the classifier in its metrics; empty uses "LLMClassifier#<n>".
	Name string
	// Instructions is the system instruction for the LLM. Empty uses
	// DefaultInstructions, which asks for one JSON object with an answer per
	// question.
	Instructions string
	// MaxTokens caps the reply's length, for services that take one. Zero
	// leaves the service's own bound in place.
	MaxTokens int `validate:"min=0"`
	// Timeout is how long to wait for the LLM's reply before giving up; zero
	// uses 10s.
	Timeout time.Duration `validate:"min=0"`
}

// Classifier answers questions by asking an LLM for a JSON object.
//
// The probabilities are whatever the LLM wrote, so they are not calibrated. Any
// service that runs a one-shot inference can back a classifier. The reply's
// shape is enforced by the provider where the service supports a reply schema,
// and otherwise asked for in the prompt and parsed from the reply.
//
//	c, err := llm.New(llm.Config{LLM: service})
//	results, err := c.YesNo(ctx, "Hi, you've reached Dana. Leave a message.",
//		[]classifier.Named[classifier.YesNoQuestion]{
//			{Name: "voicemail", Question: classifier.YesNoQuestion{Instructions: "is this a voicemail greeting?"}},
//		})
//	results["voicemail"].Probability
type Classifier struct {
	*classifier.Base
	cfg Config
}

// New builds an LLM classifier.
func New(cfg Config) (*Classifier, error) {
	if err := validate.Struct(cfg); err != nil {
		return nil, err
	}
	if cfg.Instructions == "" {
		cfg.Instructions = DefaultInstructions
	}
	if cfg.Timeout == 0 {
		cfg.Timeout = defaultTimeout
	}
	c := &Classifier{cfg: cfg}
	c.Base = classifier.NewBase("LLMClassifier", cfg.Name, c)
	return c, nil
}

// LLM is the service that answers the questions.
func (c *Classifier) LLM() llmservice.Inferencer { return c.cfg.LLM }

// Model is the LLM service's model, or "" when it does not say.
func (c *Classifier) Model() string {
	if m, ok := c.cfg.LLM.(interface{ Model() string }); ok {
		return m.Model()
	}
	return ""
}

// AskQuestions implements classifier.Asker, answering the questions in one LLM
// call. The call reports no token usage.
func (c *Classifier) AskQuestions(
	ctx context.Context, state any, questions []classifier.Named[classifier.Question],
) (map[string]classifier.Result, *frames.LLMTokenUsage, error) {
	prompt, err := c.render(state, questions)
	if err != nil {
		return nil, nil, fmt.Errorf("%w: cannot write the question: %w", classifier.ErrClassifier, err)
	}
	schema, err := json.Marshal(c.schema(questions))
	if err != nil {
		return nil, nil, fmt.Errorf("%w: cannot write the reply schema: %w", classifier.ErrClassifier, err)
	}
	convo := frames.NewLLMContext("")
	convo.AddUserMessage(prompt)

	callCtx, cancel := context.WithTimeout(ctx, c.cfg.Timeout)
	defer cancel()
	reply, err := c.cfg.LLM.RunInference(callCtx, convo, llmservice.InferenceOptions{
		MaxTokens:         c.cfg.MaxTokens,
		SystemInstruction: c.cfg.Instructions,
		ResponseSchema:    schema,
	})
	switch {
	case err != nil && ctx.Err() == nil && errors.Is(callCtx.Err(), context.DeadlineExceeded):
		return nil, nil, fmt.Errorf("%w: %s did not answer within %s: %w",
			classifier.ErrClassifier, c.llmName(), c.cfg.Timeout, err)
	case err != nil:
		return nil, nil, fmt.Errorf("%w: %s failed to answer: %w", classifier.ErrClassifier, c.llmName(), err)
	}

	answers, err := parse(reply)
	if err != nil {
		return nil, nil, err
	}
	if len(questions) == 1 {
		if _, named := answers[questions[0].Name]; !named {
			// A lone answer often comes back without its name around it.
			answers = map[string]any{questions[0].Name: answers}
		}
	}
	results := make(map[string]classifier.Result, len(questions))
	for _, q := range questions {
		answer, ok := answers[q.Name].(map[string]any)
		if !ok {
			return nil, nil, fmt.Errorf("%w: the LLM gave no answer for %q", classifier.ErrClassifier, q.Name)
		}
		r, err := result(q.Question, answer)
		if err != nil {
			return nil, nil, err
		}
		results[q.Name] = r
	}
	return results, nil, nil
}

// llmName names the LLM in an error.
func (c *Classifier) llmName() string {
	if n, ok := c.cfg.LLM.(interface{ Name() string }); ok {
		return n.Name()
	}
	return fmt.Sprintf("%T", c.cfg.LLM)
}

// render writes the state and the questions as the message the LLM answers.
func (c *Classifier) render(state any, questions []classifier.Named[classifier.Question]) (string, error) {
	input, err := text(state)
	if err != nil {
		return "", err
	}
	parts := []string{"Input:\n" + input}
	for _, q := range questions {
		lines, err := renderQuestion(q.Name, q.Question)
		if err != nil {
			return "", err
		}
		parts = append(parts, strings.Join(lines, "\n"))
	}
	names := make([]string, 0, len(questions))
	for _, q := range questions {
		names = append(names, fmt.Sprintf("%q: <answer to %q>", q.Name, q.Name))
	}
	parts = append(parts, fmt.Sprintf("Reply with one JSON object of this shape: {%s}", strings.Join(names, ", ")))
	return strings.Join(parts, "\n\n"), nil
}

// renderQuestion writes one question, what its answers mean and the shape its
// answer takes.
func renderQuestion(name string, question classifier.Question) ([]string, error) {
	var (
		instructions any
		body         func() ([]string, error)
	)
	switch q := question.(type) {
	case classifier.YesNoQuestion:
		instructions, body = q.Instructions, func() ([]string, error) { return renderYesNo(q) }
	case classifier.ChoiceQuestion:
		instructions, body = q.Instructions, func() ([]string, error) { return renderChoice(q) }
	case classifier.ScoreQuestion:
		instructions, body = q.Instructions, func() ([]string, error) { return renderScore(q) }
	default:
		return nil, fmt.Errorf("%w: unknown question type %T", classifier.ErrClassifier, question)
	}
	head, err := text(instructions)
	if err != nil {
		return nil, err
	}
	lines, err := body()
	if err != nil {
		return nil, err
	}
	return append([]string{fmt.Sprintf("Question %q: %s", name, head)}, lines...), nil
}

// renderYesNo writes what a yes and a no mean, when the question says, and the
// answer's shape.
func renderYesNo(q classifier.YesNoQuestion) ([]string, error) {
	var lines []string
	if q.Yes != nil {
		yes, err := text(q.Yes)
		if err != nil {
			return nil, err
		}
		lines = append(lines, "Yes means: "+yes)
	}
	if q.No != nil {
		no, err := text(q.No)
		if err != nil {
			return nil, err
		}
		lines = append(lines, "No means: "+no)
	}
	return append(lines, `Answer shape: {"probability": <probability that the answer is yes>}`), nil
}

// renderChoice writes the options and the answer's shape.
func renderChoice(q classifier.ChoiceQuestion) ([]string, error) {
	lines := []string{"Options:"}
	for _, o := range q.Options {
		if o.Description == nil {
			lines = append(lines, "- "+o.Name)
			continue
		}
		desc, err := text(o.Description)
		if err != nil {
			return nil, err
		}
		lines = append(lines, fmt.Sprintf("- %s: %s", o.Name, desc))
	}
	return append(lines, `Answer shape: {"choice": <the option that fits best>, `+
		`"probabilities": {<a probability per option, summing to 1>}}`), nil
}

// renderScore writes the scale and the answer's shape.
func renderScore(q classifier.ScoreQuestion) ([]string, error) {
	lines := []string{"Scale, lowest first:"}
	for i, level := range q.Levels {
		l, err := text(level)
		if err != nil {
			return nil, err
		}
		lines = append(lines, fmt.Sprintf("%d: %s", i, l))
	}
	positions := make([]string, 0, len(q.Levels))
	for i := range q.Levels {
		positions = append(positions, fmt.Sprintf(`"%d": <probability>`, i))
	}
	return append(lines, fmt.Sprintf(`Answer shape: {"probabilities": {%s}}, one per level by `+
		"position, summing to 1", strings.Join(positions, ", "))), nil
}

// schema is the JSON schema of the reply: one answer per question, in its shape.
func (c *Classifier) schema(questions []classifier.Named[classifier.Question]) ordered.Object {
	answers := ordered.Object{}
	for _, q := range questions {
		switch question := q.Question.(type) {
		case classifier.YesNoQuestion:
			answers.Set(q.Name, object(ordered.Object{{Key: "probability", Value: number()}}))
		case classifier.ChoiceQuestion:
			options := question.OptionNames()
			probabilities := ordered.Object{}
			for _, o := range options {
				probabilities.Set(o, number())
			}
			answers.Set(q.Name, object(ordered.Object{
				{Key: "choice", Value: ordered.Object{{Key: schemaType, Value: "string"}, {Key: "enum", Value: options}}},
				{Key: "probabilities", Value: object(probabilities)},
			}))
		case classifier.ScoreQuestion:
			probabilities := ordered.Object{}
			for i := range question.Levels {
				probabilities.Set(strconv.Itoa(i), number())
			}
			answers.Set(q.Name, object(ordered.Object{{Key: "probabilities", Value: object(probabilities)}}))
		}
	}
	return object(answers)
}

// object is a schema for an object with exactly these properties, all required.
func object(properties ordered.Object) ordered.Object {
	return ordered.Object{
		{Key: schemaType, Value: "object"},
		{Key: "properties", Value: properties},
		{Key: "required", Value: properties.Keys()},
		{Key: "additionalProperties", Value: false},
	}
}

// number is the schema of a number.
func number() ordered.Object { return ordered.Object{{Key: schemaType, Value: "number"}} }

// schemaType is the key a schema names its type under.
const schemaType = "type"

// fenced finds a reply wrapped in a code fence.
var fenced = regexp.MustCompile("(?s)```(?:json)?\\s*(.*?)```")

// parse reads the JSON object in the LLM's reply, ignoring fences and prose
// around it.
func parse(reply string) (map[string]any, error) {
	text := strings.TrimSpace(reply)
	if m := fenced.FindStringSubmatch(text); m != nil {
		text = strings.TrimSpace(m[1])
	}
	start, end := strings.Index(text, "{"), strings.LastIndex(text, "}")
	if start == -1 || end == -1 || end < start {
		return nil, fmt.Errorf("%w: the LLM did not answer with a JSON object: %q", classifier.ErrClassifier, reply)
	}
	var data any
	if err := json.Unmarshal([]byte(text[start:end+1]), &data); err != nil {
		return nil, fmt.Errorf("%w: the LLM's answer is not valid JSON: %w", classifier.ErrClassifier, err)
	}
	obj, ok := data.(map[string]any)
	if !ok {
		return nil, fmt.Errorf("%w: the LLM's answer is not a JSON object", classifier.ErrClassifier)
	}
	return obj, nil
}

// result builds a result from the LLM's answer to the question.
func result(question classifier.Question, answer map[string]any) (classifier.Result, error) {
	switch q := question.(type) {
	case classifier.YesNoQuestion:
		p, err := unit(answer, "probability")
		if err != nil {
			return nil, err
		}
		return classifier.YesNoResult{Probability: p}, nil
	case classifier.ChoiceQuestion:
		choice := str(answer["choice"])
		if !q.Has(choice) {
			return nil, fmt.Errorf("%w: the LLM chose %q, which is not an option", classifier.ErrClassifier, choice)
		}
		given, _ := answer["probabilities"].(map[string]any)
		probabilities := make(map[string]float64, len(q.Options))
		for _, o := range q.Options {
			probabilities[o.Name] = clamp(given[o.Name])
		}
		return classifier.ChoiceResult{
			Choice: choice, Probabilities: probabilities, Confidence: probabilities[choice],
		}, nil
	case classifier.ScoreQuestion:
		// The LLM gives a probability per level, by position. The score is the
		// probability-weighted position and the confidence the largest share, as
		// with Jev; a distribution that does not sum to 1 is scaled to.
		given, _ := answer["probabilities"].(map[string]any)
		probabilities := make([]float64, len(q.Levels))
		total := 0.0
		for i := range q.Levels {
			probabilities[i] = clamp(given[strconv.Itoa(i)])
			total += probabilities[i]
		}
		if total <= 0 {
			return nil, fmt.Errorf("%w: the LLM gave no probability for any level", classifier.ErrClassifier)
		}
		r := classifier.ScoreResult{Levels: make([]classifier.ScoreLevel, len(q.Levels))}
		for i, level := range q.Levels {
			p := probabilities[i] / total
			r.Score += float64(i) * p
			r.Confidence = math.Max(r.Confidence, p)
			r.Levels[i] = classifier.ScoreLevel{Level: level, Probability: p}
		}
		return r, nil
	default:
		return nil, fmt.Errorf("%w: unknown question type %T", classifier.ErrClassifier, question)
	}
}

// text writes a value for the LLM: a string as is, structured data as JSON.
func text(value any) (string, error) {
	if s, ok := value.(string); ok {
		return s, nil
	}
	return ordered.Indent(value)
}

// unit reads the number the LLM wrote under key, kept between 0 and 1.
func unit(answer map[string]any, key string) (float64, error) {
	v, ok := toFloat(answer[key])
	if !ok {
		return 0, fmt.Errorf("%w: the LLM's answer has no usable '%s'", classifier.ErrClassifier, key)
	}
	return math.Min(1, math.Max(0, v)), nil
}

// clamp reads a number the LLM wrote, kept between 0 and 1; anything that is
// not a number reads as 0.
func clamp(value any) float64 {
	v, ok := toFloat(value)
	if !ok {
		return 0
	}
	return math.Min(1, math.Max(0, v))
}

// toFloat reads a number the way the LLM may have written it: as a number, a
// numeric string, or a boolean.
func toFloat(value any) (float64, bool) {
	switch v := value.(type) {
	case float64:
		return v, !math.IsNaN(v)
	case string:
		f, err := strconv.ParseFloat(strings.TrimSpace(v), 64)
		return f, err == nil && !math.IsNaN(f)
	case bool:
		if v {
			return 1, true
		}
		return 0, true
	default:
		return 0, false
	}
}

// str reads the choice the LLM wrote as text.
func str(value any) string {
	switch v := value.(type) {
	case nil:
		return ""
	case string:
		return v
	default:
		return fmt.Sprint(v)
	}
}

var _ classifier.Classifier = (*Classifier)(nil)
