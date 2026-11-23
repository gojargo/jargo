package decisions

import (
	"fmt"
	"math"
	"strconv"
	"strings"
	"unicode/utf8"

	"github.com/gojargo/jargo/classifier"
	"github.com/gojargo/jargo/internal/ordered"
)

// Questions and answers between the classifier types and the question and
// answer format of the Zoom Decisions API.

// questionsToWire is the questions in Zoom's format, in order, each named by
// the name it was asked under.
func questionsToWire(questions []classifier.Named[classifier.Question]) ([]Question, error) {
	wire := make([]Question, 0, len(questions))
	for _, q := range questions {
		w, err := questionToWire(q.Name, q.Question)
		if err != nil {
			return nil, err
		}
		wire = append(wire, w)
	}
	return wire, nil
}

// questionToWire is one question in Zoom's format.
//
// A yes or no question becomes a predicate, a choice question a choice and a
// score question a score. Structured instructions, descriptions and levels are
// sent as JSON.
func questionToWire(name string, question classifier.Question) (Question, error) {
	switch q := question.(type) {
	case classifier.YesNoQuestion:
		return predicateToWire(name, q)
	case classifier.ChoiceQuestion:
		return choiceToWire(name, q)
	case classifier.ScoreQuestion:
		instructions, err := text(q.Instructions)
		if err != nil {
			return Question{}, err
		}
		levels := make([]string, 0, len(q.Levels))
		for _, level := range q.Levels {
			label, err := text(level)
			if err != nil {
				return Question{}, err
			}
			levels = append(levels, label)
		}
		return Question{Type: typeScore, Name: name, Instructions: instructions, Levels: levels}, nil
	default:
		return Question{}, fmt.Errorf("%w: unknown question type %T", classifier.ErrClassifier, question)
	}
}

// predicateToWire writes a yes or no question. A predicate has no field for
// what counts as each answer, so it goes in the instructions.
func predicateToWire(name string, q classifier.YesNoQuestion) (Question, error) {
	instructions, err := text(q.Instructions)
	if err != nil {
		return Question{}, err
	}
	lines := []string{instructions}
	for _, meaning := range []struct {
		prefix string
		value  any
	}{{"Yes means: ", q.Yes}, {"No means: ", q.No}} {
		if meaning.value == nil {
			continue
		}
		t, err := text(meaning.value)
		if err != nil {
			return Question{}, err
		}
		lines = append(lines, meaning.prefix+t)
	}
	return Question{Type: typePredicate, Name: name, Instructions: strings.Join(lines, "\n")}, nil
}

// choiceToWire writes a choice question, its options in order.
func choiceToWire(name string, q classifier.ChoiceQuestion) (Question, error) {
	instructions, err := text(q.Instructions)
	if err != nil {
		return Question{}, err
	}
	choices := make([]Choice, 0, len(q.Options))
	for _, option := range q.Options {
		choice := Choice{Value: option.Name}
		if option.Description != nil {
			description, err := text(option.Description)
			if err != nil {
				return Question{}, err
			}
			choice.Description = &description
		}
		choices = append(choices, choice)
	}
	return Question{Type: typeChoice, Name: name, Instructions: instructions, Choices: choices}, nil
}

// resultFromWire is the result for one question, from Zoom's answer to it: of
// the type the question calls for. It fails with an error wrapping
// classifier.ErrClassifier if Zoom refused the question, or the answer doesn't
// fit it.
//
// Zoom gives no confidence. A choice's is the chosen option's probability, and
// a score's the likeliest level's.
func resultFromWire(name string, question classifier.Question, answer map[string]any) (classifier.Result, error) {
	if answer[keyType] == typeRefusal {
		return nil, fmt.Errorf("%w: Zoom Decisions refused to answer %q", classifier.ErrClassifier, name)
	}
	switch q := question.(type) {
	case classifier.YesNoQuestion:
		p, err := number(answer, "probability")
		if err != nil {
			return nil, err
		}
		return classifier.YesNoResult{Probability: p}, nil
	case classifier.ChoiceQuestion:
		choice := valueText(answer["choice"])
		if !q.Has(choice) {
			return nil, fmt.Errorf("%w: Zoom Decisions chose %q, which is not an option",
				classifier.ErrClassifier, choice)
		}
		probabilities, err := probabilitiesOf(answer, q.OptionNames())
		if err != nil {
			return nil, err
		}
		return classifier.ChoiceResult{
			Choice: choice, Probabilities: probabilities, Confidence: probabilities[choice],
		}, nil
	case classifier.ScoreQuestion:
		return scoreFromWire(q, answer)
	default:
		return nil, fmt.Errorf("%w: unknown question type %T", classifier.ErrClassifier, question)
	}
}

// scoreFromWire builds a score result. Zoom keys level probabilities by
// position.
func scoreFromWire(q classifier.ScoreQuestion, answer map[string]any) (classifier.Result, error) {
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
	levels := make([]classifier.ScoreLevel, len(q.Levels))
	confidence := math.Inf(-1)
	for i, level := range q.Levels {
		p := probabilities[positions[i]]
		levels[i] = classifier.ScoreLevel{Level: level, Probability: p}
		confidence = math.Max(confidence, p)
	}
	return classifier.ScoreResult{Score: score, Levels: levels, Confidence: confidence}, nil
}

// text is a value as text for Zoom: a string as is, structured data as JSON
// indented by two spaces, with everything outside ASCII escaped.
func text(value any) (string, error) {
	if s, ok := value.(string); ok {
		return s, nil
	}
	indented, err := ordered.Indent(value)
	if err != nil {
		return "", fmt.Errorf("%w: cannot write %T as JSON: %w", classifier.ErrClassifier, value, err)
	}
	var b strings.Builder
	for _, r := range indented {
		switch {
		case r < utf8.RuneSelf:
			b.WriteRune(r)
		case r > 0xffff:
			// Outside the basic plane: escaped as two UTF-16 code units.
			r -= 0x10000
			fmt.Fprintf(&b, `\u%04x\u%04x`, 0xd800+(r>>10), 0xdc00+(r&0x3ff))
		default:
			fmt.Fprintf(&b, `\u%04x`, r)
		}
	}
	return b.String(), nil
}

// number reads the number Zoom wrote under key, or an error if it wrote none.
func number(answer map[string]any, key string) (float64, error) {
	v, ok := toFloat(answer[key])
	if !ok {
		return 0, fmt.Errorf("%w: Zoom Decisions answer has no usable '%s'", classifier.ErrClassifier, key)
	}
	return v, nil
}

// probabilitiesOf reads the probability Zoom gave for each key, 0 for the ones
// it left out.
//
// Zoom lists probabilities as items with a value and a probability, where the
// value is a choice's option or a level's position.
func probabilitiesOf(answer map[string]any, keys []string) (map[string]float64, error) {
	given, _ := answer["probabilities"].([]any)
	byValue := make(map[string]any, len(given))
	for _, item := range given {
		if obj, ok := item.(map[string]any); ok {
			byValue[valueText(obj["value"])] = obj["probability"]
		}
	}
	out := make(map[string]float64, len(keys))
	for _, key := range keys {
		v, present := byValue[key]
		if !present {
			out[key] = 0
			continue
		}
		f, ok := toFloat(v)
		if !ok {
			return nil, fmt.Errorf("%w: Zoom Decisions answer has an unusable probability: %v",
				classifier.ErrClassifier, v)
		}
		out[key] = f
	}
	return out, nil
}

// toFloat reads a number as Zoom may have written it.
func toFloat(value any) (float64, bool) {
	switch v := value.(type) {
	case float64:
		return v, true
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

// valueText reads a choice or a probability's value as text: an option as it
// is, a level's position as its digits.
func valueText(value any) string {
	switch v := value.(type) {
	case nil:
		return ""
	case string:
		return v
	case float64:
		return strconv.FormatFloat(v, 'f', -1, 64)
	default:
		return fmt.Sprint(v)
	}
}
