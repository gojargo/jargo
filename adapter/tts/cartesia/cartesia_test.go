package cartesia_test

import (
	"reflect"
	"slices"
	"testing"

	"github.com/gojargo/jargo/adapter/tts/cartesia"
	"github.com/gojargo/jargo/expressive"
)

func names(terms []expressive.ExpressionTerm) []string {
	out := make([]string, 0, len(terms))
	for _, t := range terms {
		out = append(out, t.Name)
	}
	return out
}

func TestCapabilities(t *testing.T) {
	c := cartesia.CartesiaExpressionAdapter{}.Capabilities()
	delivery := names(c.Delivery)

	if !slices.Contains(delivery, "excited") || slices.Contains(delivery, "neutral") {
		t.Fatalf("delivery = %q", delivery)
	}
	if got := names(c.Sounds); !slices.Equal(got, []string{"laughter"}) {
		t.Fatalf("sounds = %q", got)
	}
	if c.DeliveryScope != expressive.DeliveryScopeSentence || c.Emphasis {
		t.Fatalf("capabilities = %+v", c)
	}
}

func TestEmotionFormattedWhereItChanges(t *testing.T) {
	a := cartesia.CartesiaExpressionAdapter{}
	for _, tt := range []struct {
		delivery, previous []string
		want               string
	}{
		{[]string{"excited"}, nil, `<emotion value="excited"/> Hi.`},
		{[]string{"excited"}, []string{"excited"}, "Hi."},
		{[]string{"sad"}, []string{"excited"}, `<emotion value="sad"/> Hi.`},
		{[]string{}, nil, "Hi."},
		// A return to the baseline formats neutral.
		{[]string{}, []string{"sad"}, `<emotion value="neutral"/> Hi.`},
		// Of several delivery names the first is formatted.
		{[]string{"sad", "calm"}, nil, `<emotion value="sad"/> Hi.`},
		{[]string{"sad", "tired"}, []string{"sad", "calm"}, "Hi."},
	} {
		if got := a.FormatText("Hi.", tt.delivery, tt.previous); got != tt.want {
			t.Errorf("FormatText(%q after %q) = %q, want %q", tt.delivery, tt.previous, got, tt.want)
		}
	}
}

func TestSlowAndFastSetTheSpeedNotTheEmotion(t *testing.T) {
	a := cartesia.CartesiaExpressionAdapter{}
	delivery := names(a.Capabilities().Delivery)
	if !slices.Contains(delivery, "slow") || !slices.Contains(delivery, "fast") {
		t.Fatalf("delivery = %q", delivery)
	}
	if got := a.FormatText("Hi.", []string{"slow", "sad"}, nil); got != `<emotion value="sad"/> Hi.` {
		t.Errorf("slow sad = %q", got)
	}
	if got := a.FormatText("Hi.", []string{"sad", "slow"}, []string{"sad"}); got != "Hi." {
		t.Errorf("sad slow after sad = %q", got)
	}
	if got := a.RequestSettings([]string{"sad", "slow"}); !reflect.DeepEqual(got, map[string]any{"speed": 0.7}) {
		t.Errorf("settings = %v", got)
	}
	if got := a.RequestSettings([]string{"fast"}); !reflect.DeepEqual(got, map[string]any{"speed": 1.3}) {
		t.Errorf("settings = %v", got)
	}
	if got := a.RequestSettings([]string{"sad"}); len(got) != 0 {
		t.Errorf("settings = %v", got)
	}
}

func TestSoundsAndPauses(t *testing.T) {
	a := cartesia.CartesiaExpressionAdapter{}
	if got := a.FormatSound("laughter"); got != " [laughter] " {
		t.Errorf("sound = %q", got)
	}
	if got := a.FormatPause(600); got != ` <break time="600ms"/> ` {
		t.Errorf("pause = %q", got)
	}
}
