package cartesia

import (
	"reflect"
	"testing"

	uctx "github.com/gojargo/jargo/utils/context"
)

func TestARequestInASpeedDeliveryStatesItsSpeed(t *testing.T) {
	volume, speed := 1.2, 0.9
	for _, tt := range []struct {
		config     *GenerationConfig
		expression map[string]any
		want       map[string]any
	}{
		{nil, map[string]any{"speed": 0.7}, map[string]any{"speed": 0.7}},
		{&GenerationConfig{Volume: &volume}, map[string]any{"speed": 1.3}, map[string]any{"volume": 1.2, "speed": 1.3}},
		// Without a speed delivery, the configured speed is kept.
		{&GenerationConfig{Speed: &speed}, map[string]any{}, map[string]any{"speed": 0.9}},
		{nil, nil, nil},
	} {
		if got := generationConfig(tt.config, tt.expression); !reflect.DeepEqual(got, tt.want) {
			t.Errorf("generationConfig(%+v, %v) = %v, want %v", tt.config, tt.expression, got, tt.want)
		}
	}
}

func TestTheLaughterTokenIsKept(t *testing.T) {
	// It marks when the laughter sound is performed.
	got, _ := normalizeWordTimings(&wsWordTimings{
		Words: []string{"ancient.", "[laughter]", "Let"},
		Start: []float64{0, 0.5, 1},
	}, false)

	want := []uctx.WordTiming{{Word: "ancient.", Offset: 0}, {Word: "[laughter]", Offset: 0.5}, {Word: "Let", Offset: 1}}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("words = %+v, want %+v", got, want)
	}
}
