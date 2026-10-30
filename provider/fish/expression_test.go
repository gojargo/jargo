package fish

import (
	"reflect"
	"testing"

	ttsadapter "github.com/gojargo/jargo/adapter/tts"
	ttsfish "github.com/gojargo/jargo/adapter/tts/fish"
)

func TestExpressionAdapterFollowsTheModel(t *testing.T) {
	for _, tt := range []struct {
		model   string
		adapter ttsadapter.ExpressionAdapter
	}{
		{"s2.1-pro", ttsfish.FishAudioS2ExpressionAdapter{}},
		{"s2-pro", ttsfish.FishAudioS2ExpressionAdapter{}},
		{"s1", ttsadapter.BaseExpressionAdapter{}},
		{"drama-3", ttsadapter.BaseExpressionAdapter{}},
	} {
		s := &synthesizer{cfg: Config{Model: tt.model}}
		if got := s.ExpressionAdapter(); reflect.TypeOf(got) != reflect.TypeOf(tt.adapter) {
			t.Errorf("%s: adapter %T, want %T", tt.model, got, tt.adapter)
		}
	}
}
