package mistral

import (
	"reflect"
	"testing"

	ttsadapter "github.com/gojargo/jargo/adapter/tts"
	ttsmistral "github.com/gojargo/jargo/adapter/tts/mistral"
)

func TestExpressionAdapterFollowsTheModel(t *testing.T) {
	for _, tt := range []struct {
		model   string
		adapter ttsadapter.ExpressionAdapter
	}{
		{"voxtral-mini-tts-3-0", ttsmistral.MistralExpressionAdapter{}},
		{"voxtral-mini-tts-3", ttsmistral.MistralExpressionAdapter{}},
		{"voxtral-mini-tts-2603", ttsadapter.BaseExpressionAdapter{}},
	} {
		s := &ttsSynthesizer{cfg: TTSConfig{Model: tt.model}}
		if got := s.ExpressionAdapter(); reflect.TypeOf(got) != reflect.TypeOf(tt.adapter) {
			t.Errorf("%s: adapter %T, want %T", tt.model, got, tt.adapter)
		}
	}
}
