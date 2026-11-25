package mistral

import (
	"reflect"
	"testing"

	"github.com/gojargo/jargo/expressive"
)

func TestExpressionAdapterFollowsTheModel(t *testing.T) {
	for _, tt := range []struct {
		model   string
		adapter expressive.ExpressionAdapter
	}{
		{"voxtral-mini-tts-3-0", ExpressionAdapter{}},
		{"voxtral-mini-tts-3", ExpressionAdapter{}},
		{"voxtral-mini-tts-2603", expressive.BaseExpressionAdapter{}},
	} {
		s := &ttsSynthesizer{cfg: TTSConfig{Model: tt.model}}
		if got := s.ExpressionAdapter(); reflect.TypeOf(got) != reflect.TypeOf(tt.adapter) {
			t.Errorf("%s: adapter %T, want %T", tt.model, got, tt.adapter)
		}
	}
}
