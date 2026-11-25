package fish

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
		{"s2.1-pro", S2ExpressionAdapter{}},
		{"s2-pro", S2ExpressionAdapter{}},
		{"s1", expressive.BaseExpressionAdapter{}},
		{"drama-3", expressive.BaseExpressionAdapter{}},
	} {
		s := &synthesizer{cfg: Config{Model: tt.model}}
		if got := s.ExpressionAdapter(); reflect.TypeOf(got) != reflect.TypeOf(tt.adapter) {
			t.Errorf("%s: adapter %T, want %T", tt.model, got, tt.adapter)
		}
	}
}
