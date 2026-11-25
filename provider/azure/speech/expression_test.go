package speech

import (
	"reflect"
	"testing"

	"github.com/gojargo/jargo/expressive"
)

func TestExpressionAdapterFollowsTheVoice(t *testing.T) {
	for _, tt := range []struct {
		voice   string
		adapter expressive.ExpressionAdapter
	}{
		{"en-US-Ava:DragonHDLatestNeural", HDExpressionAdapter{}},
		{"en-US-Ava:DragonHDOmniLatestNeural", HDExpressionAdapter{}},
		{"en-US-Tiana:DragonHDFlashLatestNeural", expressive.BaseExpressionAdapter{}},
		{"en-US-SaraNeural", expressive.BaseExpressionAdapter{}},
	} {
		s := &ttsSynthesizer{cfg: TTSConfig{Voice: tt.voice}}
		if got := s.ExpressionAdapter(); reflect.TypeOf(got) != reflect.TypeOf(tt.adapter) {
			t.Errorf("%s: adapter %T, want %T", tt.voice, got, tt.adapter)
		}
	}
}
