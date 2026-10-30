package speech

import (
	"reflect"
	"testing"

	ttsadapter "github.com/gojargo/jargo/adapter/tts"
	ttsazure "github.com/gojargo/jargo/adapter/tts/azure"
)

func TestExpressionAdapterFollowsTheVoice(t *testing.T) {
	for _, tt := range []struct {
		voice   string
		adapter ttsadapter.ExpressionAdapter
	}{
		{"en-US-Ava:DragonHDLatestNeural", ttsazure.AzureHDExpressionAdapter{}},
		{"en-US-Ava:DragonHDOmniLatestNeural", ttsazure.AzureHDExpressionAdapter{}},
		{"en-US-Tiana:DragonHDFlashLatestNeural", ttsadapter.BaseExpressionAdapter{}},
		{"en-US-SaraNeural", ttsadapter.BaseExpressionAdapter{}},
	} {
		s := &ttsSynthesizer{cfg: TTSConfig{Voice: tt.voice}}
		if got := s.ExpressionAdapter(); reflect.TypeOf(got) != reflect.TypeOf(tt.adapter) {
			t.Errorf("%s: adapter %T, want %T", tt.voice, got, tt.adapter)
		}
	}
}
