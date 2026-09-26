package sentencex

import (
	"slices"
	"testing"
	"time"
)

func TestLanguageFallbacks(t *testing.T) {
	if got, ok := getFallbacks("zh-cn"); !ok || !slices.Equal(got, []string{"zh-hans", "zh", "zh-hant"}) {
		t.Errorf("getFallbacks(zh-cn) = %q, %v", got, ok)
	}
	if got, ok := getFallbacks("de-at"); !ok || !slices.Equal(got, []string{"de"}) {
		t.Errorf("getFallbacks(de-at) = %q, %v", got, ok)
	}
	if got, ok := getFallbacks("en"); ok {
		t.Errorf("getFallbacks(en) = %q, want none", got)
	}
}

func TestLanguageFallbacksDirectAccess(t *testing.T) {
	if len(languageFallbacks) == 0 {
		t.Error("the fallback table is empty")
	}
}

func TestLanguageFallbacksLookupSpeed(t *testing.T) {
	start := time.Now()
	for range 1000 {
		getFallbacks("zh-cn")
		getFallbacks("de-at")
		getFallbacks("en")
		getFallbacks("non-existent")
	}
	if d := time.Since(start); d >= 100*time.Millisecond {
		t.Errorf("map lookups too slow: %v", d)
	}
}
