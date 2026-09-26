package sentencex

import "testing"

func TestGetBoundaryExtendSumsRunInBytes(t *testing.T) {
	lang := languages["ja"]()

	tests := []struct {
		word   string
		want   int
		wantOK bool
	}{
		{". X", 2, true},
		{"。次", 3, true},
		{"。。次", 6, true},
		{"\u3000\u3000X", 6, true},
		{"", 0, true},
		{" foo", 0, false},
	}
	for _, tt := range tests {
		got, ok := lang.getBoundaryExtend(tt.word)
		if got != tt.want || ok != tt.wantOK {
			t.Errorf("getBoundaryExtend(%q) = %d, %v; want %d, %v", tt.word, got, ok, tt.want, tt.wantOK)
		}
	}
}
