package embeds_test

import (
	"reflect"
	"testing"

	"github.com/gojargo/jargo/internal/embeds"
)

type base struct{ A int }

type middle struct{ base }

type leaf struct{ middle }

type other struct{ B int }

func TestIs(t *testing.T) {
	cases := []struct {
		got, want any
		is        bool
	}{
		{&base{}, &base{}, true},
		{&middle{}, &base{}, true},
		{&leaf{}, &base{}, true},
		{&leaf{}, &middle{}, true},
		{&base{}, &middle{}, false},
		{&other{}, &base{}, false},
	}
	for _, c := range cases {
		if got := embeds.Is(reflect.TypeOf(c.got), reflect.TypeOf(c.want)); got != c.is {
			t.Errorf("Is(%T, %T) = %v, want %v", c.got, c.want, got, c.is)
		}
	}
}
