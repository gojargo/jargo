// Package embeds answers whether one type is another or is built on it, the
// way a type that embeds another inherits from it: the Go counterpart of asking
// whether a value is an instance of a class or of one derived from it.
package embeds

import (
	"reflect"
	"sync"
)

//nolint:gochecknoglobals // a cache of answers about types, which never change
var cache sync.Map

// Is reports whether got is want, or a type that embeds want, directly or
// through a type it embeds. Pointers are looked through, so *LLMTextFrame is
// built on *TextFrame when LLMTextFrame embeds TextFrame.
func Is(got, want reflect.Type) bool {
	if got == nil || want == nil {
		return false
	}
	if got == want {
		return true
	}
	key := [2]reflect.Type{got, want}
	if v, ok := cache.Load(key); ok {
		b, _ := v.(bool)
		return b
	}
	is := embeds(deref(got), deref(want), 0)
	cache.Store(key, is)
	return is
}

// maxDepth bounds the walk through embedded types.
const maxDepth = 8

// embeds reports whether the struct got is want or embeds it.
func embeds(got, want reflect.Type, depth int) bool {
	if got == want {
		return true
	}
	if got.Kind() != reflect.Struct || depth >= maxDepth {
		return false
	}
	for f := range got.Fields() {
		if f.Anonymous && embeds(deref(f.Type), want, depth+1) {
			return true
		}
	}
	return false
}

// deref looks through a pointer to what it points at.
func deref(t reflect.Type) reflect.Type {
	if t.Kind() == reflect.Pointer {
		return t.Elem()
	}
	return t
}
