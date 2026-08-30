package bus

import (
	"fmt"
	"reflect"
	"sync"
)

// TypeRegistry maps the type names on the wire to the values they name.
//
// It is what stands in for reading a type out of a module path at run time,
// which is how the same machinery works in a dynamic language and is not
// available here. A type that may cross a bus registers itself, once, and the
// registry answers in both directions: the name to write for a value, and a
// fresh value to fill for a name.
//
// A name is the type's own, without its package: "FrameMessage",
// "TranscriptionFrame". Two registered types may not share one, since the name
// is all the far end has to go on.
type TypeRegistry struct {
	mu    sync.RWMutex
	build map[string]func() any
	names map[reflect.Type]string
}

// NewTypeRegistry returns an empty registry.
func NewTypeRegistry() *TypeRegistry {
	return &TypeRegistry{build: map[string]func() any{}, names: map[reflect.Type]string{}}
}

// Register records that values of the type sample points at travel as name.
//
// sample is a zero value of the type, given as a pointer to it (&FrameMessage{}),
// which is both what identifies the type and what the registry copies to build
// a fresh one. Registering the same name twice panics: it is a mistake made
// once, at startup, and a wire name that resolves to two different things
// silently corrupts every message carrying it.
func (r *TypeRegistry) Register(name string, sample any) {
	if sample == nil {
		//nolint:forbidigo // a mistake in the program, refused where it is made
		panic("bus: registering " + name + " with no sample value")
	}
	t := reflect.TypeOf(sample)
	if t.Kind() != reflect.Pointer {
		//nolint:forbidigo // a mistake in the program, refused where it is made
		panic(fmt.Sprintf("bus: %s must be registered with a pointer, got %s", name, t))
	}
	elem := t.Elem()

	r.mu.Lock()
	defer r.mu.Unlock()
	if _, taken := r.build[name]; taken {
		//nolint:forbidigo // a mistake in the program, refused where it is made
		panic("bus: type name " + name + " is already registered")
	}
	r.build[name] = func() any { return reflect.New(elem).Interface() }
	r.names[t] = name
	r.names[elem] = name
}

// NameOf is the wire name for the value's type, reporting false for a type that
// never registered one.
func (r *TypeRegistry) NameOf(v any) (string, bool) {
	if v == nil {
		return "", false
	}
	r.mu.RLock()
	defer r.mu.RUnlock()
	name, ok := r.names[reflect.TypeOf(v)]
	return name, ok
}

// New builds a fresh value of the type name refers to, as a pointer to it. It
// reports false for a name nothing registered, which is what a message from a
// peer that knows a type this one does not looks like.
func (r *TypeRegistry) New(name string) (any, bool) {
	r.mu.RLock()
	build, ok := r.build[name]
	r.mu.RUnlock()
	if !ok {
		return nil, false
	}
	return build(), true
}

// Names lists every registered wire name, for reporting what a process can
// carry.
func (r *TypeRegistry) Names() []string {
	r.mu.RLock()
	defer r.mu.RUnlock()
	out := make([]string, 0, len(r.build))
	for name := range r.build {
		out = append(out, name)
	}
	return out
}
