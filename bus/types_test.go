package bus_test

import (
	"strings"
	"testing"

	"github.com/gojargo/jargo/bus"
	"github.com/gojargo/jargo/frames"
)

// TestRegistryAnswersBothWays checks the registry does the two things the
// serializer needs of it: the name to write for a value, and a fresh value to
// fill for a name.
func TestRegistryAnswersBothWays(t *testing.T) {
	r := bus.NewTypeRegistry()
	r.Register("FrameMessage", &bus.FrameMessage{})

	name, known := r.NameOf(&bus.FrameMessage{})
	if !known || name != "FrameMessage" {
		t.Errorf("NameOf = %q/%v, want FrameMessage/true", name, known)
	}

	built, ok := r.New("FrameMessage")
	if !ok {
		t.Fatal("New returned nothing for a registered name")
	}
	if _, isMessage := built.(*bus.FrameMessage); !isMessage {
		t.Errorf("New built a %T, want a *bus.FrameMessage", built)
	}
}

// TestRegistryBuildsAFreshValueEachTime checks two messages of one type do not
// share the value they were built from, which would have a bus rewriting a
// message that had already been delivered.
func TestRegistryBuildsAFreshValueEachTime(t *testing.T) {
	r := bus.NewTypeRegistry()
	r.Register("FrameMessage", &bus.FrameMessage{})

	first, _ := r.New("FrameMessage")
	second, _ := r.New("FrameMessage")
	if first == second {
		t.Fatal("New returned the same value twice")
	}
	one, ok := first.(*bus.FrameMessage)
	if !ok {
		t.Fatalf("New built a %T", first)
	}
	other, ok := second.(*bus.FrameMessage)
	if !ok {
		t.Fatalf("New built a %T", second)
	}
	one.Bridge = "one"
	if other.Bridge != "" {
		t.Errorf("the second value saw the first's Bridge = %q", other.Bridge)
	}
}

// TestRegistryReportsAnUnknownName checks a name nothing registered is reported
// rather than guessed at, which is what a peer carrying a type this process
// does not have looks like.
func TestRegistryReportsAnUnknownName(t *testing.T) {
	r := bus.NewTypeRegistry()
	if _, ok := r.New("NoSuchThing"); ok {
		t.Error("New answered for a name nothing registered")
	}
	if _, ok := r.NameOf(&bus.FrameMessage{}); ok {
		t.Error("NameOf answered for a type nothing registered")
	}
}

// TestRegistryRefusesADuplicateName checks the same name cannot mean two
// things. It is a mistake made once, at startup, and a name that resolves to
// two types would corrupt every message carrying it.
func TestRegistryRefusesADuplicateName(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Error("registering a name twice was allowed")
		}
	}()
	r := bus.NewTypeRegistry()
	r.Register("Thing", &bus.FrameMessage{})
	r.Register("Thing", &bus.TTSSpeakMessage{})
}

// TestBuiltInTypesAreRegistered checks the registry the serializer uses by
// default knows the bus's own messages and the frames a bridged worker may be
// handed. A frame missing from it cannot cross a bus at all.
func TestBuiltInTypesAreRegistered(t *testing.T) {
	for _, sample := range []any{
		&bus.FrameMessage{},
		&bus.TTSSpeakMessage{},
		&bus.JobRequestMessage{},
		&bus.WorkerReadyMessage{},
		&frames.TranscriptionFrame{},
		&frames.LLMContextFrame{},
		&frames.InputAudioRawFrame{},
		&frames.EndFrame{},
		&frames.LLMContext{},
	} {
		if _, ok := bus.Types.NameOf(sample); !ok {
			t.Errorf("%T is not registered, so it cannot cross a bus", sample)
		}
	}
}

// TestBuiltInNamesAreBare checks a registered name is the type's own, without
// its package, since that name is all the far end has to go on.
func TestBuiltInNamesAreBare(t *testing.T) {
	for _, name := range bus.Types.Names() {
		if strings.Contains(name, ".") || strings.Contains(name, "/") {
			t.Errorf("registered name %q is qualified, want the bare type name", name)
		}
	}
}
