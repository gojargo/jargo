package llm

import (
	"context"
	"errors"
	"sync"
	"testing"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/service/settings"
)

// How an LLM service attaches, wires and composes its extensions. The tests
// that run frames through a pipeline are in extension_test.go.

// recorder is an extension that records its setup and cleanup and contributes
// the instruction it was given.
type recorder struct {
	BaseLLMExtension
	order       int
	instruction string

	mu        sync.Mutex
	setUp     bool
	cleanedUp bool
}

func (r *recorder) Order() int                { return r.order }
func (r *recorder) SystemInstruction() string { return r.instruction }

func (r *recorder) Setup(context.Context, processor.Setup) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.setUp = true
	return nil
}

func (r *recorder) Cleanup(context.Context) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.cleanedUp = true
	return nil
}

func (r *recorder) wasSetUp() bool {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.setUp
}

func (r *recorder) wasCleanedUp() bool {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.cleanedUp
}

// alpha, beta and gamma are recorders of types of their own, for a service
// that holds several: one of each type.
type (
	alpha struct{ *recorder }
	beta  struct{ *recorder }
	gamma struct{ *recorder }
)

// extensionGen stands in for a model; it keeps settings of its own, which the
// service reads its initial extensions from.
type extensionGen struct{ settings settings.LLM }

func (g *extensionGen) Settings() any { return &g.settings }

func (g *extensionGen) Generate(context.Context, *frames.LLMContext, Emit) error { return nil }

// extensionService builds a service with the given base prompt and the
// extensions its settings are built with.
func extensionService(prompt string, extensions ...any) *Base {
	gen := &extensionGen{}
	if len(extensions) > 0 {
		gen.settings.Extensions = settings.Set(settings.Extensions(extensions))
	}
	svc := New("FakeLLM", gen)
	if prompt != "" {
		svc.SetSystemInstruction(prompt)
	}
	return svc
}

// setUpService sets the service up as a pipeline would, and cleans it up when
// the test ends.
func setUpService(t *testing.T, svc *Base) {
	t.Helper()
	if err := svc.Setup(t.Context(), processor.Setup{}); err != nil {
		t.Fatalf("Setup: %v", err)
	}
	t.Cleanup(func() { _ = svc.Cleanup(context.Background()) })
}

// extensionsDelta is a settings delta adding the given extension entries.
func extensionsDelta(entries ...any) *settings.LLM {
	var delta settings.LLM
	delta.Extensions = settings.Set(settings.Extensions(entries))
	return &delta
}

// recordersOf are the recorders attached to the service, of whatever type.
func recordersOf(svc *Base) []LLMExtension {
	var out []LLMExtension
	for _, e := range svc.Extensions() {
		switch e.(type) {
		case *recorder, *alpha, *beta, *gamma:
			out = append(out, e)
		}
	}
	return out
}

func TestExtensionsAreSetUpAndCleanedUpWithTheService(t *testing.T) {
	extension := &recorder{}
	svc := extensionService("", extension)

	if err := svc.Setup(t.Context(), processor.Setup{}); err != nil {
		t.Fatal(err)
	}
	if !extension.wasSetUp() {
		t.Error("the extension was not set up with the service")
	}
	if extension.AttachedService() != svc {
		t.Error("the extension does not know the service it is attached to")
	}
	if err := svc.Cleanup(t.Context()); err != nil {
		t.Fatal(err)
	}
	if !extension.wasCleanedUp() {
		t.Error("the extension was not cleaned up with the service")
	}
	// The service's handlers are gone with it.
	if extension.AttachedService() != nil {
		t.Error("the extension still holds the service's handlers")
	}
}

func TestAnExtensionAddedInHandIsSetUpWithTheService(t *testing.T) {
	svc := extensionService("APP")
	extension := &recorder{instruction: "ADDED"}

	if err := svc.AddExtension(extension); err != nil {
		t.Fatal(err)
	}
	setUpService(t, svc)

	if !extension.wasSetUp() {
		t.Error("the extension was not set up with the service")
	}
	if got := svc.SystemInstruction(); got != "APP\n\nADDED" {
		t.Errorf("SystemInstruction() = %q, want APP then ADDED", got)
	}
}

func TestAddingAnExtensionToARunningServiceIsRefused(t *testing.T) {
	svc := extensionService("")
	setUpService(t, svc)

	if err := svc.AddExtension(&recorder{}); !errors.Is(err, ErrExtensionAfterSetup) {
		t.Errorf("AddExtension() = %v, want ErrExtensionAfterSetup", err)
	}
}

func TestInstructionsComposeAfterTheBaseInOrder(t *testing.T) {
	svc := extensionService("APP",
		&alpha{&recorder{order: 1, instruction: "LATER"}},
		&beta{&recorder{order: 0, instruction: "EARLIER"}},
		&gamma{&recorder{order: 0}},
	)

	if got := svc.SystemInstruction(); got != "APP\n\nEARLIER\n\nLATER" {
		t.Errorf("SystemInstruction() = %q, want APP, EARLIER, LATER", got)
	}
}

func TestASettingsUpdateBuildsAndAddsExtensions(t *testing.T) {
	first := &alpha{&recorder{instruction: "FIRST"}}
	svc := extensionService("", first)
	setUpService(t, svc)

	err := updateLLMSettings(t, svc, extensionsDelta(
		func() LLMExtension { return &beta{&recorder{instruction: "SECOND"}} },
	))
	if err != nil {
		t.Fatal(err)
	}

	recorders := recordersOf(svc)
	if len(recorders) != 2 || recorders[0] != LLMExtension(first) {
		t.Fatalf("recorders = %v, want the first then the one built", recorders)
	}
	second, ok := recorders[1].(*beta)
	if !ok {
		t.Fatalf("the second recorder is %T, want *beta", recorders[1])
	}
	if !second.wasSetUp() {
		t.Error("an extension attached to a running service was not set up")
	}
	if got := svc.SystemInstruction(); got != "FIRST\n\nSECOND" {
		t.Errorf("SystemInstruction() = %q, want FIRST then SECOND", got)
	}
}

func TestGetExtensionFindsTheAttachedExtensionOfAClass(t *testing.T) {
	a := &alpha{&recorder{}}
	svc := extensionService("", a)

	if got, ok := GetExtension[*alpha](svc); !ok || got != a {
		t.Errorf("GetExtension[*alpha] = %v, %v, want the one attached", got, ok)
	}
	if _, ok := GetExtension[*recorder](svc); ok {
		t.Error("GetExtension[*recorder] found an extension of another type")
	}
	if _, ok := GetExtension[*beta](svc); ok {
		t.Error("GetExtension[*beta] found one where none is attached")
	}
}

func TestASecondExtensionOfAClassIsRefused(t *testing.T) {
	// Built with two, the service holds the first and reports the second.
	built := extensionService("", &recorder{}, &recorder{})
	if n := len(recordersOf(built)); n != 1 {
		t.Errorf("a service built with two recorders holds %d, want 1", n)
	}

	held := &recorder{}
	svc := extensionService("", held)
	if err := svc.AddExtension(&recorder{}); !errors.Is(err, ErrDuplicateExtension) {
		t.Errorf("AddExtension() = %v, want ErrDuplicateExtension", err)
	}
	err := updateLLMSettings(t, svc, extensionsDelta(func() LLMExtension { return &recorder{} }))
	if !errors.Is(err, ErrDuplicateExtension) {
		t.Errorf("a settings update = %v, want ErrDuplicateExtension", err)
	}
	if recorders := recordersOf(svc); len(recorders) != 1 || recorders[0] != LLMExtension(held) {
		t.Errorf("recorders = %v, want only the one held", recorders)
	}
}

func TestConstructorSettingsBuildACallableEntry(t *testing.T) {
	svc := extensionService("", func() LLMExtension { return &recorder{} })

	if _, ok := GetExtension[*recorder](svc); !ok {
		t.Error("the function in the settings did not build an extension")
	}
}

func TestASettingsUpdateTakesAnExtensionInHand(t *testing.T) {
	svc := extensionService("")
	extension := &recorder{}

	if err := updateLLMSettings(t, svc, extensionsDelta(extension)); err != nil {
		t.Fatal(err)
	}

	if got, ok := GetExtension[*recorder](svc); !ok || got != extension {
		t.Errorf("GetExtension[*recorder] = %v, want the extension the update carried", got)
	}
}

// A second service that an update with an extension in hand reaches refuses
// it.
func TestAnExtensionBelongsToOneService(t *testing.T) {
	extension := &recorder{}
	one, another := extensionService(""), extensionService("")
	setUpService(t, one)
	setUpService(t, another)

	if err := updateLLMSettings(t, one, extensionsDelta(extension)); err != nil {
		t.Fatal(err)
	}
	if extension.AttachedService() != one {
		t.Fatal("the extension is not attached to the service the update reached")
	}
	err := updateLLMSettings(t, another, extensionsDelta(extension))
	if !errors.Is(err, ErrExtensionAttachedElsewhere) {
		t.Errorf("the second service took the extension: %v", err)
	}
	if extension.AttachedService() != one {
		t.Error("refusing the extension took it away from the service holding it")
	}
}

func TestASettingsUpdateBuildsAnExtensionPerService(t *testing.T) {
	delta := extensionsDelta(func() LLMExtension { return &recorder{} })
	one, another := extensionService(""), extensionService("")

	if err := updateLLMSettings(t, one, delta); err != nil {
		t.Fatal(err)
	}
	if err := updateLLMSettings(t, another, delta); err != nil {
		t.Fatal(err)
	}

	builtForOne, _ := GetExtension[*recorder](one)
	builtForAnother, _ := GetExtension[*recorder](another)
	if builtForOne == nil || builtForAnother == nil || builtForOne == builtForAnother {
		t.Errorf("built %p and %p, want one extension for each service", builtForOne, builtForAnother)
	}
}

// Every service holds the async-tool extension from construction, after the
// extensions an application attaches.
func TestEveryServiceHasTheAsyncToolExtension(t *testing.T) {
	svc := extensionService("", &recorder{})

	extensions := svc.Extensions()
	if _, ok := extensions[len(extensions)-1].(*AsyncToolLLMExtension); !ok {
		t.Errorf("extensions = %v, want the async-tool extension farthest from the service", extensions)
	}
}

// The settings hold the extensions attached, in the order they were attached.
func TestTheSettingsHoldTheExtensions(t *testing.T) {
	extension := &recorder{}
	svc := extensionService("", extension)

	v, _ := settings.Get(svc.CurrentSettings(), "extensions")
	held, _ := v.(settings.Extensions)
	if len(held) != 2 || held[1] != any(extension) {
		t.Errorf("the settings hold %v, want the async-tool extension then the one attached", held)
	}
}
