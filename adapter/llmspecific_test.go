package adapter_test

import (
	"errors"
	"testing"

	"github.com/gojargo/jargo/adapter"
	"github.com/gojargo/jargo/frames"
)

// namedAdapter stands in for a provider adapter, which is all
// CreateLLMSpecificMessage needs of one.
type namedAdapter struct{ id string }

func (a namedAdapter) IDForLLMSpecificMessages() string { return a.id }

// nativeMessage is a stand-in for a provider's own message type.
type nativeMessage struct{ Text string }

// TestCreateLLMSpecificMessage checks a message is written for the provider the
// adapter converts for, carrying that provider's own type.
func TestCreateLLMSpecificMessage(t *testing.T) {
	m := adapter.CreateLLMSpecificMessage(namedAdapter{id: "anthropic"}, nativeMessage{Text: "hi"})

	if !m.IsLLMSpecific() {
		t.Error("message is not provider-specific, want it written for one provider")
	}
	if m.LLM != "anthropic" {
		t.Errorf("llm = %q, want the adapter's own identifier", m.LLM)
	}
	if got, ok := m.Native.(nativeMessage); !ok || got.Text != "hi" {
		t.Errorf("native = %v, want the message it was given", m.Native)
	}
}

// TestNativeMessageReadsItsOwnType checks a native message is handed back as
// the type the adapter defines.
func TestNativeMessageReadsItsOwnType(t *testing.T) {
	m := frames.NewLLMSpecificMessage("anthropic", nativeMessage{Text: "hi"})

	got, err := adapter.NativeMessage[nativeMessage](m)
	if err != nil {
		t.Fatalf("NativeMessage: %v", err)
	}
	if got.Text != "hi" {
		t.Errorf("native = %+v, want the message it was written with", got)
	}
}

// TestNativeMessageRejectsAnotherType checks a message holding something its own
// adapter cannot read is a conversion failure rather than being sent as
// whatever it happens to be.
func TestNativeMessageRejectsAnotherType(t *testing.T) {
	m := frames.NewLLMSpecificMessage("anthropic", "just a string")

	_, err := adapter.NativeMessage[nativeMessage](m)
	if err == nil {
		t.Fatal("NativeMessage succeeded, want a conversion error")
	}
	if _, ok := errors.AsType[*adapter.ConversionError](err); !ok {
		t.Fatalf("err = %v, want an adapter.ConversionError", err)
	}
}

// TestPlainMessageIsNotLLMSpecific checks an ordinary turn is not mistaken for
// one written in a provider's own format.
func TestPlainMessageIsNotLLMSpecific(t *testing.T) {
	if (frames.Message{Role: frames.RoleUser, Text: "hi"}).IsLLMSpecific() {
		t.Error("a plain message reads as provider-specific, want it universal")
	}
}
