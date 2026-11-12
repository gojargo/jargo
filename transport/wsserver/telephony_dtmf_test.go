package wsserver_test

// Ported from upstream's telephony serializer suites: a DTMF digit that is not
// one of the twelve keypad keys is dropped, not pushed as a keypress.

import (
	"testing"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/transport/wsserver"
	"github.com/gojargo/jargo/transport/wsserver/exotel"
	"github.com/gojargo/jargo/transport/wsserver/plivo"
	"github.com/gojargo/jargo/transport/wsserver/telnyx"
	"github.com/gojargo/jargo/transport/wsserver/twilio"
)

// telephonySerializers is every telephony serializer, set up at 8 kHz.
func telephonySerializers(t *testing.T) map[string]wsserver.Serializer {
	t.Helper()
	off := false
	all := map[string]wsserver.Serializer{
		"twilio": twilio.New(twilio.Config{AutoHangUp: &off}),
		"telnyx": telnyx.New(telnyx.Config{AutoHangUp: &off}),
		"plivo":  plivo.New(plivo.Config{AutoHangUp: &off}),
		"exotel": exotel.New(exotel.Config{}),
	}
	for name, s := range all {
		if err := s.Setup(processor.Setup{AudioInSampleRate: 8000, AudioOutSampleRate: 8000}); err != nil {
			t.Fatalf("%s: setup: %v", name, err)
		}
	}
	return all
}

func TestDeserializeDropsANonKeypadDigit(t *testing.T) {
	for name, s := range telephonySerializers(t) {
		for _, digit := range []string{"A", "D", "x", "12", ""} {
			f, err := s.Deserialize([]byte(`{"event":"dtmf","dtmf":{"digit":"` + digit + `"}}`))
			if err != nil {
				t.Fatalf("%s: %v", name, err)
			}
			if f != nil {
				t.Errorf("%s: digit %q became %T, want nothing", name, digit, f)
			}
		}
		f, err := s.Deserialize([]byte(`{"event":"dtmf","dtmf":{"digit":"#"}}`))
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		if d, ok := f.(*frames.InputDTMFFrame); !ok || d.Button != frames.KeypadPound {
			t.Errorf("%s: # became %T %+v, want the keypress", name, f, f)
		}
	}
}
