package cartesia

import (
	"testing"

	"github.com/gojargo/jargo/processor"
)

// TestEachServiceIsTheProcessorItsConstructorReturns checks the value a
// constructor returns is the processor the pipeline runs, so a switcher or an
// observer comparing against it finds it.
func TestEachServiceIsTheProcessorItsConstructorReturns(t *testing.T) {
	tts := NewTTS(Config{APIKey: "k", VoiceID: testVoiceID})
	httpTTS := NewHTTPTTS(HTTPTTSConfig{APIKey: "k", VoiceID: testVoiceID})
	stt := NewSTT(STTConfig{APIKey: "k"})
	for name, c := range map[string]struct{ svc, self processor.Processor }{
		"TTSService":     {tts, tts.Self()},
		"HTTPTTSService": {httpTTS, httpTTS.Self()},
		"STTService":     {stt, stt.Self()},
	} {
		if c.self != c.svc {
			t.Errorf("%s: Self() = %T, want the value the constructor returned", name, c.self)
		}
	}
}
