package frames_test

import (
	"strings"
	"testing"

	"github.com/gojargo/jargo/frames"
)

func TestAudioNumFrames(t *testing.T) {
	// 16-bit stereo: 8 bytes -> 2 frames.
	in := frames.NewInputAudioRawFrame(make([]byte, 8), 16000, 2)
	if in.NumFrames() != 2 {
		t.Errorf("NumFrames() = %d, want 2", in.NumFrames())
	}
	// 16-bit mono: 8 bytes -> 4 frames.
	out := frames.NewOutputAudioRawFrame(make([]byte, 8), 24000, 1)
	if out.NumFrames() != 4 {
		t.Errorf("NumFrames() = %d, want 4", out.NumFrames())
	}
	// Zero channels must not panic and yields 0 frames.
	bad := frames.NewInputAudioRawFrame([]byte{1, 2}, 16000, 0)
	if bad.NumFrames() != 0 {
		t.Errorf("NumFrames() = %d, want 0 for zero channels", bad.NumFrames())
	}
}

func TestAudioCategories(t *testing.T) {
	var in frames.Frame = frames.NewInputAudioRawFrame(nil, 16000, 1)
	if _, ok := in.(frames.SystemFrame); !ok {
		t.Error("InputAudioRawFrame should be a SystemFrame")
	}
	if _, ok := in.(frames.DataFrame); ok {
		t.Error("InputAudioRawFrame should not be a DataFrame")
	}

	var out frames.Frame = frames.NewOutputAudioRawFrame(nil, 24000, 1)
	if _, ok := out.(frames.DataFrame); !ok {
		t.Error("OutputAudioRawFrame should be a DataFrame")
	}

	var tts frames.Frame = frames.NewTTSAudioRawFrame(nil, 24000, 1)
	if _, ok := tts.(frames.DataFrame); !ok {
		t.Error("TTSAudioRawFrame should be a DataFrame")
	}
	if got := tts.Name(); !strings.HasPrefix(got, "TTSAudioRawFrame#") {
		t.Errorf("Name() = %q, want TTSAudioRawFrame# prefix", got)
	}
}

func TestTextFrameDefaults(t *testing.T) {
	f := frames.NewTextFrame("hi")
	if !f.AppendToContext {
		t.Error("AppendToContext should default to true")
	}
	if f.SkipTTS != nil {
		t.Error("SkipTTS should default to nil (unset)")
	}
	if f.IncludesInterFrameSpaces {
		t.Error("IncludesInterFrameSpaces should default to false")
	}
}

func TestLLMTextFrame(t *testing.T) {
	f := frames.NewLLMTextFrame("hello")
	if !f.IncludesInterFrameSpaces {
		t.Error("LLMTextFrame should include inter-frame spaces")
	}
	if !strings.HasPrefix(f.Name(), "LLMTextFrame#") {
		t.Errorf("Name() = %q, want LLMTextFrame# prefix", f.Name())
	}
	var fr frames.Frame = f
	if _, ok := fr.(frames.DataFrame); !ok {
		t.Error("LLMTextFrame should be a DataFrame")
	}
	// Inherits TextFrame.String, but renders its own name.
	if !strings.Contains(f.String(), "LLMTextFrame#") {
		t.Errorf("String() = %q, want it to carry the LLMTextFrame name", f.String())
	}
}

func TestTranscriptionFrames(t *testing.T) {
	tf := frames.NewTranscriptionFrame("hello", "user-1", "2026-06-19T00:00:00Z")
	if tf.UserID != "user-1" || tf.Text != "hello" {
		t.Errorf("unexpected fields: user=%q text=%q", tf.UserID, tf.Text)
	}
	if tf.Finalized {
		t.Error("Finalized should default to false")
	}
	if !tf.AppendToContext {
		t.Error("TranscriptionFrame.AppendToContext should default to true")
	}
	var fr frames.Frame = tf
	if _, ok := fr.(frames.DataFrame); !ok {
		t.Error("TranscriptionFrame should be a DataFrame")
	}

	itf := frames.NewInterimTranscriptionFrame("hel", "user-1", "ts")
	var ifr frames.Frame = itf
	if _, ok := ifr.(frames.DataFrame); !ok {
		t.Error("InterimTranscriptionFrame should be a DataFrame")
	}
	if itf.IncludesInterFrameSpaces {
		t.Error("InterimTranscriptionFrame should not include inter-frame spaces")
	}
}

func TestStartFrameDefaults(t *testing.T) {
	s := frames.NewStartFrame()
	if s.AudioInSampleRate != 16000 || s.AudioOutSampleRate != 24000 {
		t.Errorf("sample rates = %d/%d, want 16000/24000", s.AudioInSampleRate, s.AudioOutSampleRate)
	}
	var fr frames.Frame = s
	if _, ok := fr.(frames.SystemFrame); !ok {
		t.Error("StartFrame should be a SystemFrame")
	}
}

func TestEndFrameIsUninterruptibleControl(t *testing.T) {
	var e frames.Frame = frames.NewEndFrame()
	if _, ok := e.(frames.ControlFrame); !ok {
		t.Error("EndFrame should be a ControlFrame")
	}
	if _, ok := e.(frames.Uninterruptible); !ok {
		t.Error("EndFrame should be Uninterruptible")
	}
}

func TestErrorFrame(t *testing.T) {
	e := frames.NewErrorFrame("boom")
	if e.Error != "boom" || e.Fatal {
		t.Errorf("unexpected: error=%q fatal=%v", e.Error, e.Fatal)
	}
	var fr frames.Frame = e
	if _, ok := fr.(frames.SystemFrame); !ok {
		t.Error("ErrorFrame should be a SystemFrame")
	}
	if got := e.String(); !strings.Contains(got, "boom") {
		t.Errorf("String() = %q, want it to contain the message", got)
	}
}

func TestSpeakingFramesAreSystem(t *testing.T) {
	for _, f := range []frames.Frame{
		frames.NewUserStartedSpeakingFrame(),
		frames.NewUserStoppedSpeakingFrame(),
		frames.NewBotStartedSpeakingFrame(),
		frames.NewBotStoppedSpeakingFrame(),
		frames.NewInterruptionFrame(),
	} {
		if _, ok := f.(frames.SystemFrame); !ok {
			t.Errorf("%s should be a SystemFrame", f.Name())
		}
	}
}

func TestResponseAndTTSControlFrames(t *testing.T) {
	for _, f := range []frames.Frame{
		frames.NewLLMFullResponseStartFrame(),
		frames.NewLLMFullResponseEndFrame(),
		frames.NewTTSStartedFrame(),
		frames.NewTTSStoppedFrame(),
	} {
		if _, ok := f.(frames.ControlFrame); !ok {
			t.Errorf("%s should be a ControlFrame", f.Name())
		}
	}
	if !frames.NewTTSStartedFrame().AppendToContext {
		t.Error("TTSStartedFrame.AppendToContext should default to true")
	}
}

func TestAudioStringFormat(t *testing.T) {
	in := frames.NewInputAudioRawFrame(make([]byte, 4), 16000, 1)
	in.SetTransportSource("mic")
	got := in.String()
	for _, want := range []string{"InputAudioRawFrame#", "source: mic", "sample_rate: 16000", "frames: 2"} {
		if !strings.Contains(got, want) {
			t.Errorf("String() = %q, missing %q", got, want)
		}
	}
}
