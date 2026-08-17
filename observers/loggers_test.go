package observers_test

import (
	"bytes"
	"encoding/json"
	"log/slog"
	"strings"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/observers"
	"github.com/gojargo/jargo/processor"
)

// debugLog is a logger writing to buf, at the level the log observers use.
func debugLog(buf *bytes.Buffer) *slog.Logger {
	return slog.New(slog.NewTextHandler(buf, &slog.HandlerOptions{Level: slog.LevelDebug}))
}

// plain is a processor that is nothing in particular, which is what the log
// observers filtering on the kind of service have to tell apart from one.
type plain struct {
	*processor.Base
}

func newPlain(name string) *plain {
	p := &plain{}
	p.Base = processor.New(name, p)
	return p
}

// fakeLLM stands in for a language-model service.
type fakeLLM struct {
	*processor.Base
}

func newFakeLLM() *fakeLLM {
	s := &fakeLLM{}
	s.Base = processor.New("FakeLLM", s)
	return s
}

func (s *fakeLLM) LLMService() {}

// fakeSTT stands in for a speech-to-text service.
type fakeSTT struct {
	*processor.Base
}

func newFakeSTT() *fakeSTT {
	s := &fakeSTT{}
	s.Base = processor.New("FakeSTT", s)
	return s
}

func (s *fakeSTT) STTService() {}

// handover reports one frame passing from src to dst.
func handover(o processor.Observer, src, dst processor.Processor, f frames.Frame, dir processor.Direction) {
	o.OnPushFrame(processor.FramePushed{
		Source:      src,
		Destination: dst,
		Frame:       f,
		Direction:   dir,
		Timestamp:   time.Second,
	})
}

// TestDebugLogRendersTheContentsOfAFrame covers what the debug log is for:
// seeing what a frame actually carried, not just that one went by.
func TestDebugLogRendersTheContentsOfAFrame(t *testing.T) {
	var buf bytes.Buffer
	o := observers.NewDebugLog(observers.DebugLogConfig{Logger: debugLog(&buf)})

	src, dst := newPlain("Src"), newPlain("Dst")
	handover(o, src, dst, frames.NewTranscriptionFrame("hello there", "user-1", ""), processor.Downstream)

	got := buf.String()
	for _, want := range []string{"TranscriptionFrame", "hello there", "user-1", src.Name(), dst.Name()} {
		if !strings.Contains(got, want) {
			t.Errorf("logged %q, want it to mention %q", got, want)
		}
	}
}

// TestDebugLogLeavesOutBinaryPayloads covers the default exclusions. Raw audio
// says nothing a reader wants and would bury what does.
func TestDebugLogLeavesOutBinaryPayloads(t *testing.T) {
	var buf bytes.Buffer
	o := observers.NewDebugLog(observers.DebugLogConfig{Logger: debugLog(&buf)})

	src := newPlain("Src")
	handover(o, src, newPlain("Dst"),
		frames.NewTTSAudioRawFrame([]byte{1, 2, 3, 4}, 24000, 1), processor.Downstream)

	got := buf.String()
	if strings.Contains(got, "Audio:") {
		t.Errorf("logged %q, want the audio payload left out", got)
	}
	if !strings.Contains(got, "SampleRate: 24000") {
		t.Errorf("logged %q, want the rest of the frame kept", got)
	}
}

// TestDebugLogReportsExcludedFieldsWhenAsked covers a caller who wants a field
// the defaults leave out, and passing no exclusions at all.
func TestDebugLogReportsExcludedFieldsWhenAsked(t *testing.T) {
	var buf bytes.Buffer
	o := observers.NewDebugLog(observers.DebugLogConfig{
		Logger:        debugLog(&buf),
		ExcludeFields: []string{},
	})

	handover(o, newPlain("Src"), newPlain("Dst"),
		frames.NewTTSAudioRawFrame([]byte{1, 2, 3, 4}, 24000, 1), processor.Downstream)

	if got := buf.String(); !strings.Contains(got, "4 bytes") {
		t.Errorf("logged %q, want the audio payload reported by its size", got)
	}
}

// TestDebugLogNarrowsToTheFramesAskedFor covers the reason the filter exists: a
// pipeline pushes far too many frames to read them all.
func TestDebugLogNarrowsToTheFramesAskedFor(t *testing.T) {
	var buf bytes.Buffer
	o := observers.NewDebugLog(observers.DebugLogConfig{
		Logger: debugLog(&buf),
		Frames: []observers.DebugFrameFilter{{Frame: &frames.BotStartedSpeakingFrame{}}},
	})

	src, dst := newPlain("Src"), newPlain("Dst")
	handover(o, src, dst, frames.NewUserStartedSpeakingFrame(), processor.Downstream)
	if buf.Len() != 0 {
		t.Errorf("logged a frame the filter rejected: %q", buf.String())
	}

	handover(o, src, dst, frames.NewBotStartedSpeakingFrame(), processor.Downstream)
	if got := buf.String(); !strings.Contains(got, "BotStartedSpeakingFrame") {
		t.Errorf("logged %q, want the frame the filter accepted", got)
	}
}

// TestDebugLogNarrowsToOneEndOfTheHandover covers the other half of the filter:
// the same frame type is interesting at one point in the pipeline and noise
// everywhere else.
func TestDebugLogNarrowsToOneEndOfTheHandover(t *testing.T) {
	var buf bytes.Buffer
	stt := newFakeSTT()
	o := observers.NewDebugLog(observers.DebugLogConfig{
		Logger: debugLog(&buf),
		Frames: []observers.DebugFrameFilter{{
			Frame: &frames.TranscriptionFrame{},
			Match: func(p processor.Processor) bool { _, ok := p.(*fakeSTT); return ok },
		}},
	})

	other := newPlain("Other")
	handover(o, other, newPlain("Dst"), frames.NewTranscriptionFrame("elsewhere", "u", ""), processor.Downstream)
	if buf.Len() != 0 {
		t.Errorf("logged a handover the filter rejected: %q", buf.String())
	}

	handover(o, stt, newPlain("Dst"), frames.NewTranscriptionFrame("from the stt", "u", ""), processor.Downstream)
	if got := buf.String(); !strings.Contains(got, "from the stt") {
		t.Errorf("logged %q, want the handover the filter accepted", got)
	}
}

// TestDebugLogMatchesTheDestinationEndpoint covers the same filter aimed at the
// receiving end rather than the sending one.
func TestDebugLogMatchesTheDestinationEndpoint(t *testing.T) {
	var buf bytes.Buffer
	llm := newFakeLLM()
	o := observers.NewDebugLog(observers.DebugLogConfig{
		Logger: debugLog(&buf),
		Frames: []observers.DebugFrameFilter{{
			Frame:    &frames.TextFrame{},
			Match:    func(p processor.Processor) bool { _, ok := p.(*fakeLLM); return ok },
			Endpoint: observers.DestinationEndpoint,
		}},
	})

	handover(o, newPlain("Src"), newPlain("Dst"), frames.NewTextFrame("nowhere"), processor.Downstream)
	if buf.Len() != 0 {
		t.Errorf("logged a handover the filter rejected: %q", buf.String())
	}

	handover(o, newPlain("Src"), llm, frames.NewTextFrame("to the model"), processor.Downstream)
	if got := buf.String(); !strings.Contains(got, "to the model") {
		t.Errorf("logged %q, want the handover the filter accepted", got)
	}
}

// TestLLMLogReportsWhatPassedThroughTheModel covers the observer's whole point:
// the tokens, the bounds of the response, and the tool calls made along the way.
func TestLLMLogReportsWhatPassedThroughTheModel(t *testing.T) {
	var buf bytes.Buffer
	o := observers.NewLLMLog(observers.LLMLogConfig{Logger: debugLog(&buf)})

	llm, dst := newFakeLLM(), newPlain("Dst")
	args := json.RawMessage(`{"location":"Atlanta"}`)

	handover(o, llm, dst, frames.NewLLMFullResponseStartFrame(), processor.Downstream)
	handover(o, llm, dst, frames.NewLLMTextFrame("the weather is"), processor.Downstream)
	call := frames.NewFunctionCallInProgressFrame("call_1", "get_weather", args, true, "")
	handover(o, llm, dst, call, processor.Upstream)
	handover(o, llm, dst, frames.NewFunctionCallResultFrame("call_1", "get_weather", args, "75"), processor.Downstream)
	handover(o, llm, dst, frames.NewLLMFullResponseEndFrame(), processor.Downstream)

	got := buf.String()
	for _, want := range []string{
		"llm response started", "llm generating", "the weather is",
		"llm function call", "get_weather", "call_1",
		"llm function call result", "llm response ended",
	} {
		if !strings.Contains(got, want) {
			t.Errorf("logged %q, want it to mention %q", got, want)
		}
	}
}

// TestLLMLogReportsWhatTheModelWasAsked covers the input side. A reply that
// reads wrong is usually a context that was, so the conversation reaching the
// model is reported whole.
func TestLLMLogReportsWhatTheModelWasAsked(t *testing.T) {
	var buf bytes.Buffer
	o := observers.NewLLMLog(observers.LLMLogConfig{Logger: debugLog(&buf)})

	convo := frames.NewLLMContext("")
	convo.AddMessage(frames.Message{Role: frames.RoleUser, Text: "book a table"})
	handover(o, newPlain("Aggregator"), newFakeLLM(), frames.NewLLMContextFrame(convo), processor.Downstream)

	got := buf.String()
	for _, want := range []string{"llm context", "book a table"} {
		if !strings.Contains(got, want) {
			t.Errorf("logged %q, want it to mention %q", got, want)
		}
	}
}

// TestLLMLogIgnoresFramesFromElsewhere covers the service filter: a context
// frame built by an aggregator carries the same shape as one reaching a model,
// and only the second says anything about what a model was asked.
func TestLLMLogIgnoresFramesFromElsewhere(t *testing.T) {
	var buf bytes.Buffer
	o := observers.NewLLMLog(observers.LLMLogConfig{Logger: debugLog(&buf)})

	handover(o, newPlain("Aggregator"), newPlain("Dst"),
		frames.NewLLMTextFrame("not from a model"), processor.Downstream)

	if buf.Len() != 0 {
		t.Errorf("logged %q, want nothing: no model was involved", buf.String())
	}
}

// TestLLMLogReportsAToolCallOnce covers the tool call going out both ways. The
// call happened once, so it is reported once.
func TestLLMLogReportsAToolCallOnce(t *testing.T) {
	var buf bytes.Buffer
	o := observers.NewLLMLog(observers.LLMLogConfig{Logger: debugLog(&buf)})

	llm := newFakeLLM()
	args := json.RawMessage(`{}`)
	f := frames.NewFunctionCallInProgressFrame("call_1", "get_weather", args, true, "")
	handover(o, llm, newPlain("Down"), f, processor.Downstream)
	handover(o, llm, newPlain("Up"), f, processor.Upstream)

	if n := strings.Count(buf.String(), "llm function call"); n != 1 {
		t.Errorf("logged the call %d times, want once", n)
	}
}

// TestTranscriptionLogReportsWhatWasHeard covers both kinds of transcript, since
// the interim ones are what show a transcriber keeping up.
func TestTranscriptionLogReportsWhatWasHeard(t *testing.T) {
	var buf bytes.Buffer
	o := observers.NewTranscriptionLog(observers.TranscriptionLogConfig{Logger: debugLog(&buf)})

	stt, dst := newFakeSTT(), newPlain("Dst")
	handover(o, stt, dst, frames.NewInterimTranscriptionFrame("book a ta", "user-1", ""), processor.Downstream)
	handover(o, stt, dst, frames.NewTranscriptionFrame("book a table", "user-1", ""), processor.Downstream)

	got := buf.String()
	for _, want := range []string{"interim transcription", "book a ta", "book a table", "user-1"} {
		if !strings.Contains(got, want) {
			t.Errorf("logged %q, want it to mention %q", got, want)
		}
	}
}

// TestTranscriptionLogIgnoresTranscriptsFromElsewhere covers a transcript that a
// transcriber did not produce, which says nothing about speech recognition.
func TestTranscriptionLogIgnoresTranscriptsFromElsewhere(t *testing.T) {
	var buf bytes.Buffer
	o := observers.NewTranscriptionLog(observers.TranscriptionLogConfig{Logger: debugLog(&buf)})

	handover(o, newPlain("Aggregator"), newPlain("Dst"),
		frames.NewTranscriptionFrame("relayed", "user-1", ""), processor.Downstream)

	if buf.Len() != 0 {
		t.Errorf("logged %q, want nothing: no transcriber was involved", buf.String())
	}
}

// TestMetricsLogReportsEveryKindOfMeasurement covers the observer rendering each
// kind on its own terms rather than dumping a struct.
func TestMetricsLogReportsEveryKindOfMeasurement(t *testing.T) {
	var buf bytes.Buffer
	o := observers.NewMetricsLog(observers.MetricsLogConfig{Logger: debugLog(&buf)})

	cached := int64(12)
	handover(o, newPlain("Src"), newPlain("Dst"), frames.NewMetricsFrame(
		frames.TTFBMetricsData{
			BaseMetricsData: frames.BaseMetricsData{Processor: "LLM#0", Model: "gpt-4o"},
			Value:           250 * time.Millisecond,
		},
		frames.LLMUsageMetricsData{
			BaseMetricsData: frames.BaseMetricsData{Processor: "LLM#0"},
			Value: frames.LLMTokenUsage{
				PromptTokens: 100, CompletionTokens: 20, TotalTokens: 120, CacheReadTokens: &cached,
			},
		},
		frames.TTSUsageMetricsData{
			BaseMetricsData: frames.BaseMetricsData{Processor: "TTS#0"},
			Value:           42,
		},
	), processor.Downstream)

	got := buf.String()
	for _, want := range []string{"ttfb", "gpt-4o", "llm token usage", "cache_read=12", "tts usage", "characters=42"} {
		if !strings.Contains(got, want) {
			t.Errorf("logged %q, want it to mention %q", got, want)
		}
	}
}

// TestMetricsLogOmitsCountsTheServiceNeverReported covers the distinction a cost
// dashboard depends on: a count that was not reported is not a measured zero.
func TestMetricsLogOmitsCountsTheServiceNeverReported(t *testing.T) {
	var buf bytes.Buffer
	o := observers.NewMetricsLog(observers.MetricsLogConfig{Logger: debugLog(&buf)})

	handover(o, newPlain("Src"), newPlain("Dst"), frames.NewMetricsFrame(
		frames.LLMUsageMetricsData{
			BaseMetricsData: frames.BaseMetricsData{Processor: "LLM#0"},
			Value:           frames.LLMTokenUsage{PromptTokens: 100, CompletionTokens: 20, TotalTokens: 120},
		},
	), processor.Downstream)

	if got := buf.String(); strings.Contains(got, "cache_read") {
		t.Errorf("logged %q, want no cache counts: the service reported none", got)
	}
}

// TestMetricsLogNarrowsToTheKindsAskedFor covers a caller watching one thing,
// usually cost, who does not want the timing measurements alongside it.
func TestMetricsLogNarrowsToTheKindsAskedFor(t *testing.T) {
	var buf bytes.Buffer
	o := observers.NewMetricsLog(observers.MetricsLogConfig{
		Logger:  debugLog(&buf),
		Include: []frames.MetricsData{frames.LLMUsageMetricsData{}},
	})

	handover(o, newPlain("Src"), newPlain("Dst"), frames.NewMetricsFrame(
		frames.TTFBMetricsData{BaseMetricsData: frames.BaseMetricsData{Processor: "LLM#0"}, Value: time.Second},
		frames.LLMUsageMetricsData{
			BaseMetricsData: frames.BaseMetricsData{Processor: "LLM#0"},
			Value:           frames.LLMTokenUsage{PromptTokens: 100, CompletionTokens: 20, TotalTokens: 120},
		},
	), processor.Downstream)

	got := buf.String()
	if strings.Contains(got, "ttfb") {
		t.Errorf("logged %q, want the kinds that were not asked for left out", got)
	}
	if !strings.Contains(got, "llm token usage") {
		t.Errorf("logged %q, want the kind that was asked for", got)
	}
}

// TestMetricsLogReportsAFrameOnce covers the same frame being reported at every
// handover it makes. The measurements it carries were made once.
func TestMetricsLogReportsAFrameOnce(t *testing.T) {
	var buf bytes.Buffer
	o := observers.NewMetricsLog(observers.MetricsLogConfig{Logger: debugLog(&buf)})

	f := frames.NewMetricsFrame(frames.TTFBMetricsData{
		BaseMetricsData: frames.BaseMetricsData{Processor: "LLM#0"},
		Value:           250 * time.Millisecond,
	})
	src, mid, dst := newPlain("Src"), newPlain("Mid"), newPlain("Dst")
	handover(o, src, mid, f, processor.Downstream)
	handover(o, mid, dst, f, processor.Downstream)

	if n := strings.Count(buf.String(), "ttfb"); n != 1 {
		t.Errorf("logged the measurement %d times, want once", n)
	}
}

// TestLogObserversDefaultToTheProcessLogger covers a caller who configures no
// destination, which must not be a crash.
func TestLogObserversDefaultToTheProcessLogger(t *testing.T) {
	for _, o := range []processor.Observer{
		observers.NewDebugLog(observers.DebugLogConfig{}),
		observers.NewLLMLog(observers.LLMLogConfig{}),
		observers.NewTranscriptionLog(observers.TranscriptionLogConfig{}),
		observers.NewMetricsLog(observers.MetricsLogConfig{}),
	} {
		handover(o, newFakeLLM(), newFakeSTT(), frames.NewTextFrame("hi"), processor.Downstream)
	}
}
