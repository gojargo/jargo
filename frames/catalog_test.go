package frames_test

// A catalog sweep over every constructible frame type.
//
// Frames are mostly declaration: a constructor that fills a few fields and a
// String that renders them. Testing each one by hand would be dozens of
// near-identical functions, so instead every frame is listed once here and the
// invariants that hold for all of them are checked in one place: the name label,
// the String prefix, and exactly one of the system/data/control categories.
// Anything a specific frame does beyond that has its own test below or in
// concrete_test.go.

import (
	"strings"
	"testing"
	"time"

	"github.com/gojargo/jargo/audio/turn"
	"github.com/gojargo/jargo/audio/vad"
	"github.com/gojargo/jargo/frames"
)

// category is the frame class a type belongs to. Every frame belongs to exactly
// one, which is what decides whether an interruption may drop it.
type category int

const (
	system category = iota
	data
	control
)

func (c category) String() string {
	switch c {
	case system:
		return "system"
	case data:
		return "data"
	case control:
		return "control"
	}
	return "unknown"
}

// catalogEntry is one frame type: how to build it, the label it should report,
// and the class it belongs to.
type catalogEntry struct {
	label string
	cat   category
	build func() frames.Frame
	// wantString, when set, must appear in the frame's String output — used for
	// the frames whose String renders their payload.
	wantString string
	// uninterruptible marks frames that must survive a barge-in.
	uninterruptible bool
}

// catalog lists every frame a caller can construct. Adding a frame type means
// adding a line here.
//
//nolint:funlen,maintidx // a flat catalog of every frame type; length is the point
func catalog() []catalogEntry {
	return []catalogEntry{
		// System frames: priority delivery, unaffected by interruptions.
		{
			label: "StartFrame", cat: system,
			build: func() frames.Frame { return frames.NewStartFrame() },
		},
		{
			label: "CancelFrame", cat: system, wantString: "reason:",
			build: func() frames.Frame { return frames.NewCancelFrame() },
		},
		{
			label: "ErrorFrame", cat: system, wantString: "boom",
			build: func() frames.Frame { return frames.NewErrorFrame("boom") },
		},
		{
			label: "FatalErrorFrame", cat: system, wantString: "fatal: true",
			build: func() frames.Frame { return frames.NewFatalErrorFrame("boom") },
		},
		{
			label: "CancelWorkerFrame", cat: system, wantString: "reason:",
			build: func() frames.Frame { return frames.NewCancelWorkerFrame() },
		},
		{
			label: "InterruptionWorkerFrame", cat: system,
			build: func() frames.Frame { return frames.NewInterruptionWorkerFrame() },
		},
		{
			label: "InterruptionFrame", cat: system,
			build: func() frames.Frame { return frames.NewInterruptionFrame() },
		},
		{
			label: "UserStartedSpeakingFrame", cat: system,
			build: func() frames.Frame { return frames.NewUserStartedSpeakingFrame() },
		},
		{
			label: "UserStoppedSpeakingFrame", cat: system,
			build: func() frames.Frame { return frames.NewUserStoppedSpeakingFrame() },
		},
		{
			label: "BotStartedSpeakingFrame", cat: system,
			build: func() frames.Frame { return frames.NewBotStartedSpeakingFrame() },
		},
		{
			label: "BotStoppedSpeakingFrame", cat: system,
			build: func() frames.Frame { return frames.NewBotStoppedSpeakingFrame() },
		},
		{
			label: "UserSpeakingFrame", cat: system,
			build: func() frames.Frame { return frames.NewUserSpeakingFrame() },
		},
		{
			label: "BotSpeakingFrame", cat: system,
			build: func() frames.Frame { return frames.NewBotSpeakingFrame() },
		},
		{
			label: "UserMuteStartedFrame", cat: system,
			build: func() frames.Frame { return frames.NewUserMuteStartedFrame() },
		},
		{
			label: "UserMuteStoppedFrame", cat: system,
			build: func() frames.Frame { return frames.NewUserMuteStoppedFrame() },
		},
		{
			label: "VADUserStartedSpeakingFrame", cat: system, wantString: "start_secs: 1.500",
			build: func() frames.Frame { return frames.NewVADUserStartedSpeakingFrame(1.5) },
		},
		{
			label: "VADUserStoppedSpeakingFrame", cat: system, wantString: "stop_secs: 2.250",
			build: func() frames.Frame { return frames.NewVADUserStoppedSpeakingFrame(2.25, "ts") },
		},
		{
			label: "STTMetadataFrame", cat: system, wantString: "ttfs_p99: 300ms",
			build: func() frames.Frame { return frames.NewSTTMetadataFrame(300 * time.Millisecond) },
		},
		{
			label: "UserIdleTimeoutUpdateFrame", cat: system, wantString: "timeout: 5s",
			build: func() frames.Frame { return frames.NewUserIdleTimeoutUpdateFrame(5 * time.Second) },
		},
		{
			label: "SpeechControlParamsFrame", cat: system,
			build: func() frames.Frame {
				return frames.NewSpeechControlParamsFrame(&vad.Params{StopSecs: 0.8}, &turn.Params{StopSecs: 0.8})
			},
		},
		{
			label: "ServiceMetadataFrame", cat: system, wantString: "service: stt-1",
			build: func() frames.Frame { return frames.NewServiceMetadataFrame("stt-1") },
		},
		{
			label: "LLMServiceMetadataFrame", cat: system, wantString: "service: llm-1",
			build: func() frames.Frame { return frames.NewLLMServiceMetadataFrame("llm-1") },
		},
		{
			label: "InputAudioRawFrame", cat: system, wantString: "sample_rate: 16000",
			build: func() frames.Frame { return frames.NewInputAudioRawFrame(make([]byte, 4), 16000, 1) },
		},
		{
			label: "InputDTMFFrame", cat: system, wantString: "button: 7",
			build: func() frames.Frame { return frames.NewInputDTMFFrame(frames.KeypadSeven) },
		},
		{
			label: "InputTransportMessageFrame", cat: system, wantString: "size: 3",
			build: func() frames.Frame { return frames.NewInputTransportMessageFrame([]byte("abc")) },
		},
		{
			label: "OutputTransportMessageUrgentFrame", cat: system, wantString: "message:",
			build: func() frames.Frame {
				return frames.NewOutputTransportMessageUrgentFrame(map[string]any{"a": 1})
			},
		},
		{
			label: "MetricsFrame", cat: system, wantString: "processor: tts-1",
			build: func() frames.Frame { return frames.NewMetricsFrame("tts-1") },
		},

		// Data frames: carried in order, dropped on interruption.
		{
			label: "TextFrame", cat: data, wantString: "hello",
			build: func() frames.Frame { return frames.NewTextFrame("hello") },
		},
		{
			label: "LLMTextFrame", cat: data, wantString: "hello",
			build: func() frames.Frame { return frames.NewLLMTextFrame("hello") },
		},
		{
			label: "TTSTextFrame", cat: data, wantString: "hello",
			build: func() frames.Frame { return frames.NewTTSTextFrame("hello") },
		},
		{
			label: "TTSSpeakFrame", cat: data, wantString: "hello",
			build: func() frames.Frame { return frames.NewTTSSpeakFrame("hello") },
		},
		{
			label: "TranscriptionFrame", cat: data, wantString: "hello",
			build: func() frames.Frame { return frames.NewTranscriptionFrame("hello", "user-1", "ts") },
		},
		{
			label: "InterimTranscriptionFrame", cat: data, wantString: "hello",
			build: func() frames.Frame { return frames.NewInterimTranscriptionFrame("hello", "user-1", "ts") },
		},
		{
			label: "OutputAudioRawFrame", cat: data, wantString: "sample_rate: 24000",
			build: func() frames.Frame { return frames.NewOutputAudioRawFrame(make([]byte, 4), 24000, 1) },
		},
		{
			label: "TTSAudioRawFrame", cat: data, wantString: "sample_rate: 24000",
			build: func() frames.Frame { return frames.NewTTSAudioRawFrame(make([]byte, 4), 24000, 1) },
		},
		{
			label: "LLMContextFrame", cat: data,
			build: func() frames.Frame { return frames.NewLLMContextFrame(frames.NewLLMContext("be brief")) },
		},
		{
			label: "LLMMarkerFrame", cat: data,
			build: func() frames.Frame { return frames.NewLLMMarkerFrame("check") },
		},
		{
			label: "LLMRunFrame", cat: data,
			build: func() frames.Frame { return frames.NewLLMRunFrame() },
		},
		{
			label: "OutputTransportMessageFrame", cat: data, wantString: "message:",
			build: func() frames.Frame { return frames.NewOutputTransportMessageFrame(map[string]any{"a": 1}) },
		},
		{
			label: "LLMSetToolsFrame", cat: data, wantString: "tools: 1",
			build: func() frames.Frame {
				return frames.NewLLMSetToolsFrame([]frames.Tool{{Name: "get_weather"}})
			},
		},
		{
			label: "LLMSetToolChoiceFrame", cat: data, wantString: "choice: required",
			build: func() frames.Frame {
				return frames.NewLLMSetToolChoiceFrame(frames.ToolChoiceRequired)
			},
		},
		{
			label: "LLMMessagesUpdateFrame", cat: data, wantString: "messages: 1",
			build: func() frames.Frame {
				return frames.NewLLMMessagesUpdateFrame([]frames.Message{{Role: frames.RoleUser, Text: "hi"}})
			},
		},

		// Control frames: in order like data frames, but carrying instructions.
		{
			label: "EndFrame", cat: control, uninterruptible: true,
			build: func() frames.Frame { return frames.NewEndFrame() },
		},
		{
			label: "StopFrame", cat: control, uninterruptible: true,
			build: func() frames.Frame { return frames.NewStopFrame() },
		},
		{
			label: "StopWorkerFrame", cat: control, uninterruptible: true,
			build: func() frames.Frame { return frames.NewStopWorkerFrame() },
		},
		{
			label: "EndWorkerFrame", cat: control, uninterruptible: true, wantString: "reason:",
			build: func() frames.Frame { return frames.NewEndWorkerFrame() },
		},
		{
			label: "PipelineFlushFrame", cat: control, uninterruptible: true,
			build: func() frames.Frame { return frames.NewPipelineFlushFrame() },
		},
		{
			label: "LLMFullResponseStartFrame", cat: control,
			build: func() frames.Frame { return frames.NewLLMFullResponseStartFrame() },
		},
		{
			label: "LLMFullResponseEndFrame", cat: control,
			build: func() frames.Frame { return frames.NewLLMFullResponseEndFrame() },
		},
		{
			label: "TTSStartedFrame", cat: control,
			build: func() frames.Frame { return frames.NewTTSStartedFrame() },
		},
		{
			label: "TTSStoppedFrame", cat: control,
			build: func() frames.Frame { return frames.NewTTSStoppedFrame() },
		},
		{
			label: "UserTurnInferenceCompletedFrame", cat: control,
			build: func() frames.Frame { return frames.NewUserTurnInferenceCompletedFrame() },
		},
		{
			label: "OutputDTMFFrame", cat: control, wantString: "button: 7",
			build: func() frames.Frame { return frames.NewOutputDTMFFrame(frames.KeypadSeven) },
		},
		{
			label: "MixerUpdateSettingsFrame", cat: control, wantString: "settings: 1",
			build: func() frames.Frame {
				return frames.NewMixerUpdateSettingsFrame(map[string]any{"volume": 0.5})
			},
		},
		{
			label: "MixerEnableFrame", cat: control, wantString: "enable: true",
			build: func() frames.Frame { return frames.NewMixerEnableFrame(true) },
		},
		{
			label: "LLMMessagesAppendFrame", cat: control,
			build: func() frames.Frame {
				return frames.NewLLMMessagesAppendFrame([]frames.Message{{Role: frames.RoleUser, Text: "hi"}})
			},
		},
		{
			label: "FunctionCallsStartedFrame", cat: control, wantString: "calls: 1",
			build: func() frames.Frame {
				return frames.NewFunctionCallsStartedFrame("", []frames.ToolCall{{ID: "a", Name: "get_weather"}})
			},
		},
		{
			label: "FunctionCallInProgressFrame", cat: control, wantString: "get_weather",
			build: func() frames.Frame { return frames.NewFunctionCallInProgressFrame("a", "get_weather") },
		},
		{
			label: "FunctionCallResultFrame", cat: control, wantString: "get_weather",
			build: func() frames.Frame { return frames.NewFunctionCallResultFrame("a", "get_weather", "sunny", false) },
		},
		{
			label: "FunctionCallCancelFrame", cat: control, wantString: "get_weather",
			build: func() frames.Frame { return frames.NewFunctionCallCancelFrame("a", "get_weather") },
		},
		{
			label: "AudioBufferStartRecordingFrame", cat: control, uninterruptible: true,
			build: func() frames.Frame { return frames.NewAudioBufferStartRecordingFrame() },
		},
		{
			label: "AudioBufferStopRecordingFrame", cat: control, uninterruptible: true,
			build: func() frames.Frame { return frames.NewAudioBufferStopRecordingFrame() },
		},
	}
}

// TestCatalogInvariants checks the properties every frame must have, whatever it
// carries.
func TestCatalogInvariants(t *testing.T) {
	for _, entry := range catalog() {
		t.Run(entry.label, func(t *testing.T) {
			f := entry.build()

			if f.ID() == 0 {
				t.Error("ID() = 0; every frame must get a nonzero id")
			}
			if got := f.Name(); !strings.HasPrefix(got, entry.label+"#") {
				t.Errorf("Name() = %q, want the %q label", got, entry.label)
			}
			// Every String renders the name first, so a log line always
			// identifies the frame before its payload.
			if got := f.String(); !strings.HasPrefix(got, f.Name()) {
				t.Errorf("String() = %q, want it to start with Name() %q", got, f.Name())
			}
			if entry.wantString != "" {
				if got := f.String(); !strings.Contains(got, entry.wantString) {
					t.Errorf("String() = %q, want it to contain %q", got, entry.wantString)
				}
			}
			assertCategory(t, f, entry.cat)

			if _, ok := f.(frames.Uninterruptible); ok != entry.uninterruptible {
				t.Errorf("Uninterruptible = %v, want %v", ok, entry.uninterruptible)
			}
		})
	}
}

// assertCategory checks a frame is in want and in neither of the other two.
func assertCategory(t *testing.T, f frames.Frame, want category) {
	t.Helper()
	_, isSystem := f.(frames.SystemFrame)
	_, isData := f.(frames.DataFrame)
	_, isControl := f.(frames.ControlFrame)

	got := map[category]bool{system: isSystem, data: isData, control: isControl}
	for _, c := range []category{system, data, control} {
		if got[c] != (c == want) {
			t.Errorf("%s category = %v, want %v (frame should be %s)", c, got[c], c == want, want)
		}
	}
}

// TestCatalogIDsAreUnique checks the process-wide id source hands out a distinct
// id per frame, which is what lets observers correlate frames across a pipeline.
func TestCatalogIDsAreUnique(t *testing.T) {
	seen := map[uint64]string{}
	for _, entry := range catalog() {
		f := entry.build()
		if prev, dup := seen[f.ID()]; dup {
			t.Errorf("%s reused id %d, already held by %s", entry.label, f.ID(), prev)
		}
		seen[f.ID()] = entry.label
	}
}

func TestConstructorFields(t *testing.T) {
	t.Run("DTMF", func(t *testing.T) {
		if got := frames.NewInputDTMFFrame(frames.KeypadStar).Button; got != frames.KeypadStar {
			t.Errorf("Button = %q, want *", got)
		}
		if got := frames.NewOutputDTMFFrame(frames.KeypadPound).Button; got != frames.KeypadPound {
			t.Errorf("Button = %q, want #", got)
		}
	})

	t.Run("function calls", func(t *testing.T) {
		calls := []frames.ToolCall{{ID: "a", Name: "one"}, {ID: "b", Name: "two"}}
		started := frames.NewFunctionCallsStartedFrame("thinking", calls)
		if started.PreambleText != "thinking" || len(started.Calls) != 2 {
			t.Errorf("started = %+v, want the preamble and both calls", started)
		}

		prog := frames.NewFunctionCallInProgressFrame("a", "one")
		if prog.ToolCallID != "a" || prog.ToolName != "one" {
			t.Errorf("in-progress = %+v", prog)
		}

		res := frames.NewFunctionCallResultFrame("a", "one", "done", false)
		if res.ToolCallID != "a" || res.Result != "done" || res.IsError {
			t.Errorf("result = %+v", res)
		}
		// Generation re-runs after a tool result unless a handler stops the turn.
		if !res.RunLLM {
			t.Error("RunLLM should default to true")
		}

		cancel := frames.NewFunctionCallCancelFrame("a", "one")
		if cancel.ToolCallID != "a" || cancel.ToolName != "one" {
			t.Errorf("cancel = %+v", cancel)
		}
	})

	t.Run("speech control params", func(t *testing.T) {
		vadParams := &vad.Params{StopSecs: 0.8}
		turnParams := &turn.Params{StopSecs: 0.8, PreSpeechMs: 200, MaxDurationSecs: 8}
		f := frames.NewSpeechControlParamsFrame(vadParams, turnParams)
		if f.VADParams == nil || f.VADParams.StopSecs != 0.8 {
			t.Errorf("VADParams = %+v", f.VADParams)
		}
		if f.TurnParams == nil || f.TurnParams.PreSpeechMs != 200 || f.TurnParams.MaxDurationSecs != 8 {
			t.Errorf("TurnParams = %+v", f.TurnParams)
		}
	})

	t.Run("mixer settings", func(t *testing.T) {
		f := frames.NewMixerUpdateSettingsFrame(map[string]any{"volume": 0.5})
		if f.Settings["volume"] != 0.5 {
			t.Errorf("Settings = %v", f.Settings)
		}
		if !frames.NewMixerEnableFrame(true).Enable {
			t.Error("Enable should carry the constructor argument")
		}
	})

	t.Run("transport messages", func(t *testing.T) {
		in := frames.NewInputTransportMessageFrame([]byte("payload"))
		if string(in.Message) != "payload" {
			t.Errorf("Message = %q", in.Message)
		}
		out := frames.NewOutputTransportMessageFrame("payload")
		if out.Message != "payload" {
			t.Errorf("Message = %v", out.Message)
		}
	})

	t.Run("messages append", func(t *testing.T) {
		msgs := []frames.Message{{Role: frames.RoleUser, Text: "hi"}}
		if got := frames.NewLLMMessagesAppendFrame(msgs); len(got.Messages) != 1 {
			t.Errorf("Messages = %v", got.Messages)
		}
	})

	t.Run("LLM marker", func(t *testing.T) {
		if got := frames.NewLLMMarkerFrame("check").Marker; got != "check" {
			t.Errorf("Marker = %q", got)
		}
	})
}

// TestServiceMetadataInterface checks both metadata frames satisfy the interface
// downstream processors assert on, rather than each carrying its own accessors.
func TestServiceMetadataInterface(t *testing.T) {
	stt := frames.NewSTTMetadataFrame(250 * time.Millisecond)
	stt.ServiceName = "deepgram"
	stt.UserTurns = frames.UserTurnExternal

	llm := frames.NewLLMServiceMetadataFrame("openai-realtime")
	llm.UserTurns = frames.UserTurnUnspecified

	tests := []struct {
		name      string
		meta      frames.ServiceMetadata
		wantName  string
		wantTurns frames.UserTurnRecommendation
	}{
		{"STT", stt, "deepgram", frames.UserTurnExternal},
		{"LLM", llm, "openai-realtime", frames.UserTurnUnspecified},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.meta.Service(); got != tt.wantName {
				t.Errorf("Service() = %q, want %q", got, tt.wantName)
			}
			if got := tt.meta.RecommendedUserTurns(); got != tt.wantTurns {
				t.Errorf("RecommendedUserTurns() = %v, want %v", got, tt.wantTurns)
			}
		})
	}

	if got := stt.TTFSP99Latency; got != 250*time.Millisecond {
		t.Errorf("TTFSP99Latency = %v, want 250ms", got)
	}
}

func TestUserTurnRecommendationString(t *testing.T) {
	tests := []struct {
		in   frames.UserTurnRecommendation
		want string
	}{
		{frames.UserTurnExternal, "external"},
		{frames.UserTurnUnspecified, "unspecified"},
		{frames.UserTurnRecommendation(99), "unspecified"},
	}
	for _, tt := range tests {
		if got := tt.in.String(); got != tt.want {
			t.Errorf("String() = %q, want %q", got, tt.want)
		}
	}
}

// TestTTSTextFrameOriginal checks which text is recorded in the LLM context: the
// raw text when the TTS rewrote it, otherwise the spoken text.
func TestTTSTextFrameOriginal(t *testing.T) {
	f := frames.NewTTSTextFrame("one hundred")
	if got := f.Original(); got != "one hundred" {
		t.Errorf("Original() = %q, want the spoken text when RawText is unset", got)
	}
	if !f.AppendToContext {
		t.Error("AppendToContext should default to true")
	}

	f.RawText = "100"
	if got := f.Original(); got != "100" {
		t.Errorf("Original() = %q, want the raw text once set", got)
	}
	if got := f.String(); !strings.Contains(got, "raw: [100]") {
		t.Errorf("String() = %q, want it to render RawText", got)
	}
}

func TestTTSSpeakFrameDefaults(t *testing.T) {
	f := frames.NewTTSSpeakFrame("welcome")
	if f.Text != "welcome" {
		t.Errorf("Text = %q", f.Text)
	}
	if !f.AppendToContext {
		t.Error("AppendToContext should default to true")
	}
}

// TestMetricsFrameString covers both renderings: with and without token usage.
func TestMetricsFrameString(t *testing.T) {
	f := frames.NewMetricsFrame("llm-1")
	if got := f.String(); !strings.Contains(got, "processor: llm-1") || strings.Contains(got, "tokens") {
		t.Errorf("String() = %q, want the processor only", got)
	}

	f.Tokens = &frames.LLMTokenUsage{PromptTokens: 12, CompletionTokens: 34}
	if got := f.String(); !strings.Contains(got, "12 in / 34 out") {
		t.Errorf("String() = %q, want the token counts", got)
	}
}

// TestFramePTSRendering checks the presentation timestamp shows as "none" until
// it is set, so an unset PTS is never confused with zero.
func TestFramePTSRendering(t *testing.T) {
	f := frames.NewInputAudioRawFrame(nil, 16000, 1)
	if got := f.String(); !strings.Contains(got, "pts: none") {
		t.Errorf("String() = %q, want an unset pts to render as none", got)
	}
	f.SetPTS(1234)
	if got := f.String(); !strings.Contains(got, "pts: 1234") {
		t.Errorf("String() = %q, want the pts value", got)
	}
}
