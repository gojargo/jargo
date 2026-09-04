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
	"encoding/json"
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

// namedProcessor is a stand-in for the processor a pause or resume frame is
// addressed to; the frames only ever read its name.
type namedProcessor string

func (n namedProcessor) Name() string { return string(n) }

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
			label: "FrameProcessorPauseUrgentFrame", cat: system,
			build: func() frames.Frame { return frames.NewFrameProcessorPauseUrgentFrame(namedProcessor("p")) },
		},
		{
			label: "FrameProcessorResumeUrgentFrame", cat: system,
			build: func() frames.Frame { return frames.NewFrameProcessorResumeUrgentFrame(namedProcessor("p")) },
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
			label: "ProposedUserStartedSpeakingFrame", cat: system,
			build: func() frames.Frame { return frames.NewProposedUserStartedSpeakingFrame() },
		},
		{
			label: "ProposedUserStoppedSpeakingFrame", cat: control,
			build: func() frames.Frame { return frames.NewProposedUserStoppedSpeakingFrame() },
		},
		{
			label: "STTMuteFrame", cat: system, wantString: "mute: true",
			build: func() frames.Frame { return frames.NewSTTMuteFrame(true) },
		},
		{
			label: "ClientConnectedFrame", cat: system,
			build: func() frames.Frame { return frames.NewClientConnectedFrame() },
		},
		{
			label: "BotConnectedFrame", cat: system,
			build: func() frames.Frame { return frames.NewBotConnectedFrame() },
		},
		{
			label: "InputTextRawFrame", cat: system, wantString: "typed",
			build: func() frames.Frame { return frames.NewInputTextRawFrame("typed") },
		},
		{
			label: "UserAudioRawFrame", cat: system, wantString: "user: user-1",
			build: func() frames.Frame {
				return frames.NewUserAudioRawFrame("user-1", []byte{0, 0, 0, 0}, 16000, 1)
			},
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
			build: func() frames.Frame { return frames.NewVADUserStartedSpeakingFrame(1.5, time.Now()) },
		},
		{
			label: "VADUserStoppedSpeakingFrame", cat: system, wantString: "stop_secs: 2.250",
			build: func() frames.Frame { return frames.NewVADUserStoppedSpeakingFrame(2.25, time.Now()) },
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
			build: func() frames.Frame {
				return frames.NewMetricsFrame(frames.TTSUsageMetricsData{
					Processor: "tts-1",
					Value:     42,
				})
			},
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
			build: func() frames.Frame { return frames.NewTTSTextFrame("hello", frames.AggregationSentence) },
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
			label: "TranslationFrame", cat: data, wantString: "bonjour",
			build: func() frames.Frame { return frames.NewTranslationFrame("bonjour", "user-1", "ts") },
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
			label: "LLMConfigureOutputFrame", cat: data,
			build: func() frames.Frame { return frames.NewLLMConfigureOutputFrame(true) },
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
		{
			label: "LLMMessagesTransformFrame", cat: data, wantString: "run_llm: false",
			build: func() frames.Frame {
				return frames.NewLLMMessagesTransformFrame(func(m []frames.Message) []frames.Message { return m })
			},
		},
		{
			label: "LLMEnablePromptCachingFrame", cat: data, wantString: "enable: true",
			build: func() frames.Frame { return frames.NewLLMEnablePromptCachingFrame(true) },
		},
		{
			label: "LLMContextAssistantTurnFrame", cat: data, wantString: "said it",
			build: func() frames.Frame { return frames.NewLLMContextAssistantTurnFrame("said it", "ts") },
		},
		{
			label: "LLMContextAssistantTimestampFrame", cat: data, wantString: "timestamp: ts",
			build: func() frames.Frame { return frames.NewLLMContextAssistantTimestampFrame("ts") },
		},
		{
			label: "LLMThoughtTextFrame", cat: data, wantString: "thinking",
			build: func() frames.Frame { return frames.NewLLMThoughtTextFrame("thinking") },
		},
		{
			label: "LLMThoughtStartFrame", cat: control, wantString: "append_to_context: false",
			build: func() frames.Frame { return frames.NewLLMThoughtStartFrame() },
		},
		{
			label: "LLMThoughtEndFrame", cat: control, wantString: "signature:",
			build: func() frames.Frame { return frames.NewLLMThoughtEndFrame() },
		},

		// Control frames: in order like data frames, but carrying instructions.
		{
			label: "EndFrame", cat: control, uninterruptible: true,
			build: func() frames.Frame { return frames.NewEndFrame() },
		},
		{
			label: "FrameProcessorPauseFrame", cat: control,
			build: func() frames.Frame { return frames.NewFrameProcessorPauseFrame(namedProcessor("p")) },
		},
		{
			label: "FrameProcessorResumeFrame", cat: control,
			build: func() frames.Frame { return frames.NewFrameProcessorResumeFrame(namedProcessor("p")) },
		},
		{
			label: "StopFrame", cat: control, uninterruptible: true,
			build: func() frames.Frame { return frames.NewStopFrame() },
		},
		{
			label: "HeartbeatFrame", cat: control, wantString: "timestamp: 1.5s",
			build: func() frames.Frame { return frames.NewHeartbeatFrame(1500 * time.Millisecond) },
		},
		{
			label: "ManuallySwitchServiceFrame", cat: control, wantString: "service: svc",
			build: func() frames.Frame {
				return frames.NewManuallySwitchServiceFrame(namedProcessor("svc"))
			},
		},
		{
			label: "ServiceSwitcherRequestMetadataFrame", cat: control, wantString: "service: svc",
			build: func() frames.Frame {
				return frames.NewServiceSwitcherRequestMetadataFrame(namedProcessor("svc"))
			},
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
			label: "OutputDTMFFrame", cat: control, wantString: "buttons: 7",
			build: func() frames.Frame { return frames.NewOutputDTMFFrame(frames.KeypadSeven) },
		},
		{
			label: "OutputDTMFUrgentFrame", cat: system, wantString: "buttons: 9",
			build: func() frames.Frame { return frames.NewOutputDTMFUrgentFrame(frames.KeypadNine) },
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
			label: "LLMUpdateSettingsFrame", cat: control, uninterruptible: true,
			wantString: "service: all",
			build:      func() frames.Frame { return frames.NewLLMUpdateSettingsFrame(nil) },
		},
		{
			label: "TTSUpdateSettingsFrame", cat: control, uninterruptible: true,
			wantString: "service: all",
			build:      func() frames.Frame { return frames.NewTTSUpdateSettingsFrame(nil) },
		},
		{
			label: "STTUpdateSettingsFrame", cat: control, uninterruptible: true,
			wantString: "service: all",
			build:      func() frames.Frame { return frames.NewSTTUpdateSettingsFrame(nil) },
		},
		{
			label: "LLMMessagesAppendFrame", cat: data,
			build: func() frames.Frame {
				return frames.NewLLMMessagesAppendFrame([]frames.Message{{Role: frames.RoleUser, Text: "hi"}})
			},
		},
		// The function-call frames span three classes rather than one: the two
		// that write to the conversation are uninterruptible so they always reach
		// it, and the two that announce a change of state are system frames so
		// they overtake the queued frames of the turn they change.
		{
			label: "FunctionCallsStartedFrame", cat: system, wantString: "calls: 1",
			build: func() frames.Frame {
				return frames.NewFunctionCallsStartedFrame([]frames.ToolCall{{ID: "a", Name: "get_weather"}})
			},
		},
		{
			label: "FunctionCallInProgressFrame", cat: control, uninterruptible: true, wantString: "get_weather",
			build: func() frames.Frame { return frames.NewFunctionCallInProgressFrame("a", "get_weather", nil, true, "g1") },
		},
		{
			label: "FunctionCallResultFrame", cat: data, uninterruptible: true, wantString: "get_weather",
			build: func() frames.Frame { return frames.NewFunctionCallResultFrame("a", "get_weather", nil, "sunny") },
		},
		{
			label: "FunctionCallCancelFrame", cat: system, wantString: "get_weather",
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
	t.Run("function calls", func(t *testing.T) {
		calls := []frames.ToolCall{{ID: "a", Name: "one"}, {ID: "b", Name: "two"}}
		started := frames.NewFunctionCallsStartedFrame(calls)
		if len(started.Calls) != 2 {
			t.Errorf("started = %+v, want both calls", started)
		}

		args := json.RawMessage(`{"x":1}`)
		prog := frames.NewFunctionCallInProgressFrame("a", "one", args, true, "g1")
		if prog.ToolCallID != "a" || prog.ToolName != "one" || string(prog.Args) != `{"x":1}` {
			t.Errorf("in-progress = %+v", prog)
		}
		if !prog.CancelOnInterruption || prog.GroupID != "g1" {
			t.Errorf("in-progress call options = %+v", prog)
		}

		res := frames.NewFunctionCallResultFrame("a", "one", args, "done")
		if res.ToolCallID != "a" || res.Result != "done" || string(res.Args) != `{"x":1}` {
			t.Errorf("result = %+v", res)
		}
		// Whether generation re-runs is left to the aggregator unless something
		// says otherwise, so neither the frame nor absent properties decide it.
		if res.RunLLM != nil || res.Properties != nil {
			t.Errorf("result should carry no decision by default: %+v", res)
		}
		if !res.Properties.Final() {
			t.Error("absent properties describe a final result")
		}
		notFinal := false
		if (&frames.FunctionCallResultProperties{IsFinal: &notFinal}).Final() {
			t.Error("IsFinal false describes an intermediate result")
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
	// The strategies travel untyped, so the frame stays free of the package
	// that defines them. Any value stands in for them here.
	recommended := "external strategies"

	stt := frames.NewSTTMetadataFrame(250 * time.Millisecond)
	stt.ServiceName = "deepgram"
	stt.UserTurnStrategies = recommended

	llm := frames.NewLLMServiceMetadataFrame("openai-realtime")

	tests := []struct {
		name      string
		meta      frames.ServiceMetadata
		wantName  string
		wantTurns any
	}{
		{"STT", stt, "deepgram", recommended},
		{"LLM", llm, "openai-realtime", nil},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.meta.Service(); got != tt.wantName {
				t.Errorf("Service() = %q, want %q", got, tt.wantName)
			}
			if got := tt.meta.RecommendedUserTurnStrategies(); got != tt.wantTurns {
				t.Errorf("RecommendedUserTurnStrategies() = %v, want %v", got, tt.wantTurns)
			}
		})
	}

	if got := stt.TTFSP99Latency; got != 250*time.Millisecond {
		t.Errorf("TTFSP99Latency = %v, want 250ms", got)
	}
}

// TestTTSTextFrameOriginal checks which text is recorded in the LLM context: the
// raw text when the TTS rewrote it, otherwise the spoken text.
func TestTTSTextFrameOriginal(t *testing.T) {
	f := frames.NewTTSTextFrame("one hundred", frames.AggregationSentence)
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

// TestMetricsFrameString covers both renderings: one measurement names its
// processor, several are counted.
func TestMetricsFrameString(t *testing.T) {
	base := frames.BaseMetricsData{Processor: "llm-1"}
	f := frames.NewMetricsFrame(frames.TTFBMetricsData{BaseMetricsData: base, Value: time.Second})
	if got := f.String(); !strings.Contains(got, "processor: llm-1") {
		t.Errorf("String() = %q, want the processor named", got)
	}

	f = frames.NewMetricsFrame(
		frames.TTFBMetricsData{BaseMetricsData: base, Value: time.Second},
		frames.ProcessingMetricsData{BaseMetricsData: base, Value: time.Second},
	)
	if got := f.String(); !strings.Contains(got, "2 measurements") {
		t.Errorf("String() = %q, want the measurement count", got)
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

// TestDTMFConstructorFields checks the keypad frames carry the keys they were
// built with. The output frames take a run of keys, since a caller entering an
// account number presses several and the order is the number.
func TestDTMFConstructorFields(t *testing.T) {
	if got := frames.NewInputDTMFFrame(frames.KeypadStar).Button; got != frames.KeypadStar {
		t.Errorf("Button = %q, want *", got)
	}
	if got := frames.NewOutputDTMFFrame(frames.KeypadPound).Buttons; len(got) != 1 || got[0] != frames.KeypadPound {
		t.Errorf("Buttons = %q, want #", got)
	}
	seq := []frames.KeypadEntry{frames.KeypadOne, frames.KeypadTwo, frames.KeypadPound}
	if got := frames.NewOutputDTMFSequenceFrame(seq).Keys(); frames.KeypadString(got) != "12#" {
		t.Errorf("Keys() = %q, want 12#", frames.KeypadString(got))
	}
	if got := frames.NewOutputDTMFUrgentFrame(frames.KeypadStar).Keys(); frames.KeypadString(got) != "*" {
		t.Errorf("urgent Keys() = %q, want *", frames.KeypadString(got))
	}
}
