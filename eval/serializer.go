package eval

import (
	"encoding/base64"
	"encoding/json"
	"log/slog"
	"strconv"

	"github.com/gojargo/jargo/audio/resample"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor/rtvi"
	"github.com/gojargo/jargo/transport/wsclient/rtviclient"
	"github.com/gojargo/jargo/transport/wsserver"
	"github.com/gojargo/jargo/transport/wsserver/evalws"
)

// The two ends of the eval wire. The bot's end, the serializer of its eval
// transport, lives in transport/wsserver/evalws so a bot can serve the harness
// without importing it; ClientSerializer, the harness's end, is here. Both
// speak RTVI, and the eval adds messages of its own on top.

// FunctionCallCancelFrame is a function call the bot reported canceled, as the
// harness receives it.
//
// The bot's own FunctionCallCancelFrame is a system frame, which would overtake
// the bot's earlier reports on its way through the harness. This one is queued
// like FunctionCallInProgressFrame and FunctionCallResultFrame, so the harness
// sees the bot's function calls in the order the bot sent them.
type FunctionCallCancelFrame struct {
	frames.BaseControlFrame
	frames.UninterruptibleMixin

	// ToolName is the name of the function that was canceled.
	ToolName string
	// ToolCallID identifies the canceled function call.
	ToolCallID string
}

// NewFunctionCallCancelFrame builds the frame for the call toolCallID to name
// the bot reported canceled.
func NewFunctionCallCancelFrame(name, toolCallID string) *FunctionCallCancelFrame {
	return &FunctionCallCancelFrame{
		BaseControlFrame: frames.NewBaseControlFrame("EvalFunctionCallCancelFrame"),
		ToolName:         name,
		ToolCallID:       toolCallID,
	}
}

// reportedEventTypes are the bot's reports about the user it is talking to,
// that is about the harness: its raw VAD, turn-level speaking, and the
// transcription of what it heard. The harness surfaces them as scenario events
// but does not let them drive its own pipeline. They are kept as raw messages,
// which the sink maps to events, for two reasons: a TranscriptionFrame in the
// harness's pipeline already means its own STT transcribed the bot's audio (the
// response), and the VAD and speaking frames are what the harness computes from
// the bot's audio, a different signal from the bot's VAD on the harness's
// audio. Routing the reports as messages keeps both without a clash of frame
// types.
//
//nolint:gochecknoglobals // fixed lookup table, read only
var reportedEventTypes = map[string]bool{
	rtvi.TypeUserStartedSpeaking: true,
	rtvi.TypeUserStoppedSpeaking: true,
	rtvi.TypeVADUserStarted:      true,
	rtvi.TypeVADUserStopped:      true,
	// The bot reporting that its output was interrupted. Kept as a message so
	// it does not become an InterruptionFrame, which the harness's own turn
	// aggregator already broadcasts when its VAD hears the bot speak.
	rtvi.TypeBotInterrupted: true,
	// What the bot said, read for its backchannels, and what the user said:
	// its transcriptions and backchannels.
	rtvi.TypeBotOutput: true,
	rtvi.TypeUserInput: true,
}

// ClientSerializer implements wsserver.Serializer.
var _ wsserver.Serializer = (*ClientSerializer)(nil)

// ClientSerializer is the harness's serializer: RTVI client messages, plus the
// eval's own.
//
// eval-bot-audio becomes an InputAudioRawFrame, so the harness's STT can
// transcribe what the bot actually said. The bot's reports about the harness
// (its transcription, VAD and speaking messages) stay raw messages rather than
// becoming frames, so they cannot be mistaken for what the harness computes from
// the bot's audio. So does eval-bot-image, which has no frame on the harness's
// side to become. The bot's metrics stay raw messages too, so the sink can
// report them as they were sent, and so does a server-message, whose data is
// the bot's own. A canceled llm-function-call-stopped becomes a
// FunctionCallCancelFrame, which keeps its place among the bot's other
// function-call reports. The deprecated user-transcription is dropped:
// user-input carries the same transcriptions.
//
// One ClientSerializer serves one connection.
type ClientSerializer struct {
	*rtviclient.Serializer

	// resampler takes the bot's audio to evalws.STTSampleRate for the
	// pipeline's VAD and STT. It is built on the first chunk of bot audio, and
	// again when the bot's rate changes; it is only used from the transport's
	// read loop.
	resampler     *resample.Resampler
	resamplerRate int
}

// NewClientSerializer builds the harness's serializer.
func NewClientSerializer() *ClientSerializer {
	return &ClientSerializer{Serializer: rtviclient.New()}
}

// Deserialize turns an RTVI server message into a frame, handling the eval's own
// message types, or returns (nil, nil) to drop the message.
func (s *ClientSerializer) Deserialize(data []byte) (frames.Frame, error) {
	var message map[string]any
	if err := json.Unmarshal(data, &message); err != nil {
		return s.Serializer.Deserialize(data)
	}
	if label, _ := message["label"].(string); label == rtvi.MessageLabel {
		msgType, _ := message["type"].(string)
		payload, _ := message["data"].(map[string]any)
		switch {
		case msgType == evalws.BotAudioType:
			return s.botAudio(payload), nil
		case msgType == rtvi.TypeUserTranscription: //nolint:staticcheck // the message is still sent
			return nil, nil //nolint:nilnil // dropped: user-input carries it
		case msgType == rtvi.TypeLLMFunctionCallStop && truthy(payload[keyCanceled]):
			name, _ := payload["function_name"].(string)
			id, _ := payload["tool_call_id"].(string)
			return NewFunctionCallCancelFrame(name, id), nil
		case reportedEventTypes[msgType] || msgType == evalws.BotImageType ||
			// The bot's metrics, reported as sent: no per-metric mapping here
			// to fall behind the protocol.
			msgType == rtvi.TypeMetrics ||
			// A message of the bot's own making, for the scenario to read.
			msgType == rtvi.TypeServerMessage:
			// Kept as the raw message so the sink maps it to a scenario event;
			// see the type's documentation for why these are not frames.
			return frames.NewInputTransportMessageFrame(data), nil
		}
	}
	return s.Serializer.Deserialize(data)
}

// botAudio is the InputAudioRawFrame for an eval-bot-audio payload, resampled
// to the harness's STT rate.
func (s *ClientSerializer) botAudio(payload map[string]any) frames.Frame {
	encoded, _ := payload["audio"].(string)
	audio, err := base64.StdEncoding.DecodeString(encoded)
	if err != nil {
		slog.Warn("eval: dropping bot audio that is not base64", "err", err)
		audio = nil
	}
	sampleRate := intValue(payload["sampleRate"])
	if sampleRate != 0 && sampleRate != evalws.STTSampleRate && len(audio) > 0 {
		audio = s.resample(audio, sampleRate)
		sampleRate = evalws.STTSampleRate
	}
	return frames.NewInputAudioRawFrame(audio, sampleRate, 1)
}

// resample takes the bot's audio from inRate to the STT rate, with a stream
// resampler so chunk boundaries do not garble it.
func (s *ClientSerializer) resample(audio []byte, inRate int) []byte {
	if s.resampler == nil || s.resamplerRate != inRate {
		r, err := resample.New(inRate, evalws.STTSampleRate, 1)
		if err != nil {
			slog.Warn("eval: cannot resample the bot's audio", "rate", inRate, "err", err)
			return audio
		}
		s.resampler, s.resamplerRate = r, inRate
	}
	return s.resampler.Process(audio)
}

// intValue reads a JSON number, or a number written as a string, as an int.
func intValue(v any) int {
	switch n := v.(type) {
	case float64:
		return int(n)
	case string:
		i, _ := strconv.Atoi(n)
		return i
	default:
		return 0
	}
}
