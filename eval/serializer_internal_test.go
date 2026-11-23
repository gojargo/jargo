package eval

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"reflect"
	"testing"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor/rtvi"
	"github.com/gojargo/jargo/transport/wsserver/evalws"
)

// serverMessage is an RTVI server message, as the bot sends it.
func serverMessage(t *testing.T, msgType string, data any) []byte {
	t.Helper()
	raw, err := json.Marshal(map[string]any{"label": rtvi.MessageLabel, "type": msgType, "data": data})
	if err != nil {
		t.Fatal(err)
	}
	return raw
}

func deserialize(t *testing.T, s *ClientSerializer, data []byte) frames.Frame {
	t.Helper()
	f, err := s.Deserialize(data)
	if err != nil {
		t.Fatal(err)
	}
	return f
}

// The bot's metrics and its server-message stay raw messages: the harness
// reports the metrics as sent, and the server-message's data as the bot's own.
func TestClientSerializerKeepsMetricsAndServerMessagesRaw(t *testing.T) {
	for msgType, data := range map[string]map[string]any{
		rtvi.TypeMetrics:       {"ttfb": []any{map[string]any{"processor": "DeepgramSTTService#0", "value": 0.31}}},
		rtvi.TypeServerMessage: {"done": true, "transcript": "hello"},
		// The bot's flow entering a node stays a raw message too.
		rtvi.TypeFlowNodeEntered: {"node": "verify.ask", "previous_node": "start", "parent": "verify"},
	} {
		f, ok := deserialize(t, NewClientSerializer(), serverMessage(t, msgType, data)).(*frames.InputTransportMessageFrame)
		var message map[string]any
		if ok {
			_ = json.Unmarshal(f.Message, &message)
		}
		if !ok || message["type"] != msgType || !reflect.DeepEqual(message["data"], data) {
			t.Fatalf("%s: frame = %+v, want the raw message with its data", msgType, f)
		}
	}
}

// The client-side serializer the harness pipeline uses to talk to the bot.
func TestClientSerializer(t *testing.T) {
	t.Run("eval-bot-audio becomes input audio", func(t *testing.T) {
		// Already at the harness STT rate: passes through unchanged.
		pcm := []byte{1, 2, 3, 4}
		f, ok := deserialize(t, NewClientSerializer(), serverMessage(t, evalws.BotAudioType, map[string]any{
			"audio": base64.StdEncoding.EncodeToString(pcm), "sampleRate": 16000,
		})).(*frames.InputAudioRawFrame)
		if !ok || !bytes.Equal(f.Audio, pcm) || f.SampleRate != 16000 || f.NumChannels != 1 {
			t.Fatalf("frame = %+v, want the audio as it came", f)
		}
	})
	t.Run("eval-bot-audio is resampled to the harness rate", func(t *testing.T) {
		pcm := make([]byte, 2*480) // 20ms at 24 kHz
		f, ok := deserialize(t, NewClientSerializer(), serverMessage(t, evalws.BotAudioType, map[string]any{
			"audio": base64.StdEncoding.EncodeToString(pcm), "sampleRate": 24000,
		})).(*frames.InputAudioRawFrame)
		if !ok || f.SampleRate != evalws.STTSampleRate {
			t.Fatalf("frame = %+v, want it at the STT rate", f)
		}
	})
	t.Run("user-transcription stays a raw message", func(t *testing.T) {
		// Not a TranscriptionFrame, which in the harness is its STT's
		// transcription of the bot's audio.
		f := deserialize(t, NewClientSerializer(), serverMessage(t, rtvi.TypeUserTranscription, //nolint:staticcheck // sent
			map[string]any{"text": "hello", "user_id": "u", "final": true}))
		if _, ok := f.(*frames.InputTransportMessageFrame); !ok {
			t.Fatalf("frame = %T, want the raw message", f)
		}
	})
	t.Run("bot-output and user-input stay raw messages", func(t *testing.T) {
		for _, msgType := range []string{rtvi.TypeBotOutput, rtvi.TypeUserInput} {
			f, ok := deserialize(t, NewClientSerializer(),
				serverMessage(t, msgType, map[string]any{"text": "mhm"})).(*frames.InputTransportMessageFrame)
			var message map[string]any
			if ok {
				_ = json.Unmarshal(f.Message, &message)
			}
			if !ok || message["type"] != msgType {
				t.Fatalf("%s: frame = %+v, want the raw message", msgType, f)
			}
		}
	})
	t.Run("function calls are kept like the bot's other reports", func(t *testing.T) {
		// The same kind of frame as the bot's other reports, so none can
		// overtake another: they reach the harness in the order the bot sent
		// them.
		for _, canceled := range []bool{true, false} {
			f, ok := deserialize(t, NewClientSerializer(), serverMessage(t, rtvi.TypeLLMFunctionCallStop, map[string]any{
				"function_name": "write_report", "tool_call_id": "call_1", keyCanceled: canceled,
			})).(*frames.InputTransportMessageFrame)
			var message struct {
				Data map[string]any `json:"data"`
			}
			if ok {
				_ = json.Unmarshal(f.Message, &message)
			}
			if !ok || message.Data[keyCanceled] != canceled {
				t.Fatalf("canceled %v: frame = %+v, want the raw message", canceled, f)
			}
		}
	})
	t.Run("generic messages delegate to the base", func(t *testing.T) {
		s := NewClientSerializer()
		started, ok := deserialize(t, s, serverMessage(t, rtvi.TypeBotLLMStarted, nil)).(*frames.InputTransportMessageFrame)
		var message map[string]any
		if ok {
			_ = json.Unmarshal(started.Message, &message)
		}
		if !ok || message["type"] != rtvi.TypeBotLLMStarted {
			t.Fatalf("bot-llm-started = %+v, want the raw message", started)
		}
		if _, ok := deserialize(t, s, serverMessage(t, rtvi.TypeBotReady, nil)).(*frames.InputTransportMessageFrame); !ok {
			t.Fatal("bot-ready is not a raw message")
		}
		if f := deserialize(t, s, []byte("not json")); f != nil {
			t.Fatalf("not json = %T, want none", f)
		}
	})
	t.Run("a bot image stays a raw message", func(t *testing.T) {
		f, ok := deserialize(t, NewClientSerializer(), serverMessage(t, evalws.BotImageType,
			map[string]any{"width": 2, "height": 2})).(*frames.InputTransportMessageFrame)
		var message map[string]any
		if ok {
			_ = json.Unmarshal(f.Message, &message)
		}
		if !ok || message["type"] != evalws.BotImageType {
			t.Fatalf("frame = %+v, want the raw message", f)
		}
	})
}
