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

// ClientSerializer implements wsserver.Serializer.
var _ wsserver.Serializer = (*ClientSerializer)(nil)

// ClientSerializer is the harness's serializer: RTVI client messages, plus the
// bot's audio.
//
// eval-bot-audio becomes an InputAudioRawFrame, so the harness's STT can
// transcribe what the bot actually said. Every other message from the bot
// stays a raw message, as rtviclient.Serializer keeps it, and the event stream
// maps the ones a scenario reads to its events. Its reports about the harness
// (its VAD, speaking and transcription of the harness's speech) cannot be
// mistaken for what the harness computes from the bot's audio that way, nor its
// reports about itself for the harness's own output.
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

// Deserialize turns an RTVI server message into a frame: the bot's audio as
// audio, any other as the raw message it is. It returns (nil, nil) for a
// message that is not an RTVI message.
func (s *ClientSerializer) Deserialize(data []byte) (frames.Frame, error) {
	var message map[string]any
	if err := json.Unmarshal(data, &message); err != nil {
		return s.Serializer.Deserialize(data)
	}
	if label, _ := message["label"].(string); label == rtvi.MessageLabel && message["type"] == evalws.BotAudioType {
		payload, _ := message["data"].(map[string]any)
		return s.botAudio(payload), nil
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
