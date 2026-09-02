// Package protobuf is the wsserver.Serializer a WebSocket client speaks: frames
// encoded as protobuf and sent in binary WebSocket messages. It is the wire for
// a client that connects over a plain WebSocket rather than WebRTC.
//
// Both directions carry the same five frames. Outbound, the bot's speech goes as
// audio, its text and transcriptions as their own frames, a barge-in as an
// interruption, and every application message as JSON inside a message frame,
// which is how the RTVI protocol travels this wire. Inbound, a client's
// microphone audio becomes an InputAudioRawFrame (VAD, turn detection and STT
// then see it as a live mic) and a message frame becomes an
// InputTransportMessageFrame for the pipeline's RTVI processor to route.
//
// Pair it with a pipeline that includes an rtvi.Processor.
package protobuf

import (
	"encoding/json"
	"errors"
	"fmt"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/transport/wsserver"
	"github.com/gojargo/jargo/transport/wsserver/protobuf/internal/framepb"
	"google.golang.org/protobuf/proto"
)

// Serializer implements wsserver.Serializer.
var _ wsserver.Serializer = (*Serializer)(nil)

// ErrUnknownFrame reports a wire message carrying none of the frames this
// format defines.
var ErrUnknownFrame = errors.New("protobuf: message carries no frame")

// Serializer converts between pipeline frames and their protobuf encoding. It
// holds no per-session state, so a single value may serve any session.
type Serializer struct {
	wsserver.BaseSerializer
}

// New builds a protobuf serializer.
func New() *Serializer {
	// This wire is the delivery channel for RTVI messages, so they are not
	// filtered off it the way they are off a telephony call.
	return &Serializer{BaseSerializer: wsserver.BaseSerializer{KeepRTVIMessages: true}}
}

// Setup is a no-op: the encoding carries the sample rate of every chunk, so
// there is nothing to take from the StartFrame.
func (*Serializer) Setup(processor.Setup) error { return nil }

// Serialize encodes an outbound frame. Frames this format has no message for are
// dropped, which is every frame the client has no use for.
func (s *Serializer) Serialize(f frames.Frame) (wsserver.Message, error) {
	out := &framepb.Frame{}
	switch fr := f.(type) {
	case *frames.TranscriptionFrame:
		out.Frame = &framepb.Frame_Transcription{Transcription: &framepb.TranscriptionFrame{
			Id:        fr.ID(),
			Name:      fr.Name(),
			Text:      fr.Text,
			UserId:    fr.UserID,
			Timestamp: fr.Timestamp,
		}}
	case *frames.TextFrame:
		out.Frame = &framepb.Frame_Text{Text: &framepb.TextFrame{
			Id:   fr.ID(),
			Name: fr.Name(),
			Text: fr.Text,
		}}
	case frames.OutputAudioFrame:
		out.Frame = &framepb.Frame_Audio{Audio: audioProto(fr)}
	case *frames.InterruptionFrame:
		out.Frame = &framepb.Frame_Interruption{Interruption: &framepb.InterruptionFrame{
			Id:   fr.ID(),
			Name: fr.Name(),
		}}
	case frames.OutputTransportMessage:
		if s.ShouldIgnoreFrame(f) {
			return wsserver.Message{}, nil
		}
		// The payload travels as its JSON inside a message frame, which is what
		// makes this the wire the RTVI protocol is carried on.
		data, err := json.Marshal(fr.TransportMessage())
		if err != nil {
			return wsserver.Message{}, err
		}
		out.Frame = &framepb.Frame_Message{Message: &framepb.MessageFrame{Data: string(data)}}
	default:
		// Not one of the five frames this format defines.
		return wsserver.Message{}, nil
	}
	return wsserver.BinaryMessage(proto.Marshal(out))
}

// audioProto encodes one chunk of the bot's audio. The rate and channel count
// travel with it, because they are what the client needs to play it.
func audioProto(f frames.OutputAudioFrame) *framepb.AudioRawFrame {
	a := f.AudioData()
	out := &framepb.AudioRawFrame{
		Id:          f.ID(),
		Name:        f.Name(),
		Audio:       a.Audio,
		SampleRate:  uint32(a.SampleRate),  //nolint:gosec // a sample rate is never negative
		NumChannels: uint32(a.NumChannels), //nolint:gosec // a channel count is never negative
	}
	if pts, ok := f.Base().PTS(); ok && pts >= 0 {
		out.Pts = new(uint64(pts)) //nolint:gosec // guarded non-negative above
	}
	return out
}

// Deserialize decodes an inbound wire message.
func (*Serializer) Deserialize(data []byte) (frames.Frame, error) {
	var in framepb.Frame
	if err := proto.Unmarshal(data, &in); err != nil {
		return nil, err
	}
	switch f := in.GetFrame().(type) {
	case *framepb.Frame_Text:
		return frames.NewTextFrame(f.Text.GetText()), nil
	case *framepb.Frame_Audio:
		return inputAudio(f.Audio), nil
	case *framepb.Frame_Transcription:
		t := f.Transcription
		return frames.NewTranscriptionFrame(t.GetText(), t.GetUserId(), t.GetTimestamp()), nil
	case *framepb.Frame_Message:
		// The pipeline's RTVI processor parses and routes it, so it travels as
		// the bytes it arrived in.
		return frames.NewInputTransportMessageFrame([]byte(f.Message.GetData())), nil
	case *framepb.Frame_Interruption:
		return frames.NewInterruptionFrame(), nil
	default:
		return nil, fmt.Errorf("%w: %T", ErrUnknownFrame, f)
	}
}

// inputAudio decodes one chunk of the client's microphone audio.
//
// The frame is given a fresh identity rather than the one on the wire: an id
// there was allocated by whatever produced the frame at the other end, and
// jargo's are a counter local to this process that tracing and the observers
// key on. The presentation timestamp is carried across, since it describes the
// audio rather than the frame that held it.
func inputAudio(a *framepb.AudioRawFrame) *frames.InputAudioRawFrame {
	f := frames.NewInputAudioRawFrame(a.GetAudio(), int(a.GetSampleRate()), int(a.GetNumChannels()))
	if a.Pts != nil {
		f.SetPTS(int64(a.GetPts())) //nolint:gosec // a presentation timestamp fits an int64
	}
	return f
}
