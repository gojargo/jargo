package livekit

import (
	"context"
	"errors"
	"os"
	"sync"
	"time"

	"github.com/gojargo/jargo/audio/opus"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/transport"
)

// readDeadline bounds a blocking RTP read so the read loop can notice
// cancellation between packets.
const readDeadline = 500 * time.Millisecond

// Transport is a LiveKit room transport. It provides the input and output
// processors for a pipeline.
type Transport struct {
	in  *inputTransport
	out *outputTransport
}

// NewTransport builds a transport over a joined LiveKit connection.
func NewTransport(conn *Connection, params transport.Params) *Transport {
	return &Transport{
		in:  newInput(conn, params),
		out: newOutput(conn, params),
	}
}

// Input returns the input processor.
func (t *Transport) Input() processor.Processor { return t.in }

// Output returns the output processor.
func (t *Transport) Output() processor.Processor { return t.out }

func channels(n int) int {
	if n == 0 {
		return 1
	}
	return n
}

// inputTransport reads Opus RTP from the subscribed remote track, decodes it to
// PCM, and pushes InputAudioRawFrames into the pipeline.
type inputTransport struct {
	*transport.BaseInput
	conn *Connection
	dec  *opus.Decoder

	readWG     sync.WaitGroup
	mu         sync.Mutex // guards readCancel
	readCancel context.CancelFunc
}

func newInput(conn *Connection, params transport.Params) *inputTransport {
	in := &inputTransport{conn: conn}
	in.BaseInput = transport.NewBaseInput("LiveKitInput", params, in)
	return in
}

// StartReading decodes incoming audio on its own goroutine.
func (in *inputTransport) StartReading(ctx context.Context) error {
	dec, err := opus.NewDecoder(channels(in.Params().AudioInChannels))
	if err != nil {
		return err
	}
	in.dec = dec

	readCtx, cancel := context.WithCancel(ctx)
	in.mu.Lock()
	in.readCancel = cancel
	in.mu.Unlock()
	in.readWG.Add(1)
	go in.readLoop(readCtx)

	in.conn.OnMessage(func(raw []byte) {
		if readCtx.Err() != nil {
			return
		}
		in.PushTransportMessage(readCtx, raw)
	})
	// A key a SIP caller presses becomes a frame, so the DTMF aggregator works
	// on a LiveKit call the same way it does on a telephony one.
	in.conn.OnDTMF(func(button frames.KeypadEntry) {
		if readCtx.Err() != nil {
			return
		}
		_ = in.PushFrame(readCtx, frames.NewInputDTMFFrame(button), processor.Downstream)
	})
	return nil
}

// StopReading stops the read goroutine.
func (in *inputTransport) StopReading(context.Context) error {
	in.mu.Lock()
	cancel := in.readCancel
	in.mu.Unlock()
	if cancel != nil {
		cancel()
	}
	in.readWG.Wait()
	return nil
}

func (in *inputTransport) readLoop(ctx context.Context) {
	defer in.readWG.Done()

	track, err := in.conn.RemoteAudioTrack(ctx)
	if err != nil {
		return
	}
	ch := channels(in.Params().AudioInChannels)

	for {
		if ctx.Err() != nil {
			return
		}
		_ = track.SetReadDeadline(time.Now().Add(readDeadline))
		pkt, _, err := track.ReadRTP()
		if err != nil {
			if errors.Is(err, os.ErrDeadlineExceeded) {
				continue
			}
			return
		}
		if len(pkt.Payload) == 0 {
			continue
		}
		pcm, err := in.dec.Decode(pkt.Payload)
		if err != nil {
			continue
		}
		in.PushAudioFrame(ctx, frames.NewInputAudioRawFrame(pcm, opus.SampleRate, ch))
	}
}

// outputTransport encodes outgoing PCM chunks into Opus and writes them to the
// published track.
type outputTransport struct {
	*transport.BaseOutput
	conn     *Connection
	enc      *opus.Encoder
	tail     []byte
	nextSend time.Time
}

func newOutput(conn *Connection, params transport.Params) *outputTransport {
	out := &outputTransport{conn: conn}
	out.BaseOutput = transport.NewBaseOutput("LiveKitOutput", params, out)
	return out
}

// SendMessage publishes an application message to the room.
func (out *outputTransport) SendMessage(_ context.Context, data []byte) error {
	return out.conn.SendMessage(data)
}

// WriteAudio encodes PCM into 20 ms Opus frames and sends them, paced to
// wall-clock time. LiveKit's sample track packetizes and sends a frame the
// instant it is called, so writing a whole utterance back-to-back floods the
// client's jitter buffer; we pace explicitly, exactly as the Pion transport
// does. Audio that does not fill a whole frame is held until the next call.
func (out *outputTransport) WriteAudio(ctx context.Context, f frames.OutputAudioFrame) (bool, error) {
	pcm := f.AudioData().Audio
	ch := channels(out.Params().AudioOutChannels)
	if out.enc == nil {
		p := out.Params()
		enc, err := opus.NewEncoder(opus.EncoderConfig{
			Channels:           ch,
			Bitrate:            p.AudioOutBitrate,
			InbandFEC:          p.AudioOutFEC,
			ExpectedPacketLoss: p.AudioOutExpectedPacketLoss,
		})
		if err != nil {
			return false, err
		}
		out.enc = enc
	}

	frameBytes := opus.FrameBytes(ch)
	out.tail = append(out.tail, pcm...)
	for len(out.tail) >= frameBytes {
		packet, err := out.enc.Encode(out.tail[:frameBytes])
		if err != nil {
			return false, err
		}
		out.pace(ctx)
		if err := out.conn.WriteAudio(packet, opus.FrameDuration); err != nil {
			return false, err
		}
		out.tail = out.tail[frameBytes:]
	}
	return true, nil
}

// pace blocks until it is time to send the next 20 ms Opus frame, keeping output
// at real time. The clock resets after a gap longer than one frame so playback
// resumes immediately rather than bursting to catch up.
func (out *outputTransport) pace(ctx context.Context) {
	now := time.Now()
	if out.nextSend.IsZero() || now.Sub(out.nextSend) > opus.FrameDuration {
		out.nextSend = now
	}
	if d := time.Until(out.nextSend); d > 0 {
		timer := time.NewTimer(d)
		defer timer.Stop()
		select {
		case <-ctx.Done():
		case <-timer.C:
		}
	}
	out.nextSend = out.nextSend.Add(opus.FrameDuration)
}
