package pionrtc

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

// Transport is a WebRTC transport backed by a Pion connection. It provides the
// input and output processors for a pipeline.
type Transport struct {
	in  *inputTransport
	out *outputTransport
}

// NewTransport builds a WebRTC transport over conn.
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

// inputTransport reads Opus RTP from the connection, decodes it to PCM, and
// pushes InputAudioRawFrames into the pipeline.
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
	in.BaseInput = transport.NewBaseInput("PionInput", params, in)
	return in
}

func channels(n int) int {
	if n == 0 {
		return 1
	}
	return n
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

	// Surface data channel messages as frames in the pipeline.
	in.conn.OnMessage(func(raw []byte) {
		if readCtx.Err() != nil {
			return
		}
		in.PushTransportMessage(readCtx, raw)
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
			// A deadline lets us re-check ctx; any other error means the track
			// is gone.
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
// connection's audio track.
type outputTransport struct {
	*transport.BaseOutput
	conn *Connection
	enc  *opus.Encoder
	tail []byte
}

func newOutput(conn *Connection, params transport.Params) *outputTransport {
	out := &outputTransport{conn: conn}
	out.BaseOutput = transport.NewBaseOutput("PionOutput", params, out)
	return out
}

// SendMessage sends an application message over the data channel.
func (out *outputTransport) SendMessage(_ context.Context, data []byte) error {
	return out.conn.SendMessage(data)
}

// WriteAudio encodes PCM into 20 ms Opus frames and sends them. Any audio that
// does not fill a whole frame is held until the next call.
func (out *outputTransport) WriteAudio(_ context.Context, pcm []byte) error {
	ch := channels(out.Params().AudioOutChannels)
	if out.enc == nil {
		enc, err := opus.NewEncoder(ch, out.Params().AudioOutBitrate)
		if err != nil {
			return err
		}
		out.enc = enc
	}

	frameBytes := opus.FrameBytes(ch)
	out.tail = append(out.tail, pcm...)
	for len(out.tail) >= frameBytes {
		packet, err := out.enc.Encode(out.tail[:frameBytes])
		if err != nil {
			return err
		}
		if err := out.conn.WriteAudio(packet, opus.FrameDuration); err != nil {
			return err
		}
		out.tail = out.tail[frameBytes:]
	}
	return nil
}
