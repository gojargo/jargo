// Package localaudio is a transport that captures from the local microphone and
// plays back through the local speaker, for running a bot on the same machine
// with no browser or telephony leg. It speaks the PulseAudio native protocol
// through the pure-Go github.com/jfreymuth/pulse client, so it stays cgo-free;
// it needs a running PulseAudio or PipeWire (pulse-compatible) server, which is
// the default on Linux desktops.
//
// Audio is 16-bit signed mono PCM. Capture runs at the pipeline's input rate and
// playback at its output rate; PulseAudio resamples to and from the device, so
// the pipeline can run at whatever rate its STT and TTS want.
package localaudio

import (
	"context"
	"encoding/binary"
	"fmt"
	"sync"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/transport"
	"github.com/jfreymuth/pulse"
	"github.com/jfreymuth/pulse/proto"
)

// Interface checks.
var (
	_ transport.Transport    = (*Transport)(nil)
	_ transport.InputDriver  = (*inputTransport)(nil)
	_ transport.OutputDriver = (*outputTransport)(nil)
)

// Default sample rates used when the StartFrame does not carry one.
const (
	defaultInRate  = 16000
	defaultOutRate = 24000
	// maxBufferSeconds bounds the playback backlog, so a stalled speaker cannot
	// grow the output buffer without limit.
	maxBufferSeconds = 10
)

// monoMap is a single-channel (mono) PulseAudio channel map.
//
//nolint:gochecknoglobals // immutable
var monoMap = proto.ChannelMap{proto.ChannelMono}

// Config configures the local-audio transport.
type Config struct {
	// AppName is the client name PulseAudio shows for the streams; "" uses "jargo".
	AppName string
	// Params configures audio input and output; see transport.DefaultParams.
	Params transport.Params
}

// Transport captures from the microphone and plays to the speaker over a single
// PulseAudio client connection.
type Transport struct {
	client *pulse.Client
	in     *inputTransport
	out    *outputTransport
}

// New connects to the local PulseAudio server and builds the transport. It
// returns an error if no server is reachable.
func New(cfg Config) (*Transport, error) {
	name := cfg.AppName
	if name == "" {
		name = "jargo"
	}
	client, err := pulse.NewClient(pulse.ClientApplicationName(name))
	if err != nil {
		return nil, fmt.Errorf("localaudio: connect to PulseAudio: %w", err)
	}
	return &Transport{
		client: client,
		in:     newInput(client, cfg.Params),
		out:    newOutput(client, cfg.Params),
	}, nil
}

// Input returns the input processor.
func (t *Transport) Input() processor.Processor { return t.in }

// Output returns the output processor.
func (t *Transport) Output() processor.Processor { return t.out }

// Close closes the PulseAudio connection. Call it after the pipeline stops.
func (t *Transport) Close() { t.client.Close() }

//
// Input: microphone capture
//

type inputTransport struct {
	*transport.BaseInput
	client *pulse.Client

	mu     sync.Mutex
	stream *pulse.RecordStream
	ctx    context.Context
}

func newInput(client *pulse.Client, params transport.Params) *inputTransport {
	in := &inputTransport{client: client}
	in.BaseInput = transport.NewBaseInput("LocalAudioInput", params, in)
	return in
}

// StartReading opens the capture stream at the input sample rate and starts it.
func (in *inputTransport) StartReading(ctx context.Context) error {
	rate := in.SampleRate()
	if rate == 0 {
		rate = defaultInRate
	}
	in.mu.Lock()
	in.ctx = ctx
	in.mu.Unlock()

	stream, err := in.client.NewRecord(
		pulse.Int16Writer(in.onSamples),
		pulse.RecordSampleRate(rate),
		pulse.RecordChannels(monoMap),
	)
	if err != nil {
		return fmt.Errorf("localaudio: open capture stream: %w", err)
	}
	in.mu.Lock()
	in.stream = stream
	in.mu.Unlock()
	stream.Start()
	return nil
}

// onSamples receives captured mono S16 samples from PulseAudio and pushes them
// downstream as an input audio frame.
func (in *inputTransport) onSamples(samples []int16) (int, error) {
	in.mu.Lock()
	ctx := in.ctx
	in.mu.Unlock()
	if ctx == nil || len(samples) == 0 {
		return len(samples), nil
	}
	rate := in.SampleRate()
	if rate == 0 {
		rate = defaultInRate
	}
	pcm := int16ToPCM(samples)
	in.PushAudioFrame(ctx, frames.NewInputAudioRawFrame(pcm, rate, 1))
	return len(samples), nil
}

// StopReading stops and closes the capture stream.
func (in *inputTransport) StopReading(context.Context) error {
	in.mu.Lock()
	stream := in.stream
	in.stream = nil
	in.mu.Unlock()
	if stream != nil {
		stream.Stop()
		stream.Close()
	}
	return nil
}

//
// Output: speaker playback
//

type outputTransport struct {
	*transport.BaseOutput
	client *pulse.Client

	mu     sync.Mutex
	stream *pulse.PlaybackStream
	buf    []byte
}

func newOutput(client *pulse.Client, params transport.Params) *outputTransport {
	out := &outputTransport{client: client}
	out.BaseOutput = transport.NewBaseOutput("LocalAudioOutput", params, out)
	return out
}

// ProcessFrame drives the playback stream's lifecycle around the base output:
// open it on start, drop buffered audio on a barge-in, and close it on end.
func (out *outputTransport) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := out.BaseOutput.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	if dir != processor.Downstream {
		return nil
	}
	switch f.(type) {
	case *frames.StartFrame:
		return out.openStream()
	case *frames.InterruptionFrame:
		out.mu.Lock()
		out.buf = nil
		out.mu.Unlock()
	case *frames.EndFrame, *frames.CancelFrame:
		out.closeStream()
	}
	return nil
}

func (out *outputTransport) openStream() error {
	rate := out.SampleRate()
	if rate == 0 {
		rate = defaultOutRate
	}
	stream, err := out.client.NewPlayback(
		pulse.Int16Reader(out.fill),
		pulse.PlaybackSampleRate(rate),
		pulse.PlaybackChannels(monoMap),
	)
	if err != nil {
		return fmt.Errorf("localaudio: open playback stream: %w", err)
	}
	out.mu.Lock()
	out.stream = stream
	out.mu.Unlock()
	stream.Start()
	return nil
}

// WriteAudio queues one chunk of S16 PCM for playback. PulseAudio pulls it back
// out through fill at the device's pace.
func (out *outputTransport) WriteAudio(_ context.Context, pcm []byte) error {
	out.mu.Lock()
	out.buf = append(out.buf, pcm...)
	if limit := maxBufferBytes(out.SampleRate()); len(out.buf) > limit {
		out.buf = out.buf[len(out.buf)-limit:]
	}
	out.mu.Unlock()
	return nil
}

// SendMessage is a no-op: the local transport has no client data channel.
func (out *outputTransport) SendMessage(context.Context, []byte) error { return nil }

// fill supplies the next samples to PulseAudio, draining the queued PCM and
// padding any shortfall with silence so the stream keeps flowing.
func (out *outputTransport) fill(samples []int16) (int, error) {
	out.mu.Lock()
	n := 0
	for n < len(samples) && len(out.buf) >= 2 {
		samples[n] = int16(binary.LittleEndian.Uint16(out.buf))
		out.buf = out.buf[2:]
		n++
	}
	out.mu.Unlock()
	for i := n; i < len(samples); i++ {
		samples[i] = 0
	}
	return len(samples), nil
}

func (out *outputTransport) closeStream() {
	out.mu.Lock()
	stream := out.stream
	out.stream = nil
	out.buf = nil
	out.mu.Unlock()
	if stream != nil {
		stream.Stop()
		stream.Close()
	}
}

// int16ToPCM packs native S16 samples into S16LE bytes, jargo's PCM layout.
func int16ToPCM(samples []int16) []byte {
	pcm := make([]byte, len(samples)*2)
	for i, s := range samples {
		binary.LittleEndian.PutUint16(pcm[2*i:], uint16(s))
	}
	return pcm
}

func maxBufferBytes(rate int) int {
	if rate == 0 {
		rate = defaultOutRate
	}
	return rate * 2 * maxBufferSeconds
}
