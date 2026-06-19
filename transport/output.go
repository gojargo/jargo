package transport

import (
	"context"
	"encoding/json"
	"log/slog"
	"sync"

	"github.com/gojargo/jargo/audio/resample"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
)

// BaseOutput is the tail of a pipeline: it buffers OutputAudioRawFrames, slices
// them into fixed-size chunks, and hands each chunk to the concrete transport
// to send. Chunking into small, uniform pieces keeps output latency low and
// makes interruptions responsive. A concrete transport embeds it and implements
// OutputDriver to send the audio.
type BaseOutput struct {
	*processor.Base
	params Params
	self   OutputDriver

	sampleRate int
	channels   int
	chunkSize  int

	resampler  *resample.Resampler
	resampleIn int

	bufMu  sync.Mutex
	buffer []byte

	audioOut    chan *frames.OutputAudioRawFrame
	audioCtx    context.Context
	audioCancel context.CancelFunc
	audioWG     sync.WaitGroup
}

// NewBaseOutput builds a BaseOutput. self is the embedding transport, used to
// dispatch WriteAudio and to process frames.
func NewBaseOutput(name string, params Params, self OutputDriver) *BaseOutput {
	bo := &BaseOutput{params: params, self: self}
	bo.Base = processor.New(name, self)
	return bo
}

// SampleRate is the output sample rate in Hz, set when the transport starts.
func (bo *BaseOutput) SampleRate() int { return bo.sampleRate }

// Params returns the transport parameters.
func (bo *BaseOutput) Params() Params { return bo.params }

// WriteAudio is the default no-op; a concrete transport overrides it.
func (bo *BaseOutput) WriteAudio(context.Context, []byte) error { return nil }

// SendMessage is the default no-op; a concrete transport overrides it.
func (bo *BaseOutput) SendMessage(context.Context, []byte) error { return nil }

// ProcessFrame handles the transport lifecycle and routes audio.
func (bo *BaseOutput) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := bo.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	switch fr := f.(type) {
	case *frames.StartFrame:
		// Initialize before forwarding so the chunk size is set before any
		// audio frame can be processed. Nothing downstream of the output
		// transport needs the StartFrame ahead of this.
		bo.startStreaming(ctx, fr)
		return bo.PushFrame(ctx, f, dir)
	case *frames.EndFrame:
		bo.stopStreaming(ctx)
		return bo.PushFrame(ctx, f, dir)
	case *frames.CancelFrame:
		bo.stopStreaming(ctx)
		return bo.PushFrame(ctx, f, dir)
	case *frames.InterruptionFrame:
		if err := bo.PushFrame(ctx, f, dir); err != nil {
			return err
		}
		bo.handleInterruption()
		return nil
	case *frames.OutputTransportMessageFrame:
		if err := bo.sendMessage(ctx, fr); err != nil {
			return err
		}
		return bo.PushFrame(ctx, f, dir)
	case *frames.TTSAudioRawFrame:
		if dir == processor.Downstream {
			bo.handleAudioFrame(fr.Audio, fr.SampleRate, fr.NumChannels)
			return nil
		}
		return bo.PushFrame(ctx, f, dir)
	case *frames.OutputAudioRawFrame:
		if dir == processor.Downstream {
			bo.handleAudioFrame(fr.Audio, fr.SampleRate, fr.NumChannels)
			return nil
		}
		return bo.PushFrame(ctx, f, dir)
	default:
		return bo.PushFrame(ctx, f, dir)
	}
}

// Cleanup stops the audio goroutine and the processor.
func (bo *BaseOutput) Cleanup(ctx context.Context) error {
	bo.stopStreaming(ctx)
	return bo.Base.Cleanup(ctx)
}

func (bo *BaseOutput) startStreaming(ctx context.Context, f *frames.StartFrame) {
	bo.sampleRate = pick(bo.params.AudioOutSampleRate, f.AudioOutSampleRate)
	bo.channels = bo.params.AudioOutChannels
	if bo.channels == 0 {
		bo.channels = 1
	}
	chunks := bo.params.AudioOut10msChunks
	if chunks == 0 {
		chunks = 2
	}
	bytesPer10ms := bo.sampleRate / 100 * bo.channels * 2
	bo.chunkSize = bytesPer10ms * chunks

	bo.bufMu.Lock()
	bo.buffer = nil
	bo.bufMu.Unlock()

	bo.audioCtx, bo.audioCancel = context.WithCancel(ctx)
	bo.audioOut = make(chan *frames.OutputAudioRawFrame, audioFrameChanCap)
	bo.audioWG.Add(1)
	go bo.audioLoop(bo.audioCtx)
}

func (bo *BaseOutput) stopStreaming(context.Context) {
	cancel := bo.audioCancel
	bo.audioCancel = nil
	if cancel != nil {
		cancel()
		bo.audioWG.Wait()
	}
}

// sendMessage serializes a transport message to JSON and hands it to the
// concrete transport.
func (bo *BaseOutput) sendMessage(ctx context.Context, f *frames.OutputTransportMessageFrame) error {
	data, err := json.Marshal(f.Message)
	if err != nil {
		return err
	}
	return bo.self.SendMessage(ctx, data)
}

// handleAudioFrame resamples incoming audio to the output rate, buffers it, and
// emits fixed-size chunks.
func (bo *BaseOutput) handleAudioFrame(audio []byte, sampleRate, channels int) {
	if !bo.params.AudioOutEnabled || bo.chunkSize == 0 {
		return
	}
	audio = bo.resample(audio, sampleRate, channels)

	bo.bufMu.Lock()
	bo.buffer = append(bo.buffer, audio...)
	var chunks [][]byte
	for len(bo.buffer) >= bo.chunkSize {
		chunk := make([]byte, bo.chunkSize)
		copy(chunk, bo.buffer[:bo.chunkSize])
		chunks = append(chunks, chunk)
		bo.buffer = bo.buffer[bo.chunkSize:]
	}
	ctx := bo.audioCtx
	bo.bufMu.Unlock()

	for _, chunk := range chunks {
		out := frames.NewOutputAudioRawFrame(chunk, bo.sampleRate, bo.channels)
		sendAudio(ctx, bo.audioOut, out)
	}
}

// resample converts audio at sampleRate to the transport output rate. The
// resampler is created lazily and reused across frames; it is only touched on
// the process goroutine, so it needs no lock.
func (bo *BaseOutput) resample(audio []byte, sampleRate, channels int) []byte {
	if sampleRate == bo.sampleRate {
		return audio
	}
	if bo.resampler == nil || bo.resampleIn != sampleRate {
		bo.resampler = resample.New(sampleRate, bo.sampleRate, channels)
		bo.resampleIn = sampleRate
	}
	return bo.resampler.Process(audio)
}

// handleInterruption drops buffered output audio so the bot stops speaking
// promptly on a barge-in.
func (bo *BaseOutput) handleInterruption() {
	bo.bufMu.Lock()
	bo.buffer = nil
	bo.bufMu.Unlock()
	for {
		select {
		case <-bo.audioOut:
		default:
			return
		}
	}
}

// audioLoop sends buffered chunks over the transport and forwards them
// downstream.
func (bo *BaseOutput) audioLoop(ctx context.Context) {
	defer bo.audioWG.Done()
	for {
		select {
		case <-ctx.Done():
			return
		case chunk := <-bo.audioOut:
			if err := bo.self.WriteAudio(ctx, chunk.Audio); err != nil {
				slog.Error("write audio to transport", "processor", bo.Name(), "err", err)
				continue
			}
			_ = bo.PushFrame(ctx, chunk, processor.Downstream)
		}
	}
}
