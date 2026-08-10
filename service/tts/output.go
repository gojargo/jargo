package tts

import (
	"context"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
)

// defaultSilenceAfterStop is how much silence pads an utterance when padding is
// asked for without a length.
const defaultSilenceAfterStop = 2 * time.Second

// SilenceOptions configures the silence a service pads the end of an utterance
// with.
type SilenceOptions struct {
	// Enabled pads each utterance.
	Enabled bool
	// Duration is how much silence to send; zero uses two seconds.
	Duration time.Duration
}

// SetSilenceAfterStop pads the end of every utterance with silence.
//
// It is for a transport that clips the tail of what it is given, an outbound
// telephony leg that stops sending the moment audio runs out, say. The padding
// keeps the last word from being cut off. It goes out ahead of the frame that
// says the utterance stopped, so it is part of the utterance rather than
// something following it.
func (b *Base) SetSilenceAfterStop(o SilenceOptions) {
	if o.Duration <= 0 {
		o.Duration = defaultSilenceAfterStop
	}
	b.outMu.Lock()
	b.silence = o
	b.outMu.Unlock()
}

// SetDestination names the transport stream this service's audio is meant for,
// so a transport carrying several can tell them apart. Empty, the default,
// leaves the audio unaddressed and the transport sends it wherever it sends
// everything.
func (b *Base) SetDestination(dest string) {
	b.outMu.Lock()
	b.destination = dest
	b.outMu.Unlock()
}

// silenceOptions reads the padding settings.
func (b *Base) silenceOptions() SilenceOptions {
	b.outMu.Lock()
	defer b.outMu.Unlock()
	return b.silence
}

// destinationName reads the transport destination.
func (b *Base) destinationName() string {
	b.outMu.Lock()
	defer b.outMu.Unlock()
	return b.destination
}

// pushTrailingSilence sends the padding that follows an utterance, when the
// service was asked for any.
func (b *Base) pushTrailingSilence(ctx context.Context) error {
	o := b.silenceOptions()
	if !o.Enabled {
		return nil
	}
	rate := b.syn.SampleRate()
	if rate <= 0 {
		return nil
	}
	// 16-bit samples, one channel.
	n := int(o.Duration.Seconds()*float64(rate)) * 2
	if n <= 0 {
		return nil
	}
	return b.PushFrame(ctx, frames.NewTTSAudioRawFrame(make([]byte, n), rate, 1), processor.Downstream)
}

// stampDestination addresses the frames a service produces to the transport
// stream it speaks on, so a transport carrying several knows which this is.
func (b *Base) stampDestination(f frames.Frame) {
	dest := b.destinationName()
	if dest == "" {
		return
	}
	switch f.(type) {
	case *frames.TTSStartedFrame, *frames.TTSStoppedFrame,
		*frames.TTSAudioRawFrame, *frames.TTSTextFrame:
		f.Base().SetTransportDestination(dest)
	}
}
