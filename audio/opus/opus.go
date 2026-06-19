// Package opus wraps the Opus codec for jargo's audio path: it decodes Opus
// packets to PCM on the way in and encodes PCM to Opus packets on the way out.
//
// The codec runs at 48 kHz — the rate WebRTC negotiates for Opus — so audio
// crosses the transport without resampling. Encoding produces 20 ms frames, the
// standard WebRTC packetization.
package opus

import (
	"encoding/binary"
	"fmt"
	"time"

	pion "github.com/pion/opus"
)

const (
	// SampleRate is the codec sample rate in Hz. Opus runs at 48 kHz.
	SampleRate = 48000
	// FrameDuration is the duration of one encoded packet.
	FrameDuration = 20 * time.Millisecond
	// FrameSamples is the number of samples per channel in one 20 ms frame.
	FrameSamples = SampleRate / 1000 * 20 // 960

	// maxFrameSamples is the largest decoded frame (120 ms) per channel.
	maxFrameSamples = SampleRate / 1000 * 120 // 5760
	// maxPacketBytes bounds an encoded Opus packet.
	maxPacketBytes = 1500
)

// FrameBytes is the size in bytes of one 20 ms S16LE frame for channels.
func FrameBytes(channels int) int { return FrameSamples * channels * 2 }

// Decoder decodes Opus packets into signed 16-bit little-endian PCM.
type Decoder struct {
	dec      pion.Decoder
	channels int
	buf      []int16
}

// NewDecoder builds a Decoder that outputs channels-channel 48 kHz PCM. It
// decodes all Opus modes (SILK, CELT and hybrid).
func NewDecoder(channels int) (*Decoder, error) {
	d, err := pion.NewDecoderWithOutput(SampleRate, channels)
	if err != nil {
		return nil, fmt.Errorf("new opus decoder: %w", err)
	}
	return &Decoder{
		dec:      d,
		channels: channels,
		buf:      make([]int16, maxFrameSamples*channels),
	}, nil
}

// Decode decodes one Opus packet into interleaved S16LE PCM bytes.
func (d *Decoder) Decode(packet []byte) ([]byte, error) {
	n, err := d.dec.DecodeToInt16(packet, d.buf)
	if err != nil {
		return nil, fmt.Errorf("opus decode: %w", err)
	}
	total := n * d.channels
	out := make([]byte, total*2)
	for i := range total {
		binary.LittleEndian.PutUint16(out[i*2:], uint16(d.buf[i]))
	}
	return out, nil
}

// Encoder encodes 48 kHz S16LE PCM into Opus packets.
type Encoder struct {
	enc      *pion.Encoder
	channels int
	out      []byte
}

// NewEncoder builds an Encoder for channels-channel 48 kHz audio at the given
// bitrate in bits per second; pass 0 for the codec default.
func NewEncoder(channels, bitrate int) (*Encoder, error) {
	opts := []pion.EncoderOption{pion.WithChannels(channels)}
	if bitrate > 0 {
		opts = append(opts, pion.WithBitrate(bitrate))
	}
	e, err := pion.NewEncoder(opts...)
	if err != nil {
		return nil, fmt.Errorf("new opus encoder: %w", err)
	}
	return &Encoder{
		enc:      e,
		channels: channels,
		out:      make([]byte, maxPacketBytes),
	}, nil
}

// Encode encodes exactly one 20 ms frame of interleaved S16LE PCM — that is
// FrameBytes(channels) bytes — into a single Opus packet.
func (e *Encoder) Encode(pcm []byte) ([]byte, error) {
	n, err := e.enc.Encode(pcm, e.out)
	if err != nil {
		return nil, fmt.Errorf("opus encode: %w", err)
	}
	packet := make([]byte, n)
	copy(packet, e.out[:n])
	return packet, nil
}
