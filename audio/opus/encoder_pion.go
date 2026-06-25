//go:build !libopus

package opus

import (
	"fmt"

	pion "github.com/pion/opus"
)

// Encoder encodes 48 kHz S16LE PCM into Opus packets using the pure-Go pion
// encoder. This is the default build. The encoder is currently CELT-only, which
// is weaker on speech than libopus; build with `-tags libopus` for the C
// library instead (see encoder_libopus.go).
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
