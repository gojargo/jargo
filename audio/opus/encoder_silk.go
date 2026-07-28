//go:build !libopus

package opus

import (
	"encoding/binary"
	"errors"
	"fmt"

	"github.com/gojargo/jargo/audio/resample"
	pion "github.com/pion/opus"
)

var errFrameSize = errors.New("opus: wrong frame size")

// silkInternalRate is SILK's wideband internal sample rate. Encode resamples the
// 48 kHz stream down to it, encodes, and the browser's Opus decoder plays the
// SILK-only packets back at 48 kHz.
const silkInternalRate = 16000

// silkFrame is the number of 16 kHz samples in one 20 ms frame.
const silkFrame = silkInternalRate / 1000 * 20 // 320

// Encoder encodes 48 kHz S16LE PCM into Opus packets using the pure-Go pion SILK
// encoder. This is the default build — SILK is tuned for speech and needs no
// cgo. The 48 kHz input is downmixed to mono and resampled to 16 kHz with the
// pure-Go resampler, then encoded as wideband SILK. Build with `-tags libopus`
// for the C library instead; both expose the same NewEncoder/Encode API.
type Encoder struct {
	enc      *pion.Encoder
	down     *resample.Resampler
	channels int
	buf      []int16 // 16 kHz mono samples awaiting a full frame
	out      []byte
}

// NewEncoder builds a SILK Encoder for 48 kHz audio from cfg.
// EncoderConfig.InbandFEC is ignored: this encoder emits no FEC redundancy.
func NewEncoder(cfg EncoderConfig) (*Encoder, error) {
	bitrate := cfg.Bitrate
	if bitrate <= 0 {
		bitrate = 24000
	}
	enc, err := pion.NewEncoder(pion.WithBitrate(bitrate))
	if err != nil {
		return nil, fmt.Errorf("new silk encoder: %w", err)
	}
	down, err := resample.New(SampleRate, silkInternalRate, 1)
	if err != nil {
		return nil, fmt.Errorf("new silk resampler: %w", err)
	}

	return &Encoder{enc: enc, down: down, channels: cfg.Channels, out: make([]byte, maxPacketBytes)}, nil
}

// Encode encodes exactly one 20 ms frame of interleaved S16LE PCM — that is
// FrameBytes(channels) bytes — into a single SILK-only Opus packet.
func (e *Encoder) Encode(pcm []byte) ([]byte, error) {
	if len(pcm) != FrameBytes(e.channels) {
		return nil, fmt.Errorf("%w: got %d bytes, want %d", errFrameSize, len(pcm), FrameBytes(e.channels))
	}

	// Downmix to mono, resample 48 kHz -> 16 kHz (stateful), and reframe to
	// exactly one 320-sample SILK frame. At the exact 3:1 ratio the resampler
	// yields 320 samples per 20 ms in steady state; only the first call is short,
	// which the zero-padding below covers.
	e.buf = append(e.buf, bytesToI16(e.down.Process(monoBytes(pcm, e.channels)))...)

	frame := make([]int16, silkFrame)
	switch {
	case len(e.buf) >= 2*silkFrame:
		// Safety valve against latency drift: drop the oldest frame to catch up.
		e.buf = e.buf[len(e.buf)-silkFrame:]
		copy(frame, e.buf)
		e.buf = e.buf[:0]
	case len(e.buf) >= silkFrame:
		copy(frame, e.buf[:silkFrame])
		e.buf = e.buf[silkFrame:]
	default:
		copy(frame, e.buf) // warmup: zero-padded remainder
		e.buf = e.buf[:0]
	}

	n, err := e.enc.EncodeSILK(frame, pion.BandwidthWideband, e.out)
	if err != nil {
		return nil, fmt.Errorf("silk encode: %w", err)
	}
	packet := make([]byte, n)
	copy(packet, e.out[:n])

	return packet, nil
}

// monoBytes downmixes interleaved S16LE PCM to mono S16LE. Mono input is
// returned unchanged.
func monoBytes(pcm []byte, channels int) []byte {
	if channels <= 1 {
		return pcm
	}
	frames := len(pcm) / (channels * 2)
	out := make([]byte, frames*2)
	for i := range frames {
		var sum int
		for c := range channels {
			sum += int(int16(binary.LittleEndian.Uint16(pcm[(i*channels+c)*2:])))
		}
		binary.LittleEndian.PutUint16(out[i*2:], uint16(int16(sum/channels)))
	}

	return out
}

func bytesToI16(b []byte) []int16 {
	s := make([]int16, len(b)/2)
	for i := range s {
		s[i] = int16(binary.LittleEndian.Uint16(b[i*2:]))
	}

	return s
}
