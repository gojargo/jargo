//go:build libopus

package opus

/*
#cgo pkg-config: opus
#include <opus.h>
#include <stdlib.h>

// opus_encoder_ctl is variadic, which cgo cannot call directly, so wrap the
// control requests we use in plain functions.
static int jargo_opus_set_bitrate(OpusEncoder *enc, opus_int32 bitrate) {
	return opus_encoder_ctl(enc, OPUS_SET_BITRATE(bitrate));
}
static int jargo_opus_set_signal_voice(OpusEncoder *enc) {
	return opus_encoder_ctl(enc, OPUS_SET_SIGNAL(OPUS_SIGNAL_VOICE));
}
static int jargo_opus_set_inband_fec(OpusEncoder *enc, int on) {
	return opus_encoder_ctl(enc, OPUS_SET_INBAND_FEC(on));
}
static int jargo_opus_set_packet_loss_perc(OpusEncoder *enc, int perc) {
	return opus_encoder_ctl(enc, OPUS_SET_PACKET_LOSS_PERC(perc));
}
*/
import "C"

import (
	"errors"
	"fmt"
	"runtime"
	"unsafe"
)

// Encoder encodes 48 kHz S16LE PCM into Opus packets using libopus (cgo). Built
// with `-tags libopus`. libopus uses SILK/hybrid for speech (OPUS_APPLICATION_VOIP
// + a voice signal hint), which sounds far more natural than the pure-Go
// CELT-only encoder. Same API as the default build (see encoder_pion.go).
type Encoder struct {
	enc      *C.OpusEncoder
	channels int
	out      []byte
}

var errFrameSize = errors.New("opus: wrong frame size")

// NewEncoder builds an Encoder for 48 kHz audio from cfg.
func NewEncoder(cfg EncoderConfig) (*Encoder, error) {
	var cerr C.int
	enc := C.opus_encoder_create(C.opus_int32(SampleRate), C.int(cfg.Channels), C.OPUS_APPLICATION_VOIP, &cerr)
	if cerr != C.OPUS_OK || enc == nil {
		return nil, fmt.Errorf("opus_encoder_create failed: %d", int(cerr))
	}
	if cfg.Bitrate > 0 {
		if rc := C.jargo_opus_set_bitrate(enc, C.opus_int32(cfg.Bitrate)); rc != C.OPUS_OK {
			C.opus_encoder_destroy(enc)
			return nil, fmt.Errorf("opus set bitrate failed: %d", int(rc))
		}
	}
	// Bias toward SILK for speech; harmless if the content is not voice.
	C.jargo_opus_set_signal_voice(enc)

	if cfg.InbandFEC {
		if rc := C.jargo_opus_set_inband_fec(enc, 1); rc != C.OPUS_OK {
			C.opus_encoder_destroy(enc)
			return nil, fmt.Errorf("opus set inband fec failed: %d", int(rc))
		}
		// Without a loss estimate libopus assumes 0% and emits no redundancy,
		// so FEC would be on in name only.
		perc := min(max(cfg.ExpectedPacketLoss, 0), 100)
		if rc := C.jargo_opus_set_packet_loss_perc(enc, C.int(perc)); rc != C.OPUS_OK {
			C.opus_encoder_destroy(enc)
			return nil, fmt.Errorf("opus set packet loss perc failed: %d", int(rc))
		}
	}

	e := &Encoder{enc: enc, channels: cfg.Channels, out: make([]byte, maxPacketBytes)}
	// libopus encoders must be freed; tie the C lifetime to the Go object.
	runtime.SetFinalizer(e, (*Encoder).free)
	return e, nil
}

func (e *Encoder) free() {
	if e.enc != nil {
		C.opus_encoder_destroy(e.enc)
		e.enc = nil
	}
}

// Encode encodes exactly one 20 ms frame of interleaved S16LE PCM — that is
// FrameBytes(channels) bytes — into a single Opus packet.
func (e *Encoder) Encode(pcm []byte) ([]byte, error) {
	// opus_encode reads a fixed FrameSamples*channels samples from the pointer
	// regardless of the slice length, so a short frame would read out of bounds.
	if len(pcm) != FrameBytes(e.channels) {
		return nil, fmt.Errorf("%w: got %d bytes, want %d", errFrameSize, len(pcm), FrameBytes(e.channels))
	}
	// S16LE interleaved bytes are native int16 on little-endian targets
	// (amd64/arm64), so the buffer can be read as opus_int16 without a copy.
	// frame_size is samples per channel.
	n := C.opus_encode(
		e.enc,
		(*C.opus_int16)(unsafe.Pointer(&pcm[0])),
		C.int(FrameSamples),
		(*C.uchar)(unsafe.Pointer(&e.out[0])),
		C.opus_int32(len(e.out)),
	)
	runtime.KeepAlive(e)
	if n < 0 {
		return nil, fmt.Errorf("opus_encode failed: %d", int(n))
	}
	packet := make([]byte, int(n))
	copy(packet, e.out[:int(n)])
	return packet, nil
}
