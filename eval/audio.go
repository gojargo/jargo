package eval

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"os"
)

// Audio files played as a user's turn.
//
// A turn's audio: names a recording to play instead of synthesizing its text.
// WAV files are read: integer PCM of 8, 16, 24 or 32 bits and IEEE float of 32
// or 64 bits, any number of channels. Other formats (MP3, FLAC, OGG) would need
// a decoder the cgo-free build does not carry, so they are refused with an error
// naming the file.

// The ways a user audio file fails to read.
var (
	// errUserAudio is a user audio file that cannot be read.
	errUserAudio = errors.New("eval: could not read user audio")
	// errNotWAV is a file in another format than WAV.
	errNotWAV = errors.New("not a WAV file (only WAV is supported)")
	// errMalformedWAV is a WAV file whose chunks do not hold together.
	errMalformedWAV = errors.New("malformed WAV file")
	// errWAVEncoding is a WAV file in an encoding this reader does not decode.
	errWAVEncoding = errors.New("unsupported WAV encoding")
)

// WAV format tags.
const (
	wavFormatPCM        = 1
	wavFormatFloat      = 3
	wavFormatExtensible = 0xFFFE
)

// wavDataChunk is the ID of the chunk holding a WAV file's samples.
const wavDataChunk = "data"

// LoadUserAudio reads an audio file as the PCM a user turn is sent as: 16-bit
// little-endian mono PCM, and its sample rate.
//
// The file keeps its own sample rate; the bot resamples on its way in. A file
// of several channels is downmixed to their mean. It fails with an error naming
// the path when the file cannot be read as audio, or holds none.
func LoadUserAudio(path string) (pcm []byte, sampleRate int, err error) {
	data, err := os.ReadFile(path) //nolint:gosec // the turn's audio, as the scenario names it
	if err != nil {
		return nil, 0, fmt.Errorf("%w %q: %w", errUserAudio, path, err)
	}
	samples, channels, sampleRate, err := decodeWAV(data)
	if err != nil {
		return nil, 0, fmt.Errorf("%w %q: %w", errUserAudio, path, err)
	}
	if len(samples) < channels || channels == 0 {
		return nil, 0, fmt.Errorf("%w %q: the file is empty", errUserAudio, path)
	}
	frameCount := len(samples) / channels
	out := make([]byte, frameCount*2)
	for i := range frameCount {
		var v int16
		if channels == 1 {
			v = samples[i]
		} else {
			// A recording of one speaker carries the same voice on every
			// channel, so the mean is the mono take on it. Summed wide to
			// avoid int16 overflow, and truncated toward zero.
			var sum float64
			for c := range channels {
				sum += float64(samples[i*channels+c])
			}
			v = int16(sum / float64(channels))
		}
		binary.LittleEndian.PutUint16(out[i*2:], uint16(v))
	}
	return out, sampleRate, nil
}

// decodeWAV reads a WAV file's interleaved samples as int16, with its channel
// count and sample rate.
func decodeWAV(data []byte) (samples []int16, channels, sampleRate int, err error) {
	if len(data) < 12 || string(data[:4]) != "RIFF" || string(data[8:12]) != "WAVE" {
		return nil, 0, 0, errNotWAV
	}
	var (
		format, bits int
		formatSeen   bool
	)
	for off := 12; off+8 <= len(data); {
		id := string(data[off : off+4])
		size := int(binary.LittleEndian.Uint32(data[off+4 : off+8]))
		body := off + 8
		if size < 0 || body+size > len(data) {
			size = len(data) - body
		}
		switch id {
		case "fmt ":
			if size < 16 {
				return nil, 0, 0, fmt.Errorf("%w: short fmt chunk", errMalformedWAV)
			}
			format = int(binary.LittleEndian.Uint16(data[body : body+2]))
			channels = int(binary.LittleEndian.Uint16(data[body+2 : body+4]))
			sampleRate = int(binary.LittleEndian.Uint32(data[body+4 : body+8]))
			bits = int(binary.LittleEndian.Uint16(data[body+14 : body+16]))
			if format == wavFormatExtensible && size >= 26 {
				// The subformat GUID starts with the format tag it stands for.
				format = int(binary.LittleEndian.Uint16(data[body+24 : body+26]))
			}
			formatSeen = true
		case wavDataChunk:
			if !formatSeen {
				return nil, 0, 0, fmt.Errorf("%w: data before fmt", errMalformedWAV)
			}
			samples, err := wavSamples(data[body:body+size], format, bits)
			return samples, channels, sampleRate, err
		}
		off = body + size + size%2
	}
	return nil, 0, 0, fmt.Errorf("%w: no data chunk", errMalformedWAV)
}

// wavSamples converts a WAV data chunk to int16 samples.
func wavSamples(data []byte, format, bits int) ([]int16, error) {
	decode := sampleDecoder(format, bits)
	if decode == nil {
		return nil, fmt.Errorf("%w (format %d, %d-bit)", errWAVEncoding, format, bits)
	}
	width := bits / 8
	out := make([]int16, len(data)/width)
	for i := range out {
		out[i] = decode(data[i*width : (i+1)*width])
	}
	return out, nil
}

// sampleDecoder is how one sample of a WAV encoding reads as int16, or nil for
// an encoding this reader does not decode.
func sampleDecoder(format, bits int) func(b []byte) int16 {
	switch {
	case format == wavFormatFloat && bits == 32:
		return func(b []byte) int16 {
			return floatSample(float64(math.Float32frombits(binary.LittleEndian.Uint32(b))))
		}
	case format == wavFormatFloat && bits == 64:
		return func(b []byte) int16 { return floatSample(math.Float64frombits(binary.LittleEndian.Uint64(b))) }
	case format != wavFormatPCM:
		return nil
	case bits == 8:
		// 8-bit PCM is unsigned.
		return func(b []byte) int16 { return int16(int(b[0])-128) << 8 }
	case bits == 16:
		return func(b []byte) int16 { return int16(binary.LittleEndian.Uint16(b)) }
	case bits == 24:
		return func(b []byte) int16 { return int16(uint16(b[1]) | uint16(b[2])<<8) }
	case bits == 32:
		return func(b []byte) int16 { return int16(binary.LittleEndian.Uint32(b) >> 16) }
	}
	return nil
}

// floatSample scales a float sample in [-1, 1] to int16, clipping.
func floatSample(v float64) int16 {
	return int16(math.Round(max(-1, min(1, v)) * math.MaxInt16))
}
