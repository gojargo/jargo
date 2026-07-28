// Throwaway: exercise jargo's audio/opus Encoder end-to-end, the way the
// transport does — feeding 20 ms frames of 48 kHz mono S16LE PCM and decoding
// the packets back. Build with `-tags silk` to run the pure-Go SILK encoder
// (our pion/opus branch), or without tags for the CELT default. It resamples a
// clip to 48 kHz with the pure-Go resampler, encodes/decodes frame by frame,
// and writes jargo_in.wav / jargo_out.wav plus a bitrate/SNR summary.
//
//	go run -tags silk ./examples/silktest [input.wav]   # needs go.work -> our branch
package main

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"

	"github.com/gojargo/jargo/audio/opus"
	"github.com/gojargo/jargo/audio/resample"
)

const frame48 = opus.SampleRate / 1000 * 20 // 960 samples / 20 ms

func main() {
	src := "/home/fallais/dev/pion-opus/speech.wav"
	if len(os.Args) > 1 {
		src = os.Args[1]
	}
	pcm, rate := readWAV(src)
	fmt.Printf("input: %s (%d Hz, %d samples)\n", src, rate, len(pcm))

	// Pure-Go resample to jargo's 48 kHz stream rate.
	pcm48 := resampleI16(pcm, rate, opus.SampleRate)

	enc, err := opus.NewEncoder(opus.EncoderConfig{Channels: 1, Bitrate: 24000})
	must(err)
	dec, err := opus.NewDecoder(1)
	must(err)

	decoded := make([]int16, 0, len(pcm48))
	frames := len(pcm48) / frame48
	totalBytes := 0
	for f := range frames {
		pkt, err := enc.Encode(i16ToBytes(pcm48[f*frame48 : (f+1)*frame48]))
		must(err)
		totalBytes += len(pkt)
		out, err := dec.Decode(pkt)
		must(err)
		decoded = append(decoded, bytesToI16(out)...)
	}

	writeWAV("jargo_in.wav", pcm48, opus.SampleRate)
	writeWAV("jargo_out.wav", decoded, opus.SampleRate)

	kbps := float64(totalBytes) * 8 / (float64(frames) * 0.02) / 1000
	fmt.Printf("encoded %d frames, ~%.1f kbit/s, SNR %.1f dB @48k\n", frames, kbps, alignedSNR(pcm48, decoded))
	fmt.Println("wrote jargo_in.wav and jargo_out.wav — play both to compare")
}

func resampleI16(in []int16, inRate, outRate int) []int16 {
	if inRate == outRate {
		return in
	}
	r, err := resample.New(inRate, outRate, 1)
	must(err)
	defer r.Close()

	return bytesToI16(r.Process(i16ToBytes(in)))
}

func i16ToBytes(s []int16) []byte {
	b := make([]byte, len(s)*2)
	for i, v := range s {
		binary.LittleEndian.PutUint16(b[i*2:], uint16(v))
	}

	return b
}

func bytesToI16(b []byte) []int16 {
	s := make([]int16, len(b)/2)
	for i := range s {
		s[i] = int16(binary.LittleEndian.Uint16(b[i*2:]))
	}

	return s
}

// alignedSNR returns the energy-weighted SNR in dB of b against a over the best
// alignment lag (the two paths carry different resampler group delays).
func alignedSNR(a, b []int16) float64 {
	best := math.Inf(-1)
	lo, hi := len(a)*20/100, len(a)*80/100
	for lag := range 600 {
		var sig, noise float64
		for i := lo; i < hi && i+lag < len(b); i++ {
			d := float64(a[i]) - float64(b[i+lag])
			sig += float64(a[i]) * float64(a[i])
			noise += d * d
		}
		if noise > 0 {
			if s := 10 * math.Log10(sig/noise); s > best {
				best = s
			}
		}
	}

	return best
}

func readWAV(path string) ([]int16, int) {
	raw, err := os.ReadFile(path) //nolint:gosec // path is a CLI argument
	must(err)
	rate := int(binary.LittleEndian.Uint32(raw[24:28]))
	i := 12
	for i+8 <= len(raw) {
		id := string(raw[i : i+4])
		sz := int(binary.LittleEndian.Uint32(raw[i+4 : i+8]))
		if id == "data" {
			return bytesToI16(raw[i+8 : i+8+sz]), rate
		}
		i += 8 + sz
	}
	panic("no data chunk in " + path)
}

func writeWAV(path string, pcm []int16, rate int) {
	f, err := os.Create(path) //nolint:gosec // path is a CLI argument
	must(err)
	defer func() { _ = f.Close() }()

	dataLen := len(pcm) * 2
	hdr := make([]byte, 44)
	copy(hdr[0:], "RIFF")
	binary.LittleEndian.PutUint32(hdr[4:], uint32(36+dataLen))
	copy(hdr[8:], "WAVEfmt ")
	binary.LittleEndian.PutUint32(hdr[16:], 16)
	binary.LittleEndian.PutUint16(hdr[20:], 1)
	binary.LittleEndian.PutUint16(hdr[22:], 1)
	binary.LittleEndian.PutUint32(hdr[24:], uint32(rate))
	binary.LittleEndian.PutUint32(hdr[28:], uint32(rate*2))
	binary.LittleEndian.PutUint16(hdr[32:], 2)
	binary.LittleEndian.PutUint16(hdr[34:], 16)
	copy(hdr[36:], "data")
	binary.LittleEndian.PutUint32(hdr[40:], uint32(dataLen))
	_, err = f.Write(hdr)
	must(err)
	_, err = f.Write(i16ToBytes(pcm))
	must(err)
}

func must(err error) {
	if err != nil {
		panic(err)
	}
}
