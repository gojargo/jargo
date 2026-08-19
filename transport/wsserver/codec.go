package wsserver

import (
	"log/slog"
	"time"

	"github.com/gojargo/jargo/audio"
	"github.com/gojargo/jargo/audio/resample"
	"github.com/gojargo/jargo/frames"
)

// DefaultWireSampleRate is the rate telephony providers stream at: 8 kHz mono,
// companded. It is the rate on the wire, not the rate the pipeline has to run
// at.
const DefaultWireSampleRate = 8000

// Encoding is the companding a provider uses on the wire.
type Encoding int

const (
	// EncodingULaw is G.711 μ-law, the North American and Japanese variant, and
	// what every provider here uses by default.
	EncodingULaw Encoding = iota
	// EncodingALaw is G.711 A-law, used elsewhere.
	EncodingALaw
	// EncodingLinear is uncompanded 16-bit signed little-endian PCM, which a
	// provider streaming raw samples uses. Only the rate is converted.
	EncodingLinear
)

// AudioConfig configures the conversion between a provider's wire audio and the
// PCM a pipeline carries. Its zero value is the right one for a telephony
// provider: 8 kHz on the wire, the pipeline's own rate on the inside.
type AudioConfig struct {
	// WireSampleRate is the rate the provider streams at; 0 uses 8 kHz.
	WireSampleRate int
	// SampleRate overrides the pipeline rate the audio is converted to and from;
	// 0 uses the rate the StartFrame carries, which is what a pipeline should
	// normally decide.
	SampleRate int
	// ResamplerClearAfter is how long the resamplers may sit idle before they
	// treat the next chunk as a fresh signal rather than the continuation of the
	// last one. 0 uses resample.DefaultClearAfter; a negative value never
	// clears, which is what a provider whose chunks arrive at irregular
	// intervals wants, since those gaps are gaps in delivery rather than gaps in
	// the audio.
	ResamplerClearAfter time.Duration
}

// Codec converts between a provider's companded wire audio and the PCM a
// pipeline carries, resampling between the wire rate and the pipeline rate.
//
// It exists because the wire rate and the pipeline rate are not the same
// question. Telephony is 8 kHz on the wire and always will be, but a pipeline
// that runs at 8 kHz throughout hands 8 kHz to its transcriber and asks its
// voice for 8 kHz back, which is audibly worse than converting once at each
// edge. A Codec is that edge.
//
// The zero value is not usable: call Setup with the pipeline's StartFrame
// first, which is what tells it the rate to convert to.
type Codec struct {
	cfg AudioConfig

	wireRate int
	inRate   int

	in *resample.Resampler
	// out is built on the first frame sent, once the rate that audio actually
	// arrives at is known, and rebuilt if it ever changes.
	out     *resample.Resampler
	outRate int
}

// NewCodec builds a Codec. Setup must be called before it converts anything.
func NewCodec(cfg AudioConfig) *Codec {
	return &Codec{cfg: cfg}
}

// Setup takes the rates from the pipeline's StartFrame and builds the inbound
// resampler. A serializer calls it from its own Setup.
func (c *Codec) Setup(f *frames.StartFrame) error {
	c.wireRate = c.cfg.WireSampleRate
	if c.wireRate <= 0 {
		c.wireRate = DefaultWireSampleRate
	}

	c.inRate = c.cfg.SampleRate
	if c.inRate <= 0 && f != nil {
		c.inRate = f.AudioInSampleRate
	}
	if c.inRate <= 0 {
		c.inRate = c.wireRate
	}

	c.closeIn()
	r, err := c.newResampler(c.wireRate, c.inRate)
	if err != nil {
		return err
	}
	c.in = r
	return nil
}

// newResampler builds one of the two stream resamplers.
func (c *Codec) newResampler(from, to int) (*resample.Resampler, error) {
	return resample.NewWithConfig(from, to, 1, resample.Config{
		ClearAfter: c.cfg.ResamplerClearAfter,
	})
}

// WireSampleRate is the rate the provider streams at.
func (c *Codec) WireSampleRate() int { return c.wireRate }

// SampleRate is the pipeline rate incoming audio is converted to, and the rate
// the frames the Codec produces are tagged with.
func (c *Codec) SampleRate() int { return c.inRate }

// Decode converts one chunk of companded wire audio to PCM at the pipeline
// rate. It returns nil when the conversion has nothing to emit yet: a resampler
// holds a filter length of audio back for the audio it expects to follow, so
// the first chunk of a stream can be swallowed whole. The caller sends no frame
// for it, exactly as it would for a message carrying no audio.
func (c *Codec) Decode(companded []byte, enc Encoding) []byte {
	switch enc {
	case EncodingALaw:
		return audio.ALawToPCM(companded, c.in)
	case EncodingLinear:
		return audio.ResamplePCM(companded, c.in)
	default:
		return audio.ULawToPCM(companded, c.in)
	}
}

// Encode converts one chunk of PCM at frameRate to companded wire audio. It
// returns nil when the conversion has nothing to emit yet; see Decode.
func (c *Codec) Encode(pcm []byte, frameRate int, enc Encoding) []byte {
	r := c.outputResampler(frameRate)
	switch enc {
	case EncodingALaw:
		return audio.PCMToALaw(pcm, r)
	case EncodingLinear:
		return audio.ResamplePCM(pcm, r)
	default:
		return audio.PCMToULaw(pcm, r)
	}
}

// outputResampler returns the resampler converting frameRate to the wire rate,
// building it the first time and again if the rate changes. The rate is taken
// from the audio rather than from the StartFrame because that is what the
// samples were actually produced at, and a pipeline may send audio a service
// generated at its own rate.
func (c *Codec) outputResampler(frameRate int) *resample.Resampler {
	if frameRate <= 0 {
		frameRate = c.inRate
	}
	if c.out != nil && c.outRate == frameRate {
		return c.out
	}
	c.closeOut()

	r, err := c.newResampler(frameRate, c.wireRate)
	if err != nil {
		slog.Error("wsserver: create outbound resampler",
			"from", frameRate, "to", c.wireRate, "err", err)
		return nil
	}
	c.out, c.outRate = r, frameRate
	return c.out
}

// Close releases the resamplers. A serializer that holds a Codec closes it when
// the session ends.
func (c *Codec) Close() {
	c.closeIn()
	c.closeOut()
}

func (c *Codec) closeIn() {
	if c.in != nil {
		c.in.Close()
		c.in = nil
	}
}

func (c *Codec) closeOut() {
	if c.out != nil {
		c.out.Close()
		c.out = nil
		c.outRate = 0
	}
}
