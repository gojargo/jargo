package kokoro

import (
	"context"
	"errors"
	"fmt"
	"iter"
	"log/slog"
	"os"
	"strings"
	"time"

	"github.com/gojargo/jargo/internal/onnxrt"
)

// SampleRate is the rate Kokoro synthesizes at: 24 kHz mono.
const SampleRate = 24000

// The pauses added after a batch that ends a sentence or a clause, in
// seconds. Trimming removes the silence the model puts at a batch end, so the
// pause the punctuation calls for is added back when the batches are joined.
const (
	sentencePause = 0.25
	clausePause   = 0.1
)

// The names of the model's tensors. The v1.0 export reads its token ids as
// "tokens", a style vector as "style" and a float speed as "speed", and writes
// the audio as "audio".
var (
	modelInputs  = []string{"tokens", "style", "speed"} //nolint:gochecknoglobals // read-only names
	modelOutputs = []string{"audio"}                    //nolint:gochecknoglobals // read-only names
)

// Errors synthesis reports.
var (
	errSpeed       = errors.New("kokoro: speed should be between 0.5 and 2.0")
	errVoice       = errors.New("kokoro: voice not found in available voices")
	errNoPhonemes  = errors.New("kokoro: nothing to synthesize")
	errNoTokens    = errors.New("kokoro: no phonemes are in the model vocabulary")
	errNoAudio     = errors.New("kokoro: the model produced no audio")
	errVoicesFile  = errors.New("kokoro: voices file not found")
	errModelFile   = errors.New("kokoro: model file not found")
	errStreamEnded = errors.New("kokoro: stream abandoned")
)

// EspeakConfig locates espeak-ng, which phonemizes the text.
type EspeakConfig struct {
	// LibPath is the libespeak-ng shared library; empty looks for it by its
	// conventional names on the loader's default search path.
	LibPath string
	// DataPath is the espeak-ng data directory, or its parent; empty uses the
	// data the library was built to read.
	DataPath string
}

// session runs the model.
type session interface {
	Run(inputs []onnxrt.Tensor) ([]onnxrt.Tensor, error)
	Close() error
}

// kokoroModel is the model loaded with its voices, as the service drives it.
type kokoroModel interface {
	// hasVoice reports whether the voices file holds the voice named.
	hasVoice(name string) bool
	// createStream synthesizes text a batch at a time.
	createStream(ctx context.Context, text, voice, lang string, speed float64) iter.Seq2[[]float32, error]
	// close releases the model.
	close() error
}

// textTokenizer turns text into phonemes and phonemes into token ids.
type textTokenizer interface {
	phonemize(text, lang string) (string, error)
	tokenize(phonemes string, limit int) ([]int64, error)
}

// kokoro is the Kokoro model loaded into the ONNX Runtime, with its voices and
// the tokenizer that phonemizes text for it.
type kokoro struct {
	sess      session
	voices    voices
	tokenizer textTokenizer
}

// batch is one pass of the model: its phonemes, and the pause after them.
type batch struct {
	phonemes string
	pause    float64
}

// newKokoro loads the model at modelPath and the voices at voicesPath, and
// espeak-ng as espeak locates it.
func newKokoro(modelPath, voicesPath string, espeak EspeakConfig) (*kokoro, error) {
	if _, err := os.Stat(voicesPath); err != nil {
		return nil, fmt.Errorf("%w at %s: %w", errVoicesFile, voicesPath, err)
	}
	if _, err := os.Stat(modelPath); err != nil {
		return nil, fmt.Errorf("%w at %s: %w", errModelFile, modelPath, err)
	}
	lib, err := openEspeak(espeak.LibPath, espeak.DataPath)
	if err != nil {
		return nil, err
	}
	f, err := os.Open(voicesPath) //nolint:gosec // the configured voices file
	if err != nil {
		return nil, fmt.Errorf("kokoro: %w", err)
	}
	defer func() { _ = f.Close() }()
	info, err := f.Stat()
	if err != nil {
		return nil, fmt.Errorf("kokoro: %w", err)
	}
	vs, err := loadVoices(f, info.Size())
	if err != nil {
		return nil, err
	}
	model, err := os.ReadFile(modelPath) //nolint:gosec // the configured model file
	if err != nil {
		return nil, fmt.Errorf("kokoro: %w", err)
	}
	sess, err := onnxrt.New(model, modelInputs, modelOutputs)
	if err != nil {
		return nil, fmt.Errorf("kokoro: loading the model: %w", err)
	}
	return &kokoro{sess: sess, voices: vs, tokenizer: newTokenizer(lib)}, nil
}

// hasVoice reports whether the voices file holds the voice named.
func (k *kokoro) hasVoice(name string) bool {
	_, ok := k.voices.lookup(name)
	return ok
}

// close releases the model.
func (k *kokoro) close() error { return k.sess.Close() }

// infer runs the model on one batch of tokens, padded with a boundary token on
// either side.
func (k *kokoro) infer(tokens []int64, style []float32, speed float64) ([]float32, error) {
	start := time.Now()
	ids := make([]int64, 0, len(tokens)+2)
	ids = append(ids, 0)
	ids = append(ids, tokens...)
	ids = append(ids, 0)
	outs, err := k.sess.Run([]onnxrt.Tensor{
		onnxrt.Int64([]int64{1, int64(len(ids))}, ids),
		onnxrt.Float32([]int64{1, styleDim}, append([]float32(nil), style...)),
		onnxrt.Float32([]int64{1}, []float32{float32(speed)}),
	})
	if err != nil {
		return nil, fmt.Errorf("kokoro: %w", err)
	}
	if len(outs) == 0 {
		return nil, errNoAudio
	}
	audio := outs[0].F32
	audioDuration := float64(len(audio)) / SampleRate
	created := time.Since(start).Seconds()
	slog.Debug(fmt.Sprintf("Created audio in length of %.2fs for %d phonemes in %.2fs (RTF: %.2f)",
		audioDuration, len(tokens), created, created/audioDuration))
	return audio, nil
}

// prepare resolves the voice and splits the text into batches and the pauses
// after them.
func (k *kokoro) prepare(text, voiceName string, speed float64, lang string) (voice, []batch, error) {
	if speed < 0.5 || speed > 2.0 {
		return voice{}, nil, fmt.Errorf("%w, got %v", errSpeed, speed)
	}
	v, ok := k.voices.lookup(voiceName)
	if !ok {
		return voice{}, nil, fmt.Errorf("%w: %s", errVoice, voiceName)
	}
	phonemes, err := k.tokenizer.phonemize(text, lang)
	if err != nil {
		return voice{}, nil, err
	}
	// Newlines are not in the vocabulary, so without this the lines of a text
	// run together with no gap for the model to pause on.
	phonemes = strings.Join(fields(phonemes), " ")
	batched := splitPhonemes(phonemes, maxPhonemeLength)
	if len(batched) == 0 {
		return voice{}, nil, fmt.Errorf("%w, %q produced no phonemes", errNoPhonemes, text)
	}
	slog.Debug(fmt.Sprintf("Creating audio for %d batches for %d phonemes", len(batched), len([]rune(phonemes))))
	batches := make([]batch, len(batched))
	for i, b := range batched {
		batches[i] = batch{phonemes: b}
		if i < len(batched)-1 {
			batches[i].pause = pauseAfter(b, sentencePause, clausePause)
		}
	}
	return v, batches, nil
}

// createBatch synthesizes one batch, trims its leading and trailing silence
// for a more natural join, and pauses after it.
func (k *kokoro) createBatch(phonemes string, v voice, speed, pause float64) ([]float32, error) {
	tokens, err := k.tokenizer.tokenize(phonemes, maxPhonemeLength)
	if err != nil {
		return nil, err
	}
	if len(tokens) == 0 {
		return nil, fmt.Errorf("%w: %q", errNoTokens, phonemes)
	}
	audio, err := k.infer(tokens, v.style(len(tokens)), speed)
	if err != nil {
		return nil, err
	}
	audio, _ = trim(audio)
	if pause > 0 {
		audio = append(append([]float32(nil), audio...), make([]float32, int(pause*SampleRate))...)
	}
	return audio, nil
}

// streamItem is a batch's audio, or the error that ended the stream.
type streamItem struct {
	audio []float32
	err   error
}

// createStream synthesizes text in the background, a batch at a time,
// yielding each batch's audio at SampleRate as it is ready. A failure is
// yielded as the last item.
//
// One batch is synthesized ahead of the one being played, so the synthesis
// cannot run away with the whole text. Abandoning the stream stops it: the
// batch already running finishes, and no further batch is started.
func (k *kokoro) createStream(
	ctx context.Context, text, voiceName, lang string, speed float64,
) iter.Seq2[[]float32, error] {
	return func(yield func([]float32, error) bool) {
		v, batches, err := k.prepare(text, voiceName, speed, lang)
		if err != nil {
			yield(nil, err)
			return
		}
		streamCtx, cancel := context.WithCancel(ctx)
		defer cancel()
		queue := make(chan streamItem, 1)
		go k.processBatches(streamCtx, batches, v, speed, queue)
		for {
			select {
			case item, ok := <-queue:
				if !ok {
					return
				}
				if !yield(item.audio, item.err) || item.err != nil {
					return
				}
			case <-streamCtx.Done():
				yield(nil, fmt.Errorf("%w: %w", errStreamEnded, streamCtx.Err()))
				return
			}
		}
	}
}

// processBatches synthesizes the batches in turn into queue, which it closes
// at the end, stopping at the first failure or once ctx is done.
func (k *kokoro) processBatches(ctx context.Context, batches []batch, v voice, speed float64, queue chan<- streamItem) {
	defer close(queue)
	for i, b := range batches {
		if ctx.Err() != nil {
			return
		}
		audio, err := k.createBatch(b.phonemes, v, speed, b.pause)
		if err != nil {
			// Hand the failure to the consumer instead of leaving it waiting.
			select {
			case queue <- streamItem{err: err}:
			case <-ctx.Done():
			}
			return
		}
		slog.Debug(fmt.Sprintf("Processed chunk %d of stream", i))
		select {
		case queue <- streamItem{audio: audio}:
		case <-ctx.Done():
			return
		}
	}
}
