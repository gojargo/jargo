// Package reader reads how each user turn was meant, from its words, with the
// tone reader model (alpha).
//
// The model (https://huggingface.co/pipecat-ai/tone-models) reads one turn of
// text, with the turn before it, and returns the speaker's emotion, whether the
// turn is sarcastic, the move it makes and whether it is only a listening
// sound, each with a calibrated confidence. It runs on the CPU in a few
// milliseconds, through the ONNX runtime.
//
// Part of expressive conversation (alpha). These types are likely to change
// between releases.
package reader

import (
	"cmp"
	"context"
	"fmt"
	"slices"

	"github.com/gojargo/jargo/expressive"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/internal/validate"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/processor/tone/internal/shared"
)

const (
	emotionDescription = "The feeling the words show. Read from the words alone, so neutral means they show no " +
		"strong feeling, not that the user feels none: go by what they say."
	sarcasmDescription = "The chance the turn is sarcastic, from 0 to 1. Read the emotion together with it: " +
		"a sarcastic turn can read happy."
)

// Config configures a Reader. The zero value downloads the default release at
// FP16 and reads with four threads.
type Config struct {
	// Name is the processor's name. Empty is "ToneReader".
	Name string
	// Revision is the model's release on Hugging Face, such as "reader-rc2".
	// Empty is Revision.
	Revision string
	// ModelDir is a folder to load the model from instead of downloading one,
	// laid out as a release's "reader" folder. Revision is then unused.
	ModelDir string
	// Precision is the weights to load. Empty is FP16.
	Precision Precision `validate:"omitempty,oneof=fp16 fp32"`
	// Threads is the CPU threads for one read. Zero is 4.
	Threads int `validate:"min=0"`
	// CacheDir is where a downloaded model is kept. Empty is Hugging Face's own
	// cache, ~/.cache/huggingface/hub.
	CacheDir string
}

// model is what the reader needs of its model. *Model implements it.
type model interface {
	Emotions() []string
	Tag(text, previous string) (Tags, error)
}

// modelKey is what identifies a loaded model and its settings.
type modelKey struct {
	dir, revision string
	precision     Precision
	threads       int
}

//nolint:gochecknoglobals // the models are loaded once a process and shared
var (
	models = &shared.Cache[modelKey, model]{}
	// loadDir and downloadRelease open the model. Variables so tests can stand
	// a model in.
	loadDir = func(dir string, precision Precision, threads int) (model, error) {
		return LoadModel(dir, precision, threads)
	}
	downloadRelease = func(
		ctx context.Context, revision string, precision Precision, threads int, cacheDir string,
	) (model, error) {
		return DownloadModel(ctx, revision, precision, threads, cacheDir)
	}
)

// Reader reads how each user turn was meant.
//
// It goes between the STT service and the user aggregator. Each final
// transcription is read, with the bot's reply before it, and the reading is
// set as its Perception: the turn's "emotion", its two likeliest labels with
// their confidences, and its "sarcasm", the chance it is sarcastic. The
// transcription waits for the read, a few milliseconds.
//
// The model loads while the pipeline is set up, once a process, and every
// reader shares it. The first load downloads it from Hugging Face, which holds
// the pipeline's start, so download it ahead, such as when building the image.
// A model that cannot be loaded is reported as an error that leaves the reader
// unusable, and transcriptions pass through unread.
//
// Turn expressive conversation on with
// pipeline.Params.EnableExpressiveConversation for the LLM to read the turns:
//
//	pipeline.New(transport.Input(), stt, toneReader, userAggregator, llm, ...)
type Reader struct {
	*processor.Base
	cfg      Config
	model    model
	previous string
}

// New builds a Reader. It fails on a configuration it cannot use.
func New(cfg Config) (*Reader, error) {
	if err := validate.Struct(cfg); err != nil {
		return nil, fmt.Errorf("reader: %w", err)
	}
	if cfg.Name == "" {
		cfg.Name = "ToneReader"
	}
	if cfg.Revision == "" {
		cfg.Revision = Revision
	}
	if cfg.Precision == "" {
		cfg.Precision = FP16
	}
	if cfg.Threads == 0 {
		cfg.Threads = 4
	}
	r := &Reader{cfg: cfg}
	r.Base = processor.New(cfg.Name, r)
	return r, nil
}

// Setup sets the processor up and loads its model.
func (r *Reader) Setup(ctx context.Context, s processor.Setup) error {
	if err := r.Base.Setup(ctx, s); err != nil {
		return err
	}
	key := modelKey{dir: r.cfg.ModelDir, precision: r.cfg.Precision, threads: r.cfg.Threads}
	if key.dir == "" {
		key.revision = r.cfg.Revision
	}
	m, err := models.Load(key, func() (model, error) {
		if r.cfg.ModelDir != "" {
			return loadDir(r.cfg.ModelDir, r.cfg.Precision, r.cfg.Threads)
		}
		return downloadRelease(ctx, r.cfg.Revision, r.cfg.Precision, r.cfg.Threads, r.cfg.CacheDir)
	})
	if err != nil {
		r.PushError(ctx, fmt.Sprintf("%s: couldn't load its model: %v", r.Name(), err), err, false,
			processor.ForceTreatAsPermanent())
		return nil
	}
	r.model = m
	return nil
}

// ProcessFrame reads final transcriptions, and keeps the bot's replies to read
// them against.
func (r *Reader) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := r.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	switch fr := f.(type) {
	case *frames.StartFrame:
		if err := r.PushFrame(ctx, f, dir); err != nil {
			return err
		}
		if r.model != nil {
			return r.PushFrame(ctx, frames.NewPerceptionMetadataFrame(r.Name(), r.capabilities()),
				processor.Downstream)
		}
		return nil
	case *frames.TranscriptionFrame:
		if shared.Strip(fr.Text) != "" && r.model != nil {
			fr.Perception = r.read(ctx, fr.Text)
		}
	case *frames.LLMContextAssistantTurnFrame:
		r.previous = fr.Text
	}
	return r.PushFrame(ctx, f, dir)
}

func (r *Reader) capabilities() expressive.PerceptionCapabilities {
	return expressive.PerceptionCapabilities{Signals: []expressive.PerceptionTerm{
		{Name: "emotion", Description: emotionDescription, Labels: slices.Clone(r.model.Emotions())},
		{Name: "sarcasm", Description: sarcasmDescription},
	}}
}

// read reads one turn, or reports why it could not.
func (r *Reader) read(ctx context.Context, text string) []expressive.PerceptionSignal {
	tags, err := r.model.Tag(text, r.previous)
	if err != nil {
		r.PushError(ctx, fmt.Sprintf("%s: couldn't read the turn: %v", r.Name(), err), err, false)
		return nil
	}
	// The two likeliest emotions, the earlier one first of two alike.
	likeliest := slices.Clone(tags.Emotion.Probabilities)
	slices.SortStableFunc(likeliest, func(a, b Probability) int { return cmp.Compare(b.Probability, a.Probability) })
	labels := make([]expressive.PerceptionLabel, 0, 2)
	for _, p := range likeliest[:min(2, len(likeliest))] {
		labels = append(labels, expressive.PerceptionLabel{Label: p.Label, Confidence: p.Probability})
	}
	sarcasm := shared.Round(tags.Sarcasm.Probability, 2)
	return []expressive.PerceptionSignal{
		{Name: "emotion", Labels: labels},
		{Name: "sarcasm", Value: &sarcasm},
	}
}
