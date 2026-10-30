// Package tokenizers encodes text into model input ids exactly as the Hugging
// Face tokenizers library (version 0.23.3, through its Python binding) does
// for the tokenizer.json files it supports: the same ids, type ids, masks,
// sequence ids and offsets, the offsets counted in Unicode code points of the
// sequence each token belongs to.
//
// It implements the components those files use (the BertNormalizer and NFC
// normalizers, the BertPreTokenizer and ByteLevel pre-tokenizers, the
// WordPiece and BPE models, the added vocabulary and the TemplateProcessing
// post-processor) and refuses any other component or option, rather than
// encoding differently. The Unicode data comes from the libraries Hugging Face
// tokenizers links (Unicode 9.0.0 for normalization and character categories,
// Oniguruma's Unicode 16.0 for the ByteLevel pattern), not from Go's unicode
// package, whose tables are newer.
package tokenizers

import (
	"errors"
	"fmt"
	"os"
	"unicode/utf8"
)

// Strategy says which sequence of a pair truncation shortens.
type Strategy int

const (
	// LongestFirst shortens the longer sequence first, then both.
	LongestFirst Strategy = iota
	// OnlyFirst shortens the first sequence only.
	OnlyFirst
	// OnlySecond shortens the second sequence only.
	OnlySecond
)

// Direction says which end of a sequence truncation removes tokens from.
type Direction int

const (
	// Right removes tokens from the end.
	Right Direction = iota
	// Left removes tokens from the start.
	Left
)

// Truncation limits the length of an encoding, special tokens included.
type Truncation struct {
	MaxLength int
	Strategy  Strategy
	Direction Direction
}

// Offset is a token's range in the sequence it belongs to, in Unicode code
// points. Special tokens added by the template have (0, 0).
type Offset struct{ Start, End int }

// Encoding is the encoding of one sequence or a pair, as Hugging Face
// tokenizers' Encoding reports it. All slices have one element per token.
type Encoding struct {
	IDs               []int64
	TypeIDs           []int64
	AttentionMask     []int64
	Tokens            []string
	Offsets           []Offset
	SpecialTokensMask []int
	// SequenceIDs is the sequence (0 or 1) of each token, -1 for the special
	// tokens.
	SequenceIDs []int
}

var (
	// ErrSequenceTooShort is returned (wrapped in Hugging Face tokenizers'
	// message) when the sequence a strategy truncates is too short to bring
	// the encoding down to the maximum length.
	ErrSequenceTooShort = errors.New("tokenizers: sequence to truncate too short to respect the provided max_length")
	// ErrSecondSequenceNotProvided is returned (wrapped in Hugging Face
	// tokenizers' message) when OnlySecond has to truncate a single sequence.
	ErrSecondSequenceNotProvided = errors.New("tokenizers: second sequence not provided")
	// ErrUnsupported is returned by New for a component, option or value
	// this package does not implement.
	ErrUnsupported = errors.New("tokenizers: unsupported tokenizer.json")
	// ErrInvalidFile is returned by New for a tokenizer.json that is not
	// well formed, or that Hugging Face tokenizers would refuse.
	ErrInvalidFile = errors.New("tokenizers: invalid tokenizer.json")
	// ErrInvalidInput is returned for text that is not valid UTF-8, or a
	// negative maximum length.
	ErrInvalidInput = errors.New("tokenizers: invalid input")
	// ErrInternal is returned if encoding hits a state Hugging Face tokenizers
	// would panic on.
	ErrInternal = errors.New("tokenizers: internal error")
)

// Tokenizer encodes text. It is immutable after New, and safe for concurrent
// use.
type Tokenizer struct {
	normalizer   normalizer
	preTokenizer preTokenizer
	model        model
	added        *addedVocabulary
	template     *template
	truncation   *Truncation
}

// FromFile reads and parses a tokenizer.json file.
func FromFile(path string) (*Tokenizer, error) {
	data, err := os.ReadFile(path) //nolint:gosec // the caller names the file
	if err != nil {
		return nil, fmt.Errorf("tokenizers: read %s: %w", path, err)
	}
	return New(data)
}

// Truncation is the truncation the file sets, or nil.
func (t *Tokenizer) Truncation() *Truncation {
	if t.truncation == nil {
		return nil
	}
	c := *t.truncation
	return &c
}

// Encode encodes one sequence with the special tokens of the template, and
// truncates it by trunc (nil: no truncation). It never pads.
func (t *Tokenizer) Encode(text string, trunc *Truncation) (Encoding, error) {
	return t.encode(trunc, text)
}

// EncodePair encodes a pair of sequences with the special tokens of the
// template, and truncates them by trunc (nil: no truncation). It never pads.
func (t *Tokenizer) EncodePair(first, second string, trunc *Truncation) (Encoding, error) {
	return t.encode(trunc, first, second)
}

func (t *Tokenizer) encode(trunc *Truncation, seqs ...string) (enc Encoding, err error) {
	for _, s := range seqs {
		if !utf8.ValidString(s) {
			return Encoding{}, fmt.Errorf("%w: text is not valid UTF-8", ErrInvalidInput)
		}
	}
	if trunc != nil && trunc.MaxLength < 0 {
		return Encoding{}, fmt.Errorf("%w: negative maximum length %d", ErrInvalidInput, trunc.MaxLength)
	}
	defer func() {
		if r := recover(); r != nil {
			enc, err = Encoding{}, fmt.Errorf("%w: %v", ErrInternal, r)
		}
	}()

	encs := make([]*encoding, len(seqs))
	for i, s := range seqs {
		e, err := t.encodeSequence(s, int64(i))
		if err != nil {
			return Encoding{}, err
		}
		encs[i] = e
	}
	var pair *encoding
	if len(encs) == 2 {
		pair = encs[1]
	}
	if trunc != nil {
		// Hugging Face tokenizers takes the special tokens off the maximum
		// length in unsigned arithmetic, so a maximum below their count wraps
		// around and truncates nothing.
		if n := t.template.added(pair != nil); trunc.MaxLength >= n {
			if err := truncateEncodings(encs[0], pair, trunc.MaxLength-n, trunc.Strategy, trunc.Direction); err != nil {
				return Encoding{}, err
			}
		}
	}
	return t.template.process(encs).export(), nil
}

func (t *Tokenizer) encodeSequence(s string, typeID int64) (*encoding, error) {
	p := t.added.extractAndNormalize(t.normalizer, s)
	if t.preTokenizer != nil {
		t.preTokenizer.preTokenize(p)
	}
	if err := p.tokenize(t.model.tokenize); err != nil {
		return nil, err
	}
	return p.encoding(typeID), nil
}

// internalError is what panicf panics with; encode turns it back into an
// error.
type internalError string

func panicf(format string, args ...any) {
	panic(internalError(fmt.Sprintf(format, args...))) //nolint:forbidigo // recovered by encode
}
