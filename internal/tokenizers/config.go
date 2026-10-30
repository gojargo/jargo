package tokenizers

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"
)

// The tokenizer.json layout, as far as this package reads it.

type fileJSON struct {
	Version       string           `json:"version"`
	Truncation    *truncationJSON  `json:"truncation"`
	AddedTokens   []addedTokenJSON `json:"added_tokens"`
	Normalizer    json.RawMessage  `json:"normalizer"`
	PreTokenizer  json.RawMessage  `json:"pre_tokenizer"`
	PostProcessor json.RawMessage  `json:"post_processor"`
	Model         json.RawMessage  `json:"model"`
}

type truncationJSON struct {
	Direction *string `json:"direction"`
	MaxLength *int    `json:"max_length"`
	Strategy  *string `json:"strategy"`
	Stride    *int    `json:"stride"`
}

type addedTokenJSON struct {
	ID         int    `json:"id"`
	Content    string `json:"content"`
	SingleWord *bool  `json:"single_word"`
	Lstrip     *bool  `json:"lstrip"`
	Rstrip     *bool  `json:"rstrip"`
	Normalized *bool  `json:"normalized"`
	Special    *bool  `json:"special"`
}

type typeJSON struct {
	Type string `json:"type"`
}

type bertNormalizerJSON struct {
	CleanText          bool  `json:"clean_text"`
	HandleChineseChars bool  `json:"handle_chinese_chars"`
	StripAccents       *bool `json:"strip_accents"`
	Lowercase          bool  `json:"lowercase"`
}

type byteLevelJSON struct {
	AddPrefixSpace bool  `json:"add_prefix_space"`
	TrimOffsets    bool  `json:"trim_offsets"`
	UseRegex       *bool `json:"use_regex"`
}

type wordPieceJSON struct {
	UnkToken                string         `json:"unk_token"`
	ContinuingSubwordPrefix string         `json:"continuing_subword_prefix"`
	MaxInputCharsPerWord    int            `json:"max_input_chars_per_word"`
	Vocab                   map[string]int `json:"vocab"`
}

type bpeJSON struct {
	Dropout                 *float64          `json:"dropout"`
	UnkToken                *string           `json:"unk_token"`
	ContinuingSubwordPrefix *string           `json:"continuing_subword_prefix"`
	EndOfWordSuffix         *string           `json:"end_of_word_suffix"`
	FuseUnk                 bool              `json:"fuse_unk"`
	ByteFallback            bool              `json:"byte_fallback"`
	IgnoreMerges            bool              `json:"ignore_merges"`
	Vocab                   map[string]int    `json:"vocab"`
	Merges                  []json.RawMessage `json:"merges"`
}

type templateJSON struct {
	Single        []map[string]pieceJSON      `json:"single"`
	Pair          []map[string]pieceJSON      `json:"pair"`
	SpecialTokens map[string]specialTokenJSON `json:"special_tokens"`
}

type pieceJSON struct {
	ID     string `json:"id"`
	TypeID int64  `json:"type_id"`
}

type specialTokenJSON struct {
	ID     string   `json:"id"`
	IDs    []int64  `json:"ids"`
	Tokens []string `json:"tokens"`
}

func unsupported(format string, args ...any) error {
	return fmt.Errorf("%w: %s", ErrUnsupported, fmt.Sprintf(format, args...))
}

func invalid(format string, args ...any) error {
	return fmt.Errorf("%w: %s", ErrInvalidFile, fmt.Sprintf(format, args...))
}

// isNull reports whether a component is absent or null.
func isNull(raw json.RawMessage) bool {
	t := bytes.TrimSpace(raw)
	return len(t) == 0 || bytes.Equal(t, []byte("null"))
}

func componentType(raw json.RawMessage, what string) (string, error) {
	var t typeJSON
	if err := json.Unmarshal(raw, &t); err != nil {
		return "", invalid("%s: %v", what, err)
	}
	return t.Type, nil
}

// New parses tokenizer.json bytes. It returns an error naming any component,
// option or value it doesn't implement, rather than encoding differently.
func New(data []byte) (*Tokenizer, error) {
	var f fileJSON
	if err := json.Unmarshal(data, &f); err != nil {
		return nil, invalid("%v", err)
	}
	if f.Version != "1.0" {
		return nil, unsupported("version %q", f.Version)
	}
	t := &Tokenizer{}
	var err error
	if t.truncation, err = parseTruncation(f.Truncation); err != nil {
		return nil, err
	}
	if t.normalizer, err = parseNormalizer(f.Normalizer); err != nil {
		return nil, err
	}
	if t.preTokenizer, err = parsePreTokenizer(f.PreTokenizer); err != nil {
		return nil, err
	}
	if t.model, err = parseModel(f.Model); err != nil {
		return nil, err
	}
	if t.template, err = parseTemplate(f.PostProcessor); err != nil {
		return nil, err
	}
	tokens, err := parseAddedTokens(f.AddedTokens)
	if err != nil {
		return nil, err
	}
	if t.added, err = newAddedVocabulary(tokens, t.model.tokenID, t.model.vocabSize(), t.normalizer); err != nil {
		return nil, err
	}
	return t, nil
}

func parseAddedTokens(js []addedTokenJSON) ([]addedToken, error) {
	tokens := make([]addedToken, 0, len(js))
	for _, a := range js {
		if a.SingleWord == nil || a.Lstrip == nil || a.Rstrip == nil || a.Normalized == nil || a.Special == nil {
			return nil, invalid("added token %q: missing field", a.Content)
		}
		if *a.SingleWord {
			return nil, unsupported("added token %q: single_word", a.Content)
		}
		tokens = append(tokens, addedToken{
			content: a.Content, lstrip: *a.Lstrip, rstrip: *a.Rstrip,
			normalized: *a.Normalized, special: *a.Special,
		})
	}
	return tokens, nil
}

func parseTruncation(j *truncationJSON) (*Truncation, error) {
	if j == nil {
		return nil, nil //nolint:nilnil // no truncation is a valid result
	}
	if j.MaxLength == nil || j.Strategy == nil || j.Stride == nil {
		return nil, unsupported("truncation: missing field")
	}
	if *j.Stride != 0 {
		return nil, unsupported("truncation stride %d", *j.Stride)
	}
	if *j.MaxLength < 0 {
		return nil, unsupported("truncation max_length %d", *j.MaxLength)
	}
	t := &Truncation{MaxLength: *j.MaxLength}
	switch *j.Strategy {
	case "LongestFirst":
		t.Strategy = LongestFirst
	case "OnlyFirst":
		t.Strategy = OnlyFirst
	case "OnlySecond":
		t.Strategy = OnlySecond
	default:
		return nil, unsupported("truncation strategy %q", *j.Strategy)
	}
	if j.Direction != nil {
		switch *j.Direction {
		case "Right":
			t.Direction = Right
		case "Left":
			t.Direction = Left
		default:
			return nil, unsupported("truncation direction %q", *j.Direction)
		}
	}
	return t, nil
}

func parseNormalizer(raw json.RawMessage) (normalizer, error) {
	if isNull(raw) {
		return nil, nil //nolint:nilnil // no normalizer is a valid result
	}
	typ, err := componentType(raw, "normalizer")
	if err != nil {
		return nil, err
	}
	switch typ {
	case "BertNormalizer":
		var j bertNormalizerJSON
		if err := json.Unmarshal(raw, &j); err != nil {
			return nil, invalid("normalizer: %v", err)
		}
		strip := j.Lowercase
		if j.StripAccents != nil {
			strip = *j.StripAccents
		}
		return bertNormalizer{
			cleanText: j.CleanText, handleChineseChars: j.HandleChineseChars,
			stripAccents: strip, lowercase: j.Lowercase,
		}, nil
	case "NFC":
		return nfcNormalizer{}, nil
	}
	return nil, unsupported("normalizer %q", typ)
}

func parsePreTokenizer(raw json.RawMessage) (preTokenizer, error) {
	if isNull(raw) {
		return nil, nil //nolint:nilnil // no pre-tokenizer is a valid result
	}
	typ, err := componentType(raw, "pre_tokenizer")
	if err != nil {
		return nil, err
	}
	switch typ {
	case "BertPreTokenizer":
		return bertPreTokenizer{}, nil
	case "ByteLevel":
		var j byteLevelJSON
		if err := json.Unmarshal(raw, &j); err != nil {
			return nil, invalid("pre_tokenizer: %v", err)
		}
		if j.AddPrefixSpace {
			return nil, unsupported("ByteLevel add_prefix_space")
		}
		if j.UseRegex != nil && !*j.UseRegex {
			return nil, unsupported("ByteLevel without use_regex")
		}
		// trim_offsets only matters to ByteLevel as a post-processor.
		return newByteLevelPreTokenizer(), nil
	}
	return nil, unsupported("pre_tokenizer %q", typ)
}

func parseModel(raw json.RawMessage) (model, error) {
	if isNull(raw) {
		return nil, unsupported("no model")
	}
	typ, err := componentType(raw, "model")
	if err != nil {
		return nil, err
	}
	switch typ {
	case "WordPiece":
		var j wordPieceJSON
		if err := json.Unmarshal(raw, &j); err != nil {
			return nil, invalid("model: %v", err)
		}
		unkID, ok := j.Vocab[j.UnkToken]
		if !ok {
			return nil, invalid("WordPiece unknown token %q missing from the vocabulary", j.UnkToken)
		}
		return &wordPiece{
			vocab: j.Vocab, unk: j.UnkToken, unkID: unkID,
			prefix: j.ContinuingSubwordPrefix, maxInputChars: j.MaxInputCharsPerWord,
		}, nil
	case "BPE":
		return parseBPE(raw)
	}
	return nil, unsupported("model %q", typ)
}

func parseBPE(raw json.RawMessage) (model, error) {
	var j bpeJSON
	if err := json.Unmarshal(raw, &j); err != nil {
		return nil, invalid("model: %v", err)
	}
	if err := checkBPEOptions(&j); err != nil {
		return nil, err
	}
	b := &bpe{
		vocab: j.Vocab, vocabR: make([]string, 0, len(j.Vocab)),
		merges: make(map[[2]int]bpeMerge, len(j.Merges)), cache: map[string][]token{},
	}
	for k, v := range j.Vocab {
		if v < 0 || v > 1<<24 {
			return nil, unsupported("BPE token id %d", v)
		}
		for len(b.vocabR) <= v {
			b.vocabR = append(b.vocabR, "")
		}
		b.vocabR[v] = k
	}
	for rank, m := range j.Merges {
		a, c, err := parseMerge(m)
		if err != nil {
			return nil, err
		}
		aID, ok1 := j.Vocab[a]
		cID, ok2 := j.Vocab[c]
		newID, ok3 := j.Vocab[a+c]
		if !ok1 || !ok2 || !ok3 {
			return nil, invalid("BPE merge %q %q: token out of vocabulary", a, c)
		}
		b.merges[[2]int{aID, cID}] = bpeMerge{rank: rank, id: newID}
	}
	return b, nil
}

// checkBPEOptions refuses the BPE options this package does not implement.
func checkBPEOptions(j *bpeJSON) error {
	switch {
	case j.Dropout != nil && *j.Dropout != 0:
		return unsupported("BPE dropout %v", *j.Dropout)
	case j.UnkToken != nil:
		return unsupported("BPE unk_token")
	case j.ContinuingSubwordPrefix != nil:
		return unsupported("BPE continuing_subword_prefix")
	case j.EndOfWordSuffix != nil:
		return unsupported("BPE end_of_word_suffix")
	case j.ByteFallback:
		return unsupported("BPE byte_fallback")
	case j.IgnoreMerges:
		return unsupported("BPE ignore_merges")
	}
	return nil
}

// parseMerge reads a merge as a pair, or in the legacy "a b" form.
func parseMerge(raw json.RawMessage) (string, string, error) {
	var pair []string
	if err := json.Unmarshal(raw, &pair); err == nil {
		if len(pair) != 2 {
			return "", "", invalid("BPE merge %s: not a pair", raw)
		}
		return pair[0], pair[1], nil
	}
	var s string
	if err := json.Unmarshal(raw, &s); err != nil {
		return "", "", invalid("BPE merge %s: %v", raw, err)
	}
	parts := strings.Split(s, " ")
	if len(parts) != 2 {
		return "", "", invalid("BPE merge %q: not a pair", s)
	}
	return parts[0], parts[1], nil
}

func parseTemplate(raw json.RawMessage) (*template, error) {
	if isNull(raw) {
		return nil, unsupported("no post_processor")
	}
	typ, err := componentType(raw, "post_processor")
	if err != nil {
		return nil, err
	}
	if typ != "TemplateProcessing" {
		return nil, unsupported("post_processor %q", typ)
	}
	var j templateJSON
	if err = json.Unmarshal(raw, &j); err != nil {
		return nil, invalid("post_processor: %v", err)
	}
	specials := map[string]*specialToken{}
	for name, s := range j.SpecialTokens {
		if len(s.IDs) != len(s.Tokens) {
			return nil, unsupported("special token %q: ids and tokens differ in length", name)
		}
		specials[name] = &specialToken{ids: s.IDs, tokens: s.Tokens}
	}
	single, err := parsePieces(j.Single, specials, false)
	if err != nil {
		return nil, err
	}
	pair, err := parsePieces(j.Pair, specials, true)
	if err != nil {
		return nil, err
	}
	return &template{single: single, pair: pair}, nil
}

func parsePieces(js []map[string]pieceJSON, specials map[string]*specialToken, pair bool) ([]piece, error) {
	out := make([]piece, 0, len(js))
	seen := [2]bool{}
	for _, m := range js {
		if len(m) != 1 {
			return nil, unsupported("template piece %v", m)
		}
		for kind, p := range m {
			switch kind {
			case "Sequence":
				seq := map[string]int{"A": 0, "B": 1}
				i, ok := seq[p.ID]
				if !ok || (i == 1 && !pair) || seen[i] {
					return nil, unsupported("template sequence %q", p.ID)
				}
				seen[i] = true
				out = append(out, piece{seq: i, typeID: p.TypeID})
			case "SpecialToken":
				s, ok := specials[p.ID]
				if !ok {
					return nil, unsupported("template special token %q missing", p.ID)
				}
				out = append(out, piece{special: s, typeID: p.TypeID})
			default:
				return nil, unsupported("template piece %q", kind)
			}
		}
	}
	if !seen[0] || (pair && !seen[1]) {
		return nil, unsupported("template does not use every sequence")
	}
	return out, nil
}
