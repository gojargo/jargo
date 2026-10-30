# Test data

## Tokenizer files

Both come from the Hugging Face repository
[`pipecat-ai/tone-models`](https://huggingface.co/pipecat-ai/tone-models),
licensed BSD-2-Clause, and are stored gzipped and unmodified:

- `reader.tokenizer.json.gz`: `reader/tokenizer.json` at revision `reader-rc2`
  (commit `2ba272b19642f77bf72e2e47fcf1b9d1d65c398d`).
- `tagger.tokenizer.json.gz`: `tagger/17m/tokenizer.json` at revision
  `tagger-rc1` (commit `6f4bb8b077d7ef9cec4ceb75d0c67beb8deb76c1`).
  `tagger/32m/tokenizer.json` is byte-for-byte the same file.

## Fixtures

Both were produced by the Python package `tokenizers` 0.23.3 (the version is
recorded in each file and checked by the tests), loading the two files above
with `Tokenizer.from_file`.

- `fixtures.json.gz`: for each tokenizer, a list of cases, each a first
  sequence, an optional second one and a truncation (`max_length`, `strategy`,
  `direction`, or null for none), with either the `encoding` that
  `Tokenizer.encode` returned (`ids`, `type_ids`, `tokens`, `offsets`,
  `special_tokens_mask`, `attention_mask`, `sequence_ids` with -1 for `None`)
  or the text of the `error` it raised. Padding is off. The cases cover
  hand-written texts (accents and combining marks in composed and decomposed
  form, case mappings, CJK, Thai, Arabic, Devanagari and other scripts, emoji
  sequences, control, invisible and private-use characters, whitespace runs,
  contractions, numbers, the literal added tokens, words over the WordPiece
  length limit, the text shapes the models are fed), pairs of them, every
  truncation strategy and direction at maximum lengths from 0 to 40 and at the
  models' 160, a seeded fuzz of combining-mark sequences and a seeded fuzz of
  random texts.
- `unicode.json.gz`: what each Unicode-dependent step does to every code point
  but the surrogates: `BertNormalizer` (as the reader configures it) and `NFC`
  on the code point alone (`[first, last, output]` runs, for the code points
  that change), how `BertPreTokenizer` cuts `"a" + c + "a"` (`w` whitespace,
  `p` punctuation) and how `ByteLevel` cuts `"a" + c`, `"1" + c` and `"!" + c`
  (`L` letter, `N` number, `s` whitespace), and `NFC` of the NFD form of every
  code point that has one, as `[input, output]` pairs.
