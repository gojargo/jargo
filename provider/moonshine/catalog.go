package moonshine

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"strings"
)

// modelArch is a Moonshine model architecture, as the library numbers it.
type modelArch uint32

// The architectures the library knows.
const (
	archTiny modelArch = iota
	archBase
	archTinyStreaming
	archBaseStreaming
	archSmallStreaming
	archMediumStreaming
)

// Errors the catalog reports.
var (
	// errInvalidArch reports a model name that is no Moonshine architecture.
	errInvalidArch = errors.New("moonshine: invalid model architecture string")
	// errLanguageNotFound reports a language the catalog has no models for.
	errLanguageNotFound = errors.New("moonshine: language not found")
	// errModelNotFound reports an architecture the language does not publish.
	errModelNotFound = errors.New("moonshine: model not found for language")
)

// String is the architecture's name, the model string the settings carry.
func (a modelArch) String() string {
	switch a {
	case archTiny:
		return string(ModelTiny)
	case archBase:
		return string(ModelBase)
	case archTinyStreaming:
		return string(ModelTinyStreaming)
	case archBaseStreaming:
		return string(ModelBaseStreaming)
	case archSmallStreaming:
		return string(ModelSmallStreaming)
	case archMediumStreaming:
		return string(ModelMediumStreaming)
	default:
		return fmt.Sprintf("modelArch(%d)", uint32(a))
	}
}

// stringToModelArch is the architecture a model string names.
func stringToModelArch(s string) (modelArch, error) {
	switch Model(s) {
	case ModelTiny:
		return archTiny, nil
	case ModelBase:
		return archBase, nil
	case ModelTinyStreaming:
		return archTinyStreaming, nil
	case ModelBaseStreaming:
		return archBaseStreaming, nil
	case ModelSmallStreaming:
		return archSmallStreaming, nil
	case ModelMediumStreaming:
		return archMediumStreaming, nil
	default:
		return 0, fmt.Errorf("%w: %s", errInvalidArch, s)
	}
}

// modelCatalog resolves a language and an architecture to a model directory
// the library can load, downloading the model first when it has to.
type modelCatalog interface {
	// supportedLanguages lists the language codes with at least one model.
	supportedLanguages() []string
	// supportedLanguagesFriendly lists them for a person to read.
	supportedLanguagesFriendly() string
	// modelForLanguage returns the directory holding the model for lang and
	// the architecture it is. A nil arch takes the language's first model.
	modelForLanguage(ctx context.Context, lang string, arch *modelArch) (string, modelArch, error)
}

// modelFile is one file of a published model, with what it is checked
// against once downloaded: its size in bytes and its CRC32C, base64 encoded
// big-endian as the CDN reports it.
type modelFile struct {
	name   string
	size   int64
	crc32c string
}

// catalogModel is one published model: its architecture, the address its
// files are under, and the files.
type catalogModel struct {
	arch    modelArch
	baseURL string
	files   []modelFile
}

// catalogLanguage is a language and the models published for it, the default
// first.
type catalogLanguage struct {
	code        string
	englishName string
	models      []catalogModel
}

// sttCatalog is the published speech-to-text models, in the library's
// catalog order, with each model's download manifest: the files the
// transcriber loads, without the optional word-timestamp decoder. It mirrors
// the catalog and file registry compiled into the library at C ABI version
// 30000 (moonshine-voice 0.1.6), which the library serves through
// moonshine_get_stt_catalog and moonshine_get_stt_dependencies.
//
//nolint:gochecknoglobals,lll,goconst // a read-only table of published models
var sttCatalog = []catalogLanguage{
	{code: "ar", englishName: "Arabic", models: []catalogModel{
		{arch: archTinyStreaming, baseURL: "https://download.moonshine.ai/model/tiny-streaming-ar/quantized_26_08_24", files: []modelFile{
			{name: "adapter.ort", size: 1318472, crc32c: "Ba/M3A=="},
			{name: "cross_kv.ort", size: 1288120, crc32c: "BXUe/w=="},
			{name: "decoder_kv.ort", size: 19717336, crc32c: "s5bwZA=="},
			{name: "encoder.ort", size: 7772792, crc32c: "6tp7Mg=="},
			{name: "frontend.model.ort", size: 23176, crc32c: "lLmrsA=="},
			{name: "frontend.weights.ort", size: 2093280, crc32c: "exI8jg=="},
			{name: "streaming_config.json", size: 509, crc32c: "wH/VeA=="},
			{name: "tokenizer.bin", size: 135726, crc32c: "y80toQ=="},
		}},
		{arch: archBase, baseURL: "https://download.moonshine.ai/model/base-ar/quantized/base-ar", files: []modelFile{
			{name: "encoder_model.ort", size: 31326824, crc32c: "QgxL3Q=="},
			{name: "decoder_model_merged.ort", size: 109424552, crc32c: "u17PkA=="},
			{name: "tokenizer.bin", size: 249974, crc32c: "B7s10Q=="},
		}},
	}},
	{code: "es", englishName: "Spanish", models: []catalogModel{
		{arch: archSmallStreaming, baseURL: "https://download.moonshine.ai/model/small-streaming-es/quantized_26_08_24", files: []modelFile{
			{name: "adapter.ort", size: 2869296, crc32c: "ZCOnJw=="},
			{name: "cross_kv.ort", size: 5358752, crc32c: "B6G3LA=="},
			{name: "decoder_kv.ort", size: 61314512, crc32c: "Q6TnrQ=="},
			{name: "encoder.ort", size: 44358376, crc32c: "EYLK1A=="},
			{name: "frontend.model.ort", size: 26776, crc32c: "bvdZXg=="},
			{name: "frontend.weights.ort", size: 7769280, crc32c: "ITuX4A=="},
			{name: "streaming_config.json", size: 512, crc32c: "Y/oEHw=="},
			{name: "tokenizer.bin", size: 102888, crc32c: "/7v8NQ=="},
		}},
		{arch: archTinyStreaming, baseURL: "https://download.moonshine.ai/model/tiny-streaming-es/quantized_26_08_24", files: []modelFile{
			{name: "adapter.ort", size: 1318472, crc32c: "+JOBGg=="},
			{name: "cross_kv.ort", size: 1288120, crc32c: "lvRCDw=="},
			{name: "decoder_kv.ort", size: 19717336, crc32c: "KEm/xA=="},
			{name: "encoder.ort", size: 7772792, crc32c: "AAXCVA=="},
			{name: "frontend.model.ort", size: 23176, crc32c: "v7nfyQ=="},
			{name: "frontend.weights.ort", size: 2093280, crc32c: "LYbiaA=="},
			{name: "streaming_config.json", size: 509, crc32c: "wH/VeA=="},
			{name: "tokenizer.bin", size: 102888, crc32c: "/7v8NQ=="},
		}},
		{arch: archBase, baseURL: "https://download.moonshine.ai/model/base-es/quantized/base-es", files: []modelFile{
			{name: "encoder_model.ort", size: 20964320, crc32c: "2ngM7g=="},
			{name: "decoder_model_merged.ort", size: 43612200, crc32c: "9NUuYQ=="},
			{name: "tokenizer.bin", size: 241639, crc32c: "EBC+Hw=="},
		}},
	}},
	{code: "de", englishName: "German", models: []catalogModel{
		{arch: archSmallStreaming, baseURL: "https://download.moonshine.ai/model/small-streaming-de/quantized_26_08_24", files: []modelFile{
			{name: "adapter.ort", size: 2869296, crc32c: "rx5eAg=="},
			{name: "cross_kv.ort", size: 5358752, crc32c: "B9ac2w=="},
			{name: "decoder_kv.ort", size: 61314512, crc32c: "IRMZCQ=="},
			{name: "encoder.ort", size: 44358376, crc32c: "AI1jZg=="},
			{name: "frontend.model.ort", size: 26776, crc32c: "nGFUEw=="},
			{name: "frontend.weights.ort", size: 7769280, crc32c: "kUSrbA=="},
			{name: "streaming_config.json", size: 512, crc32c: "Y/oEHw=="},
			{name: "tokenizer.bin", size: 103319, crc32c: "x219Qw=="},
		}},
		{arch: archTinyStreaming, baseURL: "https://download.moonshine.ai/model/tiny-streaming-de/quantized_26_08_24", files: []modelFile{
			{name: "adapter.ort", size: 1318472, crc32c: "Q5RVow=="},
			{name: "cross_kv.ort", size: 1288120, crc32c: "Y8sggQ=="},
			{name: "decoder_kv.ort", size: 19717336, crc32c: "0Btj7Q=="},
			{name: "encoder.ort", size: 7772792, crc32c: "UMxvQg=="},
			{name: "frontend.model.ort", size: 23176, crc32c: "jBdx4Q=="},
			{name: "frontend.weights.ort", size: 2093280, crc32c: "Ljirxw=="},
			{name: "streaming_config.json", size: 509, crc32c: "wH/VeA=="},
			{name: "tokenizer.bin", size: 103319, crc32c: "x219Qw=="},
		}},
	}},
	{code: "en", englishName: "English", models: []catalogModel{
		{arch: archMediumStreaming, baseURL: "https://download.moonshine.ai/model/medium-streaming-en/quantized_26_11_02", files: []modelFile{
			{name: "adapter.ort", size: 3651256, crc32c: "z/YpJw=="},
			{name: "cross_kv.ort", size: 11643720, crc32c: "B8yR7Q=="},
			{name: "decoder_kv.ort", size: 147137432, crc32c: "XHfhHg=="},
			{name: "encoder.ort", size: 94705336, crc32c: "NOcutw=="},
			{name: "frontend.model.ort", size: 28768, crc32c: "8DJD0w=="},
			{name: "frontend.weights.ort", size: 11889512, crc32c: "IvbcFg=="},
			{name: "streaming_config.json", size: 513, crc32c: "+1bE9Q=="},
			{name: "tokenizer.bin", size: 249974, crc32c: "B7s10Q=="},
		}},
		{arch: archSmallStreaming, baseURL: "https://download.moonshine.ai/model/small-streaming-en/quantized_26_11_02", files: []modelFile{
			{name: "adapter.ort", size: 2870320, crc32c: "yw6hxQ=="},
			{name: "cross_kv.ort", size: 5356496, crc32c: "1rJswQ=="},
			{name: "decoder_kv.ort", size: 82043616, crc32c: "+Q3U0Q=="},
			{name: "encoder.ort", size: 44148536, crc32c: "vJtYUQ=="},
			{name: "frontend.model.ort", size: 26992, crc32c: "PJW8eA=="},
			{name: "frontend.weights.ort", size: 7769416, crc32c: "Fdny+w=="},
			{name: "streaming_config.json", size: 512, crc32c: "dPbFiw=="},
			{name: "tokenizer.bin", size: 249974, crc32c: "B7s10Q=="},
		}},
		{arch: archBase, baseURL: "https://download.moonshine.ai/model/base-en/quantized/base-en", files: []modelFile{
			{name: "encoder_model.ort", size: 31326816, crc32c: "+MATVw=="},
			{name: "decoder_model_merged.ort", size: 109424400, crc32c: "71VpRQ=="},
			{name: "tokenizer.bin", size: 249974, crc32c: "B7s10Q=="},
		}},
		{arch: archTinyStreaming, baseURL: "https://download.moonshine.ai/model/tiny-streaming-en/quantized_26_11_02", files: []modelFile{
			{name: "adapter.ort", size: 1319616, crc32c: "8tL9VQ=="},
			{name: "cross_kv.ort", size: 1287504, crc32c: "nwGoTw=="},
			{name: "decoder_kv.ort", size: 32748720, crc32c: "rpn9LA=="},
			{name: "encoder.ort", size: 7675392, crc32c: "9j4G6g=="},
			{name: "frontend.model.ort", size: 23392, crc32c: "AxTn9Q=="},
			{name: "frontend.weights.ort", size: 2093416, crc32c: "lNoOfg=="},
			{name: "streaming_config.json", size: 509, crc32c: "HGL0Ug=="},
			{name: "tokenizer.bin", size: 249974, crc32c: "B7s10Q=="},
		}},
		{arch: archTiny, baseURL: "https://download.moonshine.ai/model/tiny-en/quantized/tiny-en", files: []modelFile{
			{name: "encoder_model.ort", size: 13281600, crc32c: "jmn6JA=="},
			{name: "decoder_model_merged.ort", size: 30412256, crc32c: "LngmQA=="},
			{name: "tokenizer.bin", size: 249974, crc32c: "B7s10Q=="},
		}},
	}},
	{code: "ja", englishName: "Japanese", models: []catalogModel{
		{arch: archSmallStreaming, baseURL: "https://download.moonshine.ai/model/small-streaming-ja/quantized_26_08_23", files: []modelFile{
			{name: "adapter.ort", size: 2869296, crc32c: "GzlnQA=="},
			{name: "cross_kv.ort", size: 5358752, crc32c: "0btjiw=="},
			{name: "decoder_kv.ort", size: 61314512, crc32c: "qXEgyA=="},
			{name: "encoder.ort", size: 44358376, crc32c: "fltsVQ=="},
			{name: "frontend.model.ort", size: 31208, crc32c: "RhG3gA=="},
			{name: "frontend.weights.ort", size: 7769288, crc32c: "VjqzBw=="},
			{name: "streaming_config.json", size: 512, crc32c: "Y/oEHw=="},
			{name: "tokenizer.bin", size: 101836, crc32c: "M94Ogg=="},
		}},
		{arch: archTinyStreaming, baseURL: "https://download.moonshine.ai/model/tiny-streaming-ja/quantized_26_08_23", files: []modelFile{
			{name: "adapter.ort", size: 1318472, crc32c: "obzOjg=="},
			{name: "cross_kv.ort", size: 1288120, crc32c: "gbYCsw=="},
			{name: "decoder_kv.ort", size: 19717336, crc32c: "Mc/YqQ=="},
			{name: "encoder.ort", size: 7772792, crc32c: "ZS0TPg=="},
			{name: "frontend.model.ort", size: 27608, crc32c: "BTpcfg=="},
			{name: "frontend.weights.ort", size: 2093288, crc32c: "T4l/3Q=="},
			{name: "streaming_config.json", size: 509, crc32c: "wH/VeA=="},
			{name: "tokenizer.bin", size: 101836, crc32c: "M94Ogg=="},
		}},
		{arch: archBase, baseURL: "https://download.moonshine.ai/model/base-ja/quantized/base-ja", files: []modelFile{
			{name: "encoder_model.ort", size: 31326816, crc32c: "LErsPw=="},
			{name: "decoder_model_merged.ort", size: 109424424, crc32c: "FLBs0A=="},
			{name: "tokenizer.bin", size: 249974, crc32c: "B7s10Q=="},
		}},
		{arch: archTiny, baseURL: "https://download.moonshine.ai/model/tiny-ja/quantized/tiny-ja", files: []modelFile{
			{name: "encoder_model.ort", size: 13238184, crc32c: "UJX6fA=="},
			{name: "decoder_model_merged.ort", size: 58327272, crc32c: "yiH1qg=="},
			{name: "tokenizer.bin", size: 249974, crc32c: "B7s10Q=="},
		}},
	}},
	{code: "ko", englishName: "Korean", models: []catalogModel{
		{arch: archTiny, baseURL: "https://download.moonshine.ai/model/tiny-ko/quantized/tiny-ko", files: []modelFile{
			{name: "encoder_model.ort", size: 13238176, crc32c: "kczhwQ=="},
			{name: "decoder_model_merged.ort", size: 58327336, crc32c: "sjwMCA=="},
			{name: "tokenizer.bin", size: 249974, crc32c: "B7s10Q=="},
		}},
	}},
	{code: "vi", englishName: "Vietnamese", models: []catalogModel{
		{arch: archTinyStreaming, baseURL: "https://download.moonshine.ai/model/tiny-streaming-vi/quantized_26_08_24", files: []modelFile{
			{name: "adapter.ort", size: 1318472, crc32c: "cdV7Qw=="},
			{name: "cross_kv.ort", size: 1288120, crc32c: "aeNa0A=="},
			{name: "decoder_kv.ort", size: 19717336, crc32c: "792dSQ=="},
			{name: "encoder.ort", size: 7772792, crc32c: "5vA9hw=="},
			{name: "frontend.model.ort", size: 23176, crc32c: "T4ZpJw=="},
			{name: "frontend.weights.ort", size: 2093280, crc32c: "QeZo8Q=="},
			{name: "streaming_config.json", size: 509, crc32c: "wH/VeA=="},
			{name: "tokenizer.bin", size: 95323, crc32c: "7zvjaA=="},
		}},
		{arch: archBase, baseURL: "https://download.moonshine.ai/model/base-vi/quantized/base-vi", files: []modelFile{
			{name: "encoder_model.ort", size: 31326816, crc32c: "v/aGsg=="},
			{name: "decoder_model_merged.ort", size: 109424520, crc32c: "0Q2nLA=="},
			{name: "tokenizer.bin", size: 249974, crc32c: "B7s10Q=="},
		}},
	}},
	{code: "uk", englishName: "Ukrainian", models: []catalogModel{
		{arch: archBase, baseURL: "https://download.moonshine.ai/model/base-uk/quantized/base-uk", files: []modelFile{
			{name: "encoder_model.ort", size: 31326816, crc32c: "V6nZMg=="},
			{name: "decoder_model_merged.ort", size: 109424424, crc32c: "mAf6aA=="},
			{name: "tokenizer.bin", size: 249974, crc32c: "B7s10Q=="},
		}},
	}},
	{code: "zh", englishName: "Chinese", models: []catalogModel{
		{arch: archTinyStreaming, baseURL: "https://download.moonshine.ai/model/tiny-streaming-zh/quantized_26_08_24", files: []modelFile{
			{name: "adapter.ort", size: 1318472, crc32c: "hr7j6g=="},
			{name: "cross_kv.ort", size: 1288120, crc32c: "fiGyHw=="},
			{name: "decoder_kv.ort", size: 19717336, crc32c: "rr9O2A=="},
			{name: "encoder.ort", size: 7772792, crc32c: "BHPY3w=="},
			{name: "frontend.model.ort", size: 27608, crc32c: "4gF+xg=="},
			{name: "frontend.weights.ort", size: 2090728, crc32c: "Kv1kYQ=="},
			{name: "streaming_config.json", size: 509, crc32c: "wH/VeA=="},
			{name: "tokenizer.bin", size: 74587, crc32c: "nQJxPQ=="},
		}},
		{arch: archBase, baseURL: "https://download.moonshine.ai/model/base-zh/quantized/base-zh", files: []modelFile{
			{name: "encoder_model.ort", size: 31326816, crc32c: "MHjyrQ=="},
			{name: "decoder_model_merged.ort", size: 109424520, crc32c: "CbdJ2A=="},
			{name: "tokenizer.bin", size: 249974, crc32c: "B7s10Q=="},
		}},
	}},
	{code: "tl", englishName: "Tagalog", models: []catalogModel{
		{arch: archTinyStreaming, baseURL: "https://download.moonshine.ai/model/tiny-streaming-tl/quantized_26_08_24", files: []modelFile{
			{name: "adapter.ort", size: 1318472, crc32c: "Zau+Ig=="},
			{name: "cross_kv.ort", size: 1288120, crc32c: "iwurdw=="},
			{name: "decoder_kv.ort", size: 19717336, crc32c: "jsOHgA=="},
			{name: "encoder.ort", size: 7772792, crc32c: "nx0MFw=="},
			{name: "frontend.model.ort", size: 27608, crc32c: "s9Ro7g=="},
			{name: "frontend.weights.ort", size: 2093288, crc32c: "UEAV9Q=="},
			{name: "streaming_config.json", size: 509, crc32c: "wH/VeA=="},
			{name: "tokenizer.bin", size: 91356, crc32c: "4uONfQ=="},
		}},
	}},
}

// findLanguage is the catalog entry for a language code or English name.
func findLanguage(lang string) (catalogLanguage, bool) {
	wanted := strings.ToLower(lang)
	for _, l := range sttCatalog {
		if l.code == wanted {
			return l, true
		}
	}
	for _, l := range sttCatalog {
		if strings.ToLower(l.englishName) == wanted {
			return l, true
		}
	}
	return catalogLanguage{}, false
}

// catalogLanguages is the language codes with at least one model, in catalog
// order.
func catalogLanguages() []string {
	codes := make([]string, 0, len(sttCatalog))
	for _, l := range sttCatalog {
		codes = append(codes, l.code)
	}
	return codes
}

// catalogLanguagesFriendly lists the languages with their English names.
func catalogLanguagesFriendly() string {
	names := make([]string, 0, len(sttCatalog))
	for _, l := range sttCatalog {
		names = append(names, fmt.Sprintf("%s (%s)", l.code, l.englishName))
	}
	return strings.Join(names, ", ")
}

// findModelInfo is the published model for a language and architecture. A nil
// arch takes the language's first model, its default.
func findModelInfo(lang string, arch *modelArch) (catalogLanguage, catalogModel, error) {
	entry, ok := findLanguage(lang)
	if !ok {
		return catalogLanguage{}, catalogModel{}, fmt.Errorf("%w: %s. Supported languages: %s",
			errLanguageNotFound, lang, catalogLanguagesFriendly())
	}
	if arch == nil {
		return entry, entry.models[0], nil
	}
	available := make([]string, 0, len(entry.models))
	for _, m := range entry.models {
		if m.arch == *arch {
			return entry, m, nil
		}
		available = append(available, m.arch.String())
	}
	return catalogLanguage{}, catalogModel{}, fmt.Errorf("%w: %s and model arch: %s. Available models: %s",
		errModelNotFound, lang, arch, strings.Join(available, ", "))
}

// licenseNotice is logged whenever a model for a language other than English
// is fetched, as the model's publisher asks.
const licenseNotice = "Using a model released under the non-commercial Moonshine Community License. " +
	"See https://www.moonshine.ai/license for details."

// downloadCatalog is the published catalog, downloading each model into a
// local cache the first time it is asked for.
type downloadCatalog struct {
	dl *downloader
}

func (downloadCatalog) supportedLanguages() []string { return catalogLanguages() }

func (downloadCatalog) supportedLanguagesFriendly() string { return catalogLanguagesFriendly() }

// modelForLanguage downloads the model for lang and arch, unless the cache
// already holds it, and returns the directory holding its files.
func (c downloadCatalog) modelForLanguage(
	ctx context.Context, lang string, arch *modelArch,
) (string, modelArch, error) {
	entry, model, err := findModelInfo(lang, arch)
	if err != nil {
		return "", 0, err
	}
	if lang != "en" {
		slog.Info(licenseNotice, "language", entry.code)
	}
	root, err := c.dl.downloadModel(ctx, model)
	if err != nil {
		return "", 0, err
	}
	return root, model.arch, nil
}

// pathCatalog loads a model from a directory the application names, rather
// than one from the published catalog. It still answers which languages
// Moonshine has models for.
type pathCatalog struct {
	path string
}

func (pathCatalog) supportedLanguages() []string { return catalogLanguages() }

func (pathCatalog) supportedLanguagesFriendly() string { return catalogLanguagesFriendly() }

// modelForLanguage returns the named directory, taken to hold the model asked
// for. Asked for no architecture in particular, it takes the language's
// default.
func (c pathCatalog) modelForLanguage(_ context.Context, lang string, arch *modelArch) (string, modelArch, error) {
	if arch != nil {
		return c.path, *arch, nil
	}
	_, model, err := findModelInfo(lang, nil)
	if err != nil {
		return "", 0, err
	}
	return c.path, model.arch, nil
}
