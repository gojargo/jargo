package moonshine

import (
	"context"
	"encoding/base64"
	"errors"
	"hash/crc32"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync/atomic"
	"testing"
)

// The catalog lists the languages in the library's order, each with its
// default model first.
func TestCatalogListsTheLanguagesInOrder(t *testing.T) {
	want := []string{"ar", "es", "de", "en", "ja", "ko", "vi", "uk", "zh", "tl"}
	if got := catalogLanguages(); !slices.Equal(got, want) {
		t.Errorf("languages = %v, want %v", got, want)
	}
	for lang, def := range map[string]modelArch{
		"ar": archTinyStreaming, "de": archSmallStreaming, "en": archMediumStreaming,
		"ko": archTiny, "uk": archBase, "tl": archTinyStreaming,
	} {
		if _, m, err := findModelInfo(lang, nil); err != nil || m.arch != def {
			t.Errorf("%s default = %v, %v; want %v", lang, m.arch, err, def)
		}
	}
	if !strings.HasPrefix(catalogLanguagesFriendly(), "ar (Arabic), es (Spanish)") {
		t.Errorf("friendly = %q", catalogLanguagesFriendly())
	}
}

// Every model lists the files its kind of model loads, each with a size and a
// four-byte CRC32C, at an address on the model CDN.
func TestCatalogManifestsAreComplete(t *testing.T) {
	streaming := []string{
		"adapter.ort", "cross_kv.ort", "decoder_kv.ort", "encoder.ort",
		"frontend.model.ort", "frontend.weights.ort", "streaming_config.json", "tokenizer.bin",
	}
	legacy := []string{"encoder_model.ort", "decoder_model_merged.ort", "tokenizer.bin"}
	for _, l := range sttCatalog {
		for _, m := range l.models {
			want := legacy
			if strings.Contains(m.arch.String(), "streaming") {
				want = streaming
			}
			var names []string
			for _, f := range m.files {
				names = append(names, f.name)
				sum, err := base64.StdEncoding.DecodeString(f.crc32c)
				if f.size <= 0 || err != nil || len(sum) != 4 {
					t.Errorf("%s %s: size %d, crc32c %q", l.code, f.name, f.size, f.crc32c)
				}
			}
			if !slices.Equal(names, want) {
				t.Errorf("%s %s files = %v", l.code, m.arch, names)
			}
			if !strings.HasPrefix(m.baseURL, "https://download.moonshine.ai/model/") {
				t.Errorf("%s %s address = %q", l.code, m.arch, m.baseURL)
			}
		}
	}
}

func TestFindModelInfoReportsWhatIsNotPublished(t *testing.T) {
	medium := archMediumStreaming
	if _, _, err := findModelInfo("uk", &medium); !errors.Is(err, errModelNotFound) ||
		!strings.Contains(err.Error(), "Available models: base") {
		t.Errorf("uk medium: %v", err)
	}
	if _, _, err := findModelInfo("fr", nil); !errors.Is(err, errLanguageNotFound) {
		t.Errorf("fr: %v", err)
	}
	if l, _, err := findModelInfo("Japanese", nil); err != nil || l.code != "ja" {
		t.Errorf("Japanese = %q, %v; want the language found by its English name", l.code, err)
	}
}

// A model directory the application names is used as it is, for the
// architecture asked for or the language's default.
func TestPathCatalogUsesTheNamedDirectory(t *testing.T) {
	c := pathCatalog{path: "/opt/models/base-en"}
	base := archBase
	if p, a, err := c.modelForLanguage(context.Background(), "en", &base); err != nil || p != c.path || a != archBase {
		t.Errorf("base = %q, %v, %v", p, a, err)
	}
	if _, a, err := c.modelForLanguage(context.Background(), "uk", nil); err != nil || a != archBase {
		t.Errorf("uk default = %v, %v", a, err)
	}
}

// The checksum is encoded as the CDN publishes it.
func TestCRC32CIsEncodedAsPublished(t *testing.T) {
	// The CRC32C check value, 0xE3069283, big-endian in base64.
	if got := encodeCRC32C(crc32.Checksum([]byte("123456789"), castagnoli)); got != "4waSgw==" {
		t.Errorf("crc32c = %q", got)
	}
}

// cdn serves files by name and counts the requests.
func cdn(t *testing.T, files map[string]string) (*httptest.Server, *atomic.Int32) {
	t.Helper()
	var hits atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)
		body, ok := files[filepath.Base(r.URL.Path)]
		if !ok {
			http.NotFound(w, r)
			return
		}
		_, _ = w.Write([]byte(body))
	}))
	t.Cleanup(srv.Close)
	return srv, &hits
}

func published(name, body string) modelFile {
	return modelFile{name: name, size: int64(len(body)), crc32c: encodeCRC32C(crc32.Checksum([]byte(body), castagnoli))}
}

// A model is downloaded into the cache under its address, and once there it is
// not downloaded again.
func TestDownloadModelFillsTheCacheOnce(t *testing.T) {
	srv, hits := cdn(t, map[string]string{"encoder.ort": "encoder", "tokenizer.bin": "tok"})
	dir := t.TempDir()
	d := &downloader{dir: dir, client: srv.Client()}
	m := catalogModel{
		arch: archTinyStreaming, baseURL: srv.URL + "/model/tiny",
		files: []modelFile{published("encoder.ort", "encoder"), published("tokenizer.bin", "tok")},
	}

	root, err := d.downloadModel(context.Background(), m)
	if err != nil {
		t.Fatalf("download: %v", err)
	}
	wantRoot := filepath.Join(dir, strings.TrimPrefix(srv.URL, "http://"), "model", "tiny")
	if root != wantRoot {
		t.Errorf("root = %q", root)
	}
	b, err := os.ReadFile(filepath.Join(root, "encoder.ort")) //nolint:gosec // a file in the test's cache
	if err != nil || string(b) != "encoder" {
		t.Errorf("encoder.ort = %q, %v", b, err)
	}
	if _, err := d.downloadModel(context.Background(), m); err != nil || hits.Load() != 2 {
		t.Errorf("second download: %v after %d requests, want the cache used", err, hits.Load())
	}
	if leftovers, _ := filepath.Glob(filepath.Join(root, "*.partial")); len(leftovers) != 0 {
		t.Errorf("partial files left: %v", leftovers)
	}
}

// A file that is not the one published is refused and kept out of the cache.
func TestDownloadRefusesAFileThatIsNotThePublishedOne(t *testing.T) {
	srv, _ := cdn(t, map[string]string{"encoder.ort": "tampered"})
	for name, f := range map[string]modelFile{
		"size":     published("encoder.ort", "encoder"),
		"checksum": {name: "encoder.ort", size: int64(len("tampered")), crc32c: "AAAAAA=="},
	} {
		t.Run(name, func(t *testing.T) {
			dir := t.TempDir()
			d := &downloader{dir: dir, client: srv.Client()}
			_, err := d.downloadModel(context.Background(), catalogModel{baseURL: srv.URL + "/m", files: []modelFile{f}})
			if !errors.Is(err, errSizeMismatch) && !errors.Is(err, errChecksumMismatch) {
				t.Fatalf("err = %v", err)
			}
			if found, _ := filepath.Glob(filepath.Join(dir, "*", "m", "*")); len(found) != 0 {
				t.Errorf("the cache holds %v", found)
			}
		})
	}
}

func TestDownloadReportsAnErrorStatus(t *testing.T) {
	srv, _ := cdn(t, nil)
	d := &downloader{dir: t.TempDir(), client: srv.Client()}
	m := catalogModel{baseURL: srv.URL, files: []modelFile{published("x", "x")}}
	_, err := d.downloadModel(context.Background(), m)
	if !errors.Is(err, errDownloadStatus) {
		t.Fatalf("err = %v", err)
	}
}
