package piper

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"sync"
	"testing"
)

// Tests for the voice download, against a fake rhasspy/piper-voices
// repository.

// voicesRepo serves voice files at their repository paths, and voices.json.
type voicesRepo struct {
	files map[string]string
	index string
	// short announces more bytes than it sends.
	short bool

	mu   sync.Mutex
	gets []string
}

func (r *voicesRepo) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	r.mu.Lock()
	r.gets = append(r.gets, req.URL.Path)
	r.mu.Unlock()
	if req.URL.Query().Get("download") != "true" {
		http.Error(w, "missing download=true", http.StatusBadRequest)
		return
	}
	path := strings.TrimPrefix(req.URL.Path, "/rhasspy/piper-voices/resolve/main/")
	if path == "voices.json" && r.index != "" {
		_, _ = w.Write([]byte(r.index))
		return
	}
	content, ok := r.files[path]
	if !ok {
		http.NotFound(w, req)
		return
	}
	if r.short {
		w.Header().Set("Content-Length", strconv.Itoa(len(content)+10))
	}
	_, _ = w.Write([]byte(content))
}

// newDownloader is a downloader against repo, with a voices.json index of its
// own, since the index is kept once per address.
func newDownloader(t *testing.T, repo *voicesRepo) *voiceDownloader {
	t.Helper()
	srv := httptest.NewServer(repo)
	t.Cleanup(srv.Close)
	return &voiceDownloader{endpoint: srv.URL, client: srv.Client()}
}

// ryanFiles are en_US-ryan-high's files at their repository paths.
//
//nolint:gochecknoglobals // a read-only fixture
var ryanFiles = map[string]string{
	"en/en_US/ryan/high/en_US-ryan-high.onnx":      "model",
	"en/en_US/ryan/high/en_US-ryan-high.onnx.json": `{"audio": {"sample_rate": 22050}}`,
}

func readFile(t *testing.T, path string) string {
	t.Helper()
	b, err := os.ReadFile(path) //nolint:gosec // a test file
	if err != nil {
		t.Fatal(err)
	}
	return string(b)
}

// A voice's model and configuration are fetched from the repository layout
// into the download directory, and nothing is left beside them.
func TestDownloadVoiceFetchesTheModelAndItsConfiguration(t *testing.T) {
	repo := &voicesRepo{files: ryanFiles}
	dir := t.TempDir()

	name, err := newDownloader(t, repo).downloadVoice(context.Background(), " en_US-ryan-high ", dir, false)
	if err != nil {
		t.Fatalf("downloadVoice: %v", err)
	}
	if name != "en_US-ryan-high" {
		t.Errorf("name = %q", name)
	}
	if got := readFile(t, filepath.Join(dir, "en_US-ryan-high.onnx")); got != "model" {
		t.Errorf("model = %q", got)
	}
	if got := readFile(t, filepath.Join(dir, "en_US-ryan-high.onnx.json")); !strings.Contains(got, "22050") {
		t.Errorf("configuration = %q", got)
	}
	entries, _ := os.ReadDir(dir)
	if len(entries) != 2 {
		t.Errorf("the directory holds %d entries, want the two files", len(entries))
	}
}

// Files already downloaded are kept unless a download is forced, and an empty
// one is downloaded again.
func TestDownloadVoiceKeepsExistingFilesUnlessForced(t *testing.T) {
	repo := &voicesRepo{files: ryanFiles}
	d := newDownloader(t, repo)
	dir := t.TempDir()
	model := filepath.Join(dir, "en_US-ryan-high.onnx")
	if err := os.WriteFile(model, []byte("cached"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(model+".json", nil, 0o600); err != nil {
		t.Fatal(err)
	}

	if _, err := d.downloadVoice(context.Background(), "en_US-ryan-high", dir, false); err != nil {
		t.Fatalf("downloadVoice: %v", err)
	}
	if got := readFile(t, model); got != "cached" {
		t.Errorf("model = %q, want the cached one kept", got)
	}
	if got := readFile(t, model+".json"); got == "" {
		t.Error("the empty configuration was not downloaded again")
	}
	configPath := "/rhasspy/piper-voices/resolve/main/en/en_US/ryan/high/en_US-ryan-high.onnx.json"
	if !slices.Equal(repo.gets, []string{configPath}) {
		t.Errorf("fetched %v, want only the configuration", repo.gets)
	}

	if _, err := d.downloadVoice(context.Background(), "en_US-ryan-high", dir, true); err != nil {
		t.Fatalf("downloadVoice: %v", err)
	}
	if got := readFile(t, model); got != "model" {
		t.Errorf("model = %q, want it downloaded again when forced", got)
	}
}

// A voice that was renamed is found by its old name in voices.json and
// downloaded under its current one, whether the old name still matches the
// naming pattern or not.
func TestDownloadVoiceResolvesARenamedVoice(t *testing.T) {
	repo := &voicesRepo{
		files: map[string]string{
			"de/de_DE/karlsson/low/de_DE-karlsson-low.onnx":      "model",
			"de/de_DE/karlsson/low/de_DE-karlsson-low.onnx.json": "{}",
		},
		index: `{"de_DE-karlsson-low": {"aliases": ["de-karlsson-low", "de_DE-karl-low"]}}`,
	}
	d := newDownloader(t, repo)
	for _, old := range []string{"de-karlsson-low", "de_DE-karl-low"} {
		dir := t.TempDir()
		name, err := d.downloadVoice(context.Background(), old, dir, false)
		if err != nil {
			t.Fatalf("downloadVoice(%s): %v", old, err)
		}
		if name != "de_DE-karlsson-low" {
			t.Errorf("downloadVoice(%s) = %q, want the current name", old, name)
		}
		if got := readFile(t, filepath.Join(dir, "de_DE-karlsson-low.onnx")); got != "model" {
			t.Errorf("model = %q", got)
		}
	}
}

// A voice that is neither published nor an old name is reported, a malformed
// name as not matching the pattern.
func TestDownloadVoiceReportsAnUnknownVoice(t *testing.T) {
	repo := &voicesRepo{files: ryanFiles, index: `{"en_US-ryan-high": {"aliases": []}}`}
	d := newDownloader(t, repo)

	_, err := d.downloadVoice(context.Background(), "en_US-nobody-high", t.TempDir(), false)
	if !errors.Is(err, errDownloadStatus) || !strings.Contains(err.Error(), "404") {
		t.Errorf("err = %v, want the 404 reported", err)
	}
	_, err = d.downloadVoice(context.Background(), "nobody", t.TempDir(), false)
	if !errors.Is(err, errVoiceName) || !strings.Contains(err.Error(), "'en_US-lessac-medium'") {
		t.Errorf("err = %v, want the naming pattern reported", err)
	}
}

// A response cut short of its announced length fails the download, and leaves
// no file behind to be taken for a complete one.
func TestDownloadVoiceRejectsAShortResponse(t *testing.T) {
	repo := &voicesRepo{files: ryanFiles, short: true}
	dir := t.TempDir()
	_, err := newDownloader(t, repo).downloadVoice(context.Background(), "en_US-ryan-high", dir, false)
	if err == nil {
		t.Fatal("downloadVoice: want an error for a short response")
	}
	if entries, _ := os.ReadDir(dir); len(entries) != 0 {
		t.Errorf("the directory holds %d entries after a failed download", len(entries))
	}
}

func TestDefaultCacheDir(t *testing.T) {
	dir, err := DefaultCacheDir()
	if err != nil {
		t.Skipf("no user cache directory: %v", err)
	}
	if !strings.HasSuffix(dir, filepath.Join("jargo", "piper")) {
		t.Errorf("DefaultCacheDir = %q", dir)
	}
}
