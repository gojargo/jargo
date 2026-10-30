package hfhub_test

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/gojargo/jargo/internal/hfhub"
)

const commit = "2ba272b19642f77bf72e2e47fcf1b9d1d65c398d"

// hub serves one repository at one commit, the way the Hub answers: the
// revision API, and a resolve URL that reports each file's etag and redirects
// a large file to where its content is served.
type hub struct {
	files map[string]string
	// corrupt serves content that does not match the etag.
	corrupt bool

	mu   sync.Mutex
	gets []string
}

func (h *hub) etag(name string) string {
	content := h.files[name]
	if strings.HasSuffix(name, ".onnx") {
		sum := sha256.Sum256([]byte(content))
		return hex.EncodeToString(sum[:])
	}
	return fmt.Sprintf("%040x", len(content))
}

func (h *hub) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if rev, ok := strings.CutPrefix(r.URL.Path, "/api/models/org/model/revision/"); ok {
		if rev != "release-1" && rev != commit {
			http.NotFound(w, r)
			return
		}
		_, _ = fmt.Fprintf(w, `{"sha": %q}`, commit)
		return
	}
	if name, ok := strings.CutPrefix(r.URL.Path, "/cdn/"); ok {
		h.mu.Lock()
		h.gets = append(h.gets, name)
		h.mu.Unlock()
		content := h.files[name]
		if h.corrupt {
			content += "!"
		}
		_, _ = w.Write([]byte(content))
		return
	}
	name, ok := strings.CutPrefix(r.URL.Path, "/org/model/resolve/"+commit+"/")
	if _, exists := h.files[name]; !ok || !exists {
		http.NotFound(w, r)
		return
	}
	w.Header().Set("X-Repo-Commit", commit)
	w.Header().Set("X-Linked-Etag", `"`+h.etag(name)+`"`)
	http.Redirect(w, r, "/cdn/"+name, http.StatusFound) //nolint:gosec // a test server redirecting to itself
}

func (h *hub) downloads() []string {
	h.mu.Lock()
	defer h.mu.Unlock()
	return append([]string(nil), h.gets...)
}

func newHub(t *testing.T) (*hub, *httptest.Server) {
	t.Helper()
	h := &hub{files: map[string]string{
		"reader/config.json":     `{"max_len": 160}`,
		"reader/model_fp16.onnx": "weights",
		"other/config.json":      "{}",
	}}
	srv := httptest.NewServer(h)
	t.Cleanup(srv.Close)
	return h, srv
}

func config(srv *httptest.Server, cache string) hfhub.Config {
	return hfhub.Config{
		Repo:     "org/model",
		Revision: "release-1",
		Files:    []string{"reader/config.json", "reader/model_fp16.onnx"},
		CacheDir: cache,
		Endpoint: srv.URL,
		Client:   srv.Client(),
	}
}

func TestDownloadLaysTheFilesOutAsTheHubClientDoes(t *testing.T) {
	_, srv := newHub(t)
	cache := t.TempDir()

	dir, err := hfhub.Download(context.Background(), config(srv, cache))
	if err != nil {
		t.Fatal(err)
	}

	repo := filepath.Join(cache, "models--org--model")
	if want := filepath.Join(repo, "snapshots", commit); dir != want {
		t.Errorf("snapshot = %s, want %s", dir, want)
	}
	ref := filepath.Join(repo, "refs", "release-1")
	if b, err := os.ReadFile(ref); err != nil || string(b) != commit { //nolint:gosec // the test's cache
		t.Errorf("refs/release-1 = %q (%v), want the commit", b, err)
	}
	for name, want := range map[string]string{
		"reader/config.json":     `{"max_len": 160}`,
		"reader/model_fp16.onnx": "weights",
	} {
		path := filepath.Join(dir, filepath.FromSlash(name))
		if b, err := os.ReadFile(path); err != nil || string(b) != want { //nolint:gosec // the test's cache
			t.Errorf("%s = %q (%v), want %q", name, b, err, want)
		}
		if target, err := os.Readlink(path); err != nil || !strings.Contains(target, "blobs") {
			t.Errorf("%s links to %q (%v), want its blob", name, target, err)
		}
	}
	if _, err := os.Stat(filepath.Join(dir, "other", "config.json")); err == nil {
		t.Error("a file that was not asked for was downloaded")
	}
}

func TestDownloadUsesTheFilesAlreadyCached(t *testing.T) {
	h, srv := newHub(t)
	cache := t.TempDir()
	for range 2 {
		if _, err := hfhub.Download(context.Background(), config(srv, cache)); err != nil {
			t.Fatal(err)
		}
	}
	if got := h.downloads(); len(got) != 2 {
		t.Errorf("downloaded %v, want each file once", got)
	}
}

func TestDownloadFallsBackToTheCacheWhenTheHubIsUnreachable(t *testing.T) {
	_, srv := newHub(t)
	cache := t.TempDir()
	want, err := hfhub.Download(context.Background(), config(srv, cache))
	if err != nil {
		t.Fatal(err)
	}
	srv.Close()

	got, err := hfhub.Download(context.Background(), config(srv, cache))
	if err != nil || got != want {
		t.Errorf("offline = %s (%v), want the cached snapshot %s", got, err, want)
	}

	cfg := config(srv, cache)
	cfg.Files = append(cfg.Files, "other/config.json")
	if _, err := hfhub.Download(context.Background(), cfg); !errors.Is(err, hfhub.ErrNotCached) {
		t.Errorf("offline with a file missing = %v, want ErrNotCached", err)
	}
}

func TestDownloadRejectsContentThatDoesNotMatchItsDigest(t *testing.T) {
	h, srv := newHub(t)
	h.corrupt = true
	cache := t.TempDir()

	_, err := hfhub.Download(context.Background(), config(srv, cache))
	if !errors.Is(err, hfhub.ErrDigest) {
		t.Fatalf("err = %v, want ErrDigest", err)
	}
	entries, _ := os.ReadDir(filepath.Join(cache, "models--org--model", "blobs"))
	for _, e := range entries {
		if strings.HasSuffix(e.Name(), ".onnx") || len(e.Name()) == 64 {
			t.Errorf("blob %s kept after a bad download", e.Name())
		}
	}
}

func TestDownloadRejectsAPathOutsideTheSnapshot(t *testing.T) {
	_, srv := newHub(t)
	cfg := config(srv, t.TempDir())
	cfg.Files = []string{"../escape"}
	if _, err := hfhub.Download(context.Background(), cfg); !errors.Is(err, hfhub.ErrPath) {
		t.Errorf("err = %v, want ErrPath", err)
	}
}

func TestDownloadReportsAFileTheRepositoryDoesNotHave(t *testing.T) {
	_, srv := newHub(t)
	cfg := config(srv, t.TempDir())
	cfg.Files = []string{"reader/missing.json"}
	if _, err := hfhub.Download(context.Background(), cfg); !errors.Is(err, hfhub.ErrStatus) {
		t.Errorf("err = %v, want ErrStatus", err)
	}
}
