// Package hfhub downloads a model's files from the Hugging Face Hub into a
// local cache, laid out as the Hub's own client lays it out, so files fetched
// by either one are found by the other: a model downloaded ahead with
// `huggingface-cli download`, such as while building an image, is used in
// place without going to the network.
//
// The cache holds, for each repository, the commit each revision named last
// resolved to (refs/<revision>), the files by their content hash (blobs/), and
// each commit's files at their paths in the repository (snapshots/<commit>/),
// linked to their blobs.
package hfhub

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"regexp"
	"strings"
)

// DefaultEndpoint is the Hugging Face Hub.
const DefaultEndpoint = "https://huggingface.co"

// Config says what to download and where to keep it.
type Config struct {
	// Repo is the model repository, such as "org/model".
	Repo string
	// Revision is a branch, a tag or a commit. Empty is "main".
	Revision string
	// Files are the paths in the repository to download.
	Files []string
	// CacheDir is the cache. Empty is the Hub client's default,
	// ~/.cache/huggingface/hub.
	CacheDir string
	// Endpoint is the Hub's address. Empty is DefaultEndpoint.
	Endpoint string
	// Client makes the requests. Nil is http.DefaultClient.
	Client *http.Client
}

var (
	// ErrNotCached is returned when the Hub cannot be reached and the cache
	// does not hold every file asked for.
	ErrNotCached = errors.New("hfhub: the files are not in the cache and the hub could not be reached")
	// ErrDigest is returned when a downloaded file is not the file the Hub
	// described.
	ErrDigest = errors.New("hfhub: downloaded file does not match its digest")
	// ErrStatus is returned when the Hub answers a request with an error.
	ErrStatus = errors.New("hfhub: unexpected status")
	// ErrPath is returned for a file path that would leave the snapshot.
	ErrPath = errors.New("hfhub: invalid file path")
)

var (
	commitPattern = regexp.MustCompile(`^[0-9a-f]{40}$`)
	sha256Pattern = regexp.MustCompile(`^[0-9a-f]{64}$`)
)

// Download makes sure the files are in the cache, downloading those that are
// not, and returns the snapshot folder that holds them at their paths in the
// repository.
//
// When the Hub cannot be reached, the commit the revision last resolved to is
// used from the cache, provided it holds every file.
func Download(ctx context.Context, cfg Config) (string, error) {
	cfg, err := withDefaults(cfg)
	if err != nil {
		return "", err
	}
	repoDir := filepath.Join(cfg.CacheDir, "models--"+strings.ReplaceAll(cfg.Repo, "/", "--"))

	commit, err := resolve(ctx, cfg)
	if err != nil {
		// Offline, or the Hub is down: the cache may still hold the revision.
		cached, ok := cachedCommit(repoDir, cfg.Revision)
		if ok && complete(snapshotDir(repoDir, cached), cfg.Files) {
			return snapshotDir(repoDir, cached), nil
		}
		return "", errors.Join(ErrNotCached, err)
	}
	if commit != cfg.Revision {
		if err := writeRef(repoDir, cfg.Revision, commit); err != nil {
			return "", err
		}
	}
	snapshot := snapshotDir(repoDir, commit)
	for _, f := range cfg.Files {
		path := filepath.Join(snapshot, filepath.FromSlash(f))
		if _, err := os.Stat(path); err == nil {
			continue
		}
		if err := fetch(ctx, cfg, repoDir, commit, f, path); err != nil {
			return "", err
		}
	}
	return snapshot, nil
}

// withDefaults fills in what the config leaves empty, and checks its paths.
func withDefaults(cfg Config) (Config, error) {
	if cfg.Revision == "" {
		cfg.Revision = "main"
	}
	if cfg.Endpoint == "" {
		cfg.Endpoint = DefaultEndpoint
	}
	if cfg.Client == nil {
		cfg.Client = http.DefaultClient
	}
	if cfg.CacheDir == "" {
		home, err := os.UserHomeDir()
		if err != nil {
			return cfg, fmt.Errorf("hfhub: locating the cache: %w", err)
		}
		cfg.CacheDir = filepath.Join(home, ".cache", "huggingface", "hub")
	}
	for _, f := range cfg.Files {
		if !filepath.IsLocal(filepath.FromSlash(f)) {
			return cfg, fmt.Errorf("%w: %q", ErrPath, f)
		}
	}
	return cfg, nil
}

func snapshotDir(repoDir, commit string) string {
	return filepath.Join(repoDir, "snapshots", commit)
}

// complete reports whether the snapshot holds every file.
func complete(snapshot string, files []string) bool {
	for _, f := range files {
		if _, err := os.Stat(filepath.Join(snapshot, filepath.FromSlash(f))); err != nil {
			return false
		}
	}
	return true
}

// cachedCommit is the commit the cache holds for the revision: the revision
// itself when it is a commit, or the one it last resolved to.
func cachedCommit(repoDir, revision string) (string, bool) {
	if commitPattern.MatchString(revision) {
		return revision, true
	}
	ref := filepath.Join(repoDir, "refs", filepath.FromSlash(revision))
	b, err := os.ReadFile(ref) //nolint:gosec // a path in the cache
	if err != nil {
		return "", false
	}
	commit := strings.TrimSpace(string(b))
	return commit, commitPattern.MatchString(commit)
}

func writeRef(repoDir, revision, commit string) error {
	path := filepath.Join(repoDir, "refs", filepath.FromSlash(revision))
	if !filepath.IsLocal(filepath.FromSlash(revision)) {
		return fmt.Errorf("%w: revision %q", ErrPath, revision)
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o750); err != nil {
		return fmt.Errorf("hfhub: %w", err)
	}
	if err := os.WriteFile(path, []byte(commit), 0o600); err != nil {
		return fmt.Errorf("hfhub: %w", err)
	}
	return nil
}

// resolve asks the Hub which commit the revision names.
func resolve(ctx context.Context, cfg Config) (string, error) {
	u := fmt.Sprintf("%s/api/models/%s/revision/%s", cfg.Endpoint, cfg.Repo, url.PathEscape(cfg.Revision))
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, u, nil)
	if err != nil {
		return "", fmt.Errorf("hfhub: %w", err)
	}
	resp, err := cfg.Client.Do(req)
	if err != nil {
		return "", fmt.Errorf("hfhub: resolving %s at %s: %w", cfg.Repo, cfg.Revision, err)
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("%w: resolving %s at %s: %s", ErrStatus, cfg.Repo, cfg.Revision, resp.Status)
	}
	var info struct {
		SHA string `json:"sha"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&info); err != nil {
		return "", fmt.Errorf("hfhub: resolving %s at %s: %w", cfg.Repo, cfg.Revision, err)
	}
	if !commitPattern.MatchString(info.SHA) {
		return "", fmt.Errorf("%w: resolving %s at %s: no commit in the answer", ErrStatus, cfg.Repo, cfg.Revision)
	}
	return info.SHA, nil
}

// fetch downloads one file into the blobs, by its content hash, and links it
// at its path in the snapshot.
func fetch(ctx context.Context, cfg Config, repoDir, commit, file, path string) error {
	u := fmt.Sprintf("%s/%s/resolve/%s/%s", cfg.Endpoint, cfg.Repo, commit, file)
	etag, err := blobName(ctx, cfg, u)
	if err != nil {
		return err
	}
	blobs := filepath.Join(repoDir, "blobs")
	if err = os.MkdirAll(blobs, 0o750); err != nil {
		return fmt.Errorf("hfhub: %w", err)
	}
	blob := filepath.Join(blobs, etag)
	if _, statErr := os.Stat(blob); statErr != nil {
		if err = downloadBlob(ctx, cfg, u, blob, etag); err != nil {
			return err
		}
	}
	if err = os.MkdirAll(filepath.Dir(path), 0o750); err != nil {
		return fmt.Errorf("hfhub: %w", err)
	}
	rel, err := filepath.Rel(filepath.Dir(path), blob)
	if err != nil {
		return fmt.Errorf("hfhub: %w", err)
	}
	if err := os.Symlink(rel, path); err != nil {
		// Where links cannot be made, the snapshot holds a copy.
		return copyFile(blob, path)
	}
	return nil
}

// blobName is the name the file's blob is kept under: its content hash for a
// large file, its Git object id for another, as the Hub reports them before
// any redirect to where the content is served.
func blobName(ctx context.Context, cfg Config, u string) (string, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodHead, u, nil)
	if err != nil {
		return "", fmt.Errorf("hfhub: %w", err)
	}
	client := *cfg.Client
	client.CheckRedirect = func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }
	resp, err := client.Do(req)
	if err != nil {
		return "", fmt.Errorf("hfhub: %s: %w", u, err)
	}
	_ = resp.Body.Close()
	if resp.StatusCode >= http.StatusBadRequest {
		return "", fmt.Errorf("%w: %s: %s", ErrStatus, u, resp.Status)
	}
	etag := resp.Header.Get("X-Linked-Etag")
	if etag == "" {
		etag = resp.Header.Get("Etag")
	}
	etag = strings.Trim(strings.TrimPrefix(etag, "W/"), `"`)
	if etag == "" || !filepath.IsLocal(etag) || strings.ContainsAny(etag, `/\`) {
		return "", fmt.Errorf("%w: %s: no usable etag", ErrStatus, u)
	}
	return etag, nil
}

// downloadBlob writes the file to blob, through a temporary file so that an
// interrupted download never leaves a partial blob behind. A large file's etag
// is its SHA-256, which the content is checked against.
func downloadBlob(ctx context.Context, cfg Config, u, blob, etag string) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, u, nil)
	if err != nil {
		return fmt.Errorf("hfhub: %w", err)
	}
	resp, err := cfg.Client.Do(req)
	if err != nil {
		return fmt.Errorf("hfhub: %s: %w", u, err)
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("%w: %s: %s", ErrStatus, u, resp.Status)
	}
	tmp, err := os.CreateTemp(filepath.Dir(blob), filepath.Base(blob)+".*.incomplete")
	if err != nil {
		return fmt.Errorf("hfhub: %w", err)
	}
	defer func() { _ = os.Remove(tmp.Name()) }()
	digest := sha256.New()
	_, err = io.Copy(io.MultiWriter(tmp, digest), resp.Body)
	if cerr := tmp.Close(); err == nil {
		err = cerr
	}
	if err != nil {
		return fmt.Errorf("hfhub: %s: %w", u, err)
	}
	if sha256Pattern.MatchString(etag) && hex.EncodeToString(digest.Sum(nil)) != etag {
		return fmt.Errorf("%w: %s", ErrDigest, u)
	}
	if err := os.Rename(tmp.Name(), blob); err != nil {
		return fmt.Errorf("hfhub: %w", err)
	}
	return nil
}

func copyFile(from, to string) error {
	in, err := os.Open(from) //nolint:gosec // a blob in the cache
	if err != nil {
		return fmt.Errorf("hfhub: %w", err)
	}
	defer func() { _ = in.Close() }()
	out, err := os.Create(to) //nolint:gosec // a path in the cache
	if err != nil {
		return fmt.Errorf("hfhub: %w", err)
	}
	_, err = io.Copy(out, in)
	if cerr := out.Close(); err == nil {
		err = cerr
	}
	if err != nil {
		return fmt.Errorf("hfhub: %w", err)
	}
	return nil
}
