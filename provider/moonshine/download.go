package moonshine

import (
	"context"
	"encoding/base64"
	"encoding/binary"
	"errors"
	"fmt"
	"hash/crc32"
	"io"
	"log/slog"
	"net/http"
	"os"
	"path/filepath"
	"strings"
)

// Errors a model download reports.
var (
	// errDownloadStatus reports the model CDN answering with an error.
	errDownloadStatus = errors.New("moonshine: unexpected download status")
	// errSizeMismatch reports a downloaded file of another size than published.
	errSizeMismatch = errors.New("moonshine: size mismatch")
	// errChecksumMismatch reports a downloaded file whose CRC32C is not the
	// published one.
	errChecksumMismatch = errors.New("moonshine: CRC32C mismatch")
	// errNoCacheDir reports there being no user cache directory to download
	// the models into.
	errNoCacheDir = errors.New("moonshine: no user cache directory for the models")
)

// castagnoli is the CRC32C polynomial table the CDN's checksums use.
//
//nolint:gochecknoglobals // a read-only table, built once
var castagnoli = crc32.MakeTable(crc32.Castagnoli)

// DefaultCacheDir is where models are downloaded when the configuration names
// no directory: jargo/moonshine under the user's cache directory.
func DefaultCacheDir() (string, error) {
	base, err := os.UserCacheDir()
	if err != nil {
		return "", fmt.Errorf("%w: %w", errNoCacheDir, err)
	}
	return filepath.Join(base, "jargo", "moonshine"), nil
}

// downloader fetches published model files into a cache directory, laid out
// by their address: <dir>/<host>/<path>/<file>.
type downloader struct {
	dir    string
	client *http.Client
}

// downloadModel makes sure every file of the model is in the cache,
// downloading those that are not, and returns the directory holding them.
func (d *downloader) downloadModel(ctx context.Context, m catalogModel) (string, error) {
	address := m.baseURL
	if _, rest, ok := strings.Cut(address, "://"); ok {
		address = rest
	}
	root := filepath.Join(d.dir, filepath.FromSlash(address))
	for _, f := range m.files {
		if err := d.downloadFile(ctx, m.baseURL+"/"+f.name, filepath.Join(root, f.name), f); err != nil {
			return "", err
		}
	}
	return root, nil
}

// downloadFile fetches one file to dest, unless dest already holds it. A file
// already in the cache is taken on its size, so a cached model loads without
// reading every byte of it; one just downloaded is checked on its size and its
// CRC32C before it takes its place in the cache, which it does in one rename,
// so the cache never holds a file half written.
func (d *downloader) downloadFile(ctx context.Context, url, dest string, f modelFile) error {
	if info, err := os.Stat(dest); err == nil && info.Mode().IsRegular() && info.Size() == f.size {
		return nil
	}
	if err := os.MkdirAll(filepath.Dir(dest), 0o750); err != nil {
		return fmt.Errorf("moonshine: %w", err)
	}
	slog.Info("downloading Moonshine model file", "url", url, "bytes", f.size)

	tmp, err := os.CreateTemp(filepath.Dir(dest), filepath.Base(dest)+".*.partial")
	if err != nil {
		return fmt.Errorf("moonshine: %w", err)
	}
	keep := false
	defer func() {
		if !keep {
			_ = tmp.Close()
			_ = os.Remove(tmp.Name())
		}
	}()

	sum, size, err := d.fetch(ctx, url, tmp)
	if err != nil {
		return err
	}
	if err := tmp.Close(); err != nil {
		return fmt.Errorf("moonshine: %w", err)
	}
	if size != f.size {
		return fmt.Errorf("%w for %s: expected %d bytes, got %d", errSizeMismatch, f.name, f.size, size)
	}
	if f.crc32c != "" && sum != f.crc32c {
		return fmt.Errorf("%w for %s: expected %s, got %s", errChecksumMismatch, f.name, f.crc32c, sum)
	}
	if err := os.Rename(tmp.Name(), dest); err != nil {
		return fmt.Errorf("moonshine: %w", err)
	}
	keep = true
	return nil
}

// fetch writes what url holds to w, returning its CRC32C, encoded as the CDN
// publishes it, and its length.
func (d *downloader) fetch(ctx context.Context, url string, w io.Writer) (string, int64, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return "", 0, fmt.Errorf("moonshine: %w", err)
	}
	client := d.client
	if client == nil {
		client = http.DefaultClient
	}
	resp, err := client.Do(req) //nolint:gosec // the address comes from the published model table
	if err != nil {
		return "", 0, fmt.Errorf("moonshine: downloading %s: %w", url, err)
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusOK {
		return "", 0, fmt.Errorf("%w: %s: %s", errDownloadStatus, url, resp.Status)
	}
	h := crc32.New(castagnoli)
	n, err := io.Copy(io.MultiWriter(w, h), resp.Body)
	if err != nil {
		return "", 0, fmt.Errorf("moonshine: downloading %s: %w", url, err)
	}
	return encodeCRC32C(h.Sum32()), n, nil
}

// encodeCRC32C encodes a CRC32C the way the CDN publishes it: the four bytes
// big-endian, in base64.
func encodeCRC32C(sum uint32) string {
	var b [4]byte
	binary.BigEndian.PutUint32(b[:], sum)
	return base64.StdEncoding.EncodeToString(b[:])
}
