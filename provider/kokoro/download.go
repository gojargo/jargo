package kokoro

import (
	"context"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"path/filepath"
	"time"
)

// Where the v1.0 model and voices files are published.
const (
	ModelURL  = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx"
	VoicesURL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"
)

// The names the files are cached under.
const (
	modelFileName  = "kokoro-v1.0.onnx"
	voicesFileName = "voices-v1.0.bin"
)

// downloadTimeout bounds the wait for the server to answer a download. The
// download itself takes as long as the file does.
const downloadTimeout = 300 * time.Second

// Errors a download reports.
var (
	// errDownloadStatus reports the server answering with an error.
	errDownloadStatus = errors.New("kokoro: unexpected download status")
	// errNoCacheDir reports there being no user cache directory to download
	// the model into.
	errNoCacheDir = errors.New("kokoro: no user cache directory for the model")
)

// DefaultCacheDir is where the model and voices files are downloaded when the
// configuration names no directory: jargo/kokoro-onnx under the user's cache
// directory.
func DefaultCacheDir() (string, error) {
	base, err := os.UserCacheDir()
	if err != nil {
		return "", fmt.Errorf("%w: %w", errNoCacheDir, err)
	}
	return filepath.Join(base, "jargo", "kokoro-onnx"), nil
}

// ensureModelFiles downloads the model and voices files that do not exist.
func ensureModelFiles(ctx context.Context, client *http.Client, modelPath, voicesPath string) error {
	if _, err := os.Stat(modelPath); err != nil {
		if err := downloadFile(ctx, client, ModelURL, modelPath); err != nil {
			return err
		}
	}
	if _, err := os.Stat(voicesPath); err != nil {
		if err := downloadFile(ctx, client, VoicesURL, voicesPath); err != nil {
			return err
		}
	}
	return nil
}

// downloadFile fetches url to dest. The file is written beside dest and takes
// its place in one rename once complete, so dest never holds a file half
// written.
func downloadFile(ctx context.Context, client *http.Client, url, dest string) error {
	slog.Debug(fmt.Sprintf("Downloading %s to %s...", url, dest))
	if err := os.MkdirAll(filepath.Dir(dest), 0o750); err != nil {
		return fmt.Errorf("kokoro: %w", err)
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return fmt.Errorf("kokoro: %w", err)
	}
	if client == nil {
		transport := http.DefaultTransport.(*http.Transport).Clone() //nolint:forcetypeassert // the standard library's
		transport.ResponseHeaderTimeout = downloadTimeout
		client = &http.Client{Transport: transport}
	}
	resp, err := client.Do(req) //nolint:gosec // the address is one of the published model files
	if err != nil {
		return fmt.Errorf("kokoro: downloading %s: %w", url, err)
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("%w: %s: %s", errDownloadStatus, url, resp.Status)
	}

	tmp, err := os.CreateTemp(filepath.Dir(dest), filepath.Base(dest)+".*.partial")
	if err != nil {
		return fmt.Errorf("kokoro: %w", err)
	}
	keep := false
	defer func() {
		if !keep {
			_ = tmp.Close()
			_ = os.Remove(tmp.Name())
		}
	}()
	if _, err := io.Copy(tmp, resp.Body); err != nil {
		return fmt.Errorf("kokoro: downloading %s: %w", url, err)
	}
	if err := tmp.Close(); err != nil {
		return fmt.Errorf("kokoro: %w", err)
	}
	if err := os.Rename(tmp.Name(), dest); err != nil {
		return fmt.Errorf("kokoro: %w", err)
	}
	keep = true
	slog.Debug("Downloaded " + dest)
	return nil
}
