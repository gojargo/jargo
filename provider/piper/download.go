package piper

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"sync"

	"github.com/gojargo/jargo/errclass"
)

// huggingFace is the Hugging Face Hub, which hosts the rhasspy/piper-voices
// repository the voices are downloaded from.
const huggingFace = "https://huggingface.co"

// voiceExtensions are the files a voice is made of: the model and its
// configuration.
//
//nolint:gochecknoglobals // a read-only list
var voiceExtensions = []string{".onnx", ".onnx.json"}

// voicePattern is how a voice is named: <language>_<region>-<name>-<quality>,
// like en_US-lessac-medium.
var voicePattern = regexp.MustCompile(`^([^-]+)_([^-]+)-([^-]+)-(.+)$`)

// Errors a voice download reports.
var (
	// errVoiceName reports a voice named neither the way voices are, nor by
	// the old name of a voice that was renamed.
	errVoiceName = errors.New("piper: voice does not match the voice naming pattern")
	// errDownloadStatus reports the repository answering with an error.
	errDownloadStatus = errors.New("piper: unexpected download status")
	// errShortDownload reports a response cut short of its announced length.
	errShortDownload = errors.New("piper: download cut short")
	// errNoCacheDir reports there being no user cache directory to download
	// the voices into.
	errNoCacheDir = errors.New("piper: no user cache directory for the voices")
)

// DefaultCacheDir is where voices are downloaded when the configuration names
// no directory: jargo/piper under the user's cache directory.
func DefaultCacheDir() (string, error) {
	base, err := os.UserCacheDir()
	if err != nil {
		return "", fmt.Errorf("%w: %w", errNoCacheDir, err)
	}
	return filepath.Join(base, "jargo", "piper"), nil
}

// The voices.json index downloaded so far, by the address of the repository it
// came from, so it is downloaded once a process.
//
//nolint:gochecknoglobals // a process-wide cache of a published index
var (
	voicesIndexMu sync.Mutex
	voicesIndex   = map[string]map[string]voiceEntry{}
)

// voiceEntry is what the service reads of a voice's voices.json entry: the
// names it was published under before it was renamed.
type voiceEntry struct {
	Aliases []string `json:"aliases"`
}

// voiceDownloader fetches voices from the rhasspy/piper-voices repository.
type voiceDownloader struct {
	// endpoint is the address of the Hub, huggingFace outside tests.
	endpoint string
	client   *http.Client
}

// downloadVoice downloads the model and the configuration of voice into dir,
// returning the voice's current name. A voice that was renamed is found under
// its new name, by the old one it was published under, and downloaded as
// that. Files already downloaded are kept unless force is set.
func (d *voiceDownloader) downloadVoice(ctx context.Context, voice, dir string, force bool) (string, error) {
	voice = strings.TrimSpace(voice)

	current, err := d.currentName(ctx, voice, dir, force)
	if err != nil || current == voice {
		return current, err
	}

	slog.Warn("Piper voice has been renamed, downloading that instead", "voice", voice, "renamed", current)
	if err := d.downloadByName(ctx, current, dir, force); err != nil {
		return "", err
	}
	return current, nil
}

// currentName downloads voice when it is named the way voices are and is
// published under that name, returning it. Otherwise it is the current name of
// the voice that was published as voice, for the caller to download.
func (d *voiceDownloader) currentName(ctx context.Context, voice, dir string, force bool) (string, error) {
	if !voicePattern.MatchString(voice) {
		// Not a current-style name, but it may be the old name of a voice
		// that was renamed (like de-karlsson-low).
		current := d.resolveAlias(ctx, voice)
		if current == "" {
			return "", fmt.Errorf("%w: voice '%s' did not match pattern: "+
				"<language>-<name>-<quality> like 'en_US-lessac-medium'", errVoiceName, voice)
		}
		return current, nil
	}
	err := d.downloadByName(ctx, voice, dir, force)
	if err == nil {
		return voice, nil
	}
	if status, ok := errclass.ExtractHTTPStatusCode(err); !ok || status != http.StatusNotFound {
		return "", err
	}
	// The voice may have been renamed since this name was published.
	current := d.resolveAlias(ctx, voice)
	if current == "" {
		return "", err
	}
	return current, nil
}

// downloadByName downloads a voice by its current name. Every file is
// downloaded into a temporary directory inside dir, so the final rename stays
// on one filesystem, and the files already there are kept intact until every
// download has succeeded.
func (d *voiceDownloader) downloadByName(ctx context.Context, voice, dir string, force bool) error {
	m := voicePattern.FindStringSubmatch(voice)
	if m == nil {
		return fmt.Errorf("%w: %s", errVoiceName, voice)
	}
	langFamily, langCode, name, quality := m[1], m[1]+"_"+m[2], m[3], m[4]
	code := langCode + "-" + name + "-" + quality

	var toDownload []string
	for _, ext := range voiceExtensions {
		if force || needsDownload(filepath.Join(dir, code+ext)) {
			toDownload = append(toDownload, ext)
		}
	}
	if len(toDownload) > 0 {
		tmp, err := os.MkdirTemp(dir, ".piper-download-")
		if err != nil {
			return fmt.Errorf("piper: %w", err)
		}
		defer func() { _ = os.RemoveAll(tmp) }()
		for _, ext := range toDownload {
			url := fmt.Sprintf("%s/rhasspy/piper-voices/resolve/main/%s/%s/%s/%s/%s%s?download=true",
				d.endpoint, langFamily, langCode, name, quality, code, ext)
			slog.Debug("downloading Piper voice file", "url", url, "path", filepath.Join(dir, code+ext))
			if err := d.downloadFile(ctx, url, filepath.Join(tmp, code+ext)); err != nil {
				return err
			}
		}
		for _, ext := range toDownload {
			if err := os.Rename(filepath.Join(tmp, code+ext), filepath.Join(dir, code+ext)); err != nil {
				return fmt.Errorf("piper: %w", err)
			}
		}
	}
	slog.Info("downloaded Piper voice", "voice", voice)
	return nil
}

// downloadFile writes what url holds to path, failing when the response is
// cut short of the length it announced, so a truncated file is never taken
// for a complete one.
func (d *voiceDownloader) downloadFile(ctx context.Context, url, path string) error {
	resp, err := d.get(ctx, url)
	if err != nil {
		return err
	}
	defer func() { _ = resp.Body.Close() }()

	f, err := os.Create(path) //nolint:gosec // a file in the download directory
	if err != nil {
		return fmt.Errorf("piper: %w", err)
	}
	n, err := io.Copy(f, resp.Body)
	if cerr := f.Close(); err == nil {
		err = cerr
	}
	if err != nil {
		return fmt.Errorf("piper: downloading %s: %w", url, err)
	}
	if length := resp.Header.Get("Content-Length"); length != "" {
		if want, perr := strconv.ParseInt(length, 10, 64); perr == nil && n < want {
			return fmt.Errorf("%w: downloaded %d of %d byte(s) from '%s'", errShortDownload, n, want, url)
		}
	}
	return nil
}

// get fetches url, reporting an answer other than 200 as an error carrying its
// status.
func (d *voiceDownloader) get(ctx context.Context, url string) (*http.Response, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, fmt.Errorf("piper: %w", err)
	}
	client := d.client
	if client == nil {
		client = http.DefaultClient
	}
	resp, err := client.Do(req) //nolint:gosec // the address of the voices repository
	if err != nil {
		return nil, fmt.Errorf("piper: downloading %s: %w", url, err)
	}
	if resp.StatusCode != http.StatusOK {
		_ = resp.Body.Close()
		return nil, errclass.NewHTTPStatusError(resp.StatusCode,
			fmt.Errorf("%w: %s: %s", errDownloadStatus, url, resp.Status))
	}
	return resp, nil
}

// resolveAlias is the current name of a voice that was renamed, or "" when no
// voice was ever published as voice. Renamed voices keep their old names in
// the aliases of their voices.json entry.
func (d *voiceDownloader) resolveAlias(ctx context.Context, voice string) string {
	voices, err := d.voices(ctx)
	if err != nil {
		// Offline, or voices.json is unreachable. The caller already has a
		// download failure to report, so it is not masked with this one.
		slog.Debug("could not download voices.json to resolve the Piper voice", "voice", voice, "err", err)
		return ""
	}
	for current, entry := range voices {
		if slices.Contains(entry.Aliases, voice) {
			return current
		}
	}
	return ""
}

// voices is the contents of voices.json, downloaded once a process.
func (d *voiceDownloader) voices(ctx context.Context) (map[string]voiceEntry, error) {
	voicesIndexMu.Lock()
	defer voicesIndexMu.Unlock()
	if v, ok := voicesIndex[d.endpoint]; ok {
		return v, nil
	}
	url := d.endpoint + "/rhasspy/piper-voices/resolve/main/voices.json?download=true"
	slog.Debug("downloading voices.json", "url", url)
	resp, err := d.get(ctx, url)
	if err != nil {
		return nil, err
	}
	defer func() { _ = resp.Body.Close() }()
	var v map[string]voiceEntry
	if err := json.NewDecoder(resp.Body).Decode(&v); err != nil {
		return nil, fmt.Errorf("piper: reading voices.json: %w", err)
	}
	voicesIndex[d.endpoint] = v
	return v, nil
}

// needsDownload reports whether the file at path has to be downloaded: it does
// not exist, or is empty.
func needsDownload(path string) bool {
	info, err := os.Stat(path)
	return err != nil || info.Size() == 0
}
