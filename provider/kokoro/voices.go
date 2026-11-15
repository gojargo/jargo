package kokoro

import (
	"archive/zip"
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"strconv"
	"strings"
)

// styleDim is the length of a voice's style vector: the model reads one of
// 256 numbers.
const styleDim = 256

// Errors reading the voices file reports.
var (
	// errNpy reports an array in the voices file that is not one the model
	// can read.
	errNpy = errors.New("kokoro: unreadable voice array")
)

// voice is one voice's styles: a style vector per phoneme count, row n-1 for n
// phonemes, as many rows as the voice has.
type voice struct {
	rows int
	// styles is the rows laid end to end, styleDim numbers each.
	styles []float32
}

// style is the style vector for an utterance of length phonemes: one per
// phoneme count, so n phonemes use row n-1, and the longest row past it.
func (v voice) style(length int) []float32 {
	row := min(length, v.rows) - 1
	return v.styles[row*styleDim : (row+1)*styleDim]
}

// voices is the voices of a voices file, by name.
type voices map[string]voice

// lookup is the voice named name, which may also be given with its .npy file
// suffix.
func (vs voices) lookup(name string) (voice, bool) {
	if v, ok := vs[name]; ok {
		return v, true
	}
	v, ok := vs[strings.TrimSuffix(name, ".npy")]
	return v, ok && strings.HasSuffix(name, ".npy")
}

// loadVoices reads a voices file: a NumPy .npz archive holding one array per
// voice, named after it, of shape (rows, 1, 256) or (rows, 256).
func loadVoices(r io.ReaderAt, size int64) (voices, error) {
	zr, err := zip.NewReader(r, size)
	if err != nil {
		return nil, fmt.Errorf("kokoro: reading the voices file: %w", err)
	}
	out := voices{}
	for _, f := range zr.File {
		name := strings.TrimSuffix(f.Name, ".npy")
		rc, err := f.Open()
		if err != nil {
			return nil, fmt.Errorf("kokoro: voice %s: %w", name, err)
		}
		data, err := io.ReadAll(rc)
		_ = rc.Close()
		if err != nil {
			return nil, fmt.Errorf("kokoro: voice %s: %w", name, err)
		}
		v, err := parseVoiceNpy(data)
		if err != nil {
			return nil, fmt.Errorf("kokoro: voice %s: %w", name, err)
		}
		out[name] = v
	}
	return out, nil
}

// parseVoiceNpy reads a NumPy .npy array of little-endian floats, in C order,
// whose last dimension is the style and the others a row each.
func parseVoiceNpy(data []byte) (voice, error) {
	header, body, err := npyHeader(data)
	if err != nil {
		return voice{}, err
	}
	if strings.Contains(npyHeaderValue(header, "fortran_order"), "True") {
		return voice{}, fmt.Errorf("%w: fortran order", errNpy)
	}
	shape, err := npyShape(npyHeaderValue(header, "shape"))
	if err != nil {
		return voice{}, err
	}
	count := 1
	for _, d := range shape {
		count *= d
	}
	if len(shape) == 0 || shape[len(shape)-1] != styleDim || count == 0 {
		return voice{}, fmt.Errorf("%w: shape %v is not rows of %d", errNpy, shape, styleDim)
	}
	styles, err := npyFloats(strings.Trim(npyHeaderValue(header, "descr"), "'\" "), body, count)
	if err != nil {
		return voice{}, err
	}
	return voice{rows: count / styleDim, styles: styles}, nil
}

// npyHeader splits a .npy array into its header dictionary and its data.
func npyHeader(data []byte) (string, []byte, error) {
	const magic = "\x93NUMPY"
	if len(data) < 10 || !bytes.HasPrefix(data, []byte(magic)) {
		return "", nil, fmt.Errorf("%w: not a .npy array", errNpy)
	}
	var headerLen, offset int
	switch data[6] {
	case 1:
		headerLen, offset = int(binary.LittleEndian.Uint16(data[8:10])), 10
	case 2, 3:
		if len(data) < 12 {
			return "", nil, fmt.Errorf("%w: truncated header", errNpy)
		}
		headerLen, offset = int(binary.LittleEndian.Uint32(data[8:12])), 12
	default:
		return "", nil, fmt.Errorf("%w: format version %d", errNpy, data[6])
	}
	if offset+headerLen > len(data) {
		return "", nil, fmt.Errorf("%w: truncated header", errNpy)
	}
	return string(data[offset : offset+headerLen]), data[offset+headerLen:], nil
}

// npyFloats reads count little-endian floats of dtype descr from body, as
// float32.
func npyFloats(descr string, body []byte, count int) ([]float32, error) {
	out := make([]float32, count)
	switch descr {
	case "<f4":
		if len(body) < count*4 {
			return nil, fmt.Errorf("%w: truncated data", errNpy)
		}
		for i := range out {
			out[i] = math.Float32frombits(binary.LittleEndian.Uint32(body[i*4:]))
		}
	case "<f8":
		if len(body) < count*8 {
			return nil, fmt.Errorf("%w: truncated data", errNpy)
		}
		for i := range out {
			out[i] = float32(math.Float64frombits(binary.LittleEndian.Uint64(body[i*8:])))
		}
	default:
		return nil, fmt.Errorf("%w: dtype %s is not supported", errNpy, descr)
	}
	return out, nil
}

// npyHeaderValue is the text of key's value in a .npy header dictionary, up to
// the next key.
func npyHeaderValue(header, key string) string {
	_, rest, ok := strings.Cut(header, "'"+key+"':")
	if !ok {
		return ""
	}
	if strings.HasPrefix(strings.TrimSpace(rest), "(") {
		end := strings.Index(rest, ")")
		if end < 0 {
			return rest
		}
		return rest[:end+1]
	}
	end := strings.IndexAny(rest, ",}")
	if end < 0 {
		return rest
	}
	return rest[:end]
}

// npyShape reads a shape tuple such as "(510, 1, 256)".
func npyShape(text string) ([]int, error) {
	text = strings.Trim(strings.TrimSpace(text), "()")
	var shape []int
	for part := range strings.SplitSeq(text, ",") {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		n, err := strconv.Atoi(part)
		if err != nil {
			return nil, fmt.Errorf("%w: shape %q: %w", errNpy, text, err)
		}
		shape = append(shape, n)
	}
	return shape, nil
}
