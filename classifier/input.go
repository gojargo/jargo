package classifier

import (
	"encoding/base64"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/utils/images"
)

// ErrInvalidInput reports an input that cannot be built as given: an image
// that is not one, a file with both data and a URL or neither.
var ErrInvalidInput = errors.New("classifier: invalid input")

// Input is what can come with the state: an [Image], a clip of [Audio] or a
// [File].
type Input interface {
	isInput()
}

// The image types every classifier that sees images accepts.
const (
	ImagePNG  = "image/png"
	ImageJPEG = "image/jpeg"
	ImageWebP = "image/webp"
)

// Image is an encoded image the questions can look at, alongside the state.
// Build one with [NewImage] or [ImageFromFrame].
type Image struct {
	// Data is the encoded image: PNG, JPEG or WebP.
	Data []byte
	// ContentType is the image's MIME type: ImagePNG, ImageJPEG or ImageWebP.
	ContentType string
}

// NewImage builds an image from its encoded bytes. An empty contentType is
// read from the bytes. It fails with ErrInvalidInput for bytes that are not an
// image, or for an image that is not a PNG, a JPEG or a WebP.
func NewImage(data []byte, contentType string) (Image, error) {
	if contentType == "" {
		sniffed, err := imageType(data)
		if err != nil {
			return Image{}, err
		}
		contentType = sniffed
	}
	if !imageTypeTaken(contentType) {
		return Image{}, fmt.Errorf("%w: Image takes PNG, JPEG or WebP, got %s", ErrInvalidInput, contentType)
	}
	return Image{Data: data, ContentType: contentType}, nil
}

// ImageFromFrame builds an image from an image frame's payload, such as a frame
// from the user's camera. Raw pixels are encoded as a JPEG. A payload whose
// format is already a MIME type keeps its bytes.
func ImageFromFrame(frame frames.ImageRawData) (Image, error) {
	data, contentType, err := images.Encode(frame.Image, frame.Width, frame.Height, frame.Format)
	if err != nil {
		return Image{}, fmt.Errorf("%w: %w", ErrInvalidInput, err)
	}
	return NewImage(data, contentType)
}

// imageType is the MIME type of the encoded image, or an error if it is not
// one classifiers take.
func imageType(data []byte) (string, error) {
	contentType := http.DetectContentType(data)
	if !strings.HasPrefix(contentType, "image/") {
		return "", fmt.Errorf("%w: Image data is not an image", ErrInvalidInput)
	}
	if !imageTypeTaken(contentType) {
		return "", fmt.Errorf("%w: Image takes PNG, JPEG or WebP, got %s", ErrInvalidInput, contentType)
	}
	return contentType, nil
}

// imageTypeTaken reports whether classifiers take images of this MIME type.
func imageTypeTaken(contentType string) bool {
	switch contentType {
	case ImagePNG, ImageJPEG, ImageWebP:
		return true
	default:
		return false
	}
}

// Audio is a clip of audio the questions can hear, alongside the state.
type Audio struct {
	// Audio is the clip, as 16-bit mono PCM.
	Audio []byte
	// SampleRate is the clip's sample rate in Hz.
	SampleRate int
}

// Duration is how long the clip is.
func (a Audio) Duration() time.Duration {
	return time.Duration(a.seconds() * float64(time.Second))
}

// seconds is how long the clip is, in seconds.
func (a Audio) seconds() float64 {
	if a.SampleRate <= 0 {
		return 0
	}
	return float64(len(a.Audio)) / float64(2*a.SampleRate)
}

// File is an encoded file the questions can look at, alongside the state, such
// as a PDF or a video. It holds its bytes or the URL it can be fetched from,
// never both. Build one with [NewFile] or [FileFromFrame].
type File struct {
	// ContentType is the file's MIME type, such as "application/pdf" or
	// "video/mp4".
	ContentType string
	// Data is the file's bytes, or nil for a file given by URL.
	Data []byte
	// URL is where the file can be fetched from, for a classifier whose provider
	// fetches files itself, or empty for a file given by its bytes.
	URL string
}

// NewFile builds a file from its bytes or from a URL. It fails with
// ErrInvalidInput when given both or neither.
func NewFile(contentType string, data []byte, url string) (File, error) {
	if (data == nil) == (url == "") {
		return File{}, fmt.Errorf("%w: File takes either data or a url", ErrInvalidInput)
	}
	return File{ContentType: contentType, Data: data, URL: url}, nil
}

// FileFromFrame builds a file from a file frame's payload, such as a file the
// user sent: a base64 data URL is decoded, and a URL is kept.
func FileFromFrame(frame frames.FileRawData) (File, error) {
	if frame.Type == frames.FileSourceURL {
		return NewFile(frame.Format, nil, frame.File)
	}
	encoded := frame.File
	if strings.HasPrefix(encoded, "data:") {
		if _, after, ok := strings.Cut(encoded, ","); ok {
			encoded = after
		}
	}
	data, err := base64.StdEncoding.DecodeString(encoded)
	if err != nil {
		return File{}, fmt.Errorf("%w: File data is not base64: %w", ErrInvalidInput, err)
	}
	return NewFile(frame.Format, data, "")
}

func (Image) isInput() {}
func (Audio) isInput() {}
func (File) isInput()  {}
