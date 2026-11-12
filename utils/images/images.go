// Package images encodes images for a model to look at.
package images

import (
	"bytes"
	"errors"
	"fmt"
	"image"
	"image/color"
	"image/jpeg"
	"strings"
)

// ErrPixelFormat is returned for raw pixels in a format Encode cannot read.
var ErrPixelFormat = errors.New("images: unsupported pixel format")

// ErrPixelSize is returned for raw pixels that do not fill the size given.
var ErrPixelSize = errors.New("images: pixels do not match the image size")

// Encode encodes an image for a model to look at, and returns the encoded
// bytes and their MIME type.
//
// An image whose format is a MIME type is already encoded and keeps its bytes.
// Raw pixels are encoded as a JPEG, without their alpha channel: format is the
// pixel format, "RGB", "RGBA", "RGBX" or "L" (8-bit grey), and empty is RGB.
func Encode(img []byte, width, height int, format string) ([]byte, string, error) {
	if strings.HasPrefix(format, "image/") {
		return img, format, nil
	}
	if format == "" {
		format = "RGB"
	}
	var channels int
	switch format {
	case "RGB":
		channels = 3
	case "RGBA", "RGBX":
		channels = 4
	case "L":
		channels = 1
	default:
		return nil, "", fmt.Errorf("%w: %q", ErrPixelFormat, format)
	}
	if width <= 0 || height <= 0 || len(img) < width*height*channels {
		return nil, "", fmt.Errorf("%w: %d bytes for %dx%d %s", ErrPixelSize, len(img), width, height, format)
	}
	rgb := image.NewRGBA(image.Rect(0, 0, width, height))
	for y := range height {
		for x := range width {
			i := (y*width + x) * channels
			c := color.RGBA{A: 0xff}
			if channels == 1 {
				c.R, c.G, c.B = img[i], img[i], img[i]
			} else {
				c.R, c.G, c.B = img[i], img[i+1], img[i+2]
			}
			rgb.SetRGBA(x, y, c)
		}
	}
	var buf bytes.Buffer
	if err := jpeg.Encode(&buf, rgb, nil); err != nil {
		return nil, "", fmt.Errorf("images: %w", err)
	}
	return buf.Bytes(), "image/jpeg", nil
}
