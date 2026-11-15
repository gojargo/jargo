package images

import (
	"fmt"
	"math"
)

// bicubicA is the bicubic filter's sharpness parameter.
const bicubicA = -0.5

// formatRGB is the raw pixel format of 8-bit red, green and blue, which is what
// raw pixels with no format are.
const formatRGB = "RGB"

// Channels is the number of bytes a pixel takes in a raw pixel format, or 0 for
// a format Resize cannot read. Empty is RGB.
func Channels(format string) int {
	switch format {
	case "", formatRGB:
		return 3
	case "RGBA", "RGBX":
		return 4
	case "L":
		return 1
	default:
		return 0
	}
}

// Resize scales raw pixels to a new size with a bicubic filter, keeping their
// pixel format: "RGB", "RGBA", "RGBX" or "L" (8-bit grey), and empty is RGB.
//
// The filter widens with the scale when shrinking, so every source pixel counts
// toward the result rather than the image being sampled at a few points.
func Resize(img []byte, width, height int, format string, newWidth, newHeight int) ([]byte, error) {
	channels := Channels(format)
	if channels == 0 {
		return nil, fmt.Errorf("%w: %q", ErrPixelFormat, format)
	}
	if width <= 0 || height <= 0 || len(img) < width*height*channels {
		return nil, fmt.Errorf("%w: %d bytes for %dx%d %s", ErrPixelSize, len(img), width, height, format)
	}
	if newWidth <= 0 || newHeight <= 0 {
		return nil, fmt.Errorf("%w: cannot resize to %dx%d", ErrPixelSize, newWidth, newHeight)
	}
	if newWidth == width && newHeight == height {
		return img[:width*height*channels], nil
	}
	// Rows first, then columns: the filter is separable.
	rows := resizeRows(img, width, height, channels, newWidth)
	return resizeColumns(rows, newWidth, height, channels, newHeight), nil
}

// resizeRows scales each row of the image to newWidth pixels.
func resizeRows(img []byte, width, height, channels, newWidth int) []float64 {
	horizontal := resampleWeights(width, newWidth)
	out := make([]float64, newWidth*height*channels)
	for y := range height {
		row := img[y*width*channels:]
		for x, w := range horizontal {
			for c := range channels {
				var sum float64
				for k, weight := range w.weights {
					sum += weight * float64(row[(w.first+k)*channels+c])
				}
				out[(y*newWidth+x)*channels+c] = sum
			}
		}
	}
	return out
}

// resizeColumns scales each column of the row-scaled image to newHeight
// pixels, rounding the result to bytes.
func resizeColumns(rows []float64, width, height, channels, newHeight int) []byte {
	vertical := resampleWeights(height, newHeight)
	out := make([]byte, width*newHeight*channels)
	for y, w := range vertical {
		for x := range width {
			for c := range channels {
				var sum float64
				for k, weight := range w.weights {
					sum += weight * rows[((w.first+k)*width+x)*channels+c]
				}
				out[(y*width+x)*channels+c] = clampByte(sum)
			}
		}
	}
	return out
}

// pixelWeights are the source pixels one output pixel is made of, from first
// on, and how much each counts.
type pixelWeights struct {
	first   int
	weights []float64
}

// resampleWeights computes, for each of the out pixels along one axis, the
// source pixels it draws on and their bicubic weights, normalized to sum to 1.
func resampleWeights(in, out int) []pixelWeights {
	scale := float64(in) / float64(out)
	filterScale := math.Max(scale, 1)
	support := 2 * filterScale
	result := make([]pixelWeights, out)
	for i := range out {
		center := (float64(i) + 0.5) * scale
		first := max(int(center-support+0.5), 0)
		last := min(int(center+support+0.5), in)
		weights := make([]float64, 0, last-first)
		var total float64
		for j := first; j < last; j++ {
			w := bicubic((float64(j) - center + 0.5) / filterScale)
			weights = append(weights, w)
			total += w
		}
		if total != 0 {
			for k := range weights {
				weights[k] /= total
			}
		}
		result[i] = pixelWeights{first: first, weights: weights}
	}
	return result
}

// bicubic is the bicubic convolution kernel.
func bicubic(x float64) float64 {
	x = math.Abs(x)
	switch {
	case x < 1:
		return ((bicubicA+2)*x-(bicubicA+3))*x*x + 1
	case x < 2:
		return (((x-5)*x+8)*x - 4) * bicubicA
	default:
		return 0
	}
}

// clampByte rounds a filtered value to the nearest byte.
func clampByte(v float64) byte {
	switch {
	case v <= 0:
		return 0
	case v >= 255:
		return 255
	default:
		return byte(v + 0.5)
	}
}
