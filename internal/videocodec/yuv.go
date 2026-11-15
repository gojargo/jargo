package videocodec

import (
	"errors"
	"fmt"
	"unsafe"
)

// errRGBSize reports RGB data that does not hold a picture of the size given.
var errRGBSize = errors.New("videocodec: RGB data does not match the picture size")

// I420Frame is a picture in I420: a full-resolution luma plane followed by two
// chroma planes at half the width and height, rounded up. The rows of each
// plane are packed, with no padding.
type I420Frame struct {
	Width, Height int
	Y, U, V       []byte
}

// chromaSize is the width or height of a chroma plane for a luma size.
func chromaSize(n int) int { return (n + 1) / 2 }

// NewI420Frame allocates a black picture of the given size.
func NewI420Frame(width, height int) *I420Frame {
	cw, ch := chromaSize(width), chromaSize(height)
	f := &I420Frame{
		Width: width, Height: height,
		Y: make([]byte, width*height),
		U: make([]byte, cw*ch),
		V: make([]byte, cw*ch),
	}
	for i := range f.U {
		f.U[i], f.V[i] = 128, 128
	}
	return f
}

// Planes returns the luma and the two chroma planes.
func (f *I420Frame) Planes() [3][]byte { return [3][]byte{f.Y, f.U, f.V} }

// Stride is the size of a row of plane p.
func (f *I420Frame) Stride(p int) int {
	if p == 0 {
		return f.Width
	}
	return chromaSize(f.Width)
}

// copyI420 copies a picture a codec holds, each plane at its own stride, into
// an I420Frame.
func copyI420(width, height int, planes [3]*byte, strides [3]int) *I420Frame {
	f := NewI420Frame(width, height)
	dst := f.Planes()
	for p := range 3 {
		w, h := width, height
		if p > 0 {
			w, h = chromaSize(width), chromaSize(height)
		}
		for row := range h {
			start := unsafe.Add(unsafe.Pointer(planes[p]), row*strides[p]) //nolint:gosec // a row the codec wrote
			src := unsafe.Slice((*byte)(start), w)                         //nolint:gosec // its pixels
			copy(dst[p][row*w:(row+1)*w], src)
		}
	}
	return f
}

// The fixed-point BT.601 coefficients of the YUV to RGB conversion, scaled by
// 2^20.
const (
	bt601Shift = 20
	bt601CY    = 1220542
	bt601CUB   = 2116026
	bt601CUG   = -409993
	bt601CVG   = -852492
	bt601CVR   = 1673527
)

// saturate clamps a converted value to a byte.
func saturate(v int) byte {
	switch {
	case v < 0:
		return 0
	case v > 255:
		return 255
	default:
		return byte(v)
	}
}

// I420ToRGB converts a picture to packed 8-bit RGB, three bytes a pixel, with
// the BT.601 limited-range coefficients.
func I420ToRGB(f *I420Frame) []byte {
	out := make([]byte, f.Width*f.Height*3)
	cw := chromaSize(f.Width)
	half := 1 << (bt601Shift - 1)
	for row := range f.Height {
		for col := range f.Width {
			c := (row/2)*cw + col/2
			u, v := int(f.U[c])-128, int(f.V[c])-128
			ruv := half + bt601CVR*v
			guv := half + bt601CVG*v + bt601CUG*u
			buv := half + bt601CUB*u
			y := max(0, int(f.Y[row*f.Width+col])-16) * bt601CY
			i := (row*f.Width + col) * 3
			out[i] = saturate((y + ruv) >> bt601Shift)
			out[i+1] = saturate((y + guv) >> bt601Shift)
			out[i+2] = saturate((y + buv) >> bt601Shift)
		}
	}
	return out
}

// RGBToI420 converts packed 8-bit RGB, three bytes a pixel, to a picture, with
// the BT.601 limited-range coefficients. Each chroma sample is taken from the
// average of the pixels it covers.
func RGBToI420(rgb []byte, width, height int) (*I420Frame, error) {
	if width <= 0 || height <= 0 || len(rgb) != width*height*3 {
		return nil, fmt.Errorf("%w: %d bytes for %dx%d", errRGBSize, len(rgb), width, height)
	}
	f := NewI420Frame(width, height)
	for row := range height {
		for col := range width {
			i := (row*width + col) * 3
			r, g, b := int(rgb[i]), int(rgb[i+1]), int(rgb[i+2])
			f.Y[row*width+col] = byte(((66*r + 129*g + 25*b + 128) >> 8) + 16) //nolint:gosec // in 16 to 235
		}
	}
	cw, ch := chromaSize(width), chromaSize(height)
	for crow := range ch {
		for ccol := range cw {
			var r, g, b, n int
			for dy := range 2 {
				for dx := range 2 {
					row, col := crow*2+dy, ccol*2+dx
					if row >= height || col >= width {
						continue
					}
					i := (row*width + col) * 3
					r += int(rgb[i])
					g += int(rgb[i+1])
					b += int(rgb[i+2])
					n++
				}
			}
			r, g, b = (r+n/2)/n, (g+n/2)/n, (b+n/2)/n
			c := crow*cw + ccol
			f.U[c] = byte(((-38*r - 74*g + 112*b + 128) >> 8) + 128) //nolint:gosec // in 16 to 240
			f.V[c] = byte(((112*r - 94*g - 18*b + 128) >> 8) + 128)  //nolint:gosec // in 16 to 240
		}
	}
	return f, nil
}
