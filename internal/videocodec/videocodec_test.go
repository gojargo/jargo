package videocodec

import (
	"errors"
	"os"
	"testing"
)

// closeTo reports whether a channel value is within tol of want.
func closeTo(got, want byte, tol int) bool {
	d := int(got) - int(want)
	return d >= -tol && d <= tol
}

// solidRGB is a picture of one color.
func solidRGB(width, height int, r, g, b byte) []byte {
	rgb := make([]byte, width*height*3)
	for i := 0; i < len(rgb); i += 3 {
		rgb[i], rgb[i+1], rgb[i+2] = r, g, b
	}
	return rgb
}

// checkSolid fails unless every pixel of rgb is within tol of the color.
func checkSolid(t *testing.T, rgb []byte, r, g, b byte, tol int) {
	t.Helper()
	for i := 0; i < len(rgb); i += 3 {
		if !closeTo(rgb[i], r, tol) || !closeTo(rgb[i+1], g, tol) || !closeTo(rgb[i+2], b, tol) {
			t.Fatalf("pixel %d = (%d, %d, %d), want about (%d, %d, %d)",
				i/3, rgb[i], rgb[i+1], rgb[i+2], r, g, b)
		}
	}
}

func TestRGBToI420AndBack(t *testing.T) {
	for _, size := range [][2]int{{4, 4}, {5, 3}, {1, 1}} {
		rgb := solidRGB(size[0], size[1], 51, 102, 204)
		f, err := RGBToI420(rgb, size[0], size[1])
		if err != nil {
			t.Fatal(err)
		}
		if len(f.U) != chromaSize(size[0])*chromaSize(size[1]) {
			t.Fatalf("chroma plane of %d bytes for %v", len(f.U), size)
		}
		checkSolid(t, I420ToRGB(f), 51, 102, 204, 3)
	}
}

func TestRGBToI420RejectsAMismatchedSize(t *testing.T) {
	if _, err := RGBToI420(make([]byte, 10), 2, 2); !errors.Is(err, errRGBSize) {
		t.Fatalf("err = %v, want errRGBSize", err)
	}
}

// TestI420ToRGBLimitedRange covers the BT.601 limited range: luma 16 is black
// and 235 is white.
func TestI420ToRGBLimitedRange(t *testing.T) {
	f := NewI420Frame(2, 2)
	for i := range f.Y {
		f.Y[i] = 16
	}
	checkSolid(t, I420ToRGB(f), 0, 0, 0, 0)
	for i := range f.Y {
		f.Y[i] = 235
	}
	checkSolid(t, I420ToRGB(f), 255, 255, 255, 0)
}

func TestNumberOfThreads(t *testing.T) {
	for _, c := range []struct {
		pixels, cpus int
		want         uint32
	}{
		{1920 * 1080, 16, 8},
		{1920 * 1080, 8, 3},
		{1280 * 720, 4, 2},
		{640 * 480, 16, 1},
		{1280 * 720, 2, 1},
	} {
		if got := numberOfThreads(c.pixels, c.cpus); got != c.want {
			t.Errorf("numberOfThreads(%d, %d) = %d, want %d", c.pixels, c.cpus, got, c.want)
		}
	}
}

// vpxOrSkip loads libvpx, or skips the test when it is not installed.
func vpxOrSkip(t *testing.T) *vpxLibrary {
	t.Helper()
	lib, err := openVPX("")
	if errors.Is(err, ErrVPXNotAvailable) {
		t.Skipf("libvpx not available: %v", err)
	}
	if err != nil {
		t.Fatal(err)
	}
	return lib
}

func TestVPXMissingLibraryPath(t *testing.T) {
	_, err := NewVP8Encoder("/nonexistent/libvpx.so")
	if !errors.Is(err, ErrVPXNotAvailable) {
		t.Fatalf("err = %v, want ErrVPXNotAvailable", err)
	}
}

// TestVPXImageLayout checks vpxImage against the library: an image wrapped by
// libvpx reads back with the fields where the binding expects them.
func TestVPXImageLayout(t *testing.T) {
	lib := vpxOrSkip(t)
	buf := make([]byte, 64*48*3/2)
	img := &vpxImage{}
	lib.imgWrap(img, vpxImgFmtI420, 64, 48, 1, &buf[0])
	if img.fmt != vpxImgFmtI420 || img.w != 64 || img.h != 48 || img.dW != 64 || img.dH != 48 ||
		img.xChromaShift != 1 || img.yChromaShift != 1 || img.bps != 12 {
		t.Fatalf("image = %+v", img)
	}
	if img.planes[0] != &buf[0] || img.planes[1] != &buf[64*48] || img.stride[0] != 64 || img.stride[1] != 32 ||
		img.stride[2] != 32 {
		t.Fatalf("planes = %v, strides = %v", img.planes, img.stride)
	}
	if lib.encoderABI == 0 || lib.decoderABI == 0 {
		t.Fatalf("ABI versions %d and %d", lib.encoderABI, lib.decoderABI)
	}
}

// TestVP8RoundTrip encodes a picture of one color with libvpx and decodes it
// back.
func TestVP8RoundTrip(t *testing.T) {
	vpxOrSkip(t)
	enc, err := NewVP8Encoder("")
	if err != nil {
		t.Fatal(err)
	}
	defer enc.Close()
	dec, err := NewVP8Decoder("")
	if err != nil {
		t.Fatal(err)
	}
	defer dec.Close()

	const width, height = 64, 48
	in, err := RGBToI420(solidRGB(width, height, 51, 102, 204), width, height)
	if err != nil {
		t.Fatal(err)
	}
	var decoded []*I420Frame
	for i := range 3 {
		data, err := enc.Encode(in, int64(i*videoTimestampPerTick), i == 0)
		if err != nil {
			t.Fatal(err)
		}
		if len(data) == 0 {
			t.Fatalf("frame %d encoded to nothing", i)
		}
		// A keyframe starts with a clear inverse-keyframe bit.
		if i == 0 && data[0]&1 != 0 {
			t.Fatalf("first frame is not a keyframe: %x", data[:3])
		}
		frames, err := dec.Decode(data)
		if err != nil {
			t.Fatal(err)
		}
		decoded = append(decoded, frames...)
	}
	if q := enc.lastQuantizer(); q < 0 || q > 63 {
		t.Errorf("last quantizer = %d", q)
	}
	if len(decoded) != 3 {
		t.Fatalf("decoded %d frames, want 3", len(decoded))
	}
	for _, f := range decoded {
		if f.Width != width || f.Height != height {
			t.Fatalf("decoded %dx%d, want %dx%d", f.Width, f.Height, width, height)
		}
		checkSolid(t, I420ToRGB(f), 51, 102, 204, 8)
	}

	// A frame of another size recreates the codec.
	small, err := RGBToI420(solidRGB(32, 32, 200, 50, 50), 32, 32)
	if err != nil {
		t.Fatal(err)
	}
	data, err := enc.Encode(small, 3*videoTimestampPerTick, false)
	if err != nil {
		t.Fatal(err)
	}
	frames, err := dec.Decode(data)
	if err != nil {
		t.Fatal(err)
	}
	if len(frames) != 1 || frames[0].Width != 32 || frames[0].Height != 32 {
		t.Fatalf("resized frame decoded to %v", frames)
	}
	checkSolid(t, I420ToRGB(frames[0]), 200, 50, 50, 8)
}

func TestVP8EncoderBitrateIsHeldInRange(t *testing.T) {
	vpxOrSkip(t)
	enc, err := NewVP8Encoder("")
	if err != nil {
		t.Fatal(err)
	}
	defer enc.Close()
	if got := enc.TargetBitrate(); got != DefaultBitrate {
		t.Fatalf("default bitrate = %d", got)
	}
	enc.SetTargetBitrate(10)
	if got := enc.TargetBitrate(); got != MinBitrate {
		t.Fatalf("bitrate = %d, want %d", got, MinBitrate)
	}
	enc.SetTargetBitrate(1 << 30)
	if got := enc.TargetBitrate(); got != MaxBitrate {
		t.Fatalf("bitrate = %d, want %d", got, MaxBitrate)
	}
	frame := NewI420Frame(16, 16)
	if _, err := enc.Encode(frame, 0, false); err != nil {
		t.Fatal(err)
	}
	enc.SetTargetBitrate(800000)
	if _, err := enc.Encode(frame, videoTimestampPerTick, false); err != nil {
		t.Fatal(err)
	}
	if got := enc.cfg[cfgRcTargetBitrate]; got != 800 {
		t.Fatalf("configured bitrate = %d kbps, want 800", got)
	}
}

func TestCodecsRefuseUseAfterClose(t *testing.T) {
	vpxOrSkip(t)
	enc, err := NewVP8Encoder("")
	if err != nil {
		t.Fatal(err)
	}
	enc.Close()
	enc.Close()
	if _, err := enc.Encode(NewI420Frame(16, 16), 0, false); !errors.Is(err, errCodecClosed) {
		t.Fatalf("encode after close: %v", err)
	}
	dec, err := NewVP8Decoder("")
	if err != nil {
		t.Fatal(err)
	}
	dec.Close()
	if _, err := dec.Decode([]byte{0}); !errors.Is(err, errCodecClosed) {
		t.Fatalf("decode after close: %v", err)
	}
}

// h264OrSkip builds an H.264 decoder, or skips the test when libopenh264 is
// not installed.
func h264OrSkip(t *testing.T) *H264Decoder {
	t.Helper()
	dec, err := NewH264Decoder("")
	if errors.Is(err, ErrOpenH264NotAvailable) {
		t.Skipf("libopenh264 not available: %v", err)
	}
	if err != nil {
		t.Fatal(err)
	}
	return dec
}

func TestH264MissingLibraryPath(t *testing.T) {
	_, err := NewH264Decoder("/nonexistent/libopenh264.so")
	if !errors.Is(err, ErrOpenH264NotAvailable) {
		t.Fatalf("err = %v, want ErrOpenH264NotAvailable", err)
	}
}

// TestH264Decode decodes a one-frame stream of one color, (51, 102, 204) at 64
// by 48.
func TestH264Decode(t *testing.T) {
	dec := h264OrSkip(t)
	defer dec.Close()
	data, err := os.ReadFile("testdata/color-64x48.h264")
	if err != nil {
		t.Fatal(err)
	}
	frames, err := dec.Decode(data)
	if err != nil {
		t.Fatal(err)
	}
	if len(frames) != 1 {
		t.Fatalf("decoded %d frames, want 1", len(frames))
	}
	f := frames[0]
	if f.Width != 64 || f.Height != 48 {
		t.Fatalf("decoded %dx%d, want 64x48", f.Width, f.Height)
	}
	checkSolid(t, I420ToRGB(f), 51, 102, 204, 8)

	dec.Close()
	if _, err := dec.Decode(data); !errors.Is(err, errCodecClosed) {
		t.Fatalf("decode after close: %v", err)
	}
}
