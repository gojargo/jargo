package videocodec

import (
	"errors"
	"fmt"
	"runtime"
	"sync"
	"unsafe"

	"github.com/ebitengine/purego"
)

// The OpenH264 decoder's methods, by their place in its interface's table
// (ISVCDecoder).
const (
	h264Initialize         = 0
	h264Uninitialize       = 1
	h264DecodeFrameNoDelay = 3
)

// The OpenH264 constants the binding passes and reads.
const (
	// videoBitstreamDefault is VIDEO_BITSTREAM_DEFAULT, which reads both AVC
	// and SVC streams.
	videoBitstreamDefault int32 = 1
	// errorConDisable is ERROR_CON_DISABLE: a damaged picture is not
	// concealed.
	errorConDisable int32 = 0
	// dsErrorFree is dsErrorFree, a stream decoded with no error.
	dsErrorFree = 0
)

// Errors the OpenH264 decoder reports.
var (
	// errOpenH264 reports a decoder call that failed.
	errOpenH264 = errors.New("videocodec: openh264 error")
)

// h264DecodingParam is SDecodingParam: the reconstructed file name, the CPU
// load, the target layer, the error concealment, whether to parse only, and the
// video property (its size and the stream type).
type h264DecodingParam struct {
	_                 unsafe.Pointer
	_                 uint32
	_                 [4]uint8
	ecActiveIdc       int32
	_                 [4]uint8
	videoPropertySize uint32
	videoBsType       int32
}

// h264BufferInfo is SBufferInfo, with the system memory buffer its union holds:
// the buffer status, the input and output timestamps, the picture's size,
// format and strides, and its planes.
type h264BufferInfo struct {
	bufferStatus int32
	_            int32
	_            [2]uint64
	width        int32
	height       int32
	_            int32
	stride       [2]int32
	_            int32
	_            [3]*byte
}

// openH264Library is libopenh264, with the calls the decoder makes bound.
type openH264Library struct {
	createDecoder  func(decoder *unsafe.Pointer) int64
	destroyDecoder func(decoder unsafe.Pointer)
}

//nolint:gochecknoglobals // the loaded library is process-wide
var openH264Libraries libraryCache[openH264Library]

// openH264LibraryNames is the names libopenh264 goes by on this platform,
// looked for on the loader's default search path when no path is configured.
func openH264LibraryNames() []string {
	switch runtime.GOOS {
	case goosDarwin:
		return []string{"libopenh264.dylib", "libopenh264.8.dylib", "libopenh264.7.dylib"}
	case goosWindows:
		return []string{"openh264.dll", "libopenh264.dll", "openh264-2.6.0-win64.dll"}
	default:
		return []string{
			"libopenh264.so.8", "libopenh264.so.7", "libopenh264.so.6", "libopenh264.so.5", "libopenh264.so",
		}
	}
}

// openOpenH264 loads libopenh264 from path, or by its conventional names when
// path is empty. A library already loaded is reused.
func openOpenH264(path string) (*openH264Library, error) {
	return openH264Libraries.get(path, func(path string) (*openH264Library, error) {
		handle, err := openShared(path, openH264LibraryNames(), ErrOpenH264NotAvailable)
		if err != nil {
			return nil, err
		}
		lib := &openH264Library{}
		if err := bind(handle, map[string]any{
			"WelsCreateDecoder":  &lib.createDecoder,
			"WelsDestroyDecoder": &lib.destroyDecoder,
		}, ErrOpenH264NotAvailable); err != nil {
			return nil, err
		}
		return lib, nil
	})
}

// H264Decoder decodes H.264 access units, in Annex B form, to I420 with
// OpenH264. It is safe for concurrent use.
type H264Decoder struct {
	mu  sync.Mutex
	lib *openH264Library
	// decoder is the ISVCDecoder object: a pointer to its method table.
	decoder unsafe.Pointer

	uninitialize       func(decoder unsafe.Pointer) int64
	decodeFrameNoDelay func(decoder unsafe.Pointer, src *byte, srcLen int32, dst *[3]*byte, info *h264BufferInfo) int32

	closed bool
}

// NewH264Decoder builds an H.264 decoder over libopenh264, loaded from libPath
// or by its conventional names when libPath is empty.
func NewH264Decoder(libPath string) (*H264Decoder, error) {
	lib, err := openOpenH264(libPath)
	if err != nil {
		return nil, err
	}
	d := &H264Decoder{lib: lib}
	if code := lib.createDecoder(&d.decoder); code != 0 || d.decoder == nil {
		return nil, fmt.Errorf("%w: WelsCreateDecoder: %d", errOpenH264, code)
	}
	// The decoder object points at its method table.
	table := unsafe.Slice(*(**uintptr)(d.decoder), h264DecodeFrameNoDelay+1) //nolint:gosec // the method table
	method := func(i int) uintptr { return table[i] }
	var initialize func(decoder unsafe.Pointer, param *h264DecodingParam) int64
	purego.RegisterFunc(&initialize, method(h264Initialize))
	purego.RegisterFunc(&d.uninitialize, method(h264Uninitialize))
	purego.RegisterFunc(&d.decodeFrameNoDelay, method(h264DecodeFrameNoDelay))

	param := &h264DecodingParam{
		ecActiveIdc:       errorConDisable,
		videoPropertySize: 8,
		videoBsType:       videoBitstreamDefault,
	}
	if code := initialize(d.decoder, param); code != 0 {
		lib.destroyDecoder(d.decoder)
		return nil, fmt.Errorf("%w: Initialize: %d", errOpenH264, code)
	}
	return d, nil
}

// Decode decodes one access unit and returns the picture it completes, if
// any, copied out of the decoder.
func (d *H264Decoder) Decode(data []byte) ([]*I420Frame, error) {
	d.mu.Lock()
	defer d.mu.Unlock()
	if d.closed {
		return nil, errCodecClosed
	}
	if len(data) == 0 {
		return nil, nil
	}
	dst := &[3]*byte{}
	info := &h264BufferInfo{}
	size := int32(len(data)) //nolint:gosec // a frame size
	state := d.decodeFrameNoDelay(d.decoder, &data[0], size, dst, info)
	if state != dsErrorFree {
		return nil, fmt.Errorf("%w: DecodeFrameNoDelay: state %#x", errOpenH264, state)
	}
	if info.bufferStatus != 1 || dst[0] == nil {
		return nil, nil
	}
	return []*I420Frame{copyI420(int(info.width), int(info.height), *dst, [3]int{
		int(info.stride[0]), int(info.stride[1]), int(info.stride[1]),
	})}, nil
}

// Close frees the decoder.
func (d *H264Decoder) Close() {
	d.mu.Lock()
	defer d.mu.Unlock()
	if d.closed {
		return
	}
	d.closed = true
	d.uninitialize(d.decoder)
	d.lib.destroyDecoder(d.decoder)
}
