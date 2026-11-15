package videocodec

import (
	"errors"
	"fmt"
	"runtime"
	"sync"
	"unsafe"

	"github.com/ebitengine/purego"
)

// The libvpx constants the binding passes and reads.
const (
	vpxCodecOK          int32 = 0
	vpxCodecABIMismatch int32 = 3
	// vpxImgFmtI420 is VPX_IMG_FMT_I420: VPX_IMG_FMT_PLANAR | 2.
	vpxImgFmtI420 int32 = 0x102
	// vpxCodecCxFramePkt is VPX_CODEC_CX_FRAME_PKT, a packet of compressed
	// frame data.
	vpxCodecCxFramePkt int32 = 0
	// vpxEflagForceKF is VPX_EFLAG_FORCE_KF, which makes the frame encoded a
	// keyframe.
	vpxEflagForceKF int64 = 1
	// vpxDLRealtime is VPX_DL_REALTIME, the deadline of a live encoder.
	vpxDLRealtime = 1
	// vpxCBR is VPX_CBR, constant bitrate control.
	vpxCBR uint32 = 1
	// vpxKFAuto is VPX_KF_AUTO, keyframes placed by the encoder.
	vpxKFAuto uint32 = 1
)

// The VP8 controls the binding sets (vp8e_enc_control_id and
// vp8_com_control_id), and their values.
const (
	vp8eSetCPUUsed          int32 = 13
	vp8eSetNoiseSensitivity int32 = 15
	vp8eSetStaticThreshold  int32 = 17
	vp8eSetTokenPartitions  int32 = 18
	vp8eGetLastQuantizer64  int32 = 20
	vp8SetPostproc          int32 = 3
	vp8OneTokenPartition    int64 = 0
	vp8Deblock              int32 = 1 << 0
	vp8Demacroblock         int32 = 1 << 1
)

// The sizes of the structures handed to libvpx whole, with room to spare, and
// the defaults the VP8 encoder's configuration is checked against.
const (
	maxABIVersionProbed             = 128
	vpxCodecCtxSize                 = 64
	vpxEncCfgWords                  = 1024
	vpxDefaultConfigWidth           = 320
	vpxDefaultConfigHeight          = 240
	vpxDefaultConfigBitrateKbps     = 256
	vpxDefaultConfigKeyframeMaxDist = 128
	vpxDefaultConfigFramesPerSecond = 30
)

// The VP8 encoder's bitrates, in bits per second: the one it starts at, and the
// range a receiver's estimate is held to.
const (
	DefaultBitrate = 500000
	MinBitrate     = 250000
	MaxBitrate     = 1500000
)

// The offsets, in 32-bit words, of the vpx_codec_enc_cfg_t fields the encoder
// sets. They have not moved since libvpx 1.8.
const (
	cfgGThreads        = 1
	cfgGW              = 3
	cfgGH              = 4
	cfgGTimebaseNum    = 7
	cfgGTimebaseDen    = 8
	cfgGLagInFrames    = 11
	cfgRcResizeAllowed = 13
	cfgRcEndUsage      = 18
	cfgRcTargetBitrate = 28
	cfgRcMinQuantizer  = 29
	cfgRcMaxQuantizer  = 30
	cfgRcUndershootPct = 31
	cfgRcOvershootPct  = 32
	cfgRcBufSz         = 33
	cfgRcBufInitialSz  = 34
	cfgRcBufOptimalSz  = 35
	cfgKfMode          = 40
	cfgKfMaxDist       = 42
)

// The video clock: 90 kHz, with frames 1/30 s apart.
const (
	videoClockRate        = 90000
	videoTimestampPerTick = videoClockRate / 30
)

// vpxCodecCtx is vpx_codec_ctx_t, which libvpx fills in.
type vpxCodecCtx [vpxCodecCtxSize]byte

// vpxEncCfg is vpx_codec_enc_cfg_t, read and written as 32-bit words.
type vpxEncCfg [vpxEncCfgWords]uint32

// Errors the libvpx codecs report.
var (
	// errVPXLayout reports a libvpx whose encoder configuration is not laid
	// out as the binding expects.
	errVPXLayout = errors.New("videocodec: libvpx encoder configuration has an unexpected layout")
	// errVPXABI reports a libvpx none of whose ABI versions could be found.
	errVPXABI = errors.New("videocodec: libvpx ABI version not found")
	// errVPX reports a libvpx call that failed.
	errVPX = errors.New("videocodec: libvpx error")
	// errCodecClosed reports a codec used after it was closed.
	errCodecClosed = errors.New("videocodec: codec is closed")
)

// vpxImage is vpx_image_t.
type vpxImage struct {
	fmt          int32
	_            int32 // cs
	_            int32 // range
	w            uint32
	h            uint32
	_            uint32 // bit_depth
	dW           uint32
	dH           uint32
	_            [2]uint32 // r_w, r_h
	xChromaShift uint32
	yChromaShift uint32
	planes       [4]*byte
	stride       [4]int32
	bps          int32
	_            unsafe.Pointer // user_priv
	_            unsafe.Pointer // img_data
	_            [2]int32       // img_data_owner, self_allocd
	_            unsafe.Pointer // fb_priv
}

// vpxCxPkt is the head of vpx_codec_cx_pkt_t for a frame packet: its kind, then
// the frame's data and size.
type vpxCxPkt struct {
	kind int32
	_    int32
	buf  *byte
	sz   uint64
}

// vp8PostprocCfg is vp8_postproc_cfg_t.
type vp8PostprocCfg struct {
	postProcFlag    int32
	deblockingLevel int32
	_               int32 // noise_level
}

// vpxInit is vpx_codec_enc_init_ver or vpx_codec_dec_init_ver.
type vpxInit func(ctx *vpxCodecCtx, iface uintptr, cfg *vpxEncCfg, flags int64, ver int32) int32

// vpxLibrary is libvpx, with the calls the VP8 codecs make bound.
type vpxLibrary struct {
	vp8CX            func() uintptr
	vp8DX            func() uintptr
	encConfigDefault func(iface uintptr, cfg *vpxEncCfg, usage uint32) int32
	encConfigSet     func(ctx *vpxCodecCtx, cfg *vpxEncCfg) int32
	encInitVer       vpxInit
	decInitVer       vpxInit
	encode           func(ctx *vpxCodecCtx, img *vpxImage, pts int64, duration uint64, flags int64, deadline uint64) int32
	getCxData        func(ctx *vpxCodecCtx, iter *unsafe.Pointer) *vpxCxPkt
	decode           func(ctx *vpxCodecCtx, data *byte, size uint32, userPriv unsafe.Pointer, deadline int64) int32
	getFrame         func(ctx *vpxCodecCtx, iter *unsafe.Pointer) *vpxImage
	destroy          func(ctx *vpxCodecCtx) int32
	imgWrap          func(img *vpxImage, fmt int32, w, h, align uint32, data *byte) *vpxImage
	errToString      func(err int32) string
	// controlFn is vpx_codec_control_, which is variadic and so is called
	// through SyscallN.
	controlFn uintptr

	// encoderABI and decoderABI are the ABI versions the library was built
	// with, which its init calls insist on.
	encoderABI int32
	decoderABI int32
}

//nolint:gochecknoglobals // the loaded library is process-wide
var vpxLibraries libraryCache[vpxLibrary]

// vpxLibraryNames is the names libvpx goes by on this platform, looked for on
// the loader's default search path when no path is configured. The versioned
// names come first: a distribution's runtime package installs only those, the
// unversioned link being part of its development package.
func vpxLibraryNames() []string {
	switch runtime.GOOS {
	case goosDarwin:
		return []string{"libvpx.dylib", "libvpx.12.dylib", "libvpx.11.dylib", "libvpx.9.dylib", "libvpx.8.dylib"}
	case goosWindows:
		return []string{"vpx.dll", "libvpx.dll", "libvpx-1.dll"}
	default:
		return []string{
			"libvpx.so.12", "libvpx.so.11", "libvpx.so.9", "libvpx.so.8", "libvpx.so.7", "libvpx.so.6",
			"libvpx.so",
		}
	}
}

// openVPX loads libvpx from path, or by its conventional names when path is
// empty, and binds the calls the codecs make. A library already loaded is
// reused.
func openVPX(path string) (*vpxLibrary, error) {
	return vpxLibraries.get(path, func(path string) (*vpxLibrary, error) {
		handle, err := openShared(path, vpxLibraryNames(), ErrVPXNotAvailable)
		if err != nil {
			return nil, err
		}
		lib := &vpxLibrary{}
		if err = bind(handle, map[string]any{
			"vpx_codec_vp8_cx":             &lib.vp8CX,
			"vpx_codec_vp8_dx":             &lib.vp8DX,
			"vpx_codec_enc_config_default": &lib.encConfigDefault,
			"vpx_codec_enc_config_set":     &lib.encConfigSet,
			"vpx_codec_enc_init_ver":       &lib.encInitVer,
			"vpx_codec_dec_init_ver":       &lib.decInitVer,
			"vpx_codec_encode":             &lib.encode,
			"vpx_codec_get_cx_data":        &lib.getCxData,
			"vpx_codec_decode":             &lib.decode,
			"vpx_codec_get_frame":          &lib.getFrame,
			"vpx_codec_destroy":            &lib.destroy,
			"vpx_img_wrap":                 &lib.imgWrap,
			"vpx_codec_err_to_string":      &lib.errToString,
		}, ErrVPXNotAvailable); err != nil {
			return nil, err
		}
		if lib.controlFn, err = purego.Dlsym(handle, "vpx_codec_control_"); err != nil {
			return nil, fmt.Errorf("%w: vpx_codec_control_: %w", ErrVPXNotAvailable, err)
		}
		if err := lib.findABIVersions(); err != nil {
			return nil, err
		}
		return lib, nil
	})
}

// findABIVersions finds the ABI versions the library's init calls accept. The
// headers that define them are not at hand, and the versions change between
// releases, so each is found by asking the library: a call with another
// version is refused before anything is set up.
func (l *vpxLibrary) findABIVersions() error {
	cfg := &vpxEncCfg{}
	if code := l.encConfigDefault(l.vp8CX(), cfg, 0); code != vpxCodecOK {
		return l.err("vpx_codec_enc_config_default", code)
	}
	if err := checkEncoderConfigLayout(cfg); err != nil {
		return err
	}
	var err error
	if l.encoderABI, err = l.probeABIVersion(l.encInitVer, l.vp8CX(), cfg); err != nil {
		return err
	}
	if l.decoderABI, err = l.probeABIVersion(l.decInitVer, l.vp8DX(), nil); err != nil {
		return err
	}
	return nil
}

// probeABIVersion finds the ABI version an init call accepts.
func (l *vpxLibrary) probeABIVersion(init vpxInit, iface uintptr, cfg *vpxEncCfg) (int32, error) {
	for ver := int32(1); ver <= int32(maxABIVersionProbed); ver++ {
		ctx := &vpxCodecCtx{}
		code := init(ctx, iface, cfg, 0, ver)
		if code == vpxCodecABIMismatch {
			continue
		}
		if code != vpxCodecOK {
			return 0, l.err("probing the ABI version", code)
		}
		l.destroy(ctx)
		return ver, nil
	}
	return 0, errVPXABI
}

// checkEncoderConfigLayout checks that the VP8 encoder's default configuration
// reads as the binding expects at the offsets it writes to, so a library laid
// out otherwise is refused rather than misconfigured.
func checkEncoderConfigLayout(cfg *vpxEncCfg) error {
	if cfg[cfgGW] != vpxDefaultConfigWidth || cfg[cfgGH] != vpxDefaultConfigHeight ||
		cfg[cfgGTimebaseNum] != 1 || cfg[cfgGTimebaseDen] != vpxDefaultConfigFramesPerSecond ||
		cfg[cfgRcTargetBitrate] != vpxDefaultConfigBitrateKbps ||
		cfg[cfgKfMode] != vpxKFAuto || cfg[cfgKfMaxDist] != vpxDefaultConfigKeyframeMaxDist {
		return errVPXLayout
	}
	return nil
}

// err is the error a failed call returns.
func (l *vpxLibrary) err(call string, code int32) error {
	return fmt.Errorf("%w: %s: %s (%d)", errVPX, call, l.errToString(code), code)
}

// variadicArgsInRegisters reports whether a variadic argument is passed where
// a fixed one would be, which is how purego passes it. On Apple's arm64 it
// goes on the stack instead.
func variadicArgsInRegisters() bool {
	return runtime.GOOS != goosDarwin || runtime.GOARCH != "arm64"
}

// control sets a control of a codec. vpx_codec_control_ takes its value as a
// variadic argument; where that cannot be passed, the control is not set.
func (l *vpxLibrary) control(ctx *vpxCodecCtx, id int32, value uintptr) int32 {
	if !variadicArgsInRegisters() {
		return -1
	}
	// The context is pinned for as long as the codec lives.
	handle := uintptr(unsafe.Pointer(ctx)) //nolint:gosec // pinned
	r, _, _ := purego.SyscallN(l.controlFn, handle, uintptr(id), value)
	return int32(r) //nolint:gosec // an error code
}

// numberOfThreads is how many threads to encode a picture of pixels pixels
// with on cpus processors.
func numberOfThreads(pixels, cpus int) uint32 {
	switch {
	case pixels >= 1920*1080 && cpus > 8:
		return 8
	case pixels > 1280*960 && cpus >= 6:
		return 3
	case pixels > 640*480 && cpus >= 3:
		return 2
	default:
		return 1
	}
}

// VP8Encoder encodes I420 frames to VP8 with libvpx, configured for live,
// constant-bitrate video. It is safe for concurrent use.
type VP8Encoder struct {
	mu  sync.Mutex
	lib *vpxLibrary
	// ctx, cfg and image are handed to libvpx, so they are allocated once and
	// pinned for as long as the encoder lives.
	ctx   *vpxCodecCtx
	cfg   *vpxEncCfg
	image *vpxImage
	// placeholder is what the image is wrapped on until a frame's planes
	// replace it.
	placeholder *byte
	pinner      runtime.Pinner
	open        bool

	width, height int
	targetBitrate int
	updateConfig  bool
	closed        bool
}

// NewVP8Encoder builds a VP8 encoder over libvpx, loaded from libPath or by its
// conventional names when libPath is empty. The codec itself is created with
// the first frame, at that frame's size.
func NewVP8Encoder(libPath string) (*VP8Encoder, error) {
	lib, err := openVPX(libPath)
	if err != nil {
		return nil, err
	}
	e := &VP8Encoder{
		lib:           lib,
		ctx:           &vpxCodecCtx{},
		cfg:           &vpxEncCfg{},
		image:         &vpxImage{},
		placeholder:   new(byte),
		targetBitrate: DefaultBitrate,
	}
	e.pinner.Pin(e.ctx)
	e.pinner.Pin(e.cfg)
	e.pinner.Pin(e.image)
	e.pinner.Pin(e.placeholder)
	if code := lib.encConfigDefault(lib.vp8CX(), e.cfg, 0); code != vpxCodecOK {
		e.pinner.Unpin()
		return nil, lib.err("vpx_codec_enc_config_default", code)
	}
	return e, nil
}

// TargetBitrate is the bitrate the encoder aims for, in bits per second.
func (e *VP8Encoder) TargetBitrate() int {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.targetBitrate
}

// SetTargetBitrate changes the bitrate the encoder aims for, held between
// MinBitrate and MaxBitrate. It applies from the next frame.
func (e *VP8Encoder) SetTargetBitrate(bitrate int) {
	e.mu.Lock()
	defer e.mu.Unlock()
	bitrate = max(MinBitrate, min(bitrate, MaxBitrate))
	if bitrate != e.targetBitrate {
		e.targetBitrate = bitrate
		e.updateConfig = true
	}
}

// applyBitrateLocked writes the target bitrate into the configuration.
func (e *VP8Encoder) applyBitrateLocked() {
	e.cfg[cfgRcTargetBitrate] = uint32(e.targetBitrate / 1000) //nolint:gosec // a bitrate in range
	e.updateConfig = false
}

// createLocked creates the codec for frames of the given size.
func (e *VP8Encoder) createLocked(width, height int) error {
	e.cfg[cfgGTimebaseNum] = 1
	e.cfg[cfgGTimebaseDen] = videoClockRate
	e.cfg[cfgGLagInFrames] = 0
	e.cfg[cfgGThreads] = numberOfThreads(width*height, runtime.NumCPU())
	e.cfg[cfgGW] = uint32(width)  //nolint:gosec // a frame size
	e.cfg[cfgGH] = uint32(height) //nolint:gosec // a frame size
	e.cfg[cfgRcResizeAllowed] = 0
	e.cfg[cfgRcEndUsage] = vpxCBR
	e.cfg[cfgRcMinQuantizer] = 2
	e.cfg[cfgRcMaxQuantizer] = 56
	e.cfg[cfgRcUndershootPct] = 100
	e.cfg[cfgRcOvershootPct] = 15
	e.cfg[cfgRcBufInitialSz] = 500
	e.cfg[cfgRcBufOptimalSz] = 600
	e.cfg[cfgRcBufSz] = 1000
	e.cfg[cfgKfMode] = vpxKFAuto
	e.cfg[cfgKfMaxDist] = 3000
	e.applyBitrateLocked()

	if code := e.lib.encInitVer(e.ctx, e.lib.vp8CX(), e.cfg, 0, e.lib.encoderABI); code != vpxCodecOK {
		return e.lib.err("vpx_codec_enc_init", code)
	}
	e.open = true
	e.width, e.height = width, height

	e.lib.control(e.ctx, vp8eSetNoiseSensitivity, 4)
	e.lib.control(e.ctx, vp8eSetStaticThreshold, 1)
	cpuUsed := int64(-6)
	e.lib.control(e.ctx, vp8eSetCPUUsed, uintptr(cpuUsed))
	e.lib.control(e.ctx, vp8eSetTokenPartitions, uintptr(vp8OneTokenPartition))

	// The image is wrapped on a placeholder; each frame's planes are pointed
	// at when it is encoded.
	e.lib.imgWrap(e.image, vpxImgFmtI420, uint32(width), uint32(height), 1, e.placeholder) //nolint:gosec // a frame size
	return nil
}

// prepareLocked makes the codec ready for a frame of the given size: created
// for that size, with the configuration up to date.
func (e *VP8Encoder) prepareLocked(width, height int) error {
	if e.open && (width != e.width || height != e.height) {
		e.destroyLocked()
	}
	if !e.open {
		return e.createLocked(width, height)
	}
	if e.updateConfig {
		e.applyBitrateLocked()
		if code := e.lib.encConfigSet(e.ctx, e.cfg); code != vpxCodecOK {
			return e.lib.err("vpx_codec_enc_config_set", code)
		}
	}
	return nil
}

// destroyLocked frees the codec, if one was created.
func (e *VP8Encoder) destroyLocked() {
	if e.open {
		e.lib.destroy(e.ctx)
		e.open = false
	}
}

// Encode encodes one frame, presented at pts on the 90 kHz video clock, and
// returns the frame's compressed data. forceKeyframe makes it a keyframe.
// The codec is created again when the frame size changes.
func (e *VP8Encoder) Encode(frame *I420Frame, pts int64, forceKeyframe bool) ([]byte, error) {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.closed {
		return nil, errCodecClosed
	}
	if err := e.prepareLocked(frame.Width, frame.Height); err != nil {
		return nil, err
	}

	var pinner runtime.Pinner
	defer pinner.Unpin()
	planes := frame.Planes()
	for p := range 3 {
		pinner.Pin(&planes[p][0])
		e.image.planes[p] = &planes[p][0]
		e.image.stride[p] = int32(frame.Stride(p)) //nolint:gosec // a row size
	}
	var flags int64
	if forceKeyframe {
		flags |= vpxEflagForceKF
	}
	code := e.lib.encode(e.ctx, e.image, pts, videoTimestampPerTick, flags, vpxDLRealtime)
	for p := range 3 {
		e.image.planes[p] = nil
	}
	if code != vpxCodecOK {
		return nil, e.lib.err("vpx_codec_encode", code)
	}

	var out []byte
	var iter unsafe.Pointer
	for pkt := e.lib.getCxData(e.ctx, &iter); pkt != nil; pkt = e.lib.getCxData(e.ctx, &iter) {
		if pkt.kind == vpxCodecCxFramePkt && pkt.sz > 0 {
			out = append(out, unsafe.Slice(pkt.buf, pkt.sz)...) //nolint:gosec // the packet libvpx returned
		}
	}
	return out, nil
}

// lastQuantizer is the quantizer the last frame was encoded with, on the
// 0 to 63 scale, or -1 when it cannot be read.
func (e *VP8Encoder) lastQuantizer() int {
	e.mu.Lock()
	defer e.mu.Unlock()
	if !e.open {
		return -1
	}
	q := new(int32)
	var pinner runtime.Pinner
	pinner.Pin(q)
	defer pinner.Unpin()
	if e.lib.control(e.ctx, vp8eGetLastQuantizer64, uintptr(unsafe.Pointer(q))) != vpxCodecOK { //nolint:gosec // pinned
		return -1
	}
	return int(*q)
}

// Close frees the codec.
func (e *VP8Encoder) Close() {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.closed {
		return
	}
	e.closed = true
	e.destroyLocked()
	e.pinner.Unpin()
}

// VP8Decoder decodes VP8 frames to I420 with libvpx. It is safe for concurrent
// use.
type VP8Decoder struct {
	mu     sync.Mutex
	lib    *vpxLibrary
	ctx    *vpxCodecCtx
	pinner runtime.Pinner
	closed bool
}

// NewVP8Decoder builds a VP8 decoder over libvpx, loaded from libPath or by its
// conventional names when libPath is empty.
func NewVP8Decoder(libPath string) (*VP8Decoder, error) {
	lib, err := openVPX(libPath)
	if err != nil {
		return nil, err
	}
	d := &VP8Decoder{lib: lib, ctx: &vpxCodecCtx{}}
	d.pinner.Pin(d.ctx)
	if code := lib.decInitVer(d.ctx, lib.vp8DX(), nil, 0, lib.decoderABI); code != vpxCodecOK {
		d.pinner.Unpin()
		return nil, lib.err("vpx_codec_dec_init", code)
	}
	cfg := &vp8PostprocCfg{postProcFlag: vp8Deblock | vp8Demacroblock, deblockingLevel: 3}
	var pinner runtime.Pinner
	pinner.Pin(cfg)
	lib.control(d.ctx, vp8SetPostproc, uintptr(unsafe.Pointer(cfg))) //nolint:gosec // pinned
	pinner.Unpin()
	return d, nil
}

// Decode decodes one compressed frame and returns the pictures it yields,
// copied out of the decoder.
func (d *VP8Decoder) Decode(data []byte) ([]*I420Frame, error) {
	d.mu.Lock()
	defer d.mu.Unlock()
	if d.closed {
		return nil, errCodecClosed
	}
	if len(data) == 0 {
		return nil, nil
	}
	size := uint32(len(data)) //nolint:gosec // a frame size
	if code := d.lib.decode(d.ctx, &data[0], size, nil, vpxDLRealtime); code != vpxCodecOK {
		return nil, d.lib.err("vpx_codec_decode", code)
	}
	var out []*I420Frame
	var iter unsafe.Pointer
	for img := d.lib.getFrame(d.ctx, &iter); img != nil; img = d.lib.getFrame(d.ctx, &iter) {
		if img.fmt != vpxImgFmtI420 {
			continue
		}
		out = append(out, copyI420(int(img.dW), int(img.dH),
			[3]*byte{img.planes[0], img.planes[1], img.planes[2]},
			[3]int{int(img.stride[0]), int(img.stride[1]), int(img.stride[2])}))
	}
	return out, nil
}

// Close frees the decoder.
func (d *VP8Decoder) Close() {
	d.mu.Lock()
	defer d.mu.Unlock()
	if d.closed {
		return
	}
	d.closed = true
	d.lib.destroy(d.ctx)
	d.pinner.Unpin()
}
