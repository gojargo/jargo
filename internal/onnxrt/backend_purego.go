//go:build !cgo

// The pure-Go backend: selected by a CGO_ENABLED=0 build. It binds the ONNX
// Runtime C API directly with ebitengine/purego — no cgo, no C toolchain — and
// dlopen's the same shared library at run time.
//
// The ONNX Runtime C API is not a flat symbol table: you Dlsym exactly one
// symbol, OrtGetApiBase, walk it to *OrtApi (a struct of function pointers),
// and index that table by ordinal. The ordinals below were extracted from
// onnxruntime_c_api.h (ORT_API_VERSION 26) and verified against the canonical
// prefix (0=CreateStatus, 1=GetErrorCode, 2=GetErrorMessage, 3=CreateEnv, ...).
// ORT only appends to the table, so a table built for API 26 reads the same
// offsets on any library that supports >= 26; a wrong ordinal is a silent call
// to the wrong function, so these are covered by the smart-turn/silero tests.

package onnxrt

import (
	"errors"
	"fmt"
	"runtime"
	"unsafe"

	"github.com/ebitengine/purego"
)

// backendName identifies the compiled-in backend.
const backendName = "purego"

var (
	errNoData     = errors.New("onnxrt: input tensor has no data")
	errEmptyShape = errors.New("onnxrt: input tensor has empty shape")
	errBadOutput  = errors.New("onnxrt: output is not a float32 tensor")
	errInputCount = errors.New("onnxrt: input count does not match model")
	errNullAPI    = errors.New("onnxrt: ONNX Runtime C API unavailable")
	errRuntime    = errors.New("onnxrt: runtime error")
)

// OrtApi function-pointer ordinals.
const (
	fnGetErrorMessage        = 2
	fnCreateEnv              = 3
	fnCreateSessionFromArr   = 8
	fnRun                    = 9
	fnCreateSessionOptions   = 10
	fnSetIntraOpNumThreads   = 24
	fnGetTensorElementType   = 60
	fnGetDimensionsCount     = 61
	fnGetDimensions          = 62
	fnGetTensorShapeElemCnt  = 64
	fnGetTensorTypeAndShape  = 65
	fnCreateCPUMemoryInfo    = 69
	fnCreateTensorWithData   = 49
	fnGetTensorMutableData   = 51
	fnReleaseEnv             = 92
	fnReleaseStatus          = 93
	fnReleaseMemoryInfo      = 94
	fnReleaseSession         = 95
	fnReleaseValue           = 96
	fnReleaseTensorTypeShape = 99
	fnReleaseSessionOptions  = 100
)

// ORT enum values from the header.
const (
	loggingLevelWarning = 2 // ORT_LOGGING_LEVEL_WARNING
	tensorElemTypeFloat = 1 // ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
	tensorElemTypeInt64 = 7 // ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64
	allocatorTypeArena  = 1 // OrtArenaAllocator
	memTypeDefault      = 0 // OrtMemTypeDefault
)

// ortAPI holds the resolved function pointers. Every ORT handle (env, session,
// options, memory info, OrtValue, OrtStatus, type-and-shape info) is an opaque
// uintptr; no struct crosses by value; tensor data crosses as a pointer. That
// is squarely inside purego's well-supported zone.
type ortAPI struct {
	createEnv          func(logLevel uint32, logid *byte, out *uintptr) uintptr
	createSessionOpts  func(out *uintptr) uintptr
	setIntraOpThreads  func(opts uintptr, n int32) uintptr
	createSessionFromA func(env uintptr, data *byte, dataLen uint64, opts uintptr, out *uintptr) uintptr
	createCPUMemInfo   func(allocType int32, memType int32, out *uintptr) uintptr
	createTensor       func(
		info uintptr, data unsafe.Pointer, dataLen uint64,
		shape *int64, shapeLen uint64, dtype uint32, out *uintptr,
	) uintptr
	run func(
		sess, runOpts uintptr, inNames **byte, inputs *uintptr, inLen uint64,
		outNames **byte, outLen uint64, outputs *uintptr,
	) uintptr
	getTensorData      func(value uintptr, out *uintptr) uintptr
	getTypeAndShape    func(value uintptr, out *uintptr) uintptr
	getElementType     func(info uintptr, out *uint32) uintptr
	getDimCount        func(info uintptr, out *uint64) uintptr
	getDims            func(info uintptr, dims *int64, dimsLen uint64) uintptr
	getElementCount    func(info uintptr, out *uint64) uintptr
	getErrorMessage    func(status uintptr) uintptr
	releaseStatus      func(uintptr)
	releaseEnv         func(uintptr)
	releaseSession     func(uintptr)
	releaseSessionOpts func(uintptr)
	releaseValue       func(uintptr)
	releaseMemInfo     func(uintptr)
	releaseTypeShape   func(uintptr)
}

//nolint:gochecknoglobals // the loaded runtime is process-wide, guarded by onnxrt.Init's sync.Once
var (
	api       *ortAPI
	sharedEnv uintptr
	sharedMem uintptr
)

// member reads the function pointer at the given ordinal out of the OrtApi
// table.
func member(base uintptr, ordinal int) uintptr {
	//nolint:gosec,govet // base is a C pointer (the OrtApi table), not Go-managed memory
	return *(*uintptr)(unsafe.Pointer(base + uintptr(ordinal)*unsafe.Sizeof(uintptr(0))))
}

// backendInit dlopen's the library, resolves the OrtApi table, and creates the
// process-wide env and CPU memory info. Called once via onnxrt.Init.
func backendInit(libPath string) error {
	handle, err := purego.Dlopen(libPath, purego.RTLD_NOW|purego.RTLD_GLOBAL)
	if err != nil {
		return fmt.Errorf("dlopen %q: %w", libPath, err)
	}

	var getAPIBase func() uintptr
	purego.RegisterLibFunc(&getAPIBase, handle, "OrtGetApiBase")
	apiBase := getAPIBase() // *OrtApiBase
	if apiBase == 0 {
		return fmt.Errorf("%w: OrtGetApiBase returned null", errNullAPI)
	}
	// OrtApiBase field 0 is GetApi(uint32_t version) -> const OrtApi*. Request
	// the newest API we know, falling back so an older library still binds; all
	// functions we call have existed since early versions.
	var getAPI func(uint32) uintptr
	//nolint:gosec,govet // C pointer returned from the FFI boundary arrives as a uintptr
	purego.RegisterFunc(&getAPI, *(*uintptr)(unsafe.Pointer(apiBase)))
	var base uintptr
	for _, ver := range []uint32{26, 22, 18, 14, 11} {
		if base = getAPI(ver); base != 0 {
			break
		}
	}
	if base == 0 {
		return fmt.Errorf("%w: GetApi returned null for every known API version", errNullAPI)
	}

	a := &ortAPI{}
	purego.RegisterFunc(&a.createEnv, member(base, fnCreateEnv))
	purego.RegisterFunc(&a.createSessionOpts, member(base, fnCreateSessionOptions))
	purego.RegisterFunc(&a.setIntraOpThreads, member(base, fnSetIntraOpNumThreads))
	purego.RegisterFunc(&a.createSessionFromA, member(base, fnCreateSessionFromArr))
	purego.RegisterFunc(&a.createCPUMemInfo, member(base, fnCreateCPUMemoryInfo))
	purego.RegisterFunc(&a.createTensor, member(base, fnCreateTensorWithData))
	purego.RegisterFunc(&a.run, member(base, fnRun))
	purego.RegisterFunc(&a.getTensorData, member(base, fnGetTensorMutableData))
	purego.RegisterFunc(&a.getTypeAndShape, member(base, fnGetTensorTypeAndShape))
	purego.RegisterFunc(&a.getElementType, member(base, fnGetTensorElementType))
	purego.RegisterFunc(&a.getDimCount, member(base, fnGetDimensionsCount))
	purego.RegisterFunc(&a.getDims, member(base, fnGetDimensions))
	purego.RegisterFunc(&a.getElementCount, member(base, fnGetTensorShapeElemCnt))
	purego.RegisterFunc(&a.getErrorMessage, member(base, fnGetErrorMessage))
	purego.RegisterFunc(&a.releaseStatus, member(base, fnReleaseStatus))
	purego.RegisterFunc(&a.releaseEnv, member(base, fnReleaseEnv))
	purego.RegisterFunc(&a.releaseSession, member(base, fnReleaseSession))
	purego.RegisterFunc(&a.releaseSessionOpts, member(base, fnReleaseSessionOptions))
	purego.RegisterFunc(&a.releaseValue, member(base, fnReleaseValue))
	purego.RegisterFunc(&a.releaseMemInfo, member(base, fnReleaseMemoryInfo))
	purego.RegisterFunc(&a.releaseTypeShape, member(base, fnReleaseTensorTypeShape))

	logid := cstr("jargo")
	if err := a.check(a.createEnv(loggingLevelWarning, &logid[0], &sharedEnv), "CreateEnv"); err != nil {
		return err
	}
	memStatus := a.createCPUMemInfo(allocatorTypeArena, memTypeDefault, &sharedMem)
	if err := a.check(memStatus, "CreateCpuMemoryInfo"); err != nil {
		return err
	}
	api = a
	return nil
}

// check turns a non-null OrtStatus* into a Go error and releases it.
func (a *ortAPI) check(status uintptr, op string) error {
	if status == 0 {
		return nil
	}
	msg := goString(a.getErrorMessage(status))
	a.releaseStatus(status)
	return fmt.Errorf("%w: %s: %s", errRuntime, op, msg)
}

// goString copies a NUL-terminated C string.
func goString(p uintptr) string {
	if p == 0 {
		return ""
	}
	var b []byte
	for i := uintptr(0); ; i++ {
		//nolint:gosec,govet // p is a C string pointer returned by the runtime
		c := *(*byte)(unsafe.Pointer(p + i))
		if c == 0 {
			break
		}
		b = append(b, c)
	}
	return string(b)
}

// cstr returns a NUL-terminated copy of s. The returned slice must be kept
// alive for as long as C holds a pointer into it.
func cstr(s string) []byte { return append([]byte(s), 0) }

type puregoSession struct {
	sess     uintptr
	opts     uintptr
	inNames  []*byte // C string pointers, kept alive by the session
	outNames []*byte
	nOut     int
}

func newBackendSession(model []byte, inputNames, outputNames []string, o Options) (backendSession, error) {
	s := &puregoSession{nOut: len(outputNames)}
	if err := api.check(api.createSessionOpts(&s.opts), "CreateSessionOptions"); err != nil {
		return nil, err
	}
	if o.IntraOpThreads > 0 {
		if err := api.check(api.setIntraOpThreads(s.opts, int32(o.IntraOpThreads)), "SetIntraOpNumThreads"); err != nil {
			api.releaseSessionOpts(s.opts)
			return nil, err
		}
	}
	sessStatus := api.createSessionFromA(sharedEnv, &model[0], uint64(len(model)), s.opts, &s.sess)
	if err := api.check(sessStatus, "CreateSessionFromArray"); err != nil {
		api.releaseSessionOpts(s.opts)
		return nil, err
	}
	for _, n := range inputNames {
		b := cstr(n)
		s.inNames = append(s.inNames, &b[0])
	}
	for _, n := range outputNames {
		b := cstr(n)
		s.outNames = append(s.outNames, &b[0])
	}
	return s, nil
}

func (s *puregoSession) run(inputs []Tensor) ([]Tensor, error) {
	if len(inputs) != len(s.inNames) {
		return nil, fmt.Errorf("%w: got %d, want %d", errInputCount, len(inputs), len(s.inNames))
	}

	handles := make([]uintptr, len(inputs))
	defer func() {
		for _, h := range handles {
			if h != 0 {
				api.releaseValue(h)
			}
		}
	}()
	for i := range inputs {
		h, err := s.newInputValue(inputs[i])
		if err != nil {
			return nil, err
		}
		handles[i] = h
	}

	outs := make([]uintptr, s.nOut)
	st := api.run(s.sess, 0, &s.inNames[0], &handles[0], uint64(len(handles)), &s.outNames[0], uint64(s.nOut), &outs[0])
	runtime.KeepAlive(inputs)
	if err := api.check(st, "Run"); err != nil {
		return nil, err
	}
	defer func() {
		for _, h := range outs {
			if h != 0 {
				api.releaseValue(h)
			}
		}
	}()

	result := make([]Tensor, s.nOut)
	for i, h := range outs {
		t, err := readFloat32Output(h)
		if err != nil {
			return nil, fmt.Errorf("onnxrt: output %d: %w", i, err)
		}
		result[i] = t
	}
	return result, nil
}

// newInputValue wraps a Tensor's data as an OrtValue. ORT reads the data
// pointer for the value's lifetime, so the caller keeps the input slices alive
// (via runtime.KeepAlive) until Run returns and the value is released.
func (s *puregoSession) newInputValue(t Tensor) (uintptr, error) {
	var (
		data  unsafe.Pointer
		nByte uint64
		dtype uint32
	)
	switch {
	case t.F32 != nil:
		//nolint:gosec // wrapping a Go slice as a C tensor buffer for the duration of Run
		data, nByte, dtype = unsafe.Pointer(&t.F32[0]), uint64(len(t.F32))*4, tensorElemTypeFloat
	case t.I64 != nil:
		//nolint:gosec // wrapping a Go slice as a C tensor buffer for the duration of Run
		data, nByte, dtype = unsafe.Pointer(&t.I64[0]), uint64(len(t.I64))*8, tensorElemTypeInt64
	default:
		return 0, errNoData
	}
	if len(t.Shape) == 0 {
		return 0, errEmptyShape
	}
	var h uintptr
	st := api.createTensor(sharedMem, data, nByte, &t.Shape[0], uint64(len(t.Shape)), dtype, &h)
	return h, api.check(st, "CreateTensorWithDataAsOrtValue")
}

// readFloat32Output copies a float32 output tensor (shape + data) into a Tensor.
func readFloat32Output(h uintptr) (Tensor, error) {
	var info uintptr
	if err := api.check(api.getTypeAndShape(h, &info), "GetTensorTypeAndShape"); err != nil {
		return Tensor{}, err
	}
	defer api.releaseTypeShape(info)

	var elemType uint32
	if err := api.check(api.getElementType(info, &elemType), "GetTensorElementType"); err != nil {
		return Tensor{}, err
	}
	if elemType != tensorElemTypeFloat {
		return Tensor{}, fmt.Errorf("%w: element type %d", errBadOutput, elemType)
	}

	var ndim uint64
	if err := api.check(api.getDimCount(info, &ndim), "GetDimensionsCount"); err != nil {
		return Tensor{}, err
	}
	shape := make([]int64, ndim)
	if ndim > 0 {
		if err := api.check(api.getDims(info, &shape[0], ndim), "GetDimensions"); err != nil {
			return Tensor{}, err
		}
	}

	var count uint64
	if err := api.check(api.getElementCount(info, &count), "GetTensorShapeElementCount"); err != nil {
		return Tensor{}, err
	}

	var dataPtr uintptr
	if err := api.check(api.getTensorData(h, &dataPtr), "GetTensorMutableData"); err != nil {
		return Tensor{}, err
	}
	data := make([]float32, count)
	if count > 0 {
		//nolint:gosec,govet // dataPtr is the C tensor buffer returned by GetTensorMutableData
		copy(data, unsafe.Slice((*float32)(unsafe.Pointer(dataPtr)), int(count)))
	}
	return Tensor{Shape: shape, F32: data}, nil
}

func (s *puregoSession) close() error {
	if s.sess != 0 {
		api.releaseSession(s.sess)
		s.sess = 0
	}
	if s.opts != 0 {
		api.releaseSessionOpts(s.opts)
		s.opts = 0
	}
	return nil
}
