//go:build cgo

// The cgo backend (the default): it binds the C++ ONNX Runtime through
// yalue/onnxruntime_go, which uses cgo. A normal CGO_ENABLED=1 build selects it.

package onnxrt

import (
	"errors"
	"fmt"

	ort "github.com/yalue/onnxruntime_go"
)

// backendName identifies the compiled-in backend.
const backendName = "cgo (yalue)"

var (
	errNoData    = errors.New("onnxrt: input tensor has no data")
	errBadOutput = errors.New("onnxrt: output is not a float32 tensor")
)

func backendInit(path string) error {
	if ort.IsInitialized() {
		return nil
	}
	ort.SetSharedLibraryPath(path)
	return ort.InitializeEnvironment()
}

type cgoSession struct {
	s       *ort.DynamicAdvancedSession
	outputs int
}

func newBackendSession(model []byte, inputNames, outputNames []string, o Options) (backendSession, error) {
	opts, err := ort.NewSessionOptions()
	if err != nil {
		return nil, fmt.Errorf("onnxrt: session options: %w", err)
	}
	defer func() { _ = opts.Destroy() }()
	if o.IntraOpThreads > 0 {
		if err = opts.SetIntraOpNumThreads(o.IntraOpThreads); err != nil {
			return nil, fmt.Errorf("onnxrt: set intra-op threads: %w", err)
		}
	}

	s, err := ort.NewDynamicAdvancedSessionWithONNXData(model, inputNames, outputNames, opts)
	if err != nil {
		return nil, fmt.Errorf("onnxrt: create session: %w", err)
	}
	return &cgoSession{s: s, outputs: len(outputNames)}, nil
}

func (c *cgoSession) run(inputs []Tensor) ([]Tensor, error) {
	inVals := make([]ort.Value, 0, len(inputs))
	defer func() {
		for _, v := range inVals {
			_ = v.Destroy()
		}
	}()
	for _, t := range inputs {
		v, err := newValue(t)
		if err != nil {
			return nil, err
		}
		inVals = append(inVals, v)
	}

	outVals := make([]ort.Value, c.outputs)
	if err := c.s.Run(inVals, outVals); err != nil {
		return nil, err
	}
	defer func() {
		for _, v := range outVals {
			if v != nil {
				_ = v.Destroy()
			}
		}
	}()

	out := make([]Tensor, len(outVals))
	for i, v := range outVals {
		t, ok := v.(*ort.Tensor[float32])
		if !ok {
			return nil, fmt.Errorf("%w: output %d is %T", errBadOutput, i, v)
		}
		// Copy out before the values are destroyed above.
		data := t.GetData()
		cp := make([]float32, len(data))
		copy(cp, data)
		shape := t.GetShape()
		sh := make([]int64, len(shape))
		copy(sh, shape)
		out[i] = Tensor{Shape: sh, F32: cp}
	}
	return out, nil
}

func (c *cgoSession) close() error { return c.s.Destroy() }

func newValue(t Tensor) (ort.Value, error) {
	switch {
	case t.F32 != nil:
		return ort.NewTensor(ort.NewShape(t.Shape...), t.F32)
	case t.I64 != nil:
		return ort.NewTensor(ort.NewShape(t.Shape...), t.I64)
	default:
		return nil, errNoData
	}
}
