//go:build cgo

// The cgo backend (the default): it binds the C++ ONNX Runtime through the
// runtime's own Go binding, which uses cgo. A normal CGO_ENABLED=1 build selects
// it.

package onnxrt

import (
	"context"
	"errors"
	"fmt"

	ort "github.com/microsoft/onnxruntime/go/onnxruntime"
)

// backendName identifies the compiled-in backend.
const backendName = "cgo (onnxruntime)"

var (
	errNoData     = errors.New("onnxrt: input tensor has no data")
	errInputCount = errors.New("onnxrt: wrong number of inputs")
	errNoOutput   = errors.New("onnxrt: the model produced no output of that name")
)

func backendInit(path string) error {
	if ort.IsInitialized() {
		return nil
	}
	ort.SetSharedLibraryPath(path)
	return ort.Init()
}

// cgoSession is one loaded model, plus the names its tensors are keyed by. The
// binding addresses inputs and outputs by name where jargo's Run is positional,
// so the session holds the names and zips them itself.
type cgoSession struct {
	s           *ort.Session
	inputNames  []string
	outputNames []string
}

func newBackendSession(model []byte, inputNames, outputNames []string, o Options) (backendSession, error) {
	opts, err := ort.NewSessionOptions()
	if err != nil {
		return nil, fmt.Errorf("onnxrt: session options: %w", err)
	}
	defer func() { _ = opts.Close() }()
	if o.IntraOpThreads > 0 {
		if err = opts.SetIntraOpNumThreads(o.IntraOpThreads); err != nil {
			return nil, fmt.Errorf("onnxrt: set intra-op threads: %w", err)
		}
	}

	s, err := ort.NewSessionFromBytes(model, opts)
	if err != nil {
		return nil, fmt.Errorf("onnxrt: create session: %w", err)
	}
	return &cgoSession{
		s:           s,
		inputNames:  append([]string(nil), inputNames...),
		outputNames: append([]string(nil), outputNames...),
	}, nil
}

func (c *cgoSession) run(inputs []Tensor) ([]Tensor, error) {
	if len(inputs) != len(c.inputNames) {
		return nil, fmt.Errorf("%w: got %d, want %d", errInputCount, len(inputs), len(c.inputNames))
	}

	in := make(map[string]*ort.Tensor, len(inputs))
	// An input tensor pins the caller's slice rather than copying it, so it stays
	// alive until the tensor is closed. Closing them all together, after the run,
	// is what releases those pins.
	defer func() {
		for _, v := range in {
			_ = v.Close()
		}
	}()
	for i, t := range inputs {
		v, err := newValue(t)
		if err != nil {
			return nil, err
		}
		in[c.inputNames[i]] = v
	}

	// The binding can abandon a run on a canceled context. Nothing above here
	// carries one yet: the analyzers that own these sessions are reached through
	// interfaces that take no context, so giving them one is a change of its own.
	outs, err := c.s.Run(context.Background(), in, c.outputNames)
	if err != nil {
		return nil, err
	}
	defer func() {
		for _, v := range outs {
			_ = v.Close()
		}
	}()

	out := make([]Tensor, len(c.outputNames))
	for i, name := range c.outputNames {
		v, ok := outs[name]
		if !ok {
			return nil, fmt.Errorf("%w: %q", errNoOutput, name)
		}
		// The data is a view into the tensor's own buffer, so it is copied out
		// before the tensors are closed above.
		data, err := ort.TensorData[float32](v)
		if err != nil {
			return nil, fmt.Errorf("onnxrt: read output %q: %w", name, err)
		}
		out[i] = Tensor{
			Shape: append([]int64(nil), v.Shape()...),
			F32:   append([]float32(nil), data...),
		}
	}
	return out, nil
}

func (c *cgoSession) close() error { return c.s.Close() }

func newValue(t Tensor) (*ort.Tensor, error) {
	switch {
	case t.F32 != nil:
		return ort.CreateTensor(t.Shape, t.F32)
	case t.I64 != nil:
		return ort.CreateTensor(t.Shape, t.I64)
	default:
		return nil, errNoData
	}
}
