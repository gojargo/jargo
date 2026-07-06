//go:build !rnnoise

package rnnoise

import (
	"context"

	"github.com/gojargo/jargo/transport"
)

// New returns a no-op passthrough filter. Build with -tags rnnoise to link
// librnnoise and get the real denoiser.
func New() (transport.AudioFilter, error) {
	return passthrough{}, nil
}

type passthrough struct{}

func (passthrough) Start(context.Context, int) error { return nil }

func (passthrough) Stop(context.Context) error { return nil }

func (passthrough) Filter(_ context.Context, pcm []byte) ([]byte, error) { return pcm, nil }
