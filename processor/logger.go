package processor

import (
	"context"
	"log/slog"

	"github.com/gojargo/jargo/frames"
)

// FrameLogger logs every frame passing through it and pushes it on unchanged.
//
// It is a debugging aid: dropped into a pipeline it says what is flowing and
// which way, which is how a pipeline that produces nothing is told from one
// whose frames never arrive. The high-frequency frames are skipped by default,
// since audio and speaking frames arrive many times a second and would bury
// everything else.
type FrameLogger struct {
	*Base
	prefix  string
	ignored FrameMatcher
}

// FrameMatcher reports whether a frame is one of a set.
type FrameMatcher func(frames.Frame) bool

// LoggerOption configures a FrameLogger.
type LoggerOption func(*FrameLogger)

// WithLoggedPrefix labels each line, so several loggers in one pipeline can be
// told apart. The default is "Frame".
func WithLoggedPrefix(prefix string) LoggerOption {
	return func(l *FrameLogger) { l.prefix = prefix }
}

// WithIgnoredFrames replaces the frames left unlogged. A nil matcher logs every
// frame, audio included.
func WithIgnoredFrames(ignored FrameMatcher) LoggerOption {
	return func(l *FrameLogger) { l.ignored = ignored }
}

// defaultIgnoredFrames are the frames a logger skips unless told otherwise: the
// ones that arrive many times a second.
func defaultIgnoredFrames(f frames.Frame) bool {
	switch f.(type) {
	case *frames.BotSpeakingFrame, *frames.UserSpeakingFrame,
		*frames.InputAudioRawFrame, *frames.OutputAudioRawFrame:
		return true
	default:
		return false
	}
}

// NewFrameLogger builds a logger named name that logs what passes through it.
func NewFrameLogger(name string, opts ...LoggerOption) *FrameLogger {
	l := &FrameLogger{prefix: "Frame", ignored: defaultIgnoredFrames}
	for _, opt := range opts {
		opt(l)
	}
	l.Base = New(name, l)
	return l
}

// ProcessFrame logs the frame and pushes it on unchanged.
func (l *FrameLogger) ProcessFrame(ctx context.Context, f frames.Frame, dir Direction) error {
	if err := l.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	if l.ignored == nil || !l.ignored(f) {
		arrow := ">"
		if dir == Upstream {
			arrow = "<"
		}
		slog.DebugContext(ctx, arrow+" "+l.prefix+": "+f.Name(), "frame", f, "direction", dir)
	}
	return l.PushFrame(ctx, f, dir)
}
