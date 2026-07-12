// Package ivr navigates automated phone menus (IVR systems). The Processor
// watches the LLM's streamed response for control tags: <dtmf>N</dtmf> presses
// keypad keys to make a menu selection, and <ivr>status</ivr> reports navigation
// progress. The tags are stripped from the spoken output. Drive it with an LLM
// prompted to emit those tags as it listens to the menu.
package ivr

import (
	"context"
	"strings"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/internal/tagscan"
	"github.com/gojargo/jargo/processor"
)

// Status is the IVR navigation status the LLM reports through an <ivr> tag.
type Status string

const (
	// StatusDetected means an IVR menu was detected and navigation has begun.
	StatusDetected Status = "detected"
	// StatusCompleted means navigation reached a human or the target option.
	StatusCompleted Status = "completed"
	// StatusStuck means the menu could not be navigated.
	StatusStuck Status = "stuck"
	// StatusWait means the system is waiting (e.g. on hold) before acting.
	StatusWait Status = "wait"
)

// Config configures an IVR Processor.
type Config struct {
	// OnStatusChanged is called when the LLM reports an IVR navigation status.
	OnStatusChanged func(status Status)
}

// Processor navigates an IVR by acting on the control tags in the LLM's output:
// it emits an OutputDTMFFrame for each key in a <dtmf> tag and reports each
// <ivr> status through OnStatusChanged, forwarding the remaining text as speech.
type Processor struct {
	*processor.Base
	cfg  Config
	scan *tagscan.Scanner
}

// New builds an IVR Processor.
func New(cfg Config) *Processor {
	p := &Processor{cfg: cfg, scan: tagscan.New("dtmf", "ivr")}
	p.Base = processor.New("IVRNavigator", p)
	return p
}

// ProcessFrame strips and acts on the control tags in the LLM text stream.
func (p *Processor) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := p.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	switch fr := f.(type) {
	case *frames.LLMTextFrame:
		cleaned := p.scan.Feed(fr.Text, func(tag, value string) { p.act(ctx, tag, value) })
		if cleaned == "" {
			return nil
		}
		return p.PushFrame(ctx, frames.NewLLMTextFrame(cleaned), processor.Downstream)
	case *frames.LLMFullResponseEndFrame:
		if rest := p.scan.Flush(); rest != "" {
			if err := p.PushFrame(ctx, frames.NewLLMTextFrame(rest), processor.Downstream); err != nil {
				return err
			}
		}
		return p.PushFrame(ctx, f, dir)
	default:
		return p.PushFrame(ctx, f, dir)
	}
}

// act carries out one control tag: press DTMF keys, or report a status change.
func (p *Processor) act(ctx context.Context, tag, value string) {
	switch tag {
	case "dtmf":
		for _, r := range strings.TrimSpace(value) {
			_ = p.PushFrame(ctx, frames.NewOutputDTMFFrame(frames.KeypadEntry(string(r))), processor.Downstream)
		}
	case "ivr":
		if p.cfg.OnStatusChanged != nil {
			p.cfg.OnStatusChanged(Status(strings.TrimSpace(value)))
		}
	}
}
