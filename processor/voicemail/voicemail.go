// Package voicemail detects whether an outbound call reached a person or a
// voicemail system. The Processor watches the LLM's classification output for a
// decision tag — <voicemail></voicemail> or <human></human> — and fires the
// matching callback, so the app can leave a message and hang up, or start the
// conversation. Drive it with an LLM prompted to emit one of those tags after
// hearing the first utterance. The decision tags are stripped from the output.
package voicemail

import (
	"context"
	"sync"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/internal/tagscan"
	"github.com/gojargo/jargo/processor"
)

// Config configures a voicemail Processor.
type Config struct {
	// OnVoicemailDetected is called when the classifier decides the call reached
	// voicemail. It fires after VoicemailDelay.
	OnVoicemailDetected func()
	// OnConversationDetected is called when the classifier decides a person
	// answered.
	OnConversationDetected func()
	// VoicemailDelay delays the voicemail callback — for example to let the
	// greeting finish before leaving a message; 0 fires immediately.
	VoicemailDelay time.Duration
}

// Processor classifies a call as voicemail or human from the LLM's decision tag
// and fires the matching callback exactly once. It strips the decision tags from
// the forwarded text.
type Processor struct {
	*processor.Base
	cfg  Config
	scan *tagscan.Scanner

	mu      sync.Mutex
	decided bool
	timer   *time.Timer
}

// New builds a voicemail Processor.
func New(cfg Config) *Processor {
	p := &Processor{cfg: cfg, scan: tagscan.New("voicemail", "human", "conversation")}
	p.Base = processor.New("VoicemailDetector", p)
	return p
}

// ProcessFrame strips and acts on the decision tags in the LLM text stream.
func (p *Processor) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := p.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	switch fr := f.(type) {
	case *frames.LLMTextFrame:
		cleaned := p.scan.Feed(fr.Text, func(tag, _ string) { p.decide(tag) })
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

// Cleanup stops the pending voicemail timer and tears down the processor.
func (p *Processor) Cleanup(ctx context.Context) error {
	p.mu.Lock()
	if p.timer != nil {
		p.timer.Stop()
		p.timer = nil
	}
	p.mu.Unlock()
	return p.Base.Cleanup(ctx)
}

// decide fires the matching callback for the first decision tag seen.
func (p *Processor) decide(tag string) {
	p.mu.Lock()
	if p.decided {
		p.mu.Unlock()
		return
	}
	p.decided = true
	if tag == "voicemail" {
		if p.cfg.VoicemailDelay > 0 {
			p.timer = time.AfterFunc(p.cfg.VoicemailDelay, func() {
				if p.cfg.OnVoicemailDetected != nil {
					p.cfg.OnVoicemailDetected()
				}
			})
			p.mu.Unlock()
			return
		}
		p.mu.Unlock()
		if p.cfg.OnVoicemailDetected != nil {
			p.cfg.OnVoicemailDetected()
		}
		return
	}
	// "human" or "conversation".
	p.mu.Unlock()
	if p.cfg.OnConversationDetected != nil {
		p.cfg.OnConversationDetected()
	}
}
