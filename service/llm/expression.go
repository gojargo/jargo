package llm

import (
	"log/slog"

	"github.com/gojargo/jargo/expressive"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
)

// With expressive conversation on (alpha), the service teaches itself the
// expression tags the TTS service it feeds can perform, in its system
// instruction, so a prompt does not need to describe them. It reads them from
// the TTSMetadataFrame that TTS service broadcasts, and rebuilds the
// instruction from each one, so it follows a switch of TTS service or a
// settings change that alters what the TTS performs.

// expressionState is what the service knows of the expressive conversation it
// takes part in.
type expressionState struct {
	// expressive is whether the pipeline has expressive conversation on.
	expressive bool
	// tts is the expression the TTS service downstream performs, from its
	// latest TTSMetadataFrame.
	tts *expressive.ExpressionCapabilities
	// policy is the pipeline's policy on which tags to offer and when.
	policy *expressive.ExpressionPolicy
	// policyWarning is the last warning about the policy, so it is not
	// repeated.
	policyWarning string
}

// setupExpression records whether the pipeline has expressive conversation on,
// and its policy.
func (b *Base) setupExpression(s processor.Setup) {
	b.systemMu.Lock()
	defer b.systemMu.Unlock()
	b.expression.expressive = s.EnableExpressiveConversation
	b.expression.policy = s.ExpressionPolicy
}

// handleExpressionMetadata teaches the expression a metadata frame announces,
// with expressive conversation on. The frame travels on either way.
func (b *Base) handleExpressionMetadata(f frames.Frame) {
	md, ok := f.(*frames.TTSMetadataFrame)
	if !ok {
		return
	}
	b.systemMu.Lock()
	if !b.expression.expressive {
		b.systemMu.Unlock()
		return
	}
	// The TTS service's metadata replaces any before it.
	b.expression.tts = md.Expression
	warning := b.policyWarningLocked()
	b.systemMu.Unlock()
	if warning != "" {
		slog.Warn(warning, "service", b.Name())
	}
	b.composeSystemInstruction()
}

// policyWarningLocked is the warning about what the policy asks for that the
// TTS service cannot perform, or the empty string. Each warning is given once,
// until what is left out changes, so a switch to a service with the same gap
// does not repeat it. It runs under systemMu.
func (b *Base) policyWarningLocked() string {
	warning := expressive.ExpressionPolicyWarning(b.expression.tts, b.expression.policy)
	repeated := warning == b.expression.policyWarning
	b.expression.policyWarning = warning
	if repeated {
		return ""
	}
	return warning
}

// expressionInstructionsLocked is the part of the system instruction that
// teaches the expression tags, or the empty string. It runs under systemMu.
func (b *Base) expressionInstructionsLocked() string {
	return expressive.ExpressionInstructions(b.expression.tts, b.expression.policy)
}
