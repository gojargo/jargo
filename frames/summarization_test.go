package frames

import "testing"

// The configuration cases here are ported from upstream's summarization suite.
// Its coverage of the deprecated configuration type is left out: upstream has
// already marked that one for removal in 2.0.0.

func TestSummaryConfigDefaults(t *testing.T) {
	c := SummaryConfig{}.WithDefaults()

	if c.TargetContextTokens != 6000 {
		t.Errorf("TargetContextTokens = %d, want 6000", c.TargetContextTokens)
	}
	if c.MinMessagesAfterSummary != 4 {
		t.Errorf("MinMessagesAfterSummary = %d, want 4", c.MinMessagesAfterSummary)
	}
	if c.SummaryMessageTemplate != DefaultSummaryMessageTemplate {
		t.Errorf("SummaryMessageTemplate = %q", c.SummaryMessageTemplate)
	}
	if c.SummarizationTimeout != DefaultSummarizationTimeout {
		t.Errorf("SummarizationTimeout = %v, want %v", c.SummarizationTimeout, DefaultSummarizationTimeout)
	}
}

func TestSummaryConfigPrompt(t *testing.T) {
	if got := (SummaryConfig{}).SummaryPrompt(); got != DefaultSummarizationPrompt {
		t.Error("an unset prompt should be the default summarization prompt")
	}
	if got := (SummaryConfig{SummarizationPrompt: "custom"}).SummaryPrompt(); got != "custom" {
		t.Errorf("SummaryPrompt() = %q, want %q", got, "custom")
	}
}

func TestAutoSummarizationConfigDefaults(t *testing.T) {
	c := NewAutoSummarizationConfig().WithDefaults()

	if c.MaxContextTokens == nil || *c.MaxContextTokens != 8000 {
		t.Errorf("MaxContextTokens = %v, want 8000", c.MaxContextTokens)
	}
	if c.MaxUnsummarizedMessages == nil || *c.MaxUnsummarizedMessages != 20 {
		t.Errorf("MaxUnsummarizedMessages = %v, want 20", c.MaxUnsummarizedMessages)
	}
	if c.SummaryConfig.TargetContextTokens != 6000 {
		t.Errorf("TargetContextTokens = %d, want 6000", c.SummaryConfig.TargetContextTokens)
	}
}

// A summary allowed to be longer than the context limit it compresses into is
// pulled down to a share of that limit.
func TestAutoSummarizationConfigTargetTokensAdjusted(t *testing.T) {
	limit := 4000
	c := AutoSummarizationConfig{
		MaxContextTokens: &limit,
		SummaryConfig:    SummaryConfig{TargetContextTokens: 6000},
	}.WithDefaults()

	if c.SummaryConfig.TargetContextTokens != 3200 {
		t.Errorf("TargetContextTokens = %d, want 3200 (80%% of 4000)", c.SummaryConfig.TargetContextTokens)
	}
}

func TestAutoSummarizationConfigTargetTokensNotAdjustedWhenTokenThresholdOff(t *testing.T) {
	c := AutoSummarizationConfig{
		MaxContextTokens: nil,
		SummaryConfig:    SummaryConfig{TargetContextTokens: 6000},
	}.WithDefaults()

	if c.SummaryConfig.TargetContextTokens != 6000 {
		t.Errorf("TargetContextTokens = %d, want 6000 left alone", c.SummaryConfig.TargetContextTokens)
	}
}

func TestAutoSummarizationConfigOneThresholdMayBeDisabled(t *testing.T) {
	messages, tokens := 20, 8000
	if err := (AutoSummarizationConfig{MaxUnsummarizedMessages: &messages}).Validate(); err != nil {
		t.Errorf("disabling only the token threshold should be allowed: %v", err)
	}
	if err := (AutoSummarizationConfig{MaxContextTokens: &tokens}).Validate(); err != nil {
		t.Errorf("disabling only the message threshold should be allowed: %v", err)
	}
}

func TestAutoSummarizationConfigBothThresholdsDisabledIsRejected(t *testing.T) {
	c := AutoSummarizationConfig{}
	if err := c.Validate(); err == nil {
		t.Error("disabling both thresholds should be rejected")
	}
}
