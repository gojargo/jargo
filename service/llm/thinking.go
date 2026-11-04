package llm

import (
	"log/slog"

	"github.com/gojargo/jargo/service/settings"
)

// settingUseProviderThinkingDefault is the settings field that has a model with
// no thinking configured run at its provider's default.
const settingUseProviderThinkingDefault = "use_provider_thinking_default"

// UseProviderThinkingDefault reports whether a model with no thinking
// configured runs at its provider's default rather than at the lowest setting
// the service would otherwise send it. It is the settings field of that name,
// false when unset.
func (b *Base) UseProviderThinkingDefault() bool {
	v, _ := settings.Get(b.settingsStore(), settingUseProviderThinkingDefault)
	on, _ := v.(bool)
	return on
}

// SetUseProviderThinkingDefault sets whether a model with no thinking configured
// runs at its provider's default. A backend worker sets it on its LLM, which does
// the reasoning its frontend hands off. Set it before the pipeline runs; once it
// does, change it with an LLMUpdateSettingsFrame.
func (b *Base) SetUseProviderThinkingDefault(on bool) {
	if err := settings.SetNamed(b.settingsStore(), settingUseProviderThinkingDefault, on); err != nil {
		slog.Error("setting use_provider_thinking_default failed", "service", b.Name(), "err", err)
	}
}

// ReportThinkingDefault reports, once per model, the thinking default a service
// sends a model whose request configures no thinking of its own.
//
// Services that send each model a low-latency thinking default call it whenever
// they apply that default, or find they have none for the model, or send none
// because UseProviderThinkingDefault is set. The first request for each model
// logs the setting sent, or that the model runs at the provider's default:
// at DEBUG when UseProviderThinkingDefault asked for that, and otherwise, for a
// model that can think but has no default, as a warning, since it can think
// before replying and slow the responses down.
//
// setting is the default as it is sent, such as `thinking={"type":"disabled"}`,
// or empty when the service lists none for the model. canThink decides whether a
// model with no default is warned about. provider names the provider, and
// configure the configuration that sets thinking, such as "Thinking or Effort in
// anthropic.Config", for the log lines.
func (b *Base) ReportThinkingDefault(model, setting string, canThink bool, provider, configure string) {
	b.thinkingDefaultMu.Lock()
	if model == b.loggedThinkingDefaultModel {
		b.thinkingDefaultMu.Unlock()
		return
	}
	b.loggedThinkingDefaultModel = model
	b.thinkingDefaultMu.Unlock()

	switch {
	case b.UseProviderThinkingDefault():
		if canThink {
			slog.Debug("use_provider_thinking_default is set, so the model runs at the provider's "+
				"default thinking; to change it, configure it",
				"service", b.Name(), "model", model, "provider", provider, "configure", configure)
		}
	case setting != "":
		slog.Info("sending the model jargo's low-latency thinking default for it; "+
			"if the bot needs multistep reasoning, configure it",
			"service", b.Name(), "model", model, "setting", setting, "configure", configure)
	case canThink:
		slog.Warn("no thinking default for the model, so it runs at the provider's default, "+
			"which can think before replying and slow responses; for the fastest responses, "+
			"configure the lowest setting the model supports",
			"service", b.Name(), "model", model, "provider", provider, "configure", configure)
	}
}
