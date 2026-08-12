package stt

import (
	"context"
	"log/slog"
	"sync"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/language"
	"github.com/gojargo/jargo/service/settings"
)

// providerSettings applies a settings update to a provider's own store.
//
// Both kinds of transcription service use it, because an update means the same
// thing to either: merge what the caller gave into the store the provider reads,
// having first named the language the way that provider does, and then let the
// provider act on whatever actually changed. What differs is only what a service
// can do about a change it cannot apply to work already in flight, which is why
// apply reports that back rather than acting on it.
type providerSettings struct {
	// mu serializes the merge. A settings update arrives on the frame
	// goroutine, and the provider reads the store from its own, so without this
	// a provider could read a store that is half written.
	mu sync.Mutex

	// provider is the Transcriber or Connector the service was built with. The
	// settings interfaces are optional, so it is held as an any and asked.
	provider any
	// name identifies the service in a log line.
	name func() string
	// onModel is called with the model now in force when it changed, so the
	// service can relabel what it reports.
	onModel func(model string)
}

// hold runs fn with the settings held steady. Work that reads the provider's
// store, opening a session from it say, runs under this so it cannot see a
// store half written by an update arriving on the frame goroutine.
func (p *providerSettings) hold(fn func()) {
	p.mu.Lock()
	defer p.mu.Unlock()
	fn()
}

// store is the provider's own settings, and whether it has any.
func (p *providerSettings) store() (any, bool) {
	holder, ok := p.provider.(SettingsHolder)
	if !ok {
		return nil, false
	}
	return holder.Settings(), true
}

// apply merges an update into the provider's store and lets the provider act on
// what changed. It reports whether the provider wants its session replaced,
// which a service with no session to replace ignores.
func (p *providerSettings) apply(ctx context.Context, f *frames.STTUpdateSettingsFrame) (bool, error) {
	store, ok := p.store()
	if !ok {
		slog.Warn("settings update for a service whose provider has none", "service", p.name())
		return false, nil
	}

	p.mu.Lock()
	defer p.mu.Unlock()

	delta, ok, err := settings.Resolve(&f.ServiceUpdateSettingsFrame, store)
	if err != nil {
		return false, err
	}
	if !ok {
		return false, nil
	}

	// Naming the language the provider's way before applying is what keeps the
	// comparison honest: the store holds the provider's code, so a neutral name
	// that means the same language must be converted first or it reads as a
	// change when nothing changed.
	p.nameLanguage(delta)

	changed, err := settings.Apply(store, delta)
	if err != nil {
		return false, err
	}
	if len(changed) == 0 {
		return false, nil
	}
	slog.Info("updated settings", "service", p.name(), "fields", changed.String())

	if changed.Has("model") && p.onModel != nil {
		// The model labels the usage this service reports, and it is priced
		// against it, so a model that changed mid-call has to relabel what
		// follows or the cost lands against the wrong one.
		name, _ := settings.Get(store, "model")
		model, _ := name.(string)
		p.onModel(model)
	}

	updater, ok := p.provider.(SettingsUpdater)
	if !ok {
		return false, nil
	}
	return updater.UpdateSettings(ctx, changed)
}

// nameLanguage rewrites a language the delta gives into the code the provider
// uses, when the provider says how. A code the provider does not recognize is
// left as it came, since it may be one the service accepts directly.
func (p *providerSettings) nameLanguage(delta any) {
	namer, ok := p.provider.(LanguageNamer)
	if !ok {
		return
	}
	value, ok := settings.Get(delta, "language")
	if !ok {
		return
	}
	code, ok := value.(string)
	if !ok || code == "" {
		return
	}
	named := namer.ServiceLanguage(language.Language(code))
	if named == "" || named == code {
		return
	}
	if err := settings.SetNamed(delta, "language", named); err != nil {
		slog.Warn("naming the language the provider's way failed",
			"service", p.name(), "err", err)
	}
}

// traceSettings is what the provider's store gives, for the span covering the
// work. It is nil for a provider that holds no settings.
func (p *providerSettings) traceSettings() map[string]any {
	store, ok := p.store()
	if !ok {
		return nil
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	given, err := settings.Given(store)
	if err != nil {
		return nil
	}
	return given
}
