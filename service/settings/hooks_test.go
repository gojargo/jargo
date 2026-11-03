package settings_test

import (
	"testing"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/service/settings"
)

// session is a nested value carrying a copy of a top-level field, as a
// realtime session's properties carry the model.
type session struct {
	Model string
	Voice string
}

// mirrored keeps its session's model in step with its own, and routes the
// session's names into the session when built from plain data.
type mirrored struct {
	settings.Base
	Session settings.Opt[session] `settings:"session"`
}

func (m *mirrored) ApplyUpdate(delta any) (settings.Changed, error) {
	changed, err := settings.ApplyFields(m, delta)
	if err != nil {
		return nil, err
	}
	if model, ok := m.Model.Value(); ok {
		if sess, ok := m.Session.Value(); ok && sess.Model != model {
			sess.Model = model
			m.Session = settings.Set(sess)
		}
	}
	return changed, nil
}

func (m *mirrored) FromMap(values map[string]any) error {
	top := map[string]any{}
	for k, v := range values {
		if k == "voice" {
			voice, _ := v.(string)
			m.Session = settings.Set(session{Voice: voice})
			continue
		}
		top[k] = v
	}
	return settings.FromMapFields(m, top)
}

var (
	_ settings.Updater = (*mirrored)(nil)
	_ settings.Mapper  = (*mirrored)(nil)
)

// A store implementing Updater merges an update its own way: here the session
// it holds follows a model given at the top level.
func TestApplyDefersToAnUpdater(t *testing.T) {
	t.Parallel()
	store := &mirrored{Session: settings.Set(session{Model: "old", Voice: "alloy"})}
	store.Model = settings.Set("old")
	delta := &mirrored{}
	delta.Model = settings.Set("new")

	changed, err := settings.Apply(store, delta)
	if err != nil {
		t.Fatalf("apply: %v", err)
	}
	if !changed.Has("model") {
		t.Errorf("changed = %v, want the model", changed)
	}
	if sess, _ := store.Session.Value(); sess.Model != "new" || sess.Voice != "alloy" {
		t.Errorf("session = %+v, want the new model and the voice kept", sess)
	}
}

// A delta implementing Mapper is filled its own way, including when an update
// carrying plain data is resolved against the store.
func TestResolveDefersToAMapper(t *testing.T) {
	t.Parallel()
	store := &mirrored{}
	f := frames.ServiceUpdateSettingsFrame{Settings: map[string]any{"voice": "echo", "model": "m"}}

	delta, ok, err := settings.Resolve(&f, store)
	if err != nil || !ok {
		t.Fatalf("resolve: ok=%v err=%v", ok, err)
	}
	d, _ := delta.(*mirrored)
	if sess, _ := d.Session.Value(); sess.Voice != "echo" {
		t.Errorf("session = %+v, want the voice routed into it", sess)
	}
	if v, _ := d.Model.Value(); v != "m" {
		t.Errorf("model = %q, want it on the top-level field", v)
	}
	if len(d.Extra) != 0 {
		t.Errorf("extra = %v, want nothing left over", d.Extra)
	}
}
