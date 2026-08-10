package frames

import "fmt"

// ServiceTarget identifies one service in the pipeline, so a settings update
// meant for a single service is applied by that one and forwarded untouched by
// the rest. It is declared here, rather than imported from the processor
// package, so the frames package keeps no dependency on it; a frame processor
// satisfies it by exposing its name.
type ServiceTarget interface {
	Name() string
}

// ServiceUpdateSettingsFrame changes a service's settings while the pipeline
// runs: the voice a bot speaks in, the language it transcribes, the model it
// answers with. It is a control frame, so it is applied in order with the frames
// around it rather than jumping ahead of speech already on its way out, and it
// is uninterruptible, so a barge-in arriving at the same moment does not drop
// it.
//
// Embed it in the frame for a kind of service rather than sending it directly.
type ServiceUpdateSettingsFrame struct {
	BaseControlFrame
	UninterruptibleMixin

	// Delta is the settings to change, as a pointer to a settings value of the
	// kind the target service holds. Only the fields it gives are changed.
	Delta any
	// Settings names the same thing as plain data, for an update arriving from
	// outside the process with no typed value to carry. Delta wins when both are
	// set.
	Settings map[string]any
	// Service, when set, is the one service meant to apply this. Every other
	// service forwards the frame untouched. Leave it nil to reach every service
	// of the kind the frame names.
	Service ServiceTarget
	// ReachInactiveServices asks that the update reach every service a switcher
	// manages rather than only the one currently in use, for a setting that has
	// to survive a switch.
	ReachInactiveServices bool
}

// TargetsService reports whether this update is meant for svc. An update naming
// no service is meant for every service of its kind.
func (f *ServiceUpdateSettingsFrame) TargetsService(svc ServiceTarget) bool {
	return f.Service == nil || f.Service == svc
}

// SettingsUpdate is implemented by every settings update frame, whatever kind of
// service it names, so a processor that only routes them can read where an
// update is meant to go without knowing which kind it is.
type SettingsUpdate interface {
	Frame
	// ServiceUpdate returns the update itself.
	ServiceUpdate() *ServiceUpdateSettingsFrame
	// Copy returns the same update as a new frame with an id of its own, and of
	// the same concrete kind so whatever handles the original handles the copy.
	// It is how one update is delivered to several services and still told apart
	// afterwards.
	Copy() SettingsUpdate
}

// ServiceUpdate implements SettingsUpdate. The frame for each kind of service
// embeds this one, so each of them satisfies the interface through it.
func (f *ServiceUpdateSettingsFrame) ServiceUpdate() *ServiceUpdateSettingsFrame { return f }

// describe renders the fields shared by every settings update.
func (f *ServiceUpdateSettingsFrame) describe(name string) string {
	target := "all"
	if f.Service != nil {
		target = f.Service.Name()
	}
	switch {
	case f.Delta != nil:
		return fmt.Sprintf("%s(service: %s, delta: %v)", name, target, f.Delta)
	default:
		return fmt.Sprintf("%s(service: %s, settings: %v)", name, target, f.Settings)
	}
}

// LLMUpdateSettingsFrame changes a language model service's settings.
type LLMUpdateSettingsFrame struct {
	ServiceUpdateSettingsFrame
}

// NewLLMUpdateSettingsFrame builds an update carrying delta, a pointer to a
// settings value of the kind the LLM service holds.
func NewLLMUpdateSettingsFrame(delta any) *LLMUpdateSettingsFrame {
	return &LLMUpdateSettingsFrame{ServiceUpdateSettingsFrame{
		BaseControlFrame: NewBaseControlFrame("LLMUpdateSettingsFrame"),
		Delta:            delta,
	}}
}

// String implements fmt.Stringer.
func (f *LLMUpdateSettingsFrame) String() string { return f.describe(f.Name()) }

// Copy implements SettingsUpdate.
func (f *LLMUpdateSettingsFrame) Copy() SettingsUpdate {
	c := *f
	c.BaseControlFrame = NewBaseControlFrame("LLMUpdateSettingsFrame")
	return &c
}

// TTSUpdateSettingsFrame changes a speech synthesis service's settings.
type TTSUpdateSettingsFrame struct {
	ServiceUpdateSettingsFrame
}

// NewTTSUpdateSettingsFrame builds an update carrying delta, a pointer to a
// settings value of the kind the TTS service holds.
func NewTTSUpdateSettingsFrame(delta any) *TTSUpdateSettingsFrame {
	return &TTSUpdateSettingsFrame{ServiceUpdateSettingsFrame{
		BaseControlFrame: NewBaseControlFrame("TTSUpdateSettingsFrame"),
		Delta:            delta,
	}}
}

// String implements fmt.Stringer.
func (f *TTSUpdateSettingsFrame) String() string { return f.describe(f.Name()) }

// Copy implements SettingsUpdate.
func (f *TTSUpdateSettingsFrame) Copy() SettingsUpdate {
	c := *f
	c.BaseControlFrame = NewBaseControlFrame("TTSUpdateSettingsFrame")
	return &c
}

// STTUpdateSettingsFrame changes a transcription service's settings.
type STTUpdateSettingsFrame struct {
	ServiceUpdateSettingsFrame
}

// NewSTTUpdateSettingsFrame builds an update carrying delta, a pointer to a
// settings value of the kind the STT service holds.
func NewSTTUpdateSettingsFrame(delta any) *STTUpdateSettingsFrame {
	return &STTUpdateSettingsFrame{ServiceUpdateSettingsFrame{
		BaseControlFrame: NewBaseControlFrame("STTUpdateSettingsFrame"),
		Delta:            delta,
	}}
}

// String implements fmt.Stringer.
func (f *STTUpdateSettingsFrame) String() string { return f.describe(f.Name()) }

// Copy implements SettingsUpdate.
func (f *STTUpdateSettingsFrame) Copy() SettingsUpdate {
	c := *f
	c.BaseControlFrame = NewBaseControlFrame("STTUpdateSettingsFrame")
	return &c
}

var (
	_ SettingsUpdate  = (*LLMUpdateSettingsFrame)(nil)
	_ SettingsUpdate  = (*TTSUpdateSettingsFrame)(nil)
	_ SettingsUpdate  = (*STTUpdateSettingsFrame)(nil)
	_ Uninterruptible = (*LLMUpdateSettingsFrame)(nil)
	_ Uninterruptible = (*TTSUpdateSettingsFrame)(nil)
	_ Uninterruptible = (*STTUpdateSettingsFrame)(nil)
)
