package frames

import "fmt"

// SwitcherFrame is implemented by the frames that steer a service switcher.
// A switcher hands one to its strategy, which decides what to do with it; a
// frame the strategy does not act on travels on, so a switcher further down the
// pipeline gets its turn.
type SwitcherFrame interface {
	Frame
	isSwitcherFrame()
}

// BaseSwitcherFrame is embedded by every switcher frame.
type BaseSwitcherFrame struct {
	BaseControlFrame
}

func (*BaseSwitcherFrame) isSwitcherFrame() {}

// NewBaseSwitcherFrame builds a BaseSwitcherFrame labeled typeName.
func NewBaseSwitcherFrame(typeName string) BaseSwitcherFrame {
	return BaseSwitcherFrame{NewBaseControlFrame(typeName)}
}

// ManuallySwitchServiceFrame asks a service switcher to make Service the active
// one. A switcher that does not manage Service leaves the frame alone, so the
// request reaches the switcher that does. It is a control frame.
type ManuallySwitchServiceFrame struct {
	BaseSwitcherFrame
	// Service is the service to activate.
	Service ServiceTarget
}

// NewManuallySwitchServiceFrame builds a ManuallySwitchServiceFrame targeting
// svc.
func NewManuallySwitchServiceFrame(svc ServiceTarget) *ManuallySwitchServiceFrame {
	return &ManuallySwitchServiceFrame{
		BaseSwitcherFrame: NewBaseSwitcherFrame("ManuallySwitchServiceFrame"),
		Service:           svc,
	}
}

// String implements fmt.Stringer.
func (f *ManuallySwitchServiceFrame) String() string {
	return fmt.Sprintf("%s(service: %s)", f.Name(), targetName(f.Service))
}

// ServiceSwitcherRequestMetadataFrame asks a service to broadcast its metadata
// again. A switcher sends one to the service it has just made active, so what
// the rest of the pipeline knows describes the service now in use rather than
// the one it replaced. It is a control frame.
type ServiceSwitcherRequestMetadataFrame struct {
	BaseControlFrame
	// Service is the service that should broadcast its metadata again.
	Service ServiceTarget
}

// NewServiceSwitcherRequestMetadataFrame builds a request aimed at svc.
func NewServiceSwitcherRequestMetadataFrame(svc ServiceTarget) *ServiceSwitcherRequestMetadataFrame {
	return &ServiceSwitcherRequestMetadataFrame{
		BaseControlFrame: NewBaseControlFrame("ServiceSwitcherRequestMetadataFrame"),
		Service:          svc,
	}
}

// String implements fmt.Stringer.
func (f *ServiceSwitcherRequestMetadataFrame) String() string {
	return fmt.Sprintf("%s(service: %s)", f.Name(), targetName(f.Service))
}

// targetName renders a service target, or "none" when there is not one.
func targetName(t ServiceTarget) string {
	if t == nil {
		return "none"
	}
	return t.Name()
}

// Compile-time interface checks.
var (
	_ ControlFrame  = (*ManuallySwitchServiceFrame)(nil)
	_ SwitcherFrame = (*ManuallySwitchServiceFrame)(nil)
	_ ControlFrame  = (*ServiceSwitcherRequestMetadataFrame)(nil)
)
