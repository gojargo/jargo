package rtvi

import "testing"

// A runtime reconfiguration changes the observer it reaches, not the params
// the observer was built from: sessions sharing one params value each get
// their own settings.
func TestConfigLeavesTheCallersParamsAlone(t *testing.T) {
	params := DefaultObserverParams()
	observer := NewObserverWithParams(nil, params)
	on := true
	observer.applyConfig(NewConfigureObserverFrame(
		map[string]FunctionCallReportLevel{"*": ReportFull}, &on, &on))

	if !observer.params.VADUserSpeakingEnabled || !observer.params.BotLLMMarkerEnabled {
		t.Fatalf("observer params = %+v, want the VAD and marker reports on", observer.params)
	}
	if params.VADUserSpeakingEnabled || params.BotLLMMarkerEnabled {
		t.Errorf("the caller's params changed: %+v", params)
	}
	if got := params.FunctionCallReportLevel["*"]; got != ReportNone {
		t.Errorf("the caller's report level = %q, want %q", got, ReportNone)
	}
	next := NewObserverWithParams(nil, params)
	if next.params.VADUserSpeakingEnabled || next.params.FunctionCallReportLevel["*"] != ReportNone {
		t.Errorf("a later observer's params = %+v, want the caller's settings", next.params)
	}
}
