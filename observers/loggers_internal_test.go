package observers

import (
	awsnovasonic "github.com/gojargo/jargo/provider/aws/novasonic"
	googlelive "github.com/gojargo/jargo/provider/google/live"
	openairealtime "github.com/gojargo/jargo/provider/openai/realtime"
	xairealtime "github.com/gojargo/jargo/provider/xai/realtime"
	"github.com/gojargo/jargo/service/llm"
	"github.com/gojargo/jargo/service/stt"
)

// The service filters the LLM and transcription loggers apply are interfaces
// asserted at run time, so nothing but this pins that the services they are
// meant to select actually satisfy them. A service that stopped doing so would
// simply go unreported, which reads as a quiet pipeline rather than a defect.
var (
	_ llmService = (*llm.Base)(nil)
	_ llmService = (*openairealtime.Service)(nil)
	_ llmService = (*googlelive.Service)(nil)
	_ llmService = (*awsnovasonic.Service)(nil)
	_ llmService = (*xairealtime.Service)(nil)

	_ sttService = (*stt.StreamService)(nil)
	_ sttService = (*stt.SegmentService)(nil)
)
