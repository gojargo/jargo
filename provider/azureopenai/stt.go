package azureopenai

import (
	"net/http"
	"strings"

	"github.com/gojargo/jargo/internal/validate"
	"github.com/gojargo/jargo/provider/openai"
	"github.com/gojargo/jargo/service/stt"
)

// defaultSTTAPIVersion is a recent stable Azure OpenAI audio API version.
const defaultSTTAPIVersion = "2024-10-21"

// STTConfig configures the Azure OpenAI transcription provider. It is segmented:
// a turn detector upstream delimits each utterance, which is transcribed in one
// request against Azure OpenAI's /audio/transcriptions endpoint (Whisper or
// gpt-4o-transcribe).
type STTConfig struct {
	// Endpoint is the Azure OpenAI resource endpoint, e.g.
	// https://my-resource.openai.azure.com. Required.
	Endpoint string `validate:"required,url"`
	// Deployment is the transcription model deployment name. Required.
	Deployment string `validate:"required"`
	// APIVersion is the Azure OpenAI REST API version; empty uses a recent
	// stable. gpt-4o-transcribe deployments require a preview version.
	APIVersion string
	// STTConfig carries the shared OpenAI STT options (APIKey, Language and so
	// on). Its BaseURL and Model are ignored: the URL is built from Endpoint and
	// Deployment.
	openai.STTConfig
}

// Validate reports whether the configuration is usable.
func (c STTConfig) Validate() error { return validate.Struct(c) }

// sttShaper addresses and authorizes transcription requests the Azure way.
type sttShaper struct{ apiVersion string }

func (s sttShaper) Endpoint(baseURL string) string {
	return baseURL + "/audio/transcriptions?api-version=" + s.apiVersion
}

func (sttShaper) Authorize(req *http.Request, apiKey string) {
	req.Header.Set("api-key", apiKey)
}

// NewSTT builds an Azure OpenAI transcription service.
func NewSTT(cfg STTConfig) *stt.SegmentService {
	apiVersion := cfg.APIVersion
	if apiVersion == "" {
		apiVersion = defaultSTTAPIVersion
	}
	base := strings.TrimSuffix(cfg.Endpoint, "/") + "/openai/deployments/" + cfg.Deployment

	sttCfg := cfg.STTConfig
	sttCfg.BaseURL = "" // built from Endpoint/Deployment, not BaseURL
	sttCfg.Model = ""   // the deployment in the URL selects the model
	return openai.NewShapedSTT("AzureOpenAISTT", base, "", sttShaper{apiVersion: apiVersion}, sttCfg)
}
