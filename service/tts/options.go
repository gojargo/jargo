package tts

import "github.com/gojargo/jargo/processor"

// Option configures a Base when New builds it.
type Option func(*options)

type options struct {
	self processor.Processor
}

// WithSelf names the service type that embeds the Base: the value the pipeline
// runs, and the one frames, observers and service switchers know the service
// by. A provider whose constructor returns its own type passes that value.
// Without it, or with nil, the Base is the service.
func WithSelf(self processor.Processor) Option {
	return func(o *options) { o.self = self }
}
