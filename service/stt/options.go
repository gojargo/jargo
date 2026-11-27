package stt

import "github.com/gojargo/jargo/processor"

// Option configures a StreamService or a SegmentService when it is built.
type Option func(*options)

type options struct {
	self processor.Processor
}

// WithSelf names the service type that embeds the service: the value the pipeline
// runs, and the one frames, observers and service switchers know the service
// by. A provider whose constructor returns its own type passes that value.
// Without it, or with nil, the embedded service is the service.
func WithSelf(self processor.Processor) Option {
	return func(o *options) { o.self = self }
}

// selfOr is the processor a service registers as itself: the one WithSelf
// named, or the service.
func selfOr(service processor.Processor, opts []Option) processor.Processor {
	var o options
	for _, opt := range opts {
		opt(&o)
	}
	if o.self != nil {
		return o.self
	}
	return service
}
