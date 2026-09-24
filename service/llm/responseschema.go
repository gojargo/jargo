package llm

import (
	"context"
	"encoding/json"
	"log/slog"
)

// CheckResponseSchema returns the schema an inference should send, or nil when
// there is none or it cannot be enforced. A service calls it with the schema it
// was given before building its request.
//
// A schema the service cannot enforce is left off with a warning rather than
// failing the inference: the caller still gets an answer, and is told why it
// may not have the shape asked for.
func (b *Base) CheckResponseSchema(ctx context.Context, schema json.RawMessage) json.RawMessage {
	if len(schema) == 0 {
		return nil
	}
	s, ok := b.gen.(ResponseSchemaSupporter)
	if !ok || !s.SupportsResponseSchema() {
		slog.WarnContext(ctx, "response schema is not supported and is ignored", "service", b.Name())
		return nil
	}
	if model := b.modelName(); !s.ModelSupportsResponseSchema(model) {
		slog.WarnContext(ctx, "response schema is not supported by the model and is ignored",
			"service", b.Name(), "model", model)
		return nil
	}
	return schema
}
