package events_test

import (
	"context"
	"fmt"

	"github.com/gojargo/jargo/utils/events"
)

// An object declares the events it raises; a caller attaches typed handlers.
// This event is synchronous, so Call returns once the handler has run.
func ExampleOn() {
	var r events.Registry
	r.Register("on_order_placed", true)

	events.On(&r, "on_order_placed", func(_ context.Context, drink string) {
		fmt.Println("order placed:", drink)
	})

	r.Call(context.Background(), "on_order_placed", nil, "latte")
	// Output: order placed: latte
}
