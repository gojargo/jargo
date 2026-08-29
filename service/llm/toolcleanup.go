package llm

import (
	"context"
	"log/slog"
	"reflect"
	"sync"
)

// ToolCleanup releases a resource a tool handler works through that outlives the
// calls it serves: a connection to an MCP server, a session with some other
// system. The LLM service releases it when the pipeline tears down, so a
// developer who registered the tool does not have to wire the teardown as well.
//
// Handlers sharing one resource name the same value, and it is released once. A
// value has to be comparable for that to work, which a pointer is: naming a
// pointer to the thing that owns the resource is the ordinary shape.
type ToolCleanup interface {
	// CloseTools releases the resource. It is called once when the service is
	// cleaned up, and has to be safe to call again: the same resource may have
	// been named by a tool registered somewhere else too.
	CloseTools(ctx context.Context) error
}

// toolCleanups are the resources this service releases at teardown. They are
// kept for the service's lifetime even if the tool that named one is later
// unregistered: the resource may outlive any one tool's advertisement, and a
// tool coming and going should not close a connection something else still uses.
type toolCleanups struct {
	mu   sync.Mutex
	list []ToolCleanup
}

// recordToolCleanup remembers a resource to release at teardown, once per
// resource.
func (b *Base) recordToolCleanup(c ToolCleanup) {
	if c == nil {
		return
	}
	b.cleanups.mu.Lock()
	defer b.cleanups.mu.Unlock()
	for _, existing := range b.cleanups.list {
		if sameCleanup(existing, c) {
			return
		}
	}
	b.cleanups.list = append(b.cleanups.list, c)
}

// sameCleanup reports whether two resources are the same one. A value that
// cannot be compared is treated as its own, which at worst releases a resource
// twice; CloseTools has to tolerate that anyway.
func sameCleanup(a, b ToolCleanup) bool {
	ta, tb := reflect.TypeOf(a), reflect.TypeOf(b)
	if ta != tb || !ta.Comparable() {
		return false
	}
	return a == b
}

// ToolCleanupCount is how many distinct resources registered tools named, for a
// test checking that tools sharing one are counted once.
func (b *Base) ToolCleanupCount() int {
	b.cleanups.mu.Lock()
	defer b.cleanups.mu.Unlock()
	return len(b.cleanups.list)
}

// runToolCleanups releases every resource a registered tool named. A failure is
// logged rather than returned: the pipeline is coming down either way, and one
// resource refusing to close should not stop the next from being released.
func (b *Base) runToolCleanups(ctx context.Context) {
	b.cleanups.mu.Lock()
	list := b.cleanups.list
	b.cleanups.list = nil
	b.cleanups.mu.Unlock()

	for _, c := range list {
		if err := c.CloseTools(ctx); err != nil {
			slog.ErrorContext(ctx, "releasing a resource a tool worked through failed",
				"service", b.Name(), "err", err)
		}
	}
}
