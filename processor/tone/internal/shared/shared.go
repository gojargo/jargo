// Package shared keeps the tone models loaded once a process, so every
// processor using the same model and settings shares one.
package shared

import "sync"

// Cache holds the models loaded so far, by what identifies each one and its
// settings.
type Cache[K comparable, M any] struct {
	mu     sync.Mutex
	models map[K]M
}

// Load returns the model loaded for key, loading it first if none is. Loads
// take turns, so two processors asking for the same model at once load it
// once. A load that fails is not kept, and the next call tries again.
func (c *Cache[K, M]) Load(key K, load func() (M, error)) (M, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if m, ok := c.models[key]; ok {
		return m, nil
	}
	m, err := load()
	if err != nil {
		return m, err
	}
	if c.models == nil {
		c.models = map[K]M{}
	}
	c.models[key] = m
	return m, nil
}
