// Package benchmarks measures the overhead the framework itself adds: the cost
// of moving frames through a pipeline, of starting one, and of running many at
// once. It is a module of its own, so the tools it needs never become
// dependencies of the library.
//
// See README.md for what is measured and how to run it.
package benchmarks
