module github.com/gojargo/jargo/benchmarks

go 1.27

toolchain go1.27.3

// The suite measures the library in the tree beside it, not a published
// release, so a change and its effect on the numbers land in the same commit.
replace github.com/gojargo/jargo => ../

tool golang.org/x/perf/cmd/benchstat

require (
	github.com/aclements/go-moremath v0.0.0-20210112150236-f10218a38794 // indirect
	golang.org/x/perf v0.0.0-20261009192801-be2c69fb417e // indirect
)
