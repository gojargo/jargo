// Package frames holds the frames the Google services add to the pipeline's
// own: the search results and grounding a Gemini model reports when it answers
// from a web search.
package frames

import (
	"fmt"

	"github.com/gojargo/jargo/frames"
)

// LLMSearchResult is one piece of the answer a search supports, with the
// model's confidence in that support.
type LLMSearchResult struct {
	// Text is the part of the answer the search supports.
	Text string
	// Confidence are the model's confidence scores for that support.
	Confidence []float64
}

// LLMSearchOrigin is a web page the answer drew on, with the parts of the
// answer it supports.
type LLMSearchOrigin struct {
	// SiteURI is the page's address, when known.
	SiteURI string
	// SiteTitle is the page's title, when known.
	SiteTitle string
	// Results are the parts of the answer the page supports.
	Results []LLMSearchResult
}

// LLMSearchResponseFrame carries what a Gemini model reported about the web
// search it answered from: the answer's text, the rendered search suggestions,
// and the pages it drew on.
type LLMSearchResponseFrame struct {
	frames.BaseDataFrame
	// SearchResult is the text of the answer the search supports.
	SearchResult string
	// RenderedContent is the search suggestions from the search entry point,
	// rendered for display.
	RenderedContent string
	// Origins are the pages the answer drew on.
	Origins []LLMSearchOrigin
}

// NewLLMSearchResponseFrame builds an LLMSearchResponseFrame.
func NewLLMSearchResponseFrame(
	searchResult, renderedContent string, origins []LLMSearchOrigin,
) *LLMSearchResponseFrame {
	return &LLMSearchResponseFrame{
		BaseDataFrame:   frames.NewBaseDataFrame("LLMSearchResponseFrame"),
		SearchResult:    searchResult,
		RenderedContent: renderedContent,
		Origins:         origins,
	}
}

// String implements fmt.Stringer.
func (f *LLMSearchResponseFrame) String() string {
	return fmt.Sprintf("%s(search_result=%s, origins=%v)", f.Name(), f.SearchResult, f.Origins)
}
