package frames

import (
	"context"
	"fmt"
)

// newImageRawData builds an ImageRawData.
func newImageRawData(image []byte, width, height int, format string) ImageRawData {
	return ImageRawData{Image: image, Width: width, Height: height, Format: format}
}

// ImageData returns the image payload itself, so a frame carrying an image can
// be handled through the ImageFrame interface without knowing which kind it is.
func (d *ImageRawData) ImageData() *ImageRawData { return d }

// size renders the image's dimensions the way the image frames print them.
func (d *ImageRawData) size() string { return fmt.Sprintf("(%d, %d)", d.Width, d.Height) }

// ImageFrame is implemented by every frame that carries an image, whichever
// direction it travels in.
type ImageFrame interface {
	Frame
	ImageData() *ImageRawData
}

// OutputImageFrame is implemented by every frame carrying an image to be shown
// by an output transport: a plain output image, an image with a URL, and an
// image the assistant generated. Assert this interface rather than a concrete
// type, so an image keeps being shown whichever kind produced it.
type OutputImageFrame interface {
	ImageFrame
	// OutputImage returns the output image the frame is.
	OutputImage() *OutputImageRawFrame
}

// InputImageFrame is implemented by every frame carrying an image from an input
// transport: a plain input image, and an input image attributed to the user it
// came from. Assert this interface rather than a concrete type, so an image
// keeps being handled whichever kind the transport produced.
type InputImageFrame interface {
	ImageFrame
	isInputImage()
}

// OutputImageRawFrame is an image to be shown by an output transport. When a
// transport exposes multiple video destinations, the destination name is
// carried in TransportDestination. It is a data frame.
type OutputImageRawFrame struct {
	BaseDataFrame
	ImageRawData
	// SyncWithAudio queues the image with the audio, so it is only shown once
	// the audio ahead of it has been sent. False shows it as soon as the output
	// transport receives it.
	SyncWithAudio bool
}

// NewOutputImageRawFrame builds an OutputImageRawFrame.
func NewOutputImageRawFrame(image []byte, width, height int, format string) *OutputImageRawFrame {
	return &OutputImageRawFrame{
		BaseDataFrame: NewBaseDataFrame("OutputImageRawFrame"),
		ImageRawData:  newImageRawData(image, width, height, format),
	}
}

// OutputImage implements OutputImageFrame.
func (f *OutputImageRawFrame) OutputImage() *OutputImageRawFrame { return f }

// String implements fmt.Stringer.
func (f *OutputImageRawFrame) String() string {
	return fmt.Sprintf("%s(pts: %s, destination: %s, size: %s, format: %s)",
		f.Name(), formatPTS(f), f.TransportDestination(), f.size(), f.Format)
}

// URLImageRawFrame is an output image with the URL it can be downloaded from,
// usually one a third-party service generated.
type URLImageRawFrame struct {
	OutputImageRawFrame
	// URL is where the image can be downloaded from.
	URL string
}

// NewURLImageRawFrame builds a URLImageRawFrame.
func NewURLImageRawFrame(url string, image []byte, width, height int, format string) *URLImageRawFrame {
	return &URLImageRawFrame{
		BaseDataFrame: NewBaseDataFrame("URLImageRawFrame"),
		ImageRawData:  newImageRawData(image, width, height, format),
		URL:           url,
	}
}

// String implements fmt.Stringer.
func (f *URLImageRawFrame) String() string {
	return fmt.Sprintf("%s(pts: %s, url: %s, size: %s, format: %s)",
		f.Name(), formatPTS(f), f.URL, f.size(), f.Format)
}

// AssistantImageRawFrame is an image the assistant generated. It is shown like
// any output image, and also carries the image as it was produced, which can
// go into the conversation as it stands rather than being encoded again.
type AssistantImageRawFrame struct {
	OutputImageRawFrame
	// OriginalData is the image as it was produced, or nil.
	OriginalData []byte
	// OriginalMIMEType is the MIME type of OriginalData.
	OriginalMIMEType string
}

// NewAssistantImageRawFrame builds an AssistantImageRawFrame.
func NewAssistantImageRawFrame(image []byte, width, height int, format string) *AssistantImageRawFrame {
	return &AssistantImageRawFrame{
		BaseDataFrame: NewBaseDataFrame("AssistantImageRawFrame"),
		ImageRawData:  newImageRawData(image, width, height, format),
	}
}

// SpriteFrame is an animation of images, shown by the transport when it sends
// video, at the transport's video output framerate.
type SpriteFrame struct {
	BaseDataFrame
	// Images are the frames of the animation, in order.
	Images []*OutputImageRawFrame
}

// NewSpriteFrame builds a SpriteFrame.
func NewSpriteFrame(images []*OutputImageRawFrame) *SpriteFrame {
	return &SpriteFrame{BaseDataFrame: NewBaseDataFrame("SpriteFrame"), Images: images}
}

// String implements fmt.Stringer.
func (f *SpriteFrame) String() string {
	return fmt.Sprintf("%s(pts: %s, size: %d)", f.Name(), formatPTS(f), len(f.Images))
}

// InputImageRawFrame is an image coming from an input transport. When a
// transport exposes multiple video sources, the source name is carried in
// TransportSource. It is a system frame.
type InputImageRawFrame struct {
	BaseSystemFrame
	ImageRawData
}

// isInputImage marks this frame and the ones embedding it as input images.
func (*InputImageRawFrame) isInputImage() {}

// NewInputImageRawFrame builds an InputImageRawFrame.
func NewInputImageRawFrame(image []byte, width, height int, format string) *InputImageRawFrame {
	return &InputImageRawFrame{
		BaseSystemFrame: NewBaseSystemFrame("InputImageRawFrame"),
		ImageRawData:    newImageRawData(image, width, height, format),
	}
}

// String implements fmt.Stringer.
func (f *InputImageRawFrame) String() string {
	return fmt.Sprintf("%s(pts: %s, source: %s, size: %s, format: %s)",
		f.Name(), formatPTS(f), f.TransportSource(), f.size(), f.Format)
}

// ImageResultCallback reports a tool call's result the way the call's own
// result callback does. An image request made from a tool call carries the
// call's callback, which is told once the image has been taken, or that none
// will be.
type ImageResultCallback = func(ctx context.Context, result string, props *FunctionCallResultProperties) error

// UserImageRequestFrame asks an input transport for an image from a user's
// video. The request can come with text describing what the image is wanted
// for, which travels with the image that answers it.
//
// A tool call that needs to see the user pushes one upstream to the transport,
// naming itself in FunctionName and ToolCallID and passing its result callback,
// so the image becomes part of its answer. It is a system frame.
type UserImageRequestFrame struct {
	BaseSystemFrame
	// UserID identifies the user whose image is requested, as the transport's
	// GetClientID returns it.
	UserID string
	// Text is text to go with the image, or "".
	Text string
	// AppendToContext asks for the image to be added to the conversation.
	AppendToContext bool
	// VideoSource is the video source to take the image from, such as "camera"
	// or "screenVideo", or "" for the transport's default.
	VideoSource string
	// FunctionName is the tool that made the request, or "".
	FunctionName string
	// ToolCallID is the tool call that made the request, or "".
	ToolCallID string
	// ResultCallback, when set, is called once the image has been taken, or
	// with an error result when no image will answer the request.
	ResultCallback ImageResultCallback
}

// NewUserImageRequestFrame builds a UserImageRequestFrame.
func NewUserImageRequestFrame(userID string) *UserImageRequestFrame {
	return &UserImageRequestFrame{
		BaseSystemFrame: NewBaseSystemFrame("UserImageRequestFrame"),
		UserID:          userID,
	}
}

// String implements fmt.Stringer.
func (f *UserImageRequestFrame) String() string {
	return fmt.Sprintf("%s(user: %s, text: %s, append_to_context: %t, %s)",
		f.Name(), f.UserID, f.Text, f.AppendToContext, f.VideoSource)
}

// UserImageRawFrame is an input image attributed to the user it came from,
// possibly answering an image request.
type UserImageRawFrame struct {
	InputImageRawFrame
	// UserID identifies the user this image came from.
	UserID string
	// Text is text to go with the image, or "".
	Text string
	// AppendToContext asks for the image to be added to the conversation.
	AppendToContext bool
	// Request is the image request this image answers, or nil.
	Request *UserImageRequestFrame
	// RunLLM says whether adding the image to the conversation runs the model.
	// Nil runs it; only an explicit false holds it back.
	RunLLM *bool
}

// NewUserImageRawFrame builds a UserImageRawFrame.
func NewUserImageRawFrame(userID string, image []byte, width, height int, format string) *UserImageRawFrame {
	return &UserImageRawFrame{
		BaseSystemFrame: NewBaseSystemFrame("UserImageRawFrame"),
		ImageRawData:    newImageRawData(image, width, height, format),
		UserID:          userID,
	}
}

// String implements fmt.Stringer.
func (f *UserImageRawFrame) String() string {
	return fmt.Sprintf("%s(pts: %s, user: %s, source: %s, size: %s, format: %s, text: %s, append_to_context: %t)",
		f.Name(), formatPTS(f), f.UserID, f.TransportSource(), f.size(), f.Format, f.Text, f.AppendToContext)
}

// Compile-time interface checks.
var (
	_ DataFrame   = (*OutputImageRawFrame)(nil)
	_ DataFrame   = (*URLImageRawFrame)(nil)
	_ DataFrame   = (*AssistantImageRawFrame)(nil)
	_ DataFrame   = (*SpriteFrame)(nil)
	_ SystemFrame = (*InputImageRawFrame)(nil)
	_ SystemFrame = (*UserImageRawFrame)(nil)
	_ SystemFrame = (*UserImageRequestFrame)(nil)

	_ OutputImageFrame = (*OutputImageRawFrame)(nil)
	_ OutputImageFrame = (*URLImageRawFrame)(nil)
	_ OutputImageFrame = (*AssistantImageRawFrame)(nil)
	_ InputImageFrame  = (*InputImageRawFrame)(nil)
	_ InputImageFrame  = (*UserImageRawFrame)(nil)
)
