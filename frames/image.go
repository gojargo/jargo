package frames

// ImageRawData is the image payload an image frame carries: raw pixels, or an
// image already encoded.
type ImageRawData struct {
	// Image is the image's bytes: raw pixels, or an encoded image.
	Image []byte
	// Width and Height are the image's dimensions in pixels.
	Width, Height int
	// Format is the pixel format of raw pixels, such as "RGB" or "RGBA", or
	// the MIME type of an encoded image, such as "image/png". Empty for raw
	// pixels is RGB.
	Format string
}

// FileSourceType says how a FileRawData holds its file.
type FileSourceType string

const (
	// FileSourceBytes is a file held as a base64 data URL.
	FileSourceBytes FileSourceType = "bytes"
	// FileSourceURL is a file held as a URL it can be fetched from.
	FileSourceURL FileSourceType = "url"
)

// FileRawData is the file payload a file frame carries, such as a PDF the user
// sent.
type FileRawData struct {
	// File is a base64 data URL, or a URL, depending on Type.
	File string
	// Type says which of the two File is.
	Type FileSourceType
	// Filename is the file's name, or empty.
	Filename string
	// Format is the file's MIME type, such as "application/pdf".
	Format string
}
