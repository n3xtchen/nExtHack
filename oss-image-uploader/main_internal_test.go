
package main

import (
    "testing"
    "nExtHack/oss-image-uploader"
)

func TestDowload(t *testing.T) {
    download("https://golang.org/doc/gopher/frontpage.png", "")
}
