package lib

import (
    "log"
    "io"
	"net/http"
    "path"
    "os"
)

func Download(url string, dir string) {
    response, e := http.Get(url)
    if e != nil {
        log.Fatal(e)
    }

    defer response.Body.Close()

    //open a file for writing
    _, f := path.Split(url)
    file, err := os.Create(dir + f)
    if err != nil {
        log.Fatal(err)
    }
    // Use io.Copy to just dump the response body to the file. This supports huge files
    _, err = io.Copy(file, response.Body)
    if err != nil {
        log.Fatal(err)
    }
    file.Close()
    log.Println("Success!")
}


