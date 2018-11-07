package main

import (
    "flag"
    "fmt"
    "log"
	"net/http"
    "io"
    "io/ioutil"
    "os"
    "path"
    "nExtHack/oss-image-uploader/driver"
    "github.com/go-ini/ini"
    "mvdan.cc/xurls/v2"
)

func download(url string, dir string) {
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
    fmt.Println("Success!")
}

func main() {

    fileName := flag.String("file-name", "", "a markdown file")

    // 参数解析
    flag.Parse()

    if *fileName == "" {
        panic("Fail to read file")
        os.Exit(1)
    }

    bytes, err := ioutil.ReadFile(*fileName)
    if err != nil {
        panic(err)
        os.Exit(1)
    }

    fmt.Printf("%s is processing...", *fileName)
    urls := xurls.Strict().FindAllString(string(bytes), -1)

    for _, url := range urls {
        download(url, "./img/")
    }

    cfg, err := ini.Load("product.ini")
    if err != nil {
        fmt.Printf("Fail to read conf file: %v", err)
        os.Exit(1)
    }

    var (
        endpoint = cfg.Section("oss").Key("endpoint").String()
        access_id = cfg.Section("oss").Key("access_id").String()
        access_key = cfg.Section("oss").Key("access_key").String()
        bucket_name = cfg.Section("oss").Key("bucket_name").String()
    )

    var client = driver.AliOssClient(endpoint, access_id, access_key, bucket_name)

    files, err := ioutil.ReadDir("./img")
    if err != nil {
        log.Fatal(err)
    }

    for _, f := range files {
        client.Put("test/"+f.Name(), "./img/"+f.Name())
    }

}
