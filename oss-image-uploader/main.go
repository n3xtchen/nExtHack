package main

import (
    "flag"
    "fmt"
    "log"
    "io/ioutil"
    "os"
    "nExtHack/oss-image-uploader/driver"
    "nExtHack/oss-image-uploader/lib"
    "github.com/go-ini/ini"
    "mvdan.cc/xurls/v2"
)

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
        lib.Download(url, "./img/")
    }

    cfg, err := ini.Load("product.ini")
    if err != nil {
        fmt.Printf("Fail to read conf file: %v", err)
        os.Exit(1)
    }

    var (
        endpoint = cfg.Section("oss").Key("endpoint").String()
        accessId = cfg.Section("oss").Key("access_id").String()
        accessKey = cfg.Section("oss").Key("access_key").String()
        bucketName = cfg.Section("oss").Key("bucket_name").String()
    )

    var client = driver.AliOssClient(endpoint, accessId, accessKey, bucketName)

    files, err := ioutil.ReadDir("./img")
    if err != nil {
        log.Fatal(err)
    }

    for _, f := range files {
        client.Put("test/"+f.Name(), "./img/"+f.Name())
    }

}
