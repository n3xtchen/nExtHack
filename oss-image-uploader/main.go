package main

import (
    "flag"
    "fmt"
    "os"
    // "nExtHack/oss-image-uploader/driver"
    // "github.com/go-ini/ini"
)

func main() {

    fileName := flag.String("file-name", "", "a markdown file")

    // 参数解析
    flag.Parse()

    if *fileName == "" {
        fmt.Printf("Fail to read file: ")
        os.Exit(1)
    }

    fmt.Printf("%s is processing...", *fileName)

    // cfg, err := ini.Load("product.ini")
    // if err != nil {
    //     fmt.Printf("Fail to read conf file: %v", err)
    //     os.Exit(1)
    // }

    // var (
    //     endpoint = cfg.Section("oss").Key("endpoint").String()
    //     access_id = cfg.Section("oss").Key("access_id").String()
    //     access_key = cfg.Section("oss").Key("access_key").String()
    // )

    // var client = driver.AliOssClient(endpoint, access_id, access_key)
}
