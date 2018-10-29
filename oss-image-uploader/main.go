package main

import (
    "fmt"
    "os"
    // "regexp"
    "nExtHack/oss-image-uploader/driver"
    "github.com/go-ini/ini"
)

func main() {

    cfg, err := ini.Load("product.ini")
    if err != nil {
        fmt.Printf("Fail to read file: %v", err)
        os.Exit(1)
    }

    var (
        endpoint = cfg.Section("oss").Key("endpoint").String()
        access_id = cfg.Section("oss").Key("access_id").String()
        access_key = cfg.Section("oss").Key("access_key").String()
    )

    // client, err := oss.New(endpoint, access_id, access_key)
    // if err != nil {
    //     // HandleError(err)
    // }
    var client = driver.AliOssClient(endpoint, access_id, access_key)

    lsRes, err := client.ListBuckets()
    if err != nil {
        // HandleError(err)
    }

    for _, bucket := range lsRes.Buckets {
        fmt.Println("Buckets:", bucket.Name)
    }
}
