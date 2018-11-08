package driver_test

import (
    "github.com/go-ini/ini"
    "testing"
    "nExtHack/oss-image-uploader/driver"
)

func TestInit(t *testing.T) {
    cfg, err := ini.Load("../test.ini")
    if err != nil {
        t.Errorf("Fail to read conf file: %v", err)
    }
    var (
        endpoint = cfg.Section("oss").Key("endpoint").String()
        accessId = cfg.Section("oss").Key("access_id").String()
        accessKey = cfg.Section("oss").Key("access_key").String()
        bucketName = cfg.Section("oss").Key("bucket_name").String()
    )

    driver.AliOssClient(endpoint, accessId, accessKey, "")
    t.Log("Over!")

    driver.AliOssClient(endpoint, accessId, accessKey, bucketName)
}
