package driver

import (
    "github.com/aliyun/aliyun-oss-go-sdk/oss"
)

type aliOssClient struct {
    endpoint string
    accessId string
    accessKey string
    client  *oss.Client
}

func AliOssClient(endpoint string, accessId string, accessKey string) aliOssClient {
    client, err := oss.New(endpoint, accessId, accessKey)
    if err != nil {
        // HandleError(err)
    }
    return aliOssClient{endpoint, accessId, accessKey, client}
}

func (c aliOssClient) ListBuckets() (oss.ListBucketsResult, error) {
    return c.client.ListBuckets()
}


