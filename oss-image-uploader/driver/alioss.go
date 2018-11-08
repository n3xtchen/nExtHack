package driver

import (
    "github.com/aliyun/aliyun-oss-go-sdk/oss"
)

type aliOssClient struct {
    endpoint string
    accessId string
    accessKey string
    bucketName string
    client  *oss.Client
    bucket  *oss.Bucket
}

func AliOssClient(endpoint string, accessId string, accessKey string, bucketName string) aliOssClient {
    client, err := oss.New(endpoint, accessId, accessKey)
    if err != nil {
        // HandleError(err)
    }

    bucket, err := client.Bucket(bucketName)
    if err != nil {
        bucket = nil
        // HandleError(err)
    }

    return aliOssClient{endpoint, accessId, accessKey, bucketName, client, bucket}
}

func (c aliOssClient) ListBuckets() (oss.ListBucketsResult, error) {
    return c.client.ListBuckets()
}

func (c aliOssClient) Put(objName string, localFile string) {
    err := c.bucket.PutObjectFromFile(objName, localFile)
    if err != nil {
        // HandleError(err)
    }
}


