package main

import (
    "encoding/json"
    "fmt"
    "log"
    "io"
    "net/http"
    "os"
    // "strings"
    // wechat "github.com/silenceper/wechat/v2"
    "github.com/spf13/cobra"
)

// type Article Struct {
//     title string
//     thumb_media_id string
//     author string
//     digest string
//     show_cover_pic string
//     conent string
//     content_source_url string
// }
// 
// type Articles struct {
//     articles []Article
// }

const WECHAT_API =  "https://api.weixin.qq.com/cgi-bin"

type WechatErr struct {
    ErrCode int `json:"errcode"`
    ErrMsg string `json:"errmsg"`
}

func RespParse[T interface{}](resp io.Reader) T {
    body, err := io.ReadAll(resp)
    if err != nil {
        log.Fatalln(err)
    }

    fmt.Println(string(body))

    var response T
    json.Unmarshal(body, &response)

    return response
}

func GetToken(appID string, appSecret string) {
    authUrl := fmt.Sprintf(
        "%s/token?grant_type=client_credential&appid=%s&secret=%s",
        WECHAT_API, appID, appSecret)
    resp, err := http.Get(authUrl)
    fmt.Println(authUrl)
    fmt.Println(err)
    defer resp.Body.Close()
    fmt.Printf("%v\n", resp)
    fmt.Println(RespParse[WechatErr](resp.Body))
}

func main() {
    appID := os.Getenv("APP_ID")
    appSecret := os.Getenv("APP_SECRET")

    rootCmd := &cobra.Command{
        Use:   "wechat",
		Short: "Publish Article To Wechat",
        Run: func(cmd *cobra.Command, args []string) {
            fmt.Println("Publishing")
            GetToken(appID, appSecret)
        },
    }

    // log.Fatal(rootCmd.Execute())
    rootCmd.Execute()
}
