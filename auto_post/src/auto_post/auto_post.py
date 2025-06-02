#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2025 n3xtchen <echenwen@gmail.com>
#
# Distributed under terms of the GPL-2.0 license.

"""

"""

from time import sleep, time
from playwright.sync_api import sync_playwright

def tweet(username, password, content, file_path):
    with sync_playwright() as p:
        browser = p.chromium.launch(
                headless=False,
                proxy={
                    "server": "127.0.0.1:7890"

                    })  # 设为 False 以便调试
        context = browser.new_context()
        page = context.new_page()

        # 访问 Twitter 登录页面
        page.goto("https://twitter.com/login")
        

        # 输入用户名
        page.fill("input[name='text']", username)
        page.press("input[name='text']", "Enter")

        # 等待加载密码输入框
        page.wait_for_selector("input[name='password']")
        page.fill("input[name='password']", password)
        page.press("input[name='password']", "Enter")

        # 等待跳转到主页
        page.wait_for_selector("//span[contains(text(), 'Post')]")

        # 输入推文内容
        page.click("//span[contains(text(), 'Post')]");
        page.fill("//div[@data-viewportview='true']//div[@class='DraftEditor-editorContainer']/div[@role='textbox']",
                content)

        page.set_input_files("input[data-testid='fileInput']", file_path)
        sleep(15)

        # 点击发送按钮
        page.click("//span[contains(text(), 'Post')]")
        sleep(20)

        # 关闭浏览器
        browser.close()

# 调用函数
tweet("echenwen@gmail.com", "Punkin123@", "Hello from Playwright!", "/Users/nextchen/Downloads/playwright.png")

