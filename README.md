# 生成式聊天對話機器人
以 [NTCIR-14 Short Text Conversation Task (STC-3)](http://sakailab.com/ntcir14stc3/) 之語料為基礎，整合 [GPT2-Chinese](https://github.com/Morizeyao/GPT2-Chinese) 語言模型訓練工具，同時串接 [LINE Developer](https://developers.line.biz/console/profile) 的 Message API 服務，建立情感對話系統。

# 執行環境
- Ubuntu Server 20.04 LTS
- Python 3.9
- CUDA Version: 11.0
- NVIDIA-SMI 450.51.06
- Driver Version: 450.51.06

有使用 Conda 的話，可以透過指令來安裝環境:
```python
# 安裝 conda 環境
$ conda create --name chatbot python=3.9
...
(依安裝提示訊息完成套件安裝流程)
...
# 啟用 conda 環境
$ conda activate chatbot
```

# 下載專案
```
$ git clone 
```
沒有 Git 執行程式，可以直接 Download Zip 後，再自行解壓縮，而後**進入專案資料夾 gpt2_linebot**
```
$ cd gpt2_linebot
```

# 下載語言模型
[下載連結](https://drive.google.com/file/d/1rOJhpWwYTpt-0duqHb0nDOVFg42fz0R6/view?usp=sharing)
存放到專案資料夾當中解壓縮，會有一個獨立的 models 資料夾。

![](https://i.imgur.com/2bzNalY.png)

# 套件安裝
[先前的 PyTorch 版本安裝頁面](https://pytorch.org/get-started/previous-versions/)

## 如果擁有 GPU (以 CUDA 11.0 為例)
```python
$ pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
```

## 如果沒有 GPU (CPU only)
```python
$ pip install torch==1.7.1+cpu torchvision==0.8.2+cpu torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
```

## 安裝 requirements.txt
```python
$ pip install -r requirements.txt
```

## 確認是否啟用 GPU (電腦沒有 GPU 者，可以略過)
```python
import torch
print(torch.cuda.is_available())
```

# 設定 LINE BOT
1. 登入 [LINE Developers](https://developers.line.biz/console/profile)。
2. 新增 Provider。
3. 新增該 Provider 下的 Channel。
4. 選擇 Messaging API。
5. 依需求選擇、填寫基本資料，勾選頁面底下兩個「I have read agree to the ...」，按下 Create。
6. 選擇先前 Create 的 Channel。
7. 到 Basic settings 標籤取得 Channel secret。
8. 到 Messaging API 標籤：
  - 設定 Webhook URL，並驗證 (verify) 是否請求成功。
    - **註: 若需要測試環境，可以選擇 ngrok 工具**
  - 關閉自動回覆訊息 (Auto-reply messages)。
  - 取得 Channel access token (long-lived)。
9. **將 Channel secret 和 Channel access token 複製貼上到專案資料夾裡面的 config.py 當中**
![](https://i.imgur.com/jiSR4k1.png)


# （Optional）模擬 SSL 環境設定: ngrok
由於 LINE 的 webhook 需要使用 SSL 才能通過請求，若是在測試階段，本機沒有 SSL，此時我們需要一個暫時的 URI 來填入 Webhook URL。 

## 登入 ngrok 儀表板
- 進入 [ngrok 首頁](https://ngrok.com/)。
- 註冊一個帳號，或是使用 GitHub 或 Google 帳號來登入。
- 進入儀表板頁面：
![儀表板畫面](https://i.imgur.com/7exSHP9.png)
- 在儀表板上方，依作業系統規格來選擇 ngrok 檔案下載，並解壓縮到全域性指令集的資料夾中，或是保留在專案當中。
- 假設 ngrok 沒有執行權限，記得加上:
  - `$ chmod +x ngrok` 或 `$ chmod +x ./ngrok`
- 接下來加入 auth token 到 ngrok.yml 當中(直接複製儀表板中的內容貼上也行):
  - `$ ngrok config add-authtoken 24e35K*******************************************`
  - 註：若是放到專案資料夾下，指令可以改成 `$ ./ngrok confi add-authtoken ... ` 開頭。

## 執行 ngrok
專案使用的 port 號為 5005，可依需求調整:
```python
$ ngrok http 5005
或
$ ./ngrok http 5005
```

## ngrok 啟動成功的畫面
![](https://i.imgur.com/pczWMdG.png)
**注意: 要複製 _Forwarding_ 那一列後面的網址，例如
`https://xxxx-xxxx-xxxx-x-xxxx-xxxx-xxxx-xxxx-xxxx.jp.ngrok.io`，之後 Webhook 設定會用到。**

## 回到 LINE Developers
- 進入先前建立的 Channel。
- 在 Webhook settings 下面的 Webhook URL 當中，編輯/填寫 `https://xxxx-xxxx-xxxx-x-xxxx-xxxx-xxxx-xxxx-xxxx.jp.ngrok.io/callback`，記得後面要加上 /**callback**，這邊會跟主程式相呼應。
- 按下 Update 來儲存設定。
- 開啟 Use webhook。
![](https://i.imgur.com/hSf3fYO.png)


# 啟動服務

## 執行主程式
回到專案資料夾後，準備執行主程式，建立文字生成服務:
```python
$ python app.py
```
若執行無誤，則會出現以下結果:
![](https://i.imgur.com/1qEgOkq.png)

## 測試 Webhook
回到剛才 Message API 的 Webhook 設定，按下 Verify，如果沒有錯誤，則會出現以下結果:
![](https://i.imgur.com/H33XwAL.png)

**註: 記得加入自己建立的 LINE BOT 喔!!**

---

# Rich Menu
為了在 LINE BOT 當中可以使用指定回話情緒的選單，需要加入 RichMenu、上傳 RichMenu 圖片，並指定預設的 RichMenu。以下會提供幾個程式，可依需求自行修改內容。

**註: 記得先將 config.py 的內容設定好。** 

## 加入 Rich Menu
```python
$ python 1_addRichMenu.py
```
正常執行的結果，會輸出:
```
'{"richMenuId":"RichMenu ID 的值"}'

之後請將 richMenuId 的值，複製到 2_uploadRichMenuImage.py 和 3_setRichMenu.py 有關 richMenuId 變數當中
```

## 上傳 Rich Menu 圖片
將上傳專案資料夾當中的 richmenu.jpg，作為 Rich Menu 的背景圖片。
```python
$ python 2_uploadRichMenuImage.py
```
正常執行的結果，會輸出:
```
'{}'
```

## 指定預設 Rich Menu
```python
$ python 3_setRichMenu.py
```
正常執行的結果，會輸出:
```
'{}'
```

# 成果展示
![](https://i.imgur.com/tyrdvJm.png)
![](https://i.imgur.com/L2yLbmQ.png)
![](https://i.imgur.com/QP0k3xg.png)
![](https://i.imgur.com/c2NyQKP.png)

# 最後提醒
如果主機沒有 SSL 檔案可以用，臨時需要展示，則建議使用 ngrok，此時啟動服務的程式 **app.py**，在最底下的主程式區塊當中，就可以使用:
```python
if __name__ == "__main__":
    app.run(
        threaded=False, # 是否透過多執行緒處理請求
        host="0.0.0.0", # 設定 0.0.0.0 會對外開放
        port=5005       # 啟用 port 號
    )
```
如果主機當中，已經設置好 SSL 環境，則可以使用:
```python
if __name__ == "__main__":
    app.run(
        threaded=False, # 是否透過多執行緒處理請求
        host="0.0.0.0", # 設定 0.0.0.0 會對外開放
        port=5005,      # 啟用 port 號
        ssl_context=(
            '/root/certs/fullchain4.pem', 
            '/root/certs/privkey4.pem') 
            # 建立 SSL 證證 for LINE Bot Webhook
    )
```
注意: 使用 ssl_context 設定前，一定要先確認是否同時有 **fullchain.pem** 與 **privkey.pem** 這兩個檔案。

---

# 其它檔案說明
- checkGPU.py
  - 確認主機 GPU 是否能夠啟用。
- makeDB.py
  - 建立 sqlite3 資料庫，記錄每一個使用者的發話與回覆。
- testBert.py
  - 測試 coherence 分數是否能夠執行。