import os, re, time, argparse, traceback, sys
import sqlite3
import numpy as np
import logging
import torch
import torch.nn.functional as F
from datetime import datetime
from pprint import pprint
from tqdm import trange
from multiprocessing import freeze_support
from transformers import GPT2LMHeadModel
from simpletransformers.classification import ClassificationModel, ClassificationArgs


logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)
from flask import Flask, request, abort

# LINE Bot 套件匯入
from linebot import ( LineBotApi, WebhookHandler )
from linebot.exceptions import ( InvalidSignatureError )
from linebot.models import ( MessageEvent, TextMessage, TextSendMessage, )

# 自訂組態檔 (放置 LINE Bot 重要設定)
from config import Config

# Flask 初始設定
app = Flask(__name__)

# LINE BOT 設定
config = Config()
line_bot_api = LineBotApi(config['YOUR_CHANNEL_ACCESS_TOKEN'])
handler = WebhookHandler(config['YOUR_CHANNEL_SECRET'])

# 建立資料庫連線 (SQLite3)
conn = sqlite3.connect("linebot.db")
cursor = conn.cursor()

# 作為全域用的 dict，以 key-value 格式，放置相關參數用
dictParameter = {}

'''
simple transformers 的 model 設定
'''
model_args = ClassificationArgs()
model_args.train_batch_size = 8
model_args.num_train_epochs = 1
model_args.overwrite_output_dir = True
model_args.reprocess_input_data = True
model_args.use_multiprocessing = False

# language model 的路徑
bert_model_path = "./models/distilbert_regression_epochs_1/"

# 加入 language model 設定 (依需求來調整)
model_args.regression = True
model_args.output_dir = bert_model_path
bert_model = ClassificationModel(
    'distilbert', 
    bert_model_path, 
    use_cuda=False,
    num_labels=1, 
    args=model_args
)


'''
GPT2-Chinese 原生的函式
'''
def is_word(word):
    for item in list(word):
        if item not in 'qwertyuiopasdfghjklzxcvbnm':
            return False
    return True

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits

def sample_sequence(model, context, length, n_ctx, tokenizer, temperature=1.0, top_k=30, top_p=0.0, repitition_penalty=1.0, device='cpu'):
    context = torch.tensor(context, dtype=torch.long, device=device)
    context = context.unsqueeze(0)
    generated = context
    with torch.no_grad():
        for _ in trange(length):
            inputs = {'input_ids': generated[0][-(n_ctx - 1):].unsqueeze(0)}
            # Note: we could also use 'past' with GPT-2/Transfo-XL/XLNet (cached hidden-states)
            outputs = model(**inputs)  
            next_token_logits = outputs[0][0, -1, :]
            for id in set(generated):
                next_token_logits[id] /= repitition_penalty
            next_token_logits = next_token_logits / temperature
            next_token_logits[tokenizer.convert_tokens_to_ids('[UNK]')] = -float('Inf')
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
            next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            generated = torch.cat((generated, next_token.unsqueeze(0)), dim=1)
    return generated.tolist()[0]

def fast_sample_sequence(model, context, length, temperature=1.0, top_k=30, top_p=0.0, device='cpu'):
    inputs = torch.LongTensor(context).view(1, -1).to(device)
    if len(context) > 1:
        _, past = model(inputs[:, :-1], None)[:2]
        prev = inputs[:, -1].view(1, -1)
    else:
        past = None
        prev = inputs
    generate = [] + context
    with torch.no_grad():
        for i in trange(length):
            output = model(prev, past=past)
            output, past = output[:2]
            output = output[-1].squeeze(0) / temperature
            filtered_logits = top_k_top_p_filtering(output, top_k=top_k, top_p=top_p)
            next_token = torch.multinomial(torch.softmax(filtered_logits, dim=-1), num_samples=1)
            generate.append(next_token.item())
            prev = next_token.view(1, 1)
    return generate

def generate(n_ctx, model, context, length, tokenizer, temperature=1, top_k=0, top_p=0.0, repitition_penalty=1.0, device='cpu', is_fast_pattern=False):
    if is_fast_pattern:
        return fast_sample_sequence(model, context, length, temperature=temperature, top_k=top_k, top_p=top_p, device=device)
    else:
        return sample_sequence(model, context, length, n_ctx, tokenizer=tokenizer, temperature=temperature, top_k=top_k, top_p=top_p, repitition_penalty=repitition_penalty, device=device)

# 修改部分主程式 (加入 LINE bot 設定)
def main():
    global dictParameter

    # 取得 cmd 的參數 (此程式使用預設值)
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0', type=str, required=False, help='生成設備')
    parser.add_argument('--length', default="50", type=int, required=False, help='生成長度')
    parser.add_argument('--batch_size', default=1, type=int, required=False, help='生成的batch size')
    parser.add_argument('--nsamples', default=1, type=int, required=False, help='生成幾個樣本')
    parser.add_argument('--temperature', default=0.7, type=float, required=False, help='生成溫度')
    parser.add_argument('--topk', default=8, type=int, required=False, help='最高幾選一')
    parser.add_argument('--topp', default=0, type=float, required=False, help='最高積累機率')
    parser.add_argument('--model_config', default='config/model_config_small.json', type=str, required=False, help='模型參數')
    parser.add_argument('--tokenizer_path', default='cache/vocab_small.txt', type=str, required=False, help='詞表路徑')
    parser.add_argument('--model_path', default='models/model_stc3_cecg_2017-2019_bs-4_epoch-100/', type=str, required=False, help='模型路徑')
    parser.add_argument('--prefix', default='就是愛聊天[喜歡]', type=str, required=False, help='生成文章的前導文字')
    parser.add_argument('--no_wordpiece', action='store_true', help='不做word piece切詞')
    parser.add_argument('--segment', action='store_true', help='中文以詞為單位')
    parser.add_argument('--fast_pattern', action='store_true', help='採用更加快的方式生成文本')
    parser.add_argument('--save_samples', default=False, help='保存產生的樣本')
    parser.add_argument('--save_samples_path', default='.', type=str, required=False, help="保存樣本的路徑")
    parser.add_argument('--repetition_penalty', default=1.0, type=float, required=False)

    args = parser.parse_args()

    # 匯入分詞工具 (需要 tokenizations 資料夾)
    if args.segment:
        from tokenizations import tokenization_bert_word_level as tokenization_bert
    else:
        from tokenizations import tokenization_bert

    # 使用哪些獨立顯卡
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    # 將參數各別帶入變數中
    length = args.length
    batch_size = args.batch_size
    nsamples = args.nsamples
    temperature = args.temperature
    topk = args.topk
    topp = args.topp
    repetition_penalty = args.repetition_penalty
    fast_pattern = args.fast_pattern
    model_path = args.model_path
    tokenizer_path = args.tokenizer_path

    # 沒有 GPU，就使用 CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 使用 BERT 的 tokenizer  功能
    tokenizer = tokenization_bert.BertTokenizer(vocab_file = tokenizer_path)

    # 讀取 model
    model = GPT2LMHeadModel.from_pretrained(model_path)
    model.to(device)

    # 決定生成文字的長度
    n_ctx = model.config.n_ctx

    # 設定是否存放 samples 以及存放的路徑
    save_samples = args.save_samples
    save_samples_path = args.save_samples_path

    # 若是 length 為 -1，則以 model.confg 檔案裡面的 n_ctx 屬性的值為主
    if length == -1:
        length = model.config.n_ctx
    
    # 設定自訂參數
    dictParameter['length'] = length
    dictParameter['batch_size'] = batch_size
    dictParameter['nsamples'] = nsamples
    dictParameter['temperature'] = temperature
    dictParameter['topk'] = topk
    dictParameter['topp'] = topp
    dictParameter['repetition_penalty'] = repetition_penalty
    dictParameter['fast_pattern'] = fast_pattern
    dictParameter['tokenizer'] = tokenizer
    dictParameter['device'] = device
    dictParameter['model'] = model
    dictParameter['n_ctx'] = n_ctx
    dictParameter['save_samples'] = save_samples
    dictParameter['save_samples_path'] = save_samples_path

    # 生成 samples 的流水號
    dictParameter['generated'] = 0

    # 預設使用者 LINE 對話設定
    dictParameter['user'] = {}




'''
自訂函式 (加入部分 GPT2-Chinese 原生程式碼)
'''
# 生成文字
def _gen():
    global dictParameter

    # 回應生成對話到前端的 list
    listResult = []
    
    # 生成文字後，去除 prefix 文字的 pattern
    regex = r".+\[(其它|喜歡|悲傷|噁心|憤怒|開心)\]"

    for _ in range(dictParameter['size'] // dictParameter['batch_size']):
        out = generate(
            n_ctx = dictParameter['n_ctx'],
            model = dictParameter['model'],
            context = dictParameter['context_tokens'],
            length = dictParameter['length'],
            is_fast_pattern = dictParameter['fast_pattern'], 
            tokenizer = dictParameter['tokenizer'],
            temperature = dictParameter['temperature'], 
            top_k = dictParameter['topk'], 
            top_p = dictParameter['topp'], 
            repitition_penalty = dictParameter['repetition_penalty'], 
            device = dictParameter['device']
        )
        for i in range(dictParameter['batch_size']):
            # dictParameter['generated'] += 1
            dictParameter['generated_text'] = dictParameter['tokenizer'].convert_ids_to_tokens(out)
            for i, item in enumerate(dictParameter['generated_text'][:-1]):  # 確保英文前後有空格
                if is_word(item) and is_word(dictParameter['generated_text'][i + 1]):
                    dictParameter['generated_text'][i] = item + ' '
            for i, item in enumerate(dictParameter['generated_text']):
                if item == '[MASK]':
                    dictParameter['generated_text'][i] = ''
                elif item == '[CLS]':
                    dictParameter['generated_text'][i] = '\n\n'
                elif item == '[SEP]':
                    dictParameter['generated_text'][i] = '\n'
            # dictParameter['info'] = "=" * 40 + " SAMPLE " + str(dictParameter['generated']) + " " + "=" * 40 + "\n"
            dictParameter['generated_text'] = ''.join(dictParameter['generated_text']).replace('##', '').strip()
            dictParameter['generated_text'] = dictParameter['generated_text'].split('\n\n')[0]

            # 去掉訓練時加入的 virtual token ([喜歡]、[悲傷]等)
            dictParameter['text_b'] = re.sub(regex, "", dictParameter['generated_text'])

            # 有時候 text_b 會是 ''，所以在這裡給個預設值
            if dictParameter['text_b'] == '':
                dictParameter['text_b'] = '...'

            listResult.append(dictParameter['text_b'])

    return _getRankingList(listResult)

# 輔助 Web API 的函式: 處理 target 的判斷邏輯
def _getRankingList(listResult):
    # 執行 bert model 進行預測
    predictions, raw_outputs = bert_model.predict(listResult)
    predictions = np.array( predictions ).tolist()

    # 加入準備回傳的 list 變數
    listPredict = []
    for idx in range(len(listResult)):
        listPredict.append({
            "coherence": predictions[idx], 
            "candidate": listResult[idx]
        })

    return sorted(listPredict, key=lambda k: k['coherence'], reverse = True)[:len(listResult)]

'''
LINE BOT - Webhook
'''
# LINE BOT 的 webhook
@app.route("/callback", methods=['POST'])
def callback():
    # get X-Line-Signature header value
    signature = request.headers['X-Line-Signature']

    # get request body as text
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)

    # handle webhook body
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        print("Invalid signature. Please check your channel access token/channel secret.")
        abort(400)

    return 'OK'

# 處理訊息
@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    global dictParameter

    # 各別使用者初始情緒設定
    if event.source.user_id not in dictParameter['user']:
        dictParameter['user'][event.source.user_id] = {'emotion': '[喜歡]'}

    # 判斷是否為情緒設定
    matchEmotion = re.match(r"\[\[\[.+\]\]\]", event.message.text)
    if matchEmotion != None:
        # 取得情緒文字
        emotionText = re.sub(r'\[|\]', '', event.message.text)
        if emotionText == '喜歡':
            dictParameter['user'][event.source.user_id]['emotion'] = '[喜歡]'
        elif emotionText == '悲傷':
            dictParameter['user'][event.source.user_id]['emotion'] = '[悲傷]'
        elif emotionText == '厭惡噁心':
            dictParameter['user'][event.source.user_id]['emotion'] = '[噁心]'
        elif emotionText == '憤怒':
            dictParameter['user'][event.source.user_id]['emotion'] = '[憤怒]'
        elif emotionText == '幸福開心':
            dictParameter['user'][event.source.user_id]['emotion'] = '[開心]'
        else:
            dictParameter['user'][event.source.user_id]['emotion'] = '[其它]'

        # 回覆情緒設定結果
        replyText = f'您目前設定的回話情緒為：{emotionText}'
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text=replyText)
        )
        return True

    # 區域變數初始化
    dictParameter['size'] = 5 # 生成句子數，從這裡調整
    dictParameter['prefix_text'] = event.message.text + dictParameter['user'][event.source.user_id]['emotion']
    print(dictParameter['prefix_text'])

    # 將請求文字 tokenize
    dictParameter['context_tokens'] = dictParameter['tokenizer'].convert_tokens_to_ids(dictParameter['tokenizer'].tokenize( dictParameter['prefix_text'] ))

    # 開始時間(秒)
    time_begin = time.time()

    # 生成文字
    listResult = _gen()

    # 輸出生成花費時間
    print("[total] It took %2.4f seconds" % (time.time()-time_begin))

    # 整理 LINE 訊息
    msg_list = []
    response_list = [] # 為了寫入資料庫的 response 資料
    for m in listResult:
        m = f"{m['candidate']}\n{round(m['coherence'], 4)}"
        response_list.append(m)
        msg_list.append(TextSendMessage(text=m))

    # 回覆生成結果
    line_bot_api.reply_message(
        event.reply_token,
        msg_list   # 單句: TextSendMessage(text=event.message.text)
    )

    # 取得 LINE Profile 相關資訊
    profile = line_bot_api.get_profile(event.source.user_id)

    # 整理使用者對話與語言模型生成結果
    response = "\n\n".join(response_list)

    # 寫入對話記錄
    sql = f'''
    INSERT INTO messages (userId, displayName, request, emotion, response, created_at) 
    VALUES ( ?,?,?,?,?,? )
    '''

    # 執行 SQL 語法
    try:
        cursor.execute(sql, (
            event.source.user_id, 
            profile.display_name,
            event.message.text,
            dictParameter['user'][event.source.user_id]['emotion'],
            response,
            datetime.today().strftime("%Y-%m-%d %H-%M-%S")
        ))
        conn.commit()
    except sqlite3.Error as err: 
        # 回滾
        conn.rollback()

        # SQLite3 例外處理
        exc_type, exc_value, exc_tb = sys.exc_info()
        strErrorMsg = f'''SQLite error: {' '.join(err.args)}\n\n
        SQLite traceback: {traceback.format_exception(exc_type, exc_value, exc_tb)}
        '''
        print(strErrorMsg)
        
        # 回覆例外處理的訊息
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text=strErrorMsg)
        )

    # # 關閉 sqlite
    # conn.close()



'''
主程式
'''
if __name__ == '__main__':
    # 初始化文字生成機制
    main()

    # 建立 flask web service (使用 ngrok 的情形下)
    app.run(
        threaded=False, # 是否透過多執行緒處理請求
        host="0.0.0.0", # 設定 0.0.0.0 會對外開放
        port=5005       # 啟用 port 號
    )

    # 有自己的 SSL certs 檔案，再開啟下方設定，修正 ssl_context 的值。
    # app.run(
    #     threaded=False, # 是否透過多執行緒處理請求
    #     host="0.0.0.0", # 設定 0.0.0.0 會對外開放
    #     port=5005,      # 啟用 port 號
    #     # ssl_context=('/root/certs/fullchain4.pem', '/root/certs/privkey4.pem') # 建立 SSL 證證 for LINE Bot Webhook
    # )