from simpletransformers.classification import ClassificationModel, ClassificationArgs
from pprint import pprint
import numpy as np

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

# 加入 language model
model_args.regression = True
model_args.output_dir = bert_model_path
model = ClassificationModel(
    'distilbert', 
    bert_model_path, 
    use_cuda=False,
    num_labels=1, 
    args=model_args
)

if __name__ == '__main__':
    # 測試語句
    listResult = ['你也快樂', '我們都快樂', '謝謝親愛的,我們都要快樂樂', '祝我們~', '謝謝你，也祝你快樂！']
    
    # 儲存整合結果的變數
    listPredict = []

    # 模式執行結果
    predictions, raw_outputs = model.predict(listResult)
    predictions = np.array( predictions ).tolist()

    # 加入準備回傳的 list 變數
    for idx in range(len(listResult)):
        listPredict.append({
            "coherence": predictions[idx], 
            "candidate": listResult[idx]
        })

    pprint( sorted(listPredict, key=lambda k: k['coherence'], reverse = True)[:len(listResult)] )