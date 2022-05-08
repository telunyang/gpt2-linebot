'''
檢查是否支援 GPU

參考連結
https://pytorch.org/get-started/previous-versions/

本機測試環境所用套件 - 安裝指令
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
'''
import torch
print(torch.cuda.is_available())
# print(torch.cuda.current_device())
# print(torch.cuda.device(0))