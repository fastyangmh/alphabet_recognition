#   alphabet_recognition
[Google Colab](https://colab.research.google.com/drive/1AQAfZ466Rsrr_SPWd5YGNEVNa08fOxDM?authuser=1#scrollTo=EufszemgjjS-)

##  Abstract
    專案使用的資料來源於Kaggle的A-Z Handwritten Alphabets in .csv format(https://www.kaggle.com/sachinpatel21/az-handwritten-alphabets-in-csv-format#A_Z%20Handwritten%20Data.csv)，此數據集約有260000筆訓練資料，110000筆測試資料，下為模型效能。

    Train dataset accuracy: 0.8658
    Test dataset accuracy: 0.8623

## Model
    模型由2層全連接層組成，使用ReLU和Softmax作為激活函數，並使用Dropout降低過度凝合(over fitting)增加泛化能力。

## Train
    使用訓練集中所有資料做訓練共60000筆資料，每個類別平均10000筆資料，其中，字母Ｉ和Ｆ資料量最少約700筆，因此必須將資料做平衡後再訓練神經網路，未來將加入平衡數據的功能。

##  Predict
    給予資料，模型便會計算並輸出所有類別的機率，從中選擇機率最大值，並輸出該類別。
