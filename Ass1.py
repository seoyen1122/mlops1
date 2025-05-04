# Note: 'keras<3.x' or 'tf_keras' must be installed (legacy)
# See https://github.com/keras-team/tf-keras for more details.
from huggingface_hub import from_pretrained_keras
import tf_keras as keras
from tensorflow.keras import layers, Model
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


basemodel = from_pretrained_keras("keras-io/timeseries_forecasting_for_weather")

# 기존 LSTM까지 가져오고, 마지막 Dense는 새로 정의
x = basemodel.layers[-2].output  # LSTM의 출력
new_output = keras.layers.Dense(1)(x)
rain_model = keras.Model(inputs=basemodel.input, outputs=new_output)

rain_model.compile(optimizer="adam", loss="mse", metrics=["mae"])

# CSV 파일 불러오기
df = pd.read_csv("C:/Users/seoye/ewha/mlops/mlops1/mpi_saale_2024/mpi_saale_2024.csv", dayfirst=True)
# df["date"] = pd.to_datetime(df["Date Time"], dayfirst=True)
# df = df.drop(columns=["Date Time"])

# 사용할 피처 7개 (예시)
features = ["T (degC)", "rh (%)", "sh (g/kg)", "Tdew (degC)", "VPact (mbar)", "wv (m/s)", "rho (g/m**3)"]
target = "rain (mm)"

# train, test 데이터셋 나누기
train, test = df[df["Date Time"] < "12.09.2024"], df[df["Date Time"] >= "13.09.2024"]
seq_length = 120 # LSTM 입력 시퀀스 길이

Xtrain = []
ytrain = []
Xtest = []
ytest = []

# train, test 따로 정규화
for i in range(len(features)):
    mean = train[features[i]].mean()
    std = train[features[i]].std()
    train[features[i]] = (train[features[i]] - mean) / std  
    meantest = test[features[i]].mean()
    stdtest = test[features[i]].std()
    test[features[i]] = (test[features[i]] - meantest) / stdtest  

for i in range(len(train) - seq_length):
    x_i = train.iloc[i:i+seq_length][features].values
    y_i = train[target].iloc[i+seq_length]  
    Xtrain.append(x_i)
    ytrain.append(np.log1p(y_i))

for i in range(len(test) - seq_length):
    x_i = test.iloc[i:i+seq_length][features].values
    y_i = test[target].iloc[i+seq_length]  
    Xtest.append(x_i)
    ytest.append(np.log1p(y_i))

Xtrain = np.array(Xtrain)
ytrain = np.array(ytrain).reshape(-1, 1)
Xtest = np.array(Xtest)
ytest = np.array(ytest).reshape(-1, 1)

rain_model.fit(Xtrain, ytrain, epochs=10, batch_size=16)

# 모델 평가
predictions = rain_model.predict(Xtest)
y_true = np.expm1(ytest.reshape(-1))
y_pred = np.expm1(predictions.reshape(-1))  # 예측 결과를 1D 배열로 변환

# threshold 정의: 강수량 0.005mm 이상이면 비가 온 것으로 간주
threshold = 0.005
y_true_bin = (y_true >= threshold).astype(int)
y_pred_bin = (y_pred >= threshold).astype(int)

# 분류 지표 출력
print("Classification Report (Rain/No Rain):")
print(classification_report(y_true_bin, y_pred_bin, digits=4))

# 혼동 행렬
print("Confusion Matrix:")
print(confusion_matrix(y_true_bin, y_pred_bin))

# AUC-ROC (연속 예측값 사용)
auc = roc_auc_score(y_true_bin, y_pred)
print(f"ROC-AUC Score: {auc:.4f}")


