# 필요한 라이브러리 불러오기
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from supervised.automl import AutoML
from huggingface_hub import from_pretrained_keras

# 사전 학습된 LSTM 모델 로드
basemodel = from_pretrained_keras("keras-io/timeseries_forecasting_for_weather")

# CSV 파일 불러오기
df = pd.read_csv("C:/Users/seoye/ewha/mlops/mlops1/mpi_saale_2024/mpi_saale_2024.csv", dayfirst=True)

df['Date Time'] = pd.to_datetime(df['Date Time'], dayfirst=True)

# 사용할 피처와 타겟 (모든 피처 사용)
features = [
    "wd (deg)", "p (mbar)", "wv (m/s)", "rh (%)", "VPdef (mbar)", "VPmax (mbar)", "Tdew (degC)"
]
target = "rain (mm)"

train, test = df[df["Date Time"] < pd.Timestamp("2024-09-12")], df[df["Date Time"] >= pd.Timestamp("2024-09-13")]
seq_length = 120  # LSTM 입력 시퀀스 길이

# 데이터 정규화 함수
def normalize(df, features):
    for feature in features:
        mean = df[feature].mean()
        std = df[feature].std()
        df[feature] = (df[feature] - mean) / std
    return df

# 정규화 수행
train = normalize(train, features)
test = normalize(test, features)

# 시계열 데이터 생성 함수 수정
def create_sequences(data, features, target, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        # 시계열 길이와 피처 수를 곱하여 일관된 형태로 변환
        x_i = data.iloc[i:i+seq_length][features].values.flatten()
        y_i = data[target].iloc[i+seq_length]
        X.append(x_i)
        y.append(np.log1p(y_i))  # 로그 변환
    return np.array(X), np.array(y)

# 훈련 및 테스트 데이터 생성
Xtrain, ytrain = create_sequences(train, features, target, seq_length)
Xtest, ytest = create_sequences(test, features, target, seq_length)

# 데이터 형태 확인
print(f"훈련 데이터 형태: {Xtrain.shape}")
print(f"테스트 데이터 형태: {Xtest.shape}")

import time
timestamp = time.strftime("%Y%m%d-%H%M%S")
results_path = f"mljar_results_{timestamp}"

# AutoML 모델 설정 (Feature Engineering 및 Selection 활성화)
automl = AutoML(
    results_path=results_path,        
    mode="Perform",                    
    ml_task="regression",              
    algorithms=["Random Forest", "Xgboost"],  
    total_time_limit=3600,             
    features_selection=True,           
    start_random_models=5,             
    hill_climbing_steps=3,             
    golden_features=True,              
    train_ensemble=True,               
    stack_models=True,                 
)

# AutoML 모델 학습
automl.fit(Xtrain, ytrain)

# 모델 예측
predictions = automl.predict(Xtest)

# 로그 변환 되돌리기
y_true = np.expm1(ytest)
y_pred = np.expm1(predictions)

threshold = 0.005
y_true_bin = (y_true >= threshold).astype(int)
y_pred_bin = (y_pred >= threshold).astype(int)

# 평가
print("\nClassification Report (Rain/No Rain):")
print(classification_report(y_true_bin, y_pred_bin, digits=4))

print("\nConfusion Matrix:")
print(confusion_matrix(y_true_bin, y_pred_bin))

# AUC-ROC 계산
auc = roc_auc_score(y_true_bin, y_pred)
print(f"\nROC-AUC Score: {auc:.4f}")

# 최적 모델 정보
print("\nBest Model:", automl._best_model())

# Feature Importance 출력
print("\nFeature Importance:")
leaderboard = automl.get_leaderboard()
print(leaderboard)


