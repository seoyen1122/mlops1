# 필요한 라이브러리 불러오기
import pandas as pd
import numpy as np

# 데이터 샘플링 함수 (랜덤 샘플링 비율 조정)
def random_sample(df, fraction=0.1):
    return df.sample(frac=fraction, random_state=42).reset_index(drop=True)

def calculate_feature_importance(Xtrain, ytrain, features, n_top_features=7):
    import lightgbm as lgb
    from sklearn.model_selection import train_test_split
    import pandas as pd

    # 2차원으로 다시 변환하여 학습 (샘플 수, 피처 수)
    Xtrain_reshaped = Xtrain.reshape(Xtrain.shape[0], -1, len(features)).mean(axis=1)

    # 일부 데이터를 사용하여 피처 중요도 계산
    Xtrain_part, Xval, ytrain_part, yval = train_test_split(Xtrain_reshaped, ytrain, test_size=0.2, random_state=42)

    # LightGBM 모델 설정
    lgb_model = lgb.LGBMRegressor(
        random_state=42,
        n_estimators=300,             
        learning_rate=0.1,             
        max_depth=5,                   
        num_leaves=31,                 
        n_jobs=-1                       
    )

    # LightGBM 모델 학습 (조기 종료를 위해 검증 데이터 사용)
    lgb_model.fit(
        Xtrain_part, ytrain_part,
        eval_set=[(Xval, yval)],        # 검증 데이터 사용
        eval_metric='rmse'              # RMSE를 평가 지표로 사용
    )

    # 피처 중요도 계산
    importance = lgb_model.feature_importances_

    # 피처 중요도 데이터프레임 생성
    feature_importance_df = pd.DataFrame({
        'feature': features,
        'importance': importance
    }).sort_values(by="importance", ascending=False)

    # 중요도 상위 n개 선택
    top_features = feature_importance_df['feature'].head(n_top_features).tolist()

    print(feature_importance_df.head(n_top_features))

    # 선택된 피처 인덱스 추출
    top_feature_indices = [features.index(f) for f in top_features]
    return top_feature_indices

# 훈련 및 테스트 데이터 구성 함수
def select_top_features(X, top_feature_indices):
    return X[:, top_feature_indices]

# CSV 파일 불러오기
df = pd.read_csv("C:/Users/seoye/ewha/mlops/mlops1/mpi_saale_2024/mpi_saale_2024.csv", dayfirst=True)

# Date Time 열을 datetime 형식으로 변환
df['Date Time'] = pd.to_datetime(df['Date Time'], dayfirst=True)

# 사용할 피처와 타겟
features = [
    "p (mbar)", "T (degC)", "Tpot (K)", "Tdew (degC)", "rh (%)", 
    "VPmax (mbar)", "VPact (mbar)", "VPdef (mbar)", "sh (g/kg)", 
    "H2OC (mmol/mol)", "rho (g/m**3)", "wv (m/s)", "wd (deg)"
]
target = "rain (mm)"

train, test = df[df["Date Time"] < pd.Timestamp("2024-09-12")], df[df["Date Time"] >= pd.Timestamp("2024-09-13")]

# 데이터 샘플링 (데이터 양 줄이기)
train = random_sample(train, fraction=0.2)
test = random_sample(test, fraction=0.2)

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

# 시계열 데이터 생성 함수
def create_sequences(data, features, target, seq_length=60):
    X, y = [], []
    for i in range(len(data) - seq_length):
        x_i = data.iloc[i:i+seq_length][features].values.flatten()
        y_i = data[target].iloc[i+seq_length]
        X.append(x_i)
        y.append(np.log1p(y_i))
    return np.array(X), np.array(y)

# 훈련 및 테스트 데이터 생성
Xtrain, ytrain = create_sequences(train, features, target)
Xtest, ytest = create_sequences(test, features, target)

# Feature Importance 계산
top_feature_indices = calculate_feature_importance(Xtrain, ytrain, features, n_top_features=7)
