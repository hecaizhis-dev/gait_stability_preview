"""
步态稳定性模型服务 - 完整版
包含：实时预测接口 /predict 和 批量处理接口 /process_batch
"""

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
import joblib
import sys
import csv
from pathlib import Path
from collections import Counter
from datetime import datetime
from scipy.signal import medfilt, butter, filtfilt

# 添加项目根目录到路径，以便导入算法模块
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

# 导入算法模块
from algorithm.M1.features_extract_personalized import extract_features_algorithm
from algorithm.M1.training_generalized import drop_features

app = FastAPI(title="Gait Stability Model Service", version="1.0")

# ========== 配置 ==========
# 模型路径
MODEL_PATH = ROOT_DIR / "data" / "models" / "gait_stability_model.pkl"
# 采样率（Hz）
SAMPLE_RATE = 128
# 窗口大小（秒）
WINDOW_SECONDS = 5
# 步进大小（秒）
STEP_SECONDS = 2

# 计算窗口样本数
WINDOW_SAMPLES = int(WINDOW_SECONDS * SAMPLE_RATE)
STEP_SAMPLES = int(STEP_SECONDS * SAMPLE_RATE)

# 标签映射
LABEL_MAP = {0: "normal", 1: "distraction", 2: "fatigue"}

# ========== 全局变量 ==========
model = None

# ========== Pydantic 模型 ==========
class PredictRequest(BaseModel):
    """预测请求格式"""
    imu_data: List[List[float]]  # 二维数组，每个元素是 [acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z]
    user_id: Optional[str] = None

class PredictResponse(BaseModel):
    """预测响应格式"""
    code: int
    message: str
    data: Optional[Dict[str, Any]] = None

# ========== 保存训练数据函数 ==========
def save_training_data(imu_data: List[List[float]], user_id: str, label: str):
    """
    保存训练数据到 CSV 文件
    文件名格式: {user_id}_{label}_{timestamp}.csv
    """
    save_dir = Path(__file__).parent / "training_data"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    filename = save_dir / f"{user_id}_{label}_{timestamp}.csv"
    
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z'])
        writer.writerows(imu_data)
    
    print(f"   💾 训练数据已保存: {filename}")
    return str(filename)

# ========== 数据清洗函数 ==========
def clean_imu_data_realtime(imu_data: List[List[float]], sample_rate: int = 128) -> pd.DataFrame:
    """
    实时数据清洗函数
    1. 列名检查
    2. 清理非法字符和 inf
    3. 时间轴对齐和清理
    4. 缺失数据线性插值
    5. 时间轴等距重构
    6. 中值滤波 + 低通滤波
    """
    # 1. 转换成 DataFrame
    df = pd.DataFrame(
        imu_data,
        columns=['ACC_X_SA', 'ACC_Y_SA', 'ACC_Z_SA', 'GYR_X_SA', 'GYR_Y_SA', 'GYR_Z_SA']
    )
    
    # 2. 清理非法字符和 inf
    for col in ['ACC_X_SA', 'ACC_Y_SA', 'ACC_Z_SA', 'GYR_X_SA', 'GYR_Y_SA', 'GYR_Z_SA']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
    
    # 3. 创建时间戳
    df['Timestamp'] = np.arange(len(df)) / sample_rate
    
    # 4. 线性插值填充 NaN
    for col in ['ACC_X_SA', 'ACC_Y_SA', 'ACC_Z_SA', 'GYR_X_SA', 'GYR_Y_SA', 'GYR_Z_SA']:
        df[col] = df[col].interpolate(method='linear', limit=3)
        df[col] = df[col].fillna(0)
    
    # 5. 中值滤波 + 低通滤波
    b, a = butter(4, 20, btype='low', output='ba', fs=sample_rate)
    
    for col in ['ACC_X_SA', 'ACC_Y_SA', 'ACC_Z_SA']:
        raw = df[col].to_numpy()
        raw = medfilt(raw, kernel_size=5)
        raw = filtfilt(b, a, raw)
        df[col] = raw
    
    return df[['ACC_X_SA', 'ACC_Y_SA', 'ACC_Z_SA', 'GYR_X_SA', 'GYR_Y_SA', 'GYR_Z_SA']]

# ========== 模型加载 ==========
@app.on_event("startup")
async def load_model():
    """启动时加载模型"""
    global model
    try:
        print(f"加载模型: {MODEL_PATH}")
        model = joblib.load(MODEL_PATH)
        print("✅ 模型加载成功")
        if hasattr(model, "named_steps"):
            xgb = model.named_steps.get("xgb")
            if xgb:
                print(f"   模型类型: XGBoost")
                print(f"   特征数量: {xgb.n_features_in_}")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        model = None

# ========== 特征提取函数 ==========
def extract_features_from_imu(imu_data: List[List[float]], user_id: str) -> pd.DataFrame:
    """从IMU数据中提取特征（先清洗，再提取）"""
    # 1. 清洗数据
    cleaned_df = clean_imu_data_realtime(imu_data, SAMPLE_RATE)
    print(f"   清洗后数据 shape: {cleaned_df.shape}")
    
    # 2. 检查数据量
    if len(cleaned_df) < WINDOW_SAMPLES:
        raise ValueError(f"数据量不足，至少需要 {WINDOW_SAMPLES} 个数据点（{WINDOW_SECONDS}秒），当前 {len(cleaned_df)} 个")
    
    # 3. 包装成算法需要的格式
    temp_data = pd.DataFrame([{
        'member_id': user_id,
        'label': 'unknown',
        'data': cleaned_df
    }])
    
    # 4. 特征提取
    features_rows = extract_features_algorithm(
        temp_data, 0, SAMPLE_RATE, WINDOW_SAMPLES, STEP_SAMPLES
    )
    
    if not features_rows:
        raise ValueError("无法从IMU数据中提取有效特征")
    
    # 5. 转换为DataFrame
    features_df = pd.DataFrame(features_rows)
    print(f"   特征提取完成，得到 {len(features_df)} 个窗口")
    
    # 6. 只保留加速度特征
    features_df = drop_features(features_df)
    
    return features_df

# ========== 模型预测函数 ==========
def predict_with_model(features_df: pd.DataFrame) -> Dict[str, Any]:
    """使用模型进行预测"""
    global model
    
    meta_cols = ['member_id', 'label', 'window_start']
    X = features_df.drop([c for c in meta_cols if c in features_df.columns], axis=1)
    X = X.fillna(X.median())
    
    print(f"   特征矩阵 shape: {X.shape}")
    
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)
    
    per_window = []
    for i in range(len(y_pred)):
        per_window.append({
            "window_start": float(features_df.iloc[i]['window_start']) if 'window_start' in features_df.columns else i * STEP_SECONDS,
            "label": LABEL_MAP[y_pred[i]],
            "confidence": float(max(y_proba[i])),
            "probabilities": {
                "normal": float(y_proba[i][0]),
                "distraction": float(y_proba[i][1]),
                "fatigue": float(y_proba[i][2])
            }
        })
    
    label_counts = Counter([LABEL_MAP[p] for p in y_pred])
    overall_label = label_counts.most_common(1)[0][0]
    avg_confidence = float(np.mean([max(p) for p in y_proba]))
    avg_proba = np.mean(y_proba, axis=0)
    
    return {
        "overall_label": overall_label,
        "confidence": avg_confidence,
        "class_probabilities": {
            "normal": float(avg_proba[0]),
            "distraction": float(avg_proba[1]),
            "fatigue": float(avg_proba[2])
        },
        "per_window": per_window,
        "num_windows": len(per_window)
    }

# ========== API 接口 ==========

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """
    步态稳定性预测接口（实时预测）
    只做预测，不保存数据
    """
    if model is None:
        return PredictResponse(code=500, message="模型未加载", data=None)
    
    imu_data = request.imu_data
    user_id = request.user_id or "anonymous"
    
    print(f"\n收到预测请求")
    print(f"   用户: {user_id}")
    print(f"   数据点数: {len(imu_data)}")
    
    if len(imu_data) == 0:
        return PredictResponse(code=400, message="IMU数据不能为空", data=None)
    
    if len(imu_data) > 0:
        print(f"   第一个数据点: {imu_data[0]}")
    
    try:
        features_df = extract_features_from_imu(imu_data, user_id)
        result = predict_with_model(features_df)
        
        print(f"   预测结果: {result['overall_label']} (置信度: {result['confidence']:.3f})")
        
        return PredictResponse(code=200, message="success", data=result)
        
    except ValueError as e:
        print(f"   ❌ 数据验证失败: {e}")
        return PredictResponse(code=400, message=str(e), data=None)
    except Exception as e:
        print(f"   ❌ 预测失败: {e}")
        import traceback
        traceback.print_exc()
        return PredictResponse(code=500, message=f"预测失败: {str(e)}", data=None)


@app.post("/process_batch")
async def process_batch(request: PredictRequest):
    """
    批量处理长数据流接口
    自动分类并分别保存到三个 CSV 文件（正常/分心/疲劳）
    """
    if model is None:
        return {"code": 500, "message": "模型未加载"}
    
    imu_data = request.imu_data
    user_id = request.user_id or "anonymous"
    
    print(f"\n收到批量处理请求")
    print(f"   用户: {user_id}")
    print(f"   总数据点数: {len(imu_data)}")
    
    if len(imu_data) < WINDOW_SAMPLES:
        return {"code": 400, "message": f"数据量不足，至少需要 {WINDOW_SAMPLES} 个点（{WINDOW_SECONDS}秒）"}
    
    try:
        # 1. 提取特征和预测
        features_df = extract_features_from_imu(imu_data, user_id)
        result = predict_with_model(features_df)
        
        # 2. 按标签分组保存数据
        normal_data = []
        distraction_data = []
        fatigue_data = []
        
        for window in result['per_window']:
            window_start = window['window_start']
            label = window['label']
            window_end = window_start + WINDOW_SECONDS
            
            # 计算窗口对应的数据索引
            start_idx = int(window_start * SAMPLE_RATE)
            end_idx = int(window_end * SAMPLE_RATE)
            
            # 边界检查
            if end_idx > len(imu_data):
                end_idx = len(imu_data)
            
            # 取出这个窗口的原始数据
            window_data = imu_data[start_idx:end_idx]
            
            # 按标签放入对应容器
            if label == "normal":
                normal_data.extend(window_data)
            elif label == "distraction":
                distraction_data.extend(window_data)
            elif label == "fatigue":
                fatigue_data.extend(window_data)
        
        # 3. 保存三个 CSV 文件
        saved_files = []
        
        if normal_data:
            filename = save_training_data(normal_data, user_id, "normal")
            saved_files.append(filename)
            print(f"   ✅ 保存正常数据: {len(normal_data)} 个点")
        
        if distraction_data:
            filename = save_training_data(distraction_data, user_id, "distraction")
            saved_files.append(filename)
            print(f"   ✅ 保存分心数据: {len(distraction_data)} 个点")
        
        if fatigue_data:
            filename = save_training_data(fatigue_data, user_id, "fatigue")
            saved_files.append(filename)
            print(f"   ✅ 保存疲劳数据: {len(fatigue_data)} 个点")
        
        # 4. 统计各窗口数量
        normal_windows = sum(1 for w in result['per_window'] if w['label'] == 'normal')
        distraction_windows = sum(1 for w in result['per_window'] if w['label'] == 'distraction')
        fatigue_windows = sum(1 for w in result['per_window'] if w['label'] == 'fatigue')
        
        print(f"   📊 窗口统计: 正常={normal_windows}, 分心={distraction_windows}, 疲劳={fatigue_windows}")
        
        # 5. 返回结果
        return {
            "code": 200,
            "message": f"处理完成，共保存 {len(saved_files)} 个文件",
            "data": {
                "overall_label": result['overall_label'],
                "num_windows": result['num_windows'],
                "statistics": {
                    "normal_windows": normal_windows,
                    "distraction_windows": distraction_windows,
                    "fatigue_windows": fatigue_windows
                },
                "saved_files": saved_files
            }
        }
        
    except ValueError as e:
        print(f"   ❌ 数据验证失败: {e}")
        return {"code": 400, "message": str(e)}
    except Exception as e:
        print(f"   ❌ 处理失败: {e}")
        import traceback
        traceback.print_exc()
        return {"code": 500, "message": f"处理失败: {str(e)}"}


# ========== 健康检查 ==========
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "sample_rate": SAMPLE_RATE,
        "window_seconds": WINDOW_SECONDS
    }


# ========== 模型信息接口 ==========
@app.get("/info")
async def info():
    if model is None:
        return {"status": "model_not_loaded"}
    
    xgb = model.named_steps.get("xgb") if hasattr(model, "named_steps") else None
    
    return {
        "status": "ready",
        "model_type": "XGBoost",
        "classes": ["normal", "distraction", "fatigue"],
        "sample_rate": SAMPLE_RATE,
        "window_seconds": WINDOW_SECONDS,
        "min_data_points": WINDOW_SAMPLES,
        "feature_count": xgb.n_features_in_ if xgb else None
    }


# ========== 启动入口 ==========
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5001)