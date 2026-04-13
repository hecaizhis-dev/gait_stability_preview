# Gait Stability Preview

基于 IMU 的步态稳定性三分类：正常 / 分心 / 疲劳。 仅用作预览。

## 环境

```bash
pip install -r requirements.txt
```

## 数据

- **DUO-GAIT**：`data/dataset/IMU_processed/duo_merged_dataset`（normal / distraction / fatigue）
  

## 流程

1. **特征提取**：`python algorithm/M1/features_extract.py` → `algorithm/M1/features_full.txt`
2. **个性化模型训练**（**线上只用这一套**，不用泛化模型）：`python algorithm/M1/training_personalized.py` → `data/models/personalized/gait_stability_model_<member_id>.pkl`
 
> 仓库中有泛化训练脚本，仅作实验/对比，**不作为当前产品部署模型**。

## 配置

`config/default.yaml`：采样率、窗口、步幅检测、特征提取等参数。

## 参考

- 数据集：https://zenodo.org/records/8244887
- 步态处理：https://github.com/mad-lab-fau/gaitmap
