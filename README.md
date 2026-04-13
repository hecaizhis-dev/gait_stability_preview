# Gait Stability Preview

IMU-based gait stability 3-class classification: Normal / Distraction / Fatigue. For preview only.

## Environment

```bash
pip install -r requirements.txt
```

## Data

- **DUO-GAIT**: `data/dataset/IMU_processed/duo_merged_dataset` (normal / distraction / fatigue)

## Pipeline

1. **Feature extraction**: `python algorithm/M1/features_extract.py` → `algorithm/M1/features_full.txt`
2. **Personalized model training** (**only this is used online**, no generalization model): `python algorithm/M1/training_personalized.py` → `data/models/personalized/gait_stability_model_<member_id>.pkl`

> The repository also contains generalization training scripts for experiments/comparisons only, **not used as the currently deployed product model**.

## Configuration

`config/default.yaml`: sampling rate, window, stride detection, feature extraction parameters, etc.

## References

- Dataset: `https://zenodo.org/records/8244887`
- Gait processing: `https://github.com/mad-lab-fau/gaitmap`

