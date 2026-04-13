# 步态特征提取公式 (Features Extract Formulas)

本文档整理自 `algorithm/M1/features_extract.py` 中涉及的所有计算公式，参数以 `config/default.yaml` 为准。

---

## 1. 步幅检测 (Stride Detection)

### 1.1 步幅时间
从 `gyr_ml`（内外侧角速度）绝对值检测峰值，相邻峰值索引做差得到单步时间（秒）：
$$\Delta t_{\mathrm{step},k} = \frac{\mathrm{peaks}_{k+1} - \mathrm{peaks}_k}{f_s}$$
完整步幅时间为两个连续单步之和：
$$T_{\mathrm{stride},k} = \Delta t_{\mathrm{step},k} + \Delta t_{\mathrm{step},k+1}$$
其中 $f_s$ 为采样率 (Sam_Rate)。有效步幅过滤条件：$0.7 \leq T_{\mathrm{stride}} \leq 2.0$（秒），由 `stride_time_min`、`stride_time_max` 配置。

### 1.2 IQR 异常值过滤（当前已禁用）
IQR 过滤在代码中已注释，当前不启用。若启用，公式为：
$$\begin{aligned}
Q_1 &= \mathrm{percentile}(T_{\mathrm{stride}}, 25), \quad Q_3 = \mathrm{percentile}(T_{\mathrm{stride}}, 75) \\
\mathrm{IQR} &= Q_3 - Q_1 \\
L &= Q_1 - 1.5 \cdot \mathrm{IQR}, \quad U = Q_3 + 1.5 \cdot \mathrm{IQR}
\end{aligned}$$
保留 $L \leq T_{\mathrm{stride}} \leq U$ 的步幅时间，记为 $\{t_1, t_2, \ldots, t_N\}$。

### 1.3 步幅时间统计量
- **均值**：$\mu_{\mathrm{stride}} = \frac{1}{N}\sum_{i=1}^{N} t_i$
- **标准差**：$\sigma_{\mathrm{stride}} = \sqrt{\frac{1}{N}\sum_{i=1}^{N}(t_i - \mu_{\mathrm{stride}})^2}$
- **变异系数**：$\mathrm{CV}_{\mathrm{stride}} = \frac{\sigma_{\mathrm{stride}}}{\mu_{\mathrm{stride}} + \varepsilon_{\mathrm{cv}}}$（$\varepsilon_{\mathrm{cv}}$ 为数值稳定性常数，默认 $10^{-10}$）
- **极差**：$R_{\mathrm{stride}} = \max_i t_i - \min_i t_i$
- **相邻步幅差均值**：$\overline{|\Delta t|} = \frac{1}{N-1}\sum_{i=1}^{N-1} |t_{i+1} - t_i|$
- **相邻步幅差标准差**：$\sigma_{\Delta t} = \mathrm{std}(t_{i+1} - t_i)$（对 $i=1,\ldots,N-1$）
- **偏度**：$\mathrm{skew}(t)$（步幅时间序列的偏度）
- **峰度**：$\mathrm{kurtosis}(t)$（步幅时间序列的峰度）

---

## 2. 步频检测 (Cadence Detection)

步频定义为每分钟步数；一个步幅对应两步，故：
$$\mathrm{cadence}_i = \frac{C_{\mathrm{spm}}}{t_i + \varepsilon}, \quad i=1,\ldots,N$$
其中 $C_{\mathrm{spm}}$ 为 `steps_per_minute`（默认 120），$\varepsilon$ 为数值稳定性常数。

- **步频均值**：$\mu_{\mathrm{cadence}} = \frac{1}{N}\sum_{i=1}^{N} \mathrm{cadence}_i$
- **步频标准差**：$\sigma_{\mathrm{cadence}} = \sqrt{\frac{1}{N}\sum_{i=1}^{N}(\mathrm{cadence}_i - \mu_{\mathrm{cadence}})^2}$

---

## 3. 加速度检测 (Acceleration Detection)

### 3.1 加速度矢量模
$$a_{\mathrm{mag}} = \sqrt{a_x^2 + a_y^2 + a_z^2}$$
其中 $a_x = \mathrm{acc\_is}$（竖直），$a_y = \mathrm{acc\_ml}$（内外），$a_z = \mathrm{acc\_pa}$（前后）。

### 3.2 加速度矢量模统计
- **均值**：$\mu_a = \frac{1}{n}\sum_{i=1}^{n} a_{\mathrm{mag},i}$
- **标准差**：$\sigma_a = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(a_{\mathrm{mag},i} - \mu_a)^2}$
- **最大值**：$a_{\max} = \max_i a_{\mathrm{mag},i}$
- **极差**：$R_a = \max_i a_{\mathrm{mag},i} - \min_i a_{\mathrm{mag},i}$
- **均方根 (RMS)**：$\mathrm{acc\_rms} = \sqrt{\frac{1}{n}\sum_{i=1}^{n} a_{\mathrm{mag},i}^2}$
- **冲击程度**：$\mathrm{acc\_impact} = \max_i a_{\mathrm{mag},i} - \min_i a_{\mathrm{mag},i}$
- **变异系数**：$\mathrm{acc\_cv} = \frac{\sigma_a}{\mu_a + \varepsilon}$（当 $\mu_a > \varepsilon_{\mathrm{cv}}$ 时计算，默认 $\varepsilon_{\mathrm{cv}} = 10^{-8}$，否则为 NaN）
- **偏度**：$\mathrm{acc\_mag\_skew} = \mathrm{skew}(a_{\mathrm{mag}})$
- **峰度**：$\mathrm{acc\_mag\_kurtosis} = \mathrm{kurtosis}(a_{\mathrm{mag}})$

### 3.3 各轴加速度统计（acc_x / acc_y / acc_z）
代码中对应关系：`acc_x` = `acc_pa`（前后），`acc_y` = `acc_ml`（内外），`acc_z` = `acc_is`（竖直）。每轴计算：
- **均值**：$\mu_{\mathrm{axis}} = \frac{1}{n}\sum_{i} a_{\mathrm{axis},i}$
- **标准差**：$\sigma_{\mathrm{axis}}$
- **最大值**：$\max_i a_{\mathrm{axis},i}$
- **最小值**：$\min_i a_{\mathrm{axis},i}$
- **极差**：$R_{\mathrm{axis}} = \max - \min$
- **偏度**：$\mathrm{skew}(a_{\mathrm{axis}})$（仅 acc_x、acc_y）
- **峰度**：$\mathrm{kurtosis}(a_{\mathrm{axis}})$（仅 acc_x、acc_y）

### 3.4 Jerk（加速度导数）
Jerk 为加速度对时间的导数，$\Delta t = 1/f_s$：
$$\mathrm{jerk}_x = \frac{\mathrm{d}}{\mathrm{d}t} a_{\mathrm{pa}}, \quad \mathrm{jerk}_y = \frac{\mathrm{d}}{\mathrm{d}t} a_{\mathrm{ml}}, \quad \mathrm{jerk}_z = \frac{\mathrm{d}}{\mathrm{d}t} a_{\mathrm{is}}$$
$$\mathrm{jerk}_{\mathrm{mag}} = \sqrt{\mathrm{jerk}_x^2 + \mathrm{jerk}_y^2 + \mathrm{jerk}_z^2}$$

- **均值**：$\overline{\mathrm{jerk}_x}$、$\overline{\mathrm{jerk}_y}$、$\overline{\mathrm{jerk}_z}$、$\overline{\mathrm{jerk}_{\mathrm{mag}}}$

---

## 4. 角速度检测 (Gyroscope / Angular Velocity)

### 4.1 内外侧角速度均方根
$$\mathrm{gyr\_ml\_RMS} = \sqrt{\frac{1}{n}\sum_{i=1}^{n} \omega_{\mathrm{ml},i}^2}$$
其中 $\omega_{\mathrm{ml}}$ 为内外侧角速度 (gyr_ml)。

### 4.2 角速度峰值高度（峰值处绝对值）
记峰值为索引集合 $\mathcal{P}$，峰值高度 $h_k = |\omega_{\mathrm{ml},\mathrm{peaks}_k}|$。

- **峰值高度均值**：$\mu_{\mathrm{pah}} = \frac{1}{|\mathcal{P}|}\sum_{k} h_k$
- **峰值高度标准差**：$\sigma_{\mathrm{pah}} = \sqrt{\frac{1}{|\mathcal{P}|}\sum_{k}(h_k - \mu_{\mathrm{pah}})^2}$
- **峰值高度变异系数**：$\mathrm{gyr\_ml\_pah\_CV} = \frac{\sigma_{\mathrm{pah}}}{\mu_{\mathrm{pah}} + \varepsilon_{\mathrm{cv}}}$（$\varepsilon_{\mathrm{cv}} = 10^{-10}$）

---

## 5. 自相关系数 (Autocorrelation)

步幅时间序列 $\{t_1,\ldots,t_N\}$ 的一阶自相关系数（$N \geq 3$，由 `min_valid_stride` 配置）：
$$\rho_1 = \frac{\sum_{i=1}^{N-1} (t_i - \bar{t})(t_{i+1} - \bar{t})}{\sum_{i=1}^{N} (t_i - \bar{t})^2 + \varepsilon}$$
其中 $\bar{t} = \frac{1}{N}\sum_{i=1}^{N} t_i$。分母需 $> \varepsilon_{\mathrm{den}}$（默认 $10^{-12}$）以免除零，否则返回 NaN。

---

## 6. 谐波比 (Harmonic Ratio)

对每个完整步幅段 $\omega_{\mathrm{ml}}[\mathrm{peaks}_s : \mathrm{peaks}_{s+2}]$（两峰间隔 = 完整步幅）：
1. 重采样到固定长度 `resample_length`（默认 128）。
2. 做实数 FFT，得到幅度谱 $|X_k|$，$k=0,1,\ldots$。
3. 取前 `n_harmonics` 次谐波（默认 20，$j=1,\ldots,20$），奇次谐波和与偶次谐波和：
   $$S_{\mathrm{odd}} = \sum_{j\,\mathrm{odd},\,1\leq j\leq n} |X_j|, \quad S_{\mathrm{even}} = \sum_{j\,\mathrm{even},\,1\leq j\leq n} |X_j|$$
4. 当 $S_{\mathrm{even}} > \varepsilon_{\mathrm{even}}$（默认 $10^{-8}$）时，该步幅的谐波比：
   $$\mathrm{HR}_s = \frac{S_{\mathrm{odd}}}{S_{\mathrm{even}} + \varepsilon}$$

- **谐波比均值**：$\overline{\mathrm{HR}} = \frac{1}{|\mathrm{HR}|}\sum_s \mathrm{HR}_s$
- **谐波比标准差**：$\sigma_{\mathrm{HR}}$（在有多段时计算）

---

## 7. 疲劳变异系数 (Fatigue CV)

在步幅序列上做长度为 `fatigue_window`（默认 5）的滚动窗口：
$$\mathrm{rolling\_stride\_cv}_i = \frac{\sigma_{\mathrm{win},i}}{\mu_{\mathrm{win},i} + \varepsilon}$$
其中 $\sigma_{\mathrm{win},i}$、$\mu_{\mathrm{win},i}$ 分别为第 $i$ 个窗口内步幅时间的标准差与均值。

- **疲劳变异系数**：$\mathrm{fatigue\_cv} = \frac{1}{M}\sum_i \mathrm{rolling\_stride\_cv}_i$（$M$ 为有效窗口数，忽略 NaN）。若窗口长度 $>$ 步幅数则返回 NaN。

---

## 8. 步幅检测中的辅助参数（非特征，仅算法用）

参数来自 `config/default.yaml`：

- **峰值最小间隔（样本数）**：$\mathrm{min\_distance} = f_s \times 0.35$（`min_distance_ratio`）
- **滚动标准差窗口**：$\mathrm{window} = f_s \times 1.2$（`rolling_window_seconds`）
- **滚动最小周期**：$\mathrm{min\_periods} = f_s \times 0.5$（`rolling_min_periods_ratio`）
- **峰值最小高度**：$\mathrm{min\_height} = 0.4 \times \mathrm{rolling\_SD}$（`height_factor_1`）
- **峰值 prominence**：$\mathrm{prominence} = 25$（`prominence_1`）
- **有效步幅时间范围**：$[0.7,\, 2.0]$ 秒（`stride_time_min`、`stride_time_max`）
- **最小有效步幅数**：2（`min_valid_stride`）

---

## 9. 标准化公式（若在后续流程中使用）

$$x' = \frac{x - \mu}{\sigma}$$

---

*公式与 `algorithm/M1/features_extract.py` 实现一致，参数以 `config/default.yaml` 为准，变量名与代码中保持一致便于对照。*
