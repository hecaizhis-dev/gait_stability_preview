import sys
from pathlib import Path
_root=Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0,str(_root))

import pandas as pd
import numpy as np
import os
from scipy.signal import find_peaks
from scipy import signal
from gaitmap.utils.coordinate_conversion import convert_to_fbf
from mobgap.utils.conversions import to_body_frame as mobgap_to_body_frame
from config.loader import load_config

_cfg=load_config()



def data_load():
    root=Path(_cfg['_root'])
    file_path=root/_cfg['paths']['data_raw']   #👈文件读取路径请去config/default.yaml里修改哦
    #遍历整个文件夹，读取所有csv文档。
    data=[]
    for root,dirs,files in os.walk(file_path):
        for file_name in sorted(files):#files必须排序，因为后面要根据前xx秒行走数据进行相对特对特征计算
            if not file_name.endswith('.csv'):
                continue
            path=os.path.join(root,file_name)
            df_onefile=pd.read_csv(path)
            parts=file_name.replace('.csv','').split('_')
            label=parts[-1]
            member_id=file_name.split('_')[0].replace('sample','')
            data_file={
                'label':label,
                'member_id':member_id,
                'path':path,
                'data':df_onefile
            }
            data.append(data_file)
    df=pd.DataFrame(data)   
    return df

#1.步幅检测
def stride_detection(window_data,Sam_Rate):
    stride_detection_config=_cfg['stride_detection']
    #适用三轴加速度幅值检峰
    ax=window_data['acc_pa'].values
    ay=window_data['acc_ml'].values
    az=window_data['acc_is'].values
    acc_sig=np.sqrt(ax**2+ay**2+az**2)
    min_distance=Sam_Rate * stride_detection_config['min_distance_ratio']
    acc_std=np.std(acc_sig)
    prominence=max(acc_std * 0.15, 1e-6) if acc_std > 1e-10 else 0.01
    peaks, _=find_peaks(acc_sig, distance=min_distance, prominence=prominence)
    if len(peaks) < 3 and acc_std > 1e-10:
        prominence=max(acc_std * 0.05, 1e-6)
        peaks, _=find_peaks(acc_sig, distance=min_distance, prominence=prominence)
    # 峰值不足2个无法得到步幅时间
    if len(peaks)<2:
        stride_features_skip={'stride_times_mean':np.nan,'stride_times_std':np.nan,'stride_times_cv':np.nan,'stride_times_range':np.nan,'stride_near_diff_mean':np.nan,'stride_near_diff_std':np.nan,'stride_skew':np.nan,'stride_kurtosis':np.nan}
        return stride_features_skip,peaks,np.array([]),np.nan,True


    #用峰间隔算步幅时间，只保留在合理时长内的步幅
    step_times=np.diff(peaks) / Sam_Rate
    stride_times=(step_times[:-1] + step_times[1:]) if len(step_times) >= 2 else np.array([])
    valid_stride=stride_times[
        (stride_times >= stride_detection_config['stride_time_min']) &
        (stride_times <= stride_detection_config['stride_time_max'])
    ] if len(stride_times) > 0 else np.array([])
    if len(valid_stride) < stride_detection_config['min_valid_stride']:
        stride_features_skip={'stride_times_mean': np.nan, 'stride_times_std': np.nan, 'stride_times_cv': np.nan,
            'stride_times_range': np.nan, 'stride_near_diff_mean': np.nan, 'stride_near_diff_std': np.nan,
            'stride_skew': np.nan, 'stride_kurtosis': np.nan}
        return stride_features_skip, peaks, valid_stride, np.nan, True
    stride_mean=np.nanmean(valid_stride)
    stride_features={'stride_times_mean': stride_mean, 'stride_times_std': np.nan, 'stride_times_cv': np.nan,
        'stride_times_range': np.nan, 'stride_near_diff_mean': np.nan, 'stride_near_diff_std': np.nan,
        'stride_skew': np.nan, 'stride_kurtosis': np.nan}
    return stride_features, peaks, valid_stride, stride_mean, False

#2.步频检测 
def cadence_detection(valid_stride,stride_mean):
    steps_per_minute=_cfg['cadence_detection']['steps_per_minute']
    #步频 一分钟走多少步
    step_cadence=steps_per_minute/(valid_stride+1e-10)
    #步频平均值
    step_cadence_mean=np.nanmean(step_cadence)
    #步频标准差
    step_cadence_std=np.nanstd(step_cadence)
    
    cadence_features={
        "step_cadence_mean":step_cadence_mean,
        "step_cadence_std":step_cadence_std
    }
    return cadence_features

#3.加速度检测
def acc_detection(ax,ay,az,window_data,Sam_Rate):
    #加速度矢量
    acc_mag=np.sqrt(ax**2+ay**2+az**2)
    #加速度矢量平均值
    acc_mag_mean=np.mean(acc_mag)
    #加速度矢量标准差
    acc_mag_std=np.std(acc_mag)
    #加速度矢量最大值
    acc_mag_max=np.max(acc_mag)
    
    #加速度矢量极差
    acc_mag_range=np.max(acc_mag)-np.min(acc_mag)
    #加速度能量强度
    f_acc_rms=np.sqrt(np.mean(acc_mag**2))
    #加速度冲击力程度
    f_acc_impact=np.max(acc_mag)-np.min(acc_mag)
    #加速度变异度
    f_acc_cv=np.std(acc_mag)/(np.mean(acc_mag)+1e-10)
    #加速度偏度
    acc_mag_skew=pd.Series(acc_mag).skew()
    #加速度峰度
    acc_mag_kurtosis=pd.Series(acc_mag).kurtosis() 
    #加速度信号信号幅度面积Signal Magnitude Area
    acc_SMA=np.mean(np.abs(ax)+np.abs(ay)+np.abs(az))
    #加速度轴间相关系数
    corr_xy=float(np.nan_to_num(np.corrcoef(ax,ay)[0,1],nan=0.0))
    corr_xz=float(np.nan_to_num(np.corrcoef(ax,az)[0,1],nan=0.0))
    corr_yz=float(np.nan_to_num(np.corrcoef(ay,az)[0,1],nan=0.0))

    #X轴加速度平均值 （前后）
    acc_x_mean=np.mean(window_data['acc_pa'])
    #X轴加速度中位数
    acc_x_median=np.median(window_data['acc_pa'])
    #X轴加速度标准差
    acc_x_std=np.std(window_data['acc_pa'])
    #X轴加速度峰值
    acc_x_max=np.max(window_data['acc_pa'])
    #X轴加速度最小值
    acc_x_min=np.min(window_data['acc_pa'])
    #X轴加速度极差
    acc_x_range=acc_x_max-acc_x_min
    #X轴加速度偏度
    acc_x_skew=window_data['acc_pa'].skew()
    #X轴加速度峰度
    acc_x_kurtosis=window_data['acc_pa'].kurtosis()
    #X轴加速度变异系数
    acc_x_abs_mean=np.mean(np.abs(window_data['acc_pa']))#如果步态对称，acc_x_mean是一个趋近于0的数，所以这里取绝对值，防止下方除以1e-10数字爆炸。
    acc_x_cv=acc_x_std/(acc_x_abs_mean+1e-10)

    #Y轴加速度平均值 （左右）
    acc_y_mean=np.mean(window_data['acc_ml'])
    #Y轴加速度中位数
    acc_y_median=np.median(window_data['acc_ml'])
    #Y轴加速度标准差
    acc_y_std=np.std(window_data['acc_ml'])
    #Y轴加速度峰值
    acc_y_max=np.max(window_data['acc_ml'])
    #Y轴加速度最小值
    acc_y_min=np.min(window_data['acc_ml'])
    #Y轴加速度极差
    acc_y_range=acc_y_max-acc_y_min
    #Y轴加速度偏度
    acc_y_skew=window_data['acc_ml'].skew()
    #Y轴加速度峰度
    acc_y_kurtosis=window_data['acc_ml'].kurtosis()
    #Y轴加速度变异系数
    acc_y_abs_mean=np.mean(np.abs(window_data['acc_ml']))
    acc_y_cv=acc_y_std/(acc_y_abs_mean+1e-10)
    

    
    #Z轴加速度平均值(上下)
    acc_z_mean=np.mean(window_data['acc_is'])
    #Z轴加速度中位数
    acc_z_median=np.median(window_data['acc_is'])
    #Z轴加速度标准差
    acc_z_std=np.std(window_data['acc_is'])
    #Z轴加速度峰值
    acc_z_max=np.max(window_data['acc_is'])
    #Z轴加速度最小值
    acc_z_min=np.min(window_data['acc_is'])
    #z轴加速度极差
    acc_z_range=acc_z_max-acc_z_min
    #Z轴加速度偏度
    acc_z_skew=window_data['acc_is'].skew()
    #Z轴加速度峰度
    acc_z_kurtosis=window_data['acc_is'].kurtosis()
    #Z轴加速度变异系数
    acc_z_abs_mean=np.mean(np.abs(window_data['acc_is']))
    acc_z_cv=acc_z_std/(acc_z_abs_mean+1e-10)

    #平均Jerk（加速度导数）
    jerk_x=np.gradient(window_data['acc_pa'],1/Sam_Rate) 
    jerk_x_mean=np.mean(np.gradient(window_data['acc_pa'],1/Sam_Rate)) 
    jerk_y=np.gradient(window_data['acc_ml'],1/Sam_Rate)
    jerk_y_mean=np.mean(np.gradient(window_data['acc_ml'],1/Sam_Rate)) 
    jerk_z=np.gradient(window_data['acc_is'],1/Sam_Rate)
    jerk_z_mean=np.mean(np.gradient(window_data['acc_is'],1/Sam_Rate))
    jerk_mag=np.sqrt(jerk_x**2+jerk_y**2+jerk_z**2)  
    jerk_mag_mean=np.mean(np.sqrt(jerk_x**2+jerk_y**2+jerk_z**2))

    jerk_x_std=np.std(jerk_x,ddof=1)
    jerk_y_std=np.std(jerk_y,ddof=1)
    jerk_z_std=np.std(jerk_z,ddof=1)
    jerk_mag_std=np.std(jerk_mag,ddof=1)
    jerk_mag_max=np.max(np.abs(jerk_mag))
    jerk_mag_rms=np.sqrt(np.mean(jerk_mag**2))
    jerk_mag_cv=np.std(jerk_mag, ddof=1)/(np.mean(jerk_mag)+1e-10)
    

    
    acc_features={
        'acc_rms':f_acc_rms,
        'acc_cv':f_acc_cv,
        'acc_impact':f_acc_impact,
        'acc_mag_mean':acc_mag_mean,
        'acc_mag_std':acc_mag_std,
        'acc_mag_range':acc_mag_range,
        'acc_mag_max':acc_mag_max,
        'acc_mag_skew':acc_mag_skew,
        'acc_mag_kurtosis':acc_mag_kurtosis,
        'acc_SMA':acc_SMA,
        'corr_xy':corr_xy,
        'corr_xz':corr_xz,
        'corr_yz':corr_yz,
        'acc_x_mean':acc_x_mean,
        'acc_x_median':acc_x_median,
        'acc_x_std':acc_x_std,
        'acc_x_max':acc_x_max,
        'acc_x_min':acc_x_min,
        'acc_x_range':acc_x_range,
        'acc_x_skew':acc_x_skew,
        'acc_x_kurtosis':acc_x_kurtosis,
        'acc_x_abs_mean':acc_x_abs_mean,
        'acc_x_cv':acc_x_cv,
        'acc_y_mean':acc_y_mean,
        'acc_y_median':acc_y_median,
        'acc_y_std':acc_y_std,
        'acc_y_max':acc_y_max,
        'acc_y_min':acc_y_min,
        'acc_y_range':acc_y_range,
        'acc_y_skew':acc_y_skew,
        'acc_y_kurtosis':acc_y_kurtosis,
        'acc_y_abs_mean':acc_y_abs_mean,
        'acc_y_cv':acc_y_cv,
        'acc_z_mean':acc_z_mean,
        'acc_z_median':acc_z_median,
        'acc_z_std':acc_z_std,
        'acc_z_max':acc_z_max,
        'acc_z_min':acc_z_min,
        'acc_z_range':acc_z_range,
        'acc_z_skew':acc_z_skew,
        'acc_z_kurtosis':acc_z_kurtosis,
        'acc_z_abs_mean':acc_z_abs_mean,
        'acc_z_cv':acc_z_cv,
        'jerk_x_mean':jerk_x_mean,
        'jerk_y_mean':jerk_y_mean,
        'jerk_z_mean':jerk_z_mean,
        'jerk_mag':jerk_mag,
        'jerk_mag_mean':jerk_mag_mean,
        'jerk_x_std': jerk_x_std,
        'jerk_y_std': jerk_y_std,
        'jerk_z_std': jerk_z_std,
        'jerk_mag_std': jerk_mag_std,
        'jerk_mag_max': jerk_mag_max,
        'jerk_mag_rms': jerk_mag_rms,
        'jerk_mag_cv': jerk_mag_cv,
        

        
    }
    return acc_features

#4.角速度检测
#pah是Peak Amplitude Height的简写。
def gyr_detection(gyr_ml,peaks):
        #内外侧角速度均方根
        gyr_ml_RMS=np.sqrt(np.mean(gyr_ml**2))
        gyr_ml_abs=np.abs(gyr_ml)
        gyr_ml_peaks_abs_height=gyr_ml_abs
        #角速度峰值高度均值
        gyr_ml_pah_mean=np.mean(gyr_ml_peaks_abs_height)
        #角速度峰值高度标准差
        gyr_ml_pah_std=np.std(gyr_ml_peaks_abs_height)
        #角速度峰值高度标准差
        gyr_ml_pah_CV=gyr_ml_pah_std/(gyr_ml_pah_mean+1e-10)
        gyr_features={
            'gyr_ml_RMS':gyr_ml_RMS,
            'gyr_ml_pah_mean':gyr_ml_pah_mean,
            'gyr_ml_pah_std':gyr_ml_pah_std,
            'gyr_ml_pah_CV':gyr_ml_pah_CV
        }
        return gyr_features

#5.自相关系数检测
def autocorr_detection(valid_stride):
    autocorr_detection_config=_cfg['autocorr_detection']
    stride_autocorr=np.nan
    #当步幅个数有意义时 不使用np.corrcoef，因为归一化方式不同，np.corrcoef会出现过多极值点，在后续机器学习会导致过拟合
    if len(valid_stride)>=autocorr_detection_config['min_valid_stride']:
        x_centered=valid_stride-np.nanmean(valid_stride)
        autocorr_num=np.dot(x_centered[:-1],x_centered[1:])#分子计算
        autocorr_den=np.sum(x_centered**2)#分母计算
        if autocorr_den>1e-10:#必须注意不能为0，否则完全无法运行
            stride_autocorr=autocorr_num/(autocorr_den+1e-10)
    return stride_autocorr

#6.谐波比检测
def harmonic_ratio_detection(gyr_ml,peaks):
    harmonic_ratio_detection_config=_cfg['harmonic_ratio_detection']
    stride_HR=[]
    resample_length=harmonic_ratio_detection_config['resample_length']
    n_harmonics=harmonic_ratio_detection_config['n_harmonics']
    for s in range(len(peaks)-2):#需至少3个峰才能得到1个完整步幅
        stride_signal=gyr_ml[peaks[s]:peaks[s+2]]#两峰间隔=完整步幅，非单步
        stride_signal=signal.resample(stride_signal,resample_length)
        stride_spectrum=np.fft.rfft(stride_signal)
        stride_spectrum_magnitude=np.abs(stride_spectrum)
        even_sum=0
        odd_sum=0
        for j in range(1,min(n_harmonics+1,len(stride_spectrum_magnitude))):#j=1..20 对应第1~20次谐波，0是直波，忽略
            if j%2==0:
                even_sum+=stride_spectrum_magnitude[j]
            else:
                odd_sum+=stride_spectrum_magnitude[j]
        if even_sum>1e-10:#ML用odd/even，分母为even
            stride_HR.append(odd_sum/(even_sum+1e-10))
    stride_HR_mean=np.mean(stride_HR) if len(stride_HR)>0 else np.nan#注意stride_HR是list，要用np.mean。高=和谐，低=突兀。需要一个参照数，通常用平均数。注意这里只是一个样本数据的谐波比，且是通过将其拆分为若干个谐波的谐波比的平均值，而我们需要的是所有样本谐波比之和的平均值作为参考数据
    stride_HR_std=np.std(stride_HR) if len(stride_HR)>0 else np.nan
    hr_features={
        'HR_mean':stride_HR_mean,
        'HR_std':stride_HR_std
    }
    return hr_features

#7.疲劳检测（滚动步幅变异系数）
def fatigue_detection(valid_stride,fatigue_window=None):
    if fatigue_window is None:
        fatigue_window=_cfg['fatigue_detection']['fatigue_window']
    #疲劳检测 用滚动窗口，每几个连续步幅计算一次步幅CV，然后平均这些CV，得到一个累计变异系数，若大于某个阈值，说明疲劳。
    if fatigue_window<=len(valid_stride):
        #计算滚动步幅变异系数
        rolling_stride_cv=pd.Series(valid_stride).rolling(fatigue_window).std()/(pd.Series(valid_stride).rolling(fatigue_window).mean()+1e-10)
        #计算平局变异系数
        rolling_stride_mean_cv=np.nanmean(rolling_stride_cv)
    else:
        rolling_stride_mean_cv=np.nan
    fatigue_features={
        'fatigue_cv':rolling_stride_mean_cv
    }
    return fatigue_features


def extract_features_algorithm(df,i,Sam_Rate,window_samples,step_samples):
    rows=[]
    df_single=df.iloc[i]['data']
    #背部/骶骨SA：DUO图约定Y=上、X=前、Z=右；mobgap输入约定 x=上、y=右、z=前
    #所以acc_x=ACC_Y_SA(上),acc_y=ACC_Z_SA(右),acc_z=ACC_X_SA(前)；
    #读取数据，封装为df,给转换为bf做准备
    imu_data_SA=pd.DataFrame({
        'acc_x':np.asarray(df_single['ACC_Y_SA'],dtype=float),
        'acc_y':np.asarray(df_single['ACC_Z_SA'],dtype=float),
        'acc_z':np.asarray(df_single['ACC_X_SA'],dtype=float),
        'gyr_x':np.asarray(df_single['GYR_Y_SA'],dtype=float),
        'gyr_y':np.asarray(df_single['GYR_Z_SA'],dtype=float),
        'gyr_z':np.asarray(df_single['GYR_X_SA'],dtype=float),
    })
    SA_bf=mobgap_to_body_frame(imu_data_SA)
    data_bf_length_SA=len(SA_bf)


    start=0
    while start+window_samples<=data_bf_length_SA:
        window_data=SA_bf.iloc[start:start+window_samples]
        stride_dic,peaks,valid_stride,stride_mean,stop_extract=stride_detection(window_data,Sam_Rate)
        if stop_extract or len(peaks)<3 or len(valid_stride)==0:
            start+=step_samples
            continue
        gyr_ml_w=window_data['gyr_ml'].values
        cadence_dic=cadence_detection(valid_stride,stride_mean)
        gyr_dic=gyr_detection(gyr_ml_w,peaks)
        hr_dic=harmonic_ratio_detection(gyr_ml_w,peaks)
        fatigue_dic=fatigue_detection(valid_stride)
        stride_autocorr_val=autocorr_detection(valid_stride)
        ax=window_data['acc_is'].values
        ay=window_data['acc_ml'].values
        az=window_data['acc_pa'].values
        acc_dic=acc_detection(ax,ay,az,window_data,Sam_Rate)
        #保留所有特征
        row_base={
            'member_id':df.iloc[i]['member_id'],
            'label':df.iloc[i]['label'],
            **stride_dic,
            **cadence_dic,
            **gyr_dic,
            **hr_dic,
            **fatigue_dic,
            **acc_dic,
            
            'window_start':start/Sam_Rate,
        }
        rows.append(row_base)
        start+=int(peaks[2]-peaks[0])
    return rows

def feature_extract(original_df):
    feature_extract_config=_cfg['feature_extract']
    feature_rows=[]
    Sam_Rate=feature_extract_config['sample_rate']
    df=original_df.copy()
    window_samples=int(feature_extract_config['window_seconds']*Sam_Rate)
    step_samples=int(feature_extract_config['step_seconds']*Sam_Rate)
    for i in range(len(df)):
        feature_rows.extend(extract_features_algorithm(df,i,Sam_Rate,window_samples,step_samples))
    return pd.DataFrame(feature_rows)


if __name__=='__main__':
    df=data_load()
    features=feature_extract(df)
    root=Path(_cfg['_root'])
    m2_dir=Path(__file__).resolve().parent
    out_path=m2_dir/'features_full_generalized.txt'
    try:
        features.to_csv(out_path,index=False,encoding=_cfg['output']['encoding'])
        print('保存成功:', out_path)
    except Exception as e:
        print(f'失败因为{str(e)}')
    print(features)