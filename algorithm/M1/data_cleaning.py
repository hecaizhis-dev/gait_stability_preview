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
from scipy.signal import medfilt
from gaitmap.utils.coordinate_conversion import convert_to_fbf
from mobgap.utils.conversions import to_body_frame as mobgap_to_body_frame
from config.loader import load_config

def data_clean():
    file_path='data/dataset/stu_raw_modified' #这里设置要清理的数据集的列目录
    for root,dirs,files in os.walk(file_path):
        for files in sorted(files):
            if files.endswith('.csv'):
                label=files.replace('.csv','').split('_')[-1]
                path=os.path.join(root,files)
                print(label)
                df=pd.read_csv(path,encoding='utf-8-sig')
                #1.列名检查
                cols_needed={'Timestamp','ACC_X_SA','ACC_Y_SA','ACC_Z_SA'}
                df_cols=set(df.columns)
                cols_short=cols_needed-df_cols
                if len(cols_short)>0:
                    print(f'存在缺失列，为{cols_short}')
                    continue
                #2.清理非法字符和inf，统一变成NaN
                cols_acc={'ACC_X_SA','ACC_Y_SA','ACC_Z_SA'}
                cols_gyr={'GYR_X_SA','GYR_Y_SA','GYR_Z_SA'}
                for c in cols_acc:
                    df[c]=pd.to_numeric(df[c],errors='coerce') #coerce:转换失败的会全变成NaN
                    df[c]=df[c].replace([np.inf,-np.inf],np.nan)
                #3.时间轴清理 注意不能对单列清洗，要对整个df清洗。 df['Timestamp'].dropna(Inplace=True)是错误的。
                df.dropna(subset='Timestamp',inplace=True)
                df.sort_values(by='Timestamp',ascending=True,inplace=True)
                df.reset_index(drop=True,inplace=True)
                df.drop_duplicates(subset='Timestamp',keep='first',inplace=True)
                #4.对缺失数据进行线性插值
                for c in cols_acc:
                    df[c]=df[c].interpolate(method='linear',limit=3) #最多只允许连续修补三个空缺
                #5.时间轴等距重构（均匀重采样
                df['Timestamp']=df['Timestamp']-df['Timestamp'].iloc[0]  #让时间轴从0开始，以便后面重塑时间轴
                #创造时间尺子
                time_interval=1.0/128.0#间距
                time_end=df['Timestamp'].max()#尺子终点
                grid=np.arange(0.0,time_end,time_interval,dtype=float)
                #重塑时间轴 由于重构后行数会不一样，所以这里得重新用新容器来存放重构后的数据，最后转换为df
                df_resampled={}
                df_resampled['Timestamp']=grid
                for c in cols_acc:
                    df_resampled[c]=np.interp(grid,df['Timestamp'],df[c]) #三个参数：新尺子 旧尺子 要重塑的列
                for c in cols_gyr:
                    df_resampled[c]=np.interp(grid,df['Timestamp'],df[c]) 
                df_resampled=pd.DataFrame(df_resampled)

                #6.中值滤波（非线性滤波）->Butterworth低通滤波器(线性滤波）+filtfilt零相位滤波.注意中值滤波必须在低通滤波前
                #过滤衣服摩擦、肌肉抽搐等高频噪音，只留下人体躯干晃动的0.5~3HZ。
                b_sig,a_sig=signal.butter(4,20,btype='low',output='ba',fs=128)  #N阶数 Wn截止频率 
                for c in cols_acc:
                   raw_signal=df_resampled[c].to_numpy() #注意需要转换为数组
                   raw_signal=signal.medfilt(raw_signal,kernel_size=5)#kernel_size=5:滑动窗口长度只有5，框5个数字并从小到大排列，然后用中间数字替代脏数据
                   raw_signal=signal.filtfilt(b_sig,a_sig, raw_signal)
                   df_resampled[c]=raw_signal
                #7.输出结果 这里目录也要根据情况修改哦
                _out_root=_root/"data"/"dataset"/"stu_cleaned"
                _rel=os.path.relpath(path,os.path.abspath(file_path))
                _out=_out_root/_rel
                _out.parent.mkdir(parents=True,exist_ok=True)
                df_resampled.to_csv(_out,index=False,encoding="utf-8-sig")
                print(f"[OK] {str(_rel).replace(os.sep,'/')}->cleaned")

                   

if __name__=='__main__':
    data_clean()

                   



        

               


if __name__=='__main__':
    data_clean()
