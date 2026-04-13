import sys
from pathlib import Path
_root=Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0,str(_root))

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer,precision_score,recall_score,f1_score,accuracy_score
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve,roc_auc_score
import joblib
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier
import seaborn as sb
import optuna
import shap

#注意超参数优化和去掉无用特征的代码在training_generalized里。
from algorithm.M1.training_generalized import optuna_study_search,drop_features



#是否用内层交叉验证调参 因为建模数据多次划分导致数据太少默认不用
USE_INNER_CV_FOR_OPTUNA=True
INNER_CV_SPLITS=5        #内层交叉验证折数
RANDOM_STATE=42
#2.划分训练集
TRAIN_SIZE=0.8




#******非常非常非常重要，直接决定了模型的可用性。
#新相对特征算法：拿出最初正常行走的前times秒数据作为基线，计算出基线后将这所有分类的这times秒数据全部丢弃，防止数据泄露。
def relative_features(df,times=30):
    # 所有加速度特征
    relative_features = [
  

        'acc_rms',
        'acc_impact',
        'acc_mag_mean',
        'acc_mag_std',
        'acc_mag_range',
        'acc_mag_max',
        'acc_mag_skew',
        'acc_mag_kurtosis',
        'acc_SMA',
        'corr_xy',
        'corr_xz',
        'corr_yz',
        'acc_x_mean',
        'acc_x_median',
        'acc_x_std',
        'acc_x_max',
        'acc_x_min',
        'acc_x_range',
        'acc_x_skew',
        'acc_x_kurtosis',
        'acc_x_abs_mean',
        'acc_y_mean',
        'acc_y_median',
        'acc_y_std',
        'acc_y_max',
        'acc_y_min',
        'acc_y_range',
        'acc_y_skew',
        'acc_y_kurtosis',
        'acc_y_abs_mean',
        'acc_z_mean',
        'acc_z_median',
        'acc_z_std',
        'acc_z_max',
        'acc_z_min',
        'acc_z_range',
        'acc_z_skew',
        'acc_z_kurtosis',
        'acc_z_abs_mean',

    ]
    relative_features = [f for f in relative_features if f in df.columns]
    #每人单独计算相对特征
    #提取所有前times秒正常行走数据
    first_sec_walks=df[(df['window_start']<times)&(df['window_start']>=0)&(df['label']=='normal')]
    #按人分组
    first_sec_walks_grouped=first_sec_walks.groupby('member_id')
    #计算基准值
    baselines=first_sec_walks_grouped[relative_features].mean()
    #根据member_id把基准值广播给整张表
    for feature in relative_features:
        feature_baseline=df['member_id'].map(baselines[feature])
        #计算相对特征并增加相应列
        df[f'relative_{feature}']=df[feature]/(feature_baseline+1e-10)
    return df




def training_personalized():

    np.random.seed(RANDOM_STATE)
    #得到文件目录
    file_dic=os.path.dirname(os.path.abspath(__file__))
    file_path=os.path.join(file_dic,'features_full_personalized.txt')
    features_full=pd.read_csv(file_path)
    #转换为三分类 0:正常走路/1:分心/2:疲劳 
    features_full['triple_label']=features_full['label'].map({'normal':0,'distraction':1,'fatigue':2})
    #删除冗余特征
    features_full=drop_features(features_full)
    #相对特征算法
    features_full=relative_features(features_full)

    #1.定制化模型：按人循环，每人单独训练专属模型
    to_drop=[c for c in ['member_id','label','triple_label','file_idx','window_start'] if c in features_full.columns]
    feat_names_global=features_full.drop(columns=to_drop).columns.tolist()
    base_file_dic=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    graph_dir=os.path.join(base_file_dic,'data','graph')
    for _sub in ('CM','AUC','PR','FI','SHAP'):
        os.makedirs(os.path.join(base_file_dic,'data','graph','personalization',_sub),exist_ok=True)
    members=features_full['member_id'].unique()
    all_metrics=[]
    pooled_shap_by_class=[[],[],[]]
    pooled_Xs=[]

    #按人循环
    for member_id in members:
        features_person=features_full[features_full['member_id']==member_id].copy()
        #让样本顺序与时间对齐
        if 'window_start' in features_person.columns:
            features_person=features_person.sort_values('window_start')
        #1.处理训练集
        X_raw=features_person.drop(to_drop,axis=1)
        X=X_raw.fillna(X_raw.median())
        y=features_person['triple_label'].astype(int)
        y_np=y.values

        #2.划分训练集
        idx_all=np.arange(len(features_person))
        idx_train, idx_test=train_test_split(
                idx_all, train_size=TRAIN_SIZE, random_state=RANDOM_STATE, stratify=y)
            
       

        #3.训练/预测
        X_train_fold=X.iloc[idx_train]
        y_train_fold=y.iloc[idx_train]
        X_test_fold=X.iloc[idx_test]

        inner_cv=StratifiedKFold(n_splits=INNER_CV_SPLITS, shuffle=False) if USE_INNER_CV_FOR_OPTUNA else None
        best_params=optuna_study_search(X_train_fold, y_train_fold, inner_cv, None)

        xgb=XGBClassifier(**best_params, objective='multi:softprob', num_class=3, random_state=RANDOM_STATE)
        model=ImbPipeline([
            ('scaler', StandardScaler()),
            ('xgb', xgb),
        ])
        model.fit(X_train_fold, y_train_fold)
        y_test=y_np[idx_test]
        y_proba_test=model.predict_proba(X_test_fold)
        y_pred_train=model.predict(X_train_fold)
        y_pred_test=model.predict(X_test_fold)

        #指标只看测试集
        acc_tr=accuracy_score(y_train_fold,y_pred_train)
        acc,prec,rec,f1=accuracy_score(y_test,y_pred_test),precision_score(y_test,y_pred_test,average='macro'),recall_score(y_test,y_pred_test,average='macro'),f1_score(y_test,y_pred_test,average='macro')
        all_metrics.append({'acc':acc,'prec':prec,'rec':rec,'f1':f1})
        print(f"  [{member_id}]  train Acc:{acc_tr:.3f}  test Acc:{acc:.3f}  test F1:{f1:.3f}")

        #每个人的混淆矩阵（测试集）
        c_m=confusion_matrix(y_test,y_pred_test,labels=[0,1,2])
        c_m_label=ConfusionMatrixDisplay(
                confusion_matrix=c_m,
                display_labels=['normal','distraction','fatigue']
            )
        c_m_label.plot()
        plt.title('confusion matrix of normal and distraction and fatigue')
        plt.savefig(f'data/graph/personalization/CM/person_cm_{member_id}.png',dpi=150,bbox_inches='tight')
        #plt.show()
        plt.close()

        #每个人每一个分类画ROC
        for i in range(3):
            #ROC对于只认识是和不是，而y_test包含三分类内容，（之前的映射{'normal':0,'distraction':1,'fatigue':2}）
            #所以我们用y_test==i可以将其二分类（是和不是），如果当前是0，就是正常态。如果是1或者2，就是异常态。更笨一点的话，可以直接判断i是0/1/2 。
            fpr,tpr,threshold=roc_curve((y_test==i).astype(int),y_proba_test[:,i])
            plt.plot(
                fpr,
                tpr,
            )
            plt.plot([0,1],[0,1])  #画一条对角线
            plt.xlim([0.0,1.0])
            plt.xticks(np.arange(0,1.2,0.2))
            plt.ylim([0.0,1.0]) 
            plt.yticks(np.arange(0,1.2,0.2))
            plt.grid(0.3)
            if i==0:
                plt.title('AUC of normal')
            elif i==1:
                plt.title('AUC of distraction')
            elif i==2:
                plt.title('AUC of fatigue')
            #plt.show()
            plt.savefig(f'data/graph/personalization/AUC/person_roc_{member_id}_class{i}.png',dpi=150,bbox_inches='tight')
            plt.close()
        
        

        #每个人每一个分类画PR,计算阈值对应的最高F1。曲线越往右上靠，包围面积越大，模型越好
        for i in range(3):
            #ROC对于只认识是和不是，而y_test包含三分类内容，（之前的映射{'normal':0,'distraction':1,'fatigue':2}）
            pre,rec,threshold_pr=precision_recall_curve((y_test==i).astype(int),y_proba_test[:,i])
            plt.plot(
                rec,
                pre
            )   
            plt.xlim([0.0,1.0])
            plt.xticks(np.arange(0,1.2,0.2))
            plt.ylim([0.0,1.0])
            plt.yticks(np.arange(0,1.2,0.2))
            plt.grid(0.3)
            if i==0:
                plt.title('PR of normal')
            elif i==1:
                plt.title('PR of distraction')
            elif i==2:
                plt.title('PR of fatigue')
            #plt.show()
            plt.savefig(f'data/graph/personalization/PR/person_pr_{member_id}_class{i}.png',dpi=150,bbox_inches='tight')
            plt.close()


        #最终报告
        print(classification_report(y_test,y_pred_test,target_names=['normal','distraction','fatigue'],zero_division=0))

       
        #保存模型，在data\models\personalized里。
        out_dir=os.path.join(base_file_dic,'data','models','personalized')
        os.makedirs(out_dir,exist_ok=True)
        try:
            model_path=os.path.join(out_dir,f'gait_stability_model_{member_id}.pkl')
            joblib.dump(model,model_path)
        except Exception as e:
            print(f" [{member_id}] 模型保存失败，因为",str(e))

        #每人的特征重要性分析
        importances=model.named_steps['xgb'].feature_importances_
        #获得特征名，拼接为一个列表。
        feature_names=X.columns.tolist()
        #拼接数据
        imp_df=pd.DataFrame({
            'feature_name':feature_names,
            'importance':importances
        })
        #排序
        imp_df=imp_df.sort_values(by='importance',ascending=False)
        print(imp_df)
        #画图
        ax=sb.barplot(
            data=imp_df.head(25),
            y='feature_name',
            x='importance',
        )
        ax.tick_params(axis='y',labelsize=6)#字体改小一点防止特征挤在一起
        #sb.show()
        plt.savefig(f'data/graph/personalization/FI/person_feature_importances_{member_id}.png',dpi=150,bbox_inches='tight')
        plt.close()

        #每人SHAP 用该人专属模型上算值，拼接成整体summary
        X_scaled=model.named_steps['scaler'].transform(X)
        explainer_shap=shap.TreeExplainer(model.named_steps['xgb'])
        sv=explainer_shap.shap_values(X_scaled)
        if isinstance(sv,list):
            for ci in range(3):
                pooled_shap_by_class[ci].append(sv[ci])
        else:
            for ci in range(3):
                pooled_shap_by_class[ci].append(sv[...,ci])
        pooled_Xs.append(X_scaled)

    #全体SHAP summary
    X_all=np.vstack(pooled_Xs)
    sv_all=[np.vstack(pooled_shap_by_class[i]) for i in range(3)]
    plt.figure()
    shap.summary_plot(
        sv_all,
        X_all,
        feature_names=feat_names_global,
        class_names=['normal','distraction','fatigue'],
        show=False,
        max_display=25,
    )
    fig=plt.gcf()
    fig.tight_layout()
    fig.savefig(f'data/graph/personalization/SHAP/person_shap_summary_all.png',dpi=150,bbox_inches='tight')
    plt.close(fig)

    # 所有成员平均
    if all_metrics:
        n_ok=len(all_metrics)
        print(f"所有成员平均 n={n_ok}")
        print(f"Accuracy:{np.mean([m['acc'] for m in all_metrics]):.3f} ± {np.std([m['acc'] for m in all_metrics]):.3f}")
        print(f"Precision:{np.mean([m['prec'] for m in all_metrics]):.3f} ± {np.std([m['prec'] for m in all_metrics]):.3f}")
        print(f"Recall:{np.mean([m['rec'] for m in all_metrics]):.3f} ± {np.std([m['rec'] for m in all_metrics]):.3f}")
        print(f"F1:{np.mean([m['f1'] for m in all_metrics]):.3f} ± {np.std([m['f1'] for m in all_metrics]):.3f}")

    print(f"\n共为 {len(all_metrics)}/{len(members)} 人保存个人模型到 data/models/personalized/")


if __name__=='__main__':
    training_personalized()
