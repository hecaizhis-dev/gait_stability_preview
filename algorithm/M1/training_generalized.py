import pandas as pd
import numpy as np
import os
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict,cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold,StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer,precision_score,recall_score,f1_score,accuracy_score
from sklearn.model_selection import cross_validate
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve
import joblib
from scipy.stats import randint,uniform
from sklearn.feature_selection import RFECV
from sklearn.model_selection import KFold
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import classification_report
from sklearn.model_selection import GroupShuffleSplit
#import shap 我乱调试弄出BUG了还没修好先注释掉

#超参数优化optuna和交叉验证 
def optuna_study_search(X_train,y_train,cv,groups_split):
    classes_scorer=make_scorer(
       roc_auc_score,
       multi_class='ovr',
       response_method='predict_proba',
    )
    def objective(trial):
        optuna_params={
        'n_estimators':trial.suggest_int('n_estimators',500,3000),
        'learning_rate':trial.suggest_float('learning_rate',0.005,0.08,log=True),
        'max_depth':trial.suggest_int('max_depth',3,9),
        'min_child_weight':trial.suggest_int('min_child_weight',1,12),
        'gamma':trial.suggest_float('gamma',0.0,4.0),
        'subsample':trial.suggest_float('subsample',0.65,1.0),
        'colsample_bytree':trial.suggest_float('colsample_bytree',0.6,1.0),
        'reg_alpha':trial.suggest_float('reg_alpha',1e-8,5.0,log=True),
        'reg_lambda':trial.suggest_float('reg_lambda',0.1,10.0,log=True),  
        }
        xgb=XGBClassifier(
            **optuna_params,   
            objective='multi:softprob',
            num_class=3,
            random_state=42,
        )
        model=ImbPipeline([
        ('scaler',StandardScaler()),
        ('xgb',xgb)
        ])
        auc_scores=cross_val_score(
            model,
            X_train,y_train,
            groups=groups_split,
            cv=cv,
            scoring=classes_scorer,
        )
        mean_auc = np.nanmean(auc_scores)
        if not np.isfinite(mean_auc):
            return 0.0
        return float(mean_auc)
    study=optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=10),
    )
    study.optimize(
        objective,
        n_trials=1
    )
    print(f'最佳参数为')
    for i,j in study.best_params.items():
        print(f'{i}:{j}')
    return study.best_params    
     
# 与 features_extract_generalized 输出一致：步幅/步频/陀螺/谐波/疲劳 + acc/corr/jerk；jerk_mag 为向量列不进模型
_META_COLS = frozenset({'member_id', 'label', 'window_start', 'triple_label'})


def _keep_feature_column(name):
    if name in _META_COLS or name == 'file_idx':
        return True
    if name == 'fatigue_cv':
        return True
    if name.startswith(('stride_', 'step_cadence_', 'gyr_ml_', 'HR_')):
        return True
    if name.startswith(('acc_', 'corr_', 'jerk_')) and name != 'jerk_mag':
        return True
    return False


def drop_features(features_df):
    to_drop = [c for c in features_df.columns if not _keep_feature_column(c)]
    return features_df.drop(columns=to_drop, axis=1)


#******非常非常非常重要，直接决定了模型的可用性。
#新相对特征算法：拿出最初正常行走的前times秒数据作为基线，计算出基线后将这所有分类的这times秒数据全部丢弃，防止数据泄露。
def relative_features(df,times=30):
    # 自generalized六块特征各抽若干
    relative_features = [
        #stride_dic
        'stride_times_std',
        'stride_times_mean',
        'stride_near_diff_std',
        'stride_times_cv',
        #cadence_dic
        'step_cadence_mean',
        'step_cadence_std',
        #gyr_dic
        'gyr_ml_pah_mean',
        'gyr_ml_RMS',
        'gyr_ml_pah_std',
        #hr_dic
        'HR_mean',
        'HR_std',
        #fatigue_dic
        'fatigue_cv',
        #acc_dic
        'jerk_x_std',
        'acc_mag_max',
        'jerk_x_mean',
        'jerk_mag_cv',
        'acc_z_median',
        'acc_mag_range',
        'acc_z_min',
        'acc_y_min',
        'acc_impact',
        'acc_cv',
        'jerk_y_std',
        'acc_x_mean',
        'acc_x_median',
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
    #丢掉所有分类这times秒的行走数据（去掉预热数据+防止信息泄露）
   # first_sec_all=df[(df['window_start']<times) & (df['window_start']>=0)]
   # df=df.drop(first_sec_all.index)
   # df=df.reset_index(drop=True)#重置行号!重要，否则样本对不上
    return df


def training():
    np.random.seed(42)
    
    #得到文件目录
    file_dic=os.path.dirname(os.path.abspath(__file__))
    file_path=os.path.join(file_dic,'features_full_generalized.txt')
    features_full=pd.read_csv(file_path)
    #转换为三分类 0:正常走路/1:分心/2:疲劳 
    features_full['triple_label']=features_full['label'].map({'normal':0,'distraction':1,'fatigue':2})
    #删除冗余特征
    features_full=drop_features(features_full)
    #相对特征算法
    features_full=relative_features(features_full)
   
    

    #1.处理训练集 得到X y groups
    to_drop=[c for c in ['member_id','label','triple_label','file_idx','window_start'] if c in features_full.columns]
    X_raw=features_full.drop(to_drop,axis=1)
    X=X_raw.fillna(X_raw.median())
    y=features_full['triple_label'].astype(int)
    groups=features_full['member_id'].to_numpy()
    cv=StratifiedGroupKFold(n_splits=5,shuffle=False)
    
    #2.划分训练集
    #注意，划分X,y时，要同时划分groups
    #重新划分训练集 
    #创造划分器
    splitter=GroupShuffleSplit(n_splits=1,train_size=0.8,random_state=42)
    #得到训练索引和测试索引
    train_idx,test_idx=next(splitter.split(X,y,groups=groups))
    #通过索引得到训练集和测试集
    X_train,X_test=X.iloc[train_idx],X.iloc[test_idx]
    y_train,y_test=y.iloc[train_idx],y.iloc[test_idx]
    #划分groups
    groups_split=groups[train_idx]
    
    #3.构建管线,optuna调参+交叉验证
    xgb=XGBClassifier(
        **optuna_study_search(X_train,y_train,cv,groups_split),  
        objective='multi:softprob',
        num_class=3,
        random_state=42,
    )
    model=Pipeline([
        ('scaler',StandardScaler()),
        ('xgb',xgb),
    ])

    #4.训练
    model.fit(X_train,y_train)

    #5.模型预测
    #预测类别
    y_pred=model.predict(X_test)
    #预测概率
    y_proba=model.predict_proba(X_test)
    
   #7.混淆矩阵
    c_m=confusion_matrix(y_test,y_pred)
    c_m_label=ConfusionMatrixDisplay(
        confusion_matrix=c_m,
        display_labels=['normal','distraction','fatigue']
    )
    c_m_label.plot()
    plt.title('confusion matrix of normal and distraction and fatigue')
    plt.savefig(f'data/graph/generalization/CM/general_confusion_matrix.png',dpi=150,bbox_inches='tight')
    #plt.show()
    plt.close()
    
    #8.对多分类，每一个分类画ROC
    for i in range(3):
        #ROC对于只认识是和不是，而y_test包含三分类内容，（之前的映射{'normal':0,'distraction':1,'fatigue':2}）
        #所以我们用y_test==i可以将其二分类（是和不是），如果当前是0，就是正常态。如果是1或者2，就是异常态。更笨一点的话，可以直接判断i是0/1/2 。
        fpr,tpr,threshold=roc_curve(y_test==i,y_proba[:,i])
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
        plt.savefig(f'data/graph/generalization/AUC/general_AUC_{i}.png',dpi=150,bbox_inches='tight')
        plt.close()
    
    
    #9.对多分类，每一个分类画PR,计算阈值对应的最高F1。曲线越往右上靠，包围面积越大，模型越好

    #best_threshold_pr_all=[] 废弃 阈值点寻找图中曲线的拐点就是最佳阈值点

    for i in range(3):
        #ROC对于只认识是和不是，而y_test包含三分类内容，（之前的映射{'normal':0,'distraction':1,'fatigue':2}）
        pre,rec,threshold_pr=precision_recall_curve(y_test==i,y_proba[:,i])

        #废弃
        # #用公式计算f1
        # pr_f1=(2.0*pre*rec)/(pre+rec+1e-10)#加个极小值防止除以0
        # #找f1最好的那个索引->取出最佳f1->取出最佳阈值
        # #注意precion和recall的长度为n+1，最最后一个的点必定分别是1和0，而threshold_pr的长度为n。所以我们限制索引。
        # best_f1_pr_idx=np.argmax(pr_f1[:len(threshold_pr)])#左开右闭，[0,n)
        # best_f1_pr=pr_f1[best_f1_pr_idx]
        # best_threshold_pr=threshold_pr[best_f1_pr_idx]
        # best_threshold_pr_all.append(best_threshold_pr)

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
        plt.savefig(f'data/graph/generalization/PR/general_PR_{i}.png',dpi=150,bbox_inches='tight')
        plt.close()


    
    #10.评估
    print('最终测试报告：'+classification_report(y_test,y_pred,target_names=['normal','distraction','fatigue']))

    #11.保存模型，在data\models里。
    try:
        base_file_dic=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        model_path=os.path.join(base_file_dic,'data','models','gait_stability_model.pkl')
        print(model_path)
        joblib.dump(model,model_path)
        print("泛化模型保存成功")
    except Exception as e:
        print("泛化模型保存失败，因为",str(e))


    
    #12.总体特征重要性分析（没有分人！而是大家一起的）
    #先从pipeline中取出xgb，然后提取特征重要性
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
    plt.savefig(f'data/graph/generalization/FI/general_feature_importances.png',dpi=150,bbox_inches='tight')







#先不管下面的画图部分了。
#    #shap Xgb>=2.1.4 但<3.0.0 否则版本不兼容报错
#     #对泛化模型，用shap.explainer进行特征重要性全局解释
#     #主要shap需要的是最终的xgb模型 已经缩放的数据 以及基准数据，不能直接把pipeline传进去。
#     #基准数据：shap进行解释，相对于某个基准，每个特征对预测的贡献度是多少。
#     model_shap=model.named_steps['xgb']
#     X_test_shap=model.named_steps['scaler'].transform(X_test)
#  #要转换为dataframe!由于x_test进行了缩放，shap找不到特征名字了，要重新赋予。直接从原来的X_test提取特征名。若不带着传进去会报错
    
#     #这里传入最终模型和基准数据，传入100行数据。
#     #shap.Explainer 找参考系
#     shap_explainer=shap.Explainer(model_shap,X_test_shap[:100])#传入100行缩放后的数据作为基准数据。
#     #用参考系进行分析 得到shap explanation 
#     shap_exp=shap_explainer(X_test_shap)
#     #画图 shap.plots.bar(shap_exp) 这代码画三分类问题太多，放弃这个方法
#     shap.summary_plot(shap_exp,X_test_shap,plot_type="bar",max_display=15,
#     feature_names=X_test.columns,#传入特征名
#     class_names=['normal','distraction','fatigue']
#     )


     #图太多了这里不画了。后续定制的时候再画。
        # #5.模型预测（已由交叉验证完成，y_pred/y_proba_cv用于后续ROC/PR）
        # #对多分类，每一个分类画ROC
        # for i in range(3):
        #     #ROC对于只认识是和不是，而y_test包含三分类内容，（之前的映射{'normal':0,'distraction':1,'fatigue':2}）
        #     #所以我们用y_test==i可以将其二分类（是和不是），如果当前是0，就是正常态。如果是1或者2，就是异常态。更笨一点的话，可以直接判断i是0/1/2 。
        #     fpr,tpr,threshold=roc_curve(y==i,y_proba_cv[:,i])
        #     plt.plot(fpr,tpr)
        #     plt.plot([0,1],[0,1])  #画一条对角线
        #     plt.xlim([0.0,1.0])
        #     plt.xticks(np.arange(0,1.2,0.2))
        #     plt.ylim([0.0,1.0])
        #     plt.yticks(np.arange(0,1.2,0.2))
        #     plt.grid(0.3)
        #     if i==0:
        #         plt.title(f'AUC of normal ({member_id})')
        #         filename=f'roc_normal_{member_id}.png'
        #     elif i==1:
        #         plt.title(f'AUC of distraction ({member_id})')
        #         filename=f'roc_distraction_{member_id}.png'
        #     elif i==2:
        #         plt.title(f'AUC of fatigue ({member_id})')
        #         filename=f'roc_fatigue_{member_id}.png'
        #     plt.savefig(os.path.join(graph_dir,filename),dpi=150,bbox_inches='tight')
        #     plt.close()

        # #对多分类，每一个分类画PR,计算阈值对应的最高F1。曲线越往右上靠，包围面积越大，模型越好。图中曲线的拐点就是最佳阈值点
        # for i in range(3):
        #     #ROC对于只认识是和不是，而y_test包含三分类内容，（之前的映射{'normal':0,'distraction':1,'fatigue':2}）
        #     pre,rec,threshold_pr=precision_recall_curve(y==i,y_proba_cv[:,i])

        #     #废弃
        #     # #用公式计算f1
        #     # pr_f1=(2.0*pre*rec)/(pre+rec+1e-10)#加个极小值防止除以0
        #     # #找f1最好的那个索引->取出最佳f1->取出最佳阈值
        #     # #注意precion和recall的长度为n+1，最最后一个的点必定分别是1和0，而threshold_pr的长度为n。所以我们限制索引。
        #     # best_f1_pr_idx=np.argmax(pr_f1[:len(threshold_pr)])#左开右闭，[0,n)
        #     # best_f1_pr=pr_f1[best_f1_pr_idx]
        #     # best_threshold_pr=threshold_pr[best_f1_pr_idx]
        #     # best_threshold_pr_all.append(best_threshold_pr)

        #     plt.plot(rec,pre)
        #     plt.xlim([0.0,1.0])
        #     plt.xticks(np.arange(0,1.2,0.2))
        #     plt.ylim([0.0,1.0])
        #     plt.yticks(np.arange(0,1.2,0.2))
        #     plt.grid(0.3)
        #     if i==0:
        #         plt.title(f'PR of normal ({member_id})')
        #         filename=f'pr_normal_{member_id}.png'
        #     elif i==1:
        #         plt.title(f'PR of distraction ({member_id})')
        #         filename=f'pr_distraction_{member_id}.png'
        #     elif i==2:
        #         plt.title(f'PR of fatigue ({member_id})')
        #         filename=f'pr_fatigue_{member_id}.png'
        #     plt.savefig(os.path.join(graph_dir,filename),dpi=150,bbox_inches='tight')
        #     plt.close()

    

    
   


if __name__=='__main__':
    training()