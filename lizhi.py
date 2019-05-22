import pandas as pd
import numpy as np

f=open(r'C:\离职预测训练赛\pfm_train.csv',encoding='utf-8')
data_trian=pd.read_csv(f)
p=open(r'C:\离职预测训练赛\pfm_test.csv',encoding='utf-8')
data_test=pd.read_csv(p)
#合并训练和测试集方便统一处理特征
data=pd.concat([data_trian,data_test],axis=0,sort=True)

#查看数据是否缺失，数值型特征统计描述并排查异常值（考察数据极值是否合理）
data.info()
data.describe()

#类别型特征哑编码处理
data=data.reset_index()
data_BusinessTravel=pd.get_dummies(data['BusinessTravel'],prefix = 'BusinessTravel')
data_Department=pd.get_dummies(data['Department'],prefix = 'Department')
data_EducationField=pd.get_dummies(data['EducationField'],prefix = 'EducationField')
data_Gender=pd.get_dummies(data['Gender'],prefix = 'Gender')
data_JobRole=pd.get_dummies(data['JobRole'],prefix = 'JobRole')
data_MaritalStatus=pd.get_dummies(data['MaritalStatus'],prefix = 'MaritalStatus')
#data_NumCompaniesWorked=pd.get_dummies(data['NumCompaniesWorked'],prefix = 'NumCompaniesWorked')与Attrition相关性太低
data_OverTime=pd.get_dummies(data['OverTime'],prefix = 'OverTime')

data=data[['Attrition','Age','DistanceFromHome','Education','EnvironmentSatisfaction','JobInvolvement','JobLevel',
           'JobSatisfaction','MonthlyIncome','PercentSalaryHike','PerformanceRating','RelationshipSatisfaction',
           'StockOptionLevel','TotalWorkingYears','TrainingTimesLastYear','WorkLifeBalance','YearsAtCompany',
           'YearsInCurrentRole','YearsSinceLastPromotion','YearsWithCurrManager']]
data=pd.concat([data,data_BusinessTravel,data_Department,data_Gender,data_EducationField,data_MaritalStatus,
                data_JobRole,data_OverTime],axis=1)

#找与各特征相关系数最大的其他特征
corrDf = data_full.corr()
def func():
    for i in range(58):
         a=corrDf.iloc[:,i:i+1].drop(index=corrDf.iloc[:,i:i+1].idxmax())
         b=a.abs().idxmax()
         c=a.abs().max()
         print(b,c)
func()

#查看所有特征与离职的相关性
corrDf['Attrition'].sort_values(ascending=False)

#去除特征：与其他特征y强相关，ρ>0.5且与Attrition的|ρ|小于y的，有Age，MonthlyIncome ，PercentSalaryHike，JobLevel，YearsAtCompany，YearsSinceLastPromotion，YearsWithCurrManager，Department_Human Resources，Department_Research & Development，Department_Sales，，
data=data.drop(columns=['Age','MonthlyIncome','PercentSalaryHike','JobLevel','YearsWithCurrManager','YearsSinceLastPromotion','YearsInCurrentRole','Department_Human Resources','Department_Research & Development','Department_Sales'])

#连续特征离散化
level_workyear=pd.cut(data.TotalWorkingYears,2,retbins=True)
c=list(level_workyear)
level_workyear=pd.DataFrame(c).T
workyear=pd.get_dummies(level_workyear['TotalWorkingYears'],prefix='TotalWorkingYears')
level_YearsAtCompany=pd.cut(data.YearsAtCompany,2,retbins=True)
c=list(level_YearsAtCompany)
level_YearsAtCompany=pd.DataFrame(c).T
YearsAtCompany=pd.get_dummies(level_YearsAtCompany['YearsAtCompany'],prefix='YearsAtCompany')
level_DistanceFromHome=pd.cut(data.DistanceFromHome,2,retbins=True)
c=list(level_DistanceFromHome)
level_DistanceFromHome=pd.DataFrame(c).T
DistanceFromHome=pd.get_dummies(level_DistanceFromHome['DistanceFromHome'],prefix='DistanceFromHome')

data=data.drop(columns=['DistanceFromHome','TotalWorkingYears','YearsAtCompany'])
data=pd.concat([data,workyear,DistanceFromHome,YearsAtCompany],axis=1)

#标准化
from sklearn import preprocessing
scaler=preprocessing.StandardScaler().fit(data.iloc[:,1:].values)
X_scaler=pd.DataFrame(list(scaler.transform(data.iloc[:,1:].values)))
#划分训练集、测试集
from sklearn.cross_validation import train_test_split 
source_x= X_scaler.iloc[:1100,:]
source_y=data_change.iloc[0:1100,0:1]
train_X, test_X, train_Y, test_Y = train_test_split(source_x,source_y,train_size=.8)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit( train_X , train_Y.values.ravel() )

from sklearn.metrics import confusion_matrix
pred_Y = model.predict(test_X)
confusion_matrix(pred_Y,test_Y)

#输出预测值
pred_X=X_scaler.iloc[1100:1450,:]
pred = model.predict(pred_X)
pred = pred_Y.astype(int)
pred = pd.DataFrame(pred)
pred.to_csv('pred.csv',index=False)
