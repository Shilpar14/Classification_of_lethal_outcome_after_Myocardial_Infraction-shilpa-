#!/usr/bin/env python
# coding: utf-8

# In[25]:


import pandas as pd
import numpy as np
import streamlit as st 
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from collections import Counter
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')
import os


# In[26]:


os.chdir(r"C:\Users\Renu\Downloads\P271")
st.title(":blue[Project 271: predicting the outcome of a Myocardial Infraction]")
st.sidebar.header('User Input Parameters')
def user_input_features():
    AGE = st.sidebar.number_input("Insert the Age")
    SEX = st.sidebar.selectbox('Gender',('0','1'))
    S_AD_ORIT = st.sidebar.number_input("Insert the systolic BP at ICU")
    L_BLOOD = st.sidebar.number_input("Insert White blood cell count (billions per liter)")
    K_SH_POST = st.sidebar.selectbox('Cardiogenic Shock at the time of admission to ICU',('0','1'))
    STENOK_AN = st.sidebar.selectbox('Exertional angina pectoris',('0','1','2','3','4','5','6'))
    NITR_S = st.sidebar.selectbox('liquid nitrate in emergency',('0','1'))
    ant_im = st.sidebar.selectbox('Presence of an anterior myocardial infarction (left ventricular) ',('0','1','2','3','4')) 
    NA_R_1_n = st.sidebar.selectbox('Use of opioid drugs in the ICU in the first hours',('0','1','2','3','4'))
    ASP_S_n = st.sidebar.selectbox(' Use of acetylsalicylic acid in the ICU',('0','1'))
    MP_TP_POST = st.sidebar.selectbox('Paroxysms of atrial fibrillation at the time of admission to ICU',('0','1'))
    n_p_ecg_p_12 = st.sidebar.selectbox('Complete RBBB',('0','1'))
    DLIT_AG =  st.sidebar.selectbox('Duration of arterial hypertension',('0','1','2','3','4','5','6','7'))
    R_AB_1_n = st.sidebar.selectbox('Relapse of the pain in the 1st hours of the hospital period',('0','1','2','3'))
    R_AB_3_n = st.sidebar.selectbox('Relapse of the pain in the 3rd day of the hospital period',('0','1','2','3'))
    IM_PG_P = st.sidebar.selectbox('right ventricular myocardial infarction',('0','1'))
    ritm_ecg_p_01 = st.sidebar.selectbox('normal heart rythum(sinus)',('0','1'))
    ZSN_A = st.sidebar.selectbox('Presence of chronic Heart failure (CHF) ',('0','1','2','3','4'))
    zab_leg_02 = st.sidebar.selectbox('obstructive chronic bronchitis',('0','1'))
    IBS_POST = st.sidebar.selectbox('Coronary heart disease in recent weeks',('0','1','2'))
    zab_leg_03 = st.sidebar.selectbox('Bronchial asthma',('0','1'))
    #complications of Myocardial infraction
    P_IM_STEN = st.sidebar.selectbox('Post-infarction angina',('0','1'))
    FIBR_JELUD = st.sidebar.selectbox('Ventricular Tachycardia',('0','1'))
    RAZRIV = st.sidebar.selectbox('Myocardial Rupture',('0','1'))
    ZSN = st.sidebar.selectbox('Chronic heart faliure',('0','1'))
    REC_IM = st.sidebar.selectbox('Relapse of myocardial infraction',('0','1'))
    OTEK_LANC = st.sidebar.selectbox('Edema (either in ICU or developed)',('0','1'))
      
    data = {'RAZRIV':RAZRIV,'S_AD_ORIT':S_AD_ORIT,'AGE':AGE,'K_SH_POST':K_SH_POST,'STENOK_AN':STENOK_AN,
            'L_BLOOD':L_BLOOD,'NA_R_1_n':NA_R_1_n,'ant_im':ant_im,'IBS_POST':IBS_POST,'NITR_S':NITR_S,
            'ASP_S_n':ASP_S_n,'ritm_ecg_p_01':ritm_ecg_p_01,'R_AB_3_n':R_AB_3_n,'REC_IM':REC_IM,'n_p_ecg_p_12':n_p_ecg_p_12,
            'SEX':SEX,'DLIT_AG':DLIT_AG,'R_AB_1_n':R_AB_1_n,'IM_PG_P':IM_PG_P,'R_AB_3_n':R_AB_3_n,'zab_leg_02':zab_leg_02,
            'P_IM_STEN':P_IM_STEN,'FIBR_JELUD':FIBR_JELUD,'MP_TP_POST':MP_TP_POST,'zab_leg_03':zab_leg_03,'ZSN':ZSN,'ZSN_A':ZSN_A,
            'OTEK_LANC':OTEK_LANC}
    
    features = pd.DataFrame(data,index = [0])
    return features 


# In[27]:


df_in = user_input_features()
st.subheader('Input parameters')
st.write(df_in)


# In[28]:


Data = pd.read_csv("myocardial infarction complications.csv")
df = Data[['RAZRIV','S_AD_ORIT','AGE','K_SH_POST','STENOK_AN','L_BLOOD',
          'NA_R_1_n','ant_im','IBS_POST','NITR_S','ASP_S_n','ritm_ecg_p_01','REC_IM',
         'n_p_ecg_p_12','SEX','IM_PG_P','zab_leg_02','DLIT_AG','R_AB_1_n','R_AB_3_n','FIBR_JELUD','P_IM_STEN',
           'MP_TP_POST','zab_leg_03','ZSN','ZSN_A','OTEK_LANC','O_L_POST','LET_IS']]

#renaming columns
df.columns = ['myocar_rupture', 'sys_BP_ICU', 'age', 'cardiogenic_shock','ch_pain', 'wbc_count',
              'opioid_1st_hr', 'anterior_MI_LV', 'heart_disease_history','lq_nitrates',
              'pain_killer1', 'rytm_sinus_normal', 'mi_relapse','n_p_ecg_p_12', 'sex', 
              'RV_MI', 'obstructive_chronic_bronchitis','his_of_hbp','pain_relapse_1hr','pain_relapse_3day',
              'ven_fib','P_IM_STEN','atr_rytm_irrgularity', 'bronchial_asthma', 'CHF', 'CHF_history','edema',
              'edema_ICU','lethal_outcome']

#filling null values with most frequent

l_out = list(range(0,8))
for outcome in l_out:
    for col in df.columns:
        if df[col].isnull().sum()>0:
            most_freq = Counter(df.loc[df["lethal_outcome"]==outcome, [col]][col].values).most_common()[0][0]
            df[col].fillna(most_freq, inplace=True)
            
#log transforming wbc count
df['wbc_log'] = np.log(df['wbc_count'])
df.drop(['wbc_count'], axis =1, inplace = True)
            
#list of catagorical columns and dataframe with non catagorical variables
Y = df[["lethal_outcome"]]
df_non_cat = df[['age', 'sys_BP_ICU', 'wbc_log']]
df_cat = df.drop(['age', 'sys_BP_ICU', 'wbc_log'], axis = 1)

# normalization of non catagorical inputs
trans = MinMaxScaler()
df_non_cat_norm = pd.DataFrame(trans.fit_transform(df_non_cat))
df_non_cat_norm.columns = df_non_cat.columns
X = pd.concat([df_non_cat_norm,df_cat], axis = 1)

#detecting outliers in a column and replacing outliers with most frequent per category

for col in df_non_cat.columns:
    Q1 = X[col].quantile(0.25)
    Q3 = X[col].quantile(0.75)
    IQR = Q3 - Q1
    whisker_width = 1.5
    lower_whisker = Q1 -(whisker_width*IQR)
    upper_whisker = Q3 + (whisker_width*IQR)
    indexes=X[col][(X[col]>upper_whisker)|(X[col]<lower_whisker)].index
    index_out = []
    for outcome in l_out:
        most_common = Counter(X.loc[X["lethal_outcome"]==outcome, [col]][col].values).most_common()[0][0]
        medi = X.loc[X["lethal_outcome"]==outcome, [col]][col].median()
        for ind in indexes:
            if X.loc[ind,'lethal_outcome'] == outcome:
                index_out.append(ind)            
        X.loc[index_out, col] = most_common
        
#detecting outliers in a column and replacing outliers with most frequent per category
for col in df_cat.columns:
    least_common = Counter(X[col].values).most_common()[-1][0]
    out_ind = X.loc[(X[col]==least_common), col].index
    for outcome in l_out:
        most_common = Counter(X.loc[X["lethal_outcome"]==outcome, [col]][col].values).most_common()[0][0]
        ind_out = []
        for ind in out_ind: 
            if X.loc[ind,'lethal_outcome'] == outcome:
                ind_out.append(ind)
        if len(ind_out)/len(X.loc[X["lethal_outcome"]==outcome, [col]][col])<0.008:
            X.loc[ind_out, col] = most_common
X.drop('lethal_outcome', axis =1, inplace = True) 
# feature engineering
#merging edema and edema_ICU
test1 = []

for i in list(range(0,len(X['edema']))):
    if (X['edema'].tolist())[i]==0:
        if (X['edema_ICU'].tolist())[i]==1:
            test1.append(1)
        else:
            test1.append(0)
        
    else:
        test1.append(1)
X['Edema']=test1

#merging CHF and CHF_history
test = []

#dropping CHF, CHF_history, edema, edema_ICU from X
X.drop(['edema_ICU','edema'], axis = 1, inplace = True)

#balancing the dataset
oversample = SMOTE()
x_train1, x_test, y_train1, y_test = train_test_split(X, Y, test_size=0.20,stratify=Y, random_state=42)
x_train, y_train = oversample.fit_resample(x_train1, y_train1)

#model building
weights = {0:0.95, 1:0.85, 2:0.95, 3:0.32, 4:0.77, 5:1.43, 6:0.62, 7:0.62}
model_svm = SVC(kernel = 'linear', gamma='auto', class_weight = weights, decision_function_shape='ovr', random_state=42)
model_svm.fit(np.array(x_train),np.array(y_train).flatten())
prediction = model_svm.predict(df_in)

st.subheader('Prediction')
st.write(prediction)
if(prediction == 0):
    st.write('Death cause - :green[unknown (Alive)]')
elif(prediction == 1):
    st.write('Death cause - :red[Cardiogenic Shock]')
elif(prediction == 2):
    st.write('Death cause - :red[Pulmonary Edema]')
elif(prediction == 3):
    st.write('Death cause - :red[Myocardial Rupture]') 
elif(prediction == 4):
    st.write('Death cause - :red[Progress of Congestive Heart Failure]')
elif(prediction == 5):
    st.write('Death cause - :red[Thromboembolism]')
elif(prediction == 6):
    st.write('Death cause - :red[Asystole]')
elif(prediction == 7):
    st.write('Death cause - :red[ventricular fibrillation]')


