import pandas as pd
import numpy as np
import re

def make_categorical(df):
    cat_columns = [col for col in df.columns if df[col].dtype=="O"]
    old_columns = [col for col in df.columns]
    df = pd.get_dummies(df,drop_first=True,dummy_na=True)
    new_cats = [col for col in df.columns if col not in old_columns]
    return df,new_cats

def make_days_positive(df):
    day_reg = re.compile("DAYS|MONTHS")
    for col in df.columns:
        if re.search(day_reg, col):
            if df[col].mean()<0:
                df[col]*=-1
    return df

def replace_max_erroneous_days(df):
    day_reg = re.compile("DAY|MONTHS")
    for col in df.columns:
        if re.search(day_reg, col):
            if (df[col].max()==365423.0):
                df[col].replace(365243.0,
                                df_apps['DAYS_FIRST_DRAWING'].median(),
                                inplace=True)
            elif (df[col].max()==365423):
                df[col].replace(365243,
                                df_apps['DAYS_FIRST_DRAWING'].median(),
                                inplace=True)
    return df

def process_train_test(f):
    df = pd.read_csv(f)
   # df = df.iloc[:,1:] 
   # df_test = pd.read_csv("application_test.csv")
   # df_test.insert(1,"TARGET",np.nan)
   # df_tot = pd.concat([df,df_test])
    df_tot, cat_cols = make_categorical(df)
    #cat_cols.insert(0,"SK_ID_CURR")
    df_tot['Income_Credit_Ratio'] = (df_tot['AMT_INCOME_TOTAL']
                                     / df_tot['AMT_CREDIT'])
    df_tot['Income_per_Person'] = (df_tot['AMT_INCOME_TOTAL']/
                                        df_tot['CNT_FAM_MEMBERS'])
    df_tot['Annuity_Income_Ratio'] = (df_tot['AMT_ANNUITY']/
                                     df_tot['AMT_INCOME_TOTAL'])
    df_tot = make_days_positive(df_tot)
    replace_max_erroneous_days(df_tot)
    #median_days_employed = df_tot['DAYS_EMPLOYED'].median()
    #df_tot.loc[df_tot['DAYS_EMPLOYED']==365243,"DAYS_EMPLOYED"]=median_days_employed
    #df_app_train = df_tot.iloc[0:307511,:].copy()
    #df_app_test = df_tot.iloc[307511:,:].copy()
    del df 
    return df_tot

def process_bureau(f):
    df_bur = pd.read_csv(f)
   # df_bur = df_bur.iloc[:,1:] 
    df_bur,new_cats = make_categorical(df_bur)
    df_bur.drop("SK_ID_BUREAU",axis=1,inplace=True)
    aggs = {
        "DAYS_CREDIT": ["mean","max","min"],
        "DAYS_CREDIT_ENDDATE": ['mean',"max","min"],
        "CREDIT_DAY_OVERDUE": ["mean","max","min"],
        "CNT_CREDIT_PROLONG": ["mean","max","min"],
        "AMT_CREDIT_MAX_OVERDUE": ["mean","max","min"],
        "AMT_CREDIT_SUM": ["mean","max","min"],
        "AMT_CREDIT_SUM_DEBT": ["mean","max","min"],
        "AMT_CREDIT_SUM_LIMIT": ["mean","max","min"],
        "AMT_CREDIT_SUM_OVERDUE": ["mean","max","min"],
        "DAYS_CREDIT_UPDATE": ["mean","max","min"],
        "AMT_ANNUITY": ["mean","max","min"],   
    }
    df_bur = make_days_positive(df_bur)
    df_bur = replace_max_erroneous_days(df_bur)
    for cat in new_cats: 
           aggs[cat]=['mean']
    #df_cat_process = df_bur[['SK_ID_CURR']+new_cats].copy()
    #df_bur.drop(new_cats, axis=1,inplace=True)
    bureau_agg = df_bur.groupby("SK_ID_CURR").agg(aggs)
    bureau_agg.columns = bureau_agg.columns.map('_'.join)
    bureau_agg.reset_index(inplace=True)
    del df_bur
    return bureau_agg

def process_credit_card_balance(f):
    df_credit = pd.read_csv(f)
   # df_credit = df_credit.iloc[:,1:] 
    credit_cat,new_cats = make_categorical(df_credit)
    credit_cat.drop(['SK_ID_PREV'],axis=1,inplace=True)
    credit_cat = credit_cat.groupby("SK_ID_CURR").agg(['min','max','mean'])
    credit_cat.columns = credit_cat.columns.map('_'.join)
    credit_cat.reset_index(inplace=True)
    credit_cat = make_days_positive(credit_cat)
    credit_cat = replace_max_erroneous_days(credit_cat)
    del df_credit
    return credit_cat

def process_previous_apps(f):
    df_apps = pd.read_csv(f)
   # df_apps = df_apps.iloc[:,1:] 
    df_apps.drop('SK_ID_PREV',axis=1,inplace=True)
    print(df_apps.shape)
    df_apps, new_cats = make_categorical(df_apps)
    print(df_apps.shape)
    df_apps['DAYS_FIRST_DRAWING'].replace(365243.0,df_apps['DAYS_FIRST_DRAWING'].median(),inplace=True)
    df_apps['DAYS_FIRST_DUE'].replace(365243.0,df_apps['DAYS_FIRST_DUE'].median(),inplace=True)
    df_apps['DAYS_LAST_DUE'].replace(365243.0,df_apps['DAYS_LAST_DUE'].median(),inplace=True)
    df_apps['DAYS_TERMINATION'].replace(365243.0,df_apps['DAYS_TERMINATION'].median(),inplace=True)
    df_apps['APP_CREDIT_PERC'] = df_apps['AMT_APPLICATION'] / df_apps['AMT_CREDIT']
    aggs = {
        'AMT_ANNUITY': ['min', 'max', 'mean'],
        'AMT_APPLICATION': ['min', 'max', 'mean'],
        'AMT_CREDIT': ['min', 'max', 'mean'],
        'APP_CREDIT_PERC': ['min', 'max', 'mean', 'var'],
        'AMT_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'AMT_GOODS_PRICE': ['min', 'max', 'mean'],
        'HOUR_APPR_PROCESS_START': ['min', 'max', 'mean'],
        'RATE_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'DAYS_DECISION': ['min', 'max', 'mean'],
        'CNT_PAYMENT': ['mean', 'sum']
    }
    df_apps = make_days_positive(df_apps)
    df_apps = replace_max_erroneous_days(df_apps)
    for col in new_cats:
        aggs[col]=["mean"]
    df_apps_agg = df_apps.groupby("SK_ID_CURR").agg(aggs)
    df_apps_agg.columns = df_apps_agg.columns.map('_'.join)
    df_apps_agg.reset_index(inplace=True)
    del df_apps,new_cats
    return df_apps_agg


def process_cash_balance(f):
    df_cash = pd.read_csv(f)
   # df_cash = df_cash.iloc[:,1:] 
    df_cash, new_cats = make_categorical(df_cash)
    
    df_cash.drop("SK_ID_PREV",axis=1,inplace=True)
    aggs = {
        'MONTHS_BALANCE': ['max', 'mean', 'min'],
        'SK_DPD': ['max', 'mean'],
        'SK_DPD_DEF': ['max', 'mean']
    }
    for col in new_cats:
        aggs[col] = ['mean']
    
    df_cash = make_days_positive(df_cash)
    df_cash = replace_max_erroneous_days(df_cash)
    df_cash_agg = df_cash.groupby('SK_ID_CURR').agg(aggs)
    df_cash_agg.columns = df_cash_agg.columns.map('_'.join)
    df_cash_agg.reset_index(inplace=True)
    del df_cash
    return df_cash_agg

def process_payments(f):
    df_payments = pd.read_csv(f)
   # df_payments = df_payments.iloc[:,1:] 
    df_payments,new_cats = make_categorical(df_payments)
    df_payments['PAYMENT_PERC'] = (df_payments['AMT_PAYMENT'] / 
                                   df_payments['AMT_INSTALMENT'])
    df_payments['PAYMENT_DIFF'] = (df_payments['AMT_INSTALMENT'] - 
                                   df_payments['AMT_PAYMENT'])
    aggs = {
        'NUM_INSTALMENT_VERSION': ['nunique'],
        'PAYMENT_PERC': ['max', 'mean', 'sum', 'var'],
        'PAYMENT_DIFF': ['max', 'mean', 'sum', 'var'],
        'AMT_INSTALMENT': ['max', 'mean', 'sum'],
        'AMT_PAYMENT': ['min', 'max', 'mean', 'sum'],
        'DAYS_ENTRY_PAYMENT': ['max', 'mean', 'sum']
    }
    
    for col in new_cats:
        aggs[col] = ['mean']
    df_payments = make_days_positive(df_payments)
    df_payments_agg = df_payments.groupby("SK_ID_CURR").agg(aggs)
    df_payments_agg.columns = df_payments_agg.columns.map('_'.join)
    df_payments_agg.reset_index(inplace=True)
    del df_payments
    return df_payments_agg


def join_dfs(df_app_tot,list_dfs):
    df_tot = df_app_tot.copy()

    print(df_tot.shape)
    for df in list_dfs[1:]:
        df_tot=df_tot.merge(df,how="left",on="SK_ID_CURR")
        print(f"Size after merging: {df_tot.shape}")
    df_train = (df_tot.loc[(df_tot['TARGET']==1) |
                            (df_tot['TARGET']==0)].copy()) 
    del df_tot
    #df_test = df_tot.loc[df_tot['TARGET'].isnull()]
    return df_train

    
