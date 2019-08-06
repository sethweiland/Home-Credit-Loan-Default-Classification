import numpy as np
from flask import Flask,request,jsonify
import flask
from flask import send_file
from table import ItemTable,Item
import pandas as pd
import pickle
from lightgbm import LGBMClassifier
import csv
from process_csv import *
with open("best_lgbm.pkl",'rb') as f:
    model = pickle.load(f)

app = Flask(__name__)  
#@app.route('/predict_one',methods=['POST','GET'])
#def predict():
#    for name in ['price', 'sqrtft','income']:
#        f_value = float(
#            request.args.get(model.feature_names[name],"0")
#        )
#        x_input.append(f_value)
#    preds_probs = model.predict_proba([x_input]).flat
#    return flask.render_template("predict_one.html",
@app.route("/")
def home():
    return flask.render_template("predict_many.html")



@app.route('/predict_many',methods=['POST','GET'])
def predict_many():
   if request.method=="POST":
        csv_main = request.files['application_data']
        df_app_main = process_train_test(csv_main)
        
        csv_bureau = request.files['bureau']
        df_bureau = process_bureau(csv_bureau)

        csv_credit_balance = request.files['credit_card_balance']
        df_credit_balance = process_credit_card_balance(csv_credit_balance)

        csv_prev_apps = request.files['previous_app']
        df_prev_apps = process_previous_apps(csv_prev_apps)

        csv_cash_bal = request.files['cash_balance']
        df_cash_bal = process_cash_balance(csv_cash_bal)

        csv_payments = request.files['installments_payments']
        df_payments = process_payments(csv_payments)

        list_dfs = [df_app_main,df_bureau,df_credit_balance,df_prev_apps,df_cash_bal,df_payments]
        
        df_joint = join_dfs(df_app_main, list_dfs)
        
        application_id = df_joint['SK_ID_CURR'].values.copy()
        X = df_joint.drop(['TARGET',"SK_ID_CURR"],axis=1) 
        y = df_joint['TARGET'].values
        
        preds = model.predict_proba(X)[:,1]
        #outcome = ["Deny" if x>.75 else "Accept" for x in preds]
        outcome=[]
        for pred in preds:
            if pred<=0.40:
                outcome.append("Accept")
            elif (pred>0.40) and (pred<0.65):
                outcome.append("Loan Officer Follow Up")
            else:
                outcome.append("Reject")

        list_rows = []  
       # for id_,pred,decision in zip(application_id,preds,outcome):
        #    list_rows.append(dict(ID=id_,prediction=pred,outcome=decision))
         #   writer.write_row(id_,pred,decision) 
       # table = ItemTable(list_rows)
        
        #if request.form.get('download'):
        with open("model_predictions.csv",mode='w') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow(['Application ID','Score','Outcome'])
            for id_,pred,decision in zip(application_id,preds,outcome):
                list_rows.append(dict(ID=id_,prediction=pred,outcome=decision))
                writer.writerow([id_,pred,decision]) 
        table = ItemTable(list_rows) 
        return flask.render_template("show_preds.html",
                                     table=table,
                                    )  
@app.route("/download_predictions",methods=["POST","GET"])
def download_csv():
    path="/Users/sethweiland/Documents/metis_projects/flask_home_app/model_predictions.csv"
    return send_file(path,as_attachment=True)
                                
if __name__ == '__main__':
    app.run(port=5000,debug=True)  
    

