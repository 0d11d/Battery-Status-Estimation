from flask import Flask, render_template, request, jsonify, Response, stream_with_context

import os
import warnings
warnings.filterwarnings("ignore")
import datetime as datetime
import time 
import math
import numpy as np
import pandas as pd
import json
from pandas.core.common import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
import tensorflow as tf

import re

import sys 
print(sys.version)

# create an instance of Flask

app = Flask(__name__, template_folder='templates')

  
def get_data(mode, weather, mile):
  
    data = pd.read_csv("test.csv", index_col=False)
    data_timestamp = data.set_index("timestamp")
    df_test = data_timestamp
    print(mode)
    print(weather)
    
    #find test data
    cycle = mile//300
    print(cycle)
    t = pd.to_timedelta(cycle, unit='D')
    
    if weather=='normal':
        year = 2006
    if weather == 'hot':
        year = 2039   
    if weather == 'cold':
        year = 2043
    
    start = str(t+datetime.datetime(year, 1, 1, 0, 0, 0)) 
    end = str(t+datetime.datetime(year, 1, 1, 23, 0, 0))
    test_data = df_test.loc[start:end]

    X_test,y_test = process_data(test_data)
    
    return X_test,y_test, test_data


def process_data(df):
    df["capacity"] = df["capacity"].transform(lambda x: x/2)
    df = series_to_supervised(df, 1, 1)
    df = df.drop(["var1(t)","var2(t)","var3(t)","var4(t)","var5(t)"], axis=1)

    df = df.to_numpy()
    X, y = df[:, :-1], df[:, -1]
    X = X.reshape((X.shape[0], 1, X.shape[1]))
    
    return X,y
    


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


def cn(cycle, environment_temperature, df_nominal):
    
    lookup_table = df_nominal
    nominal_capacity = lookup_table.loc[(lookup_table["cycle"]==cycle) 
                                      & (lookup_table["environment_temperature"]==environment_temperature),
                                      "nominal_capacity"].item()
    return nominal_capacity


# conver to time
def convert_time(hour):
    second = hour*60*60
    hour = second //3600
    second %= 3600
    minute = second //60
    second %= 60
    return hour, minute, second


def predict(mode, weather, mile, battery):   
    
    X_test,y_test, test_data_raw = get_data(mode, weather, mile)
       
    # load model and predict
    json_file = open('model.json','r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = tf.keras.models.model_from_json(loaded_model_json)
    loaded_model.load_weights("model.h5")
    loaded_model.compile(loss='mae', optimizer='adam')
    res = loaded_model.predict(X_test)

    df_demo = test_data_raw.tail(-1)
    df_demo["predict_capacity"] = res*2
    df_demo.loc[df_demo["predict_capacity"]<0, "predict_capacity"]=0    

    df_demo.loc[df_demo["predict_capacity"]<0, "predict_capacity"]=0
    first_day = pd.datetime(pd.to_datetime(df_demo.index[0]).year, 1, 1).date()
    cycle = [((pd.to_datetime(df_demo.index[0]).date()-first_day).days+1) for x in range (len(df_demo))]
    df_demo["cycle"] = cycle
    df_demo["capacity"] = df_demo["capacity"].transform(lambda x: x*2)
    df_demo["environment_temperature"] = df_demo["environment_temperature"].transform(lambda x: round((x*(70-2)+2),0))
    df_demo["consumption_per_second"] = df_demo["consumption_per_second"].transform(lambda x: x*(4e-06+0.00113)-0.00113)

    df_nominal = pd.read_csv("nominal capacity table.csv")
    nominal_capacity = cn(df_demo["cycle"][0], df_demo["environment_temperature"][0], df_nominal)
    
    # add nominal capacity, SOC, plugin duration, and remain_mileage

    df_demo["SOC%"] = (df_demo["capacity"]*100/nominal_capacity).round(0)
    df_demo["plugin_duration"] = (nominal_capacity-df_demo["capacity"])/1.5
    df_demo["remain_mileage"] = (df_demo["capacity"]*500/2).round(0)
    
    # Adjust abnormal plugin_duration results
    df_demo.loc[df_demo["plugin_duration"]<0, "plugin_duration"]=0
    
    # convert charge time
    charge_time_1 = []
    for i in range(len(df_demo)):
        h, m, s = convert_time(df_demo["plugin_duration"][i])
        t = datetime. time(int(h), int(m), int(s))
        charge_time_1.append(t)

    df_demo["plugin_duration_realtime"] = charge_time_1
    
    # Adjust abnormal SOC results
    df_demo.loc[df_demo["SOC%"]<0, "SOC%"]=0
    df_demo.loc[df_demo["SOC%"]>100, "SOC%"]=100
    
    df_demo = df_demo.replace({"type": {0: "charge", 1:"discharge"}})
    
    demo1 = df_demo

    demo1 = demo1[["cycle", "type", "environment_temperature", "capacity",
        "predict_capacity", "SOC%", "plugin_duration_realtime","remain_mileage" ]]
    
    
    if mode == 'Parking': 
        demo1 = demo1.loc[(demo1["type"]== "charge") & (demo1["SOC%"] > battery)]  
    else:
        demo1 = demo1.loc[(demo1["type"]== "discharge") & (demo1["SOC%"] < battery)]
    
    prediction = demo1[[ "SOC%", "plugin_duration_realtime","remain_mileage" ]]
    
    return prediction
    
  


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict', methods = ['POST', 'GET'])
def index():
    if request.method == 'POST':
        #get form data
        mode = request.form.get('Mode')
        weather = request.form.get('Weather')
        mile = request.form.get('Miles')
        battery = request.form.get('Battery')
        
        try:
            prediction = predict(mode, weather, int(mile), int(battery))
 
            
            def generator():
            
                yield '\n\n\n\n\n\n\n\nStart!<br><br><br>'
                
                for i in range(len(prediction)):
                    
                    yield "<br>\n\n\n\n{0}%\n\n\n\n{1} remaning\n\n\n\n{2}mi\n\n\n\n <br>".format(
                        prediction.iloc[i, 0], prediction.iloc[i, 1], prediction.iloc[i, 2])
                    time.sleep(1)
                
                yield 'Prediction close\n\n'
                
                
            return Response(generator(), mimetype='text/html')
        
        except ValueError:
            return "Please enter valid values"
        pass
    pass
        

 
    
    
if __name__ == '__main__':
    app.run(debug=True)
