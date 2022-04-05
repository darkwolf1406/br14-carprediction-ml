from flask import Flask,render_template,request,redirect
from flask_cors import CORS,cross_origin
import pickle
import pandas as pd
import numpy as np

app=Flask(__name__)
cors=CORS(app)
model=pickle.load(open('RandomForestRegressor.pkl','rb'))
ds_car=pd.read_csv('Cleaneddata.csv')

@app.route('/',methods=['GET','POST'])
def index():
    
    cmodels = sorted(ds_car['Name'].unique())
    year = sorted(ds_car['Year'].unique(), reverse=True)
    fuel = sorted(ds_car['Fuel'].unique())
    kms = sorted(ds_car['KMS'].unique())
    engine = sorted(ds_car['Engine'].unique())
    owners = sorted(ds_car['Owners'].unique())
    transmission = sorted(ds_car['Transmission'].unique())
    mileage = sorted(ds_car['Mileage'].unique())
    seats = sorted(ds_car['Seats'].unique())
    ccompanies = sorted(ds_car['Company'].unique())

    ccompanies.insert(0,'Select Company')
    return render_template('main.html', cmodels=cmodels, years=year, fuels=fuel,kms=kms,engine=engine,
                           owners=owners,transmissions=transmission,mileage=mileage,seats=seats,ccompanies=ccompanies)

@app.route('/predict',methods=['POST'])
@cross_origin()
def predictPrice():

    company = request.form.get('company')
    carmodel = request.form.get('carmodel')
    year = int(request.form.get('year'))
    owner = int(request.form.get('owner'))
    seat = int(request.form.get('seat'))
    fuel_type = request.form.get('fuel_type')
    transmission = request.form.get('transmission')
    kms = int(request.form.get('kms'))
    engine = int(request.form.get('engine'))
    mileage = float(request.form.get('mileage'))
    print(company,carmodel,year,owner,seat,fuel_type,transmission,kms,engine,mileage)

    predict = model.predict(pd.DataFrame([[carmodel,year,fuel_type,kms,engine,owner,transmission,mileage,seat,company]], 
                          columns=['Name','Year','Fuel','KMS','Engine','Owners','Transmission','Mileage','Seats','Company']))
    
    return str(np.round(predict[0], 2))


if __name__=='__main__':
    app.run(debug=True)