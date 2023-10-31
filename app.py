# Importing necessary files/libraries
import pickle
import numpy as np
from flask import request, Flask, jsonify, render_template

# Importing all the files
lr = pickle.load(open("C:/Users/suman/OneDrive/Desktop/Christ/ML/linear_regressor.pickle", "rb"))
brand_le = pickle.load(open("C:/Users/suman/OneDrive/Desktop/Christ/ML/brand_le.pickle", "rb"))
gas_type_le = pickle.load(open("C:/Users/suman/OneDrive/Desktop/Christ/ML/gasType_le.pickle", "rb"))
dealing_ohe = pickle.load(open("C:/Users/suman/OneDrive/Desktop/Christ/ML/one_hot_encoder_dealing.pickle", "rb"))
gear_ohe = pickle.load(open("C:/Users/suman/OneDrive/Desktop/Christ/ML/one_hot_encoder_gear.pickle", "rb"))
owner_le = pickle.load(open("C:/Users/suman/OneDrive/Desktop/Christ/ML/owners_le.pickle", "rb"))
scaler_one = pickle.load(open("C:/Users/suman/OneDrive/Desktop/Christ/ML/scaler_one.pickle", "rb"))
scaler_two = pickle.load(open("C:/Users/suman/OneDrive/Desktop/Christ/ML/scaler_two.pickle", "rb"))

# Instantiating a flask server

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictValue', methods = ['POST'])
def predictvalue():
        
    age_of_car = int(request.form['age-of-car'])
    gas_type = str(request.form['gas-type'])
    number_of_owners = str(request.form['number-of-owners']) 
    dealing_type = str(request.form['dealing-type'])
    gear = str(request.form['gear']) 
    
    # encoding the value of gas type
    gas_type = gas_type_le.transform([gas_type])  
    gas_type = gas_type[0]  
    
    # encoding the value of number of owners
    number_of_owners = owner_le.transform([number_of_owners])
    number_of_owners = number_of_owners[0]
    
    # setting up the value of broker and direct_owner features
    broker = 0
    direct_owner = 0
    
    if dealing_type=='Broker':
        broker = 1
    elif dealing_type=="Direct Owner":
        direct_owner = 1
        
    # setting up the value of manual and automatic features
    manual = 0
    automatic = 0
        
    if gear == "Manual":
        manual=1
    elif gear == "Autmatic":
        automatic = 1 
            
    # Prediction
    y_pred = lr.predict([[age_of_car, gas_type, number_of_owners, broker, direct_owner, manual, automatic]])
    
    # returning the final output to the HTML page 
    # return jsonify({'price_button': y_pred[0]})
    
    return render_template('index.html', y_pred=y_pred[0])

if __name__ == '__main__':
    app.run(debug=True)