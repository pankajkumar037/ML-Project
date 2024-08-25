import pickle #for deseralising model
import pandas as pd
from flask import Flask,render_template,request

#creating object of falsk
app=Flask(__name__)
model=pickle.load(open('model.pkl','rb'))


@app.route('/') #means is route pe aaane se niche wal function call hoga
def index():
    return render_template('index.html')


@app.route('/predict',methods=['GET','POST'])
def predict():
    a=request.form.get('SepalLengthCm')
    b=request.form.get('SepalWidthCm')
    c=request.form.get('PetalLengthCm')
    d=request.form.get('PetalWidthCm')
    input_data = pd.DataFrame([[a,b,c,d]],
                                  columns=['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm'])
   # input_data.columns = [''] * len(input_data.columns)
    prediction = model.predict(input_data)[0]
    return render_template('index.html',prediction_text=f"The Flower Predicted is {prediction}")




#for running software
if __name__=='__main__':
    app.run(debug=True)

