import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import io
import random
from flask import Response
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure as fig
import matplotlib.pyplot as plt
import pandas as pd
import plotly
import plotly.graph_objs as go
import numpy as np
import json



app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
   
    return render_template('index.html')





@app.route('/predict',methods=['POST'])
def predict():
    feature = 'Bar'
    bar = create_plot()

    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text="Pakistan's Urban Population of the year you have chosen should be {}".format(output), plot=bar)
    
def create_plot():

    N = 6
    x = np.linspace(0, 1, N)
    y = np.random.randn(N)
    df = pd.DataFrame({'x': x, 'y': y}) # creating a sample dataframe


    data = [
        go.Bar(
            x=df['x'], # assign x as the dataframe column 'x'
            y=df['y']
        )
    ]
    

    graphJSON = json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)

    return graphJSON


df = pd.read_csv('PakUrban.csv')









if __name__ == "__main__":
    app.run(debug=True)