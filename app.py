# import Flask class from the flask module
from flask import Flask, request, render_template
import joblib
from joblib import load
import numpy as np

# Create Flask object to run
app = Flask(__name__)

# Load the model from the file
iris_model = joblib.load('model2/iris_model.pkl')

@app.route('/', methods=['GET','POST'])
def predict():
    # Get values from browser
    request_type_str = request.method
    if request_type_str == 'GET':
    	return render_template('index.html', href='static/iris.jpeg')
    else:
        sl = request.form['Sepal_length']
        sw = request.form['Sepal_width']
        pl = request.form['Petal_length']
        pw = request.form['Petal_width']

        flower_dim = []

        for i in [sl,sw,pl,pw]:
            try:
                flower_dim.append(float(i))
            except:
                return "Please enter the valid dimensions"


        iris_model = load('model2/iris_model.pkl')

        test_inp = np.array(flower_dim).reshape(1, 4)
        class_predicted = int(iris_model.predict(test_inp)[0])
        if class_predicted==0:
            name = 'Setosa'
        elif class_predicted==1:
            name = 'Versicolor'
        else:
            name = 'Virginica'

        output_pic = "static/" + name + '.jpeg'

        return render_template('out_pic.html', href=output_pic)


if __name__ == "__main__":
    app.run(debug=True)
