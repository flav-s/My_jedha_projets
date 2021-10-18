import plotly.express as px
import plotly
import json
from flask import Flask,request, render_template
from keras_preprocessing.image.utils import img_to_array
import numpy as np
import pandas
from keras.models import load_model
from PIL import Image

#data = pandas.read_csv('df_webapp.csv', index_col = 0)
data = pandas.read_csv("lastlast.csv",index_col=0)
data.to_html(header="true", table_id="table", index=False)
df = data.style.hide_index()
app = Flask(__name__)
model = load_model('model.h5')

@app.route('/', methods=["GET" , "POST"])
def home():
    return render_template("Accueil1.html", tables=[data.head(5).to_html(classes='data')], titles=data.columns.values)

@app.route('/predict',methods=["GET",'POST'])
def predict():
    imagefile = request.files['imagefile']
    #image_path = "/Users/flavienbert/Documents/Nicepage Templates/projet_final_final/images/image_app" + imagefile.filename
    image_path ="/Users/flavienbert/Documents/My_jedha_projects/creation_app/Deploiement/images/image_app" + imagefile.filename
    imagefile.save(image_path)

    image = Image.open(image_path)
    size = (224,224)
    box = (100,100,500,400)
    image = image.resize(size)
    image = img_to_array(image)
    image = np.expand_dims(image, 0)
    prediction = model.predict(image).tolist()
    liste_des_classes = ["108","208","308","2008","Aygo","3008","Berlingo","C_HR","C3","C4","C4 Cactus","C5 Aircross","Captur","Clio","Corolla","Golf","Megane","POLO","RAV 4","T-ROC",'UP!',"TIGUAN","TWINGO","YARIS","ZOE"]
    liste_des_classes.sort()
    pred_finale = liste_des_classes[np.argmax(prediction[0])] 
    return render_template('Accueil1.html', pred='Le model de votre voiture est une :{}'.format(pred_finale))
#return render_template('Accueil1.html')#, pred='Le model de votre voiture est une :{}'.format(prediction))

@app.route('/recherche',methods=['POST','GET'])
def recherche():
    df = data.copy
    if request.method == 'POST':
        State = request.form['State'].upper()
        Brand = request.form['Brand']
        Model = request.form['Model']
        Gear = request.form['Gear']
        KrM = int(request.form['KrM'])
        Energie = request.form['Energie']
        Year = int(request.form['Year'])
        Price = int(request.form['Price'])
        mask = (data['Brand'] == Brand) & (data['Model'] == Model) &\
            (data['Gear'] == Gear) &\
            (data['KM'] <= KrM) &\
            (data['Energie'] == Energie) & (data['Year'] >= Year) & \
            (data['Price'] <= Price) & (data['State'] == State)
        df = data.loc[mask,:]
        price_avg = df['Price'].mean()
        price_std = df['Price'].std()
        max_interval = price_avg+price_std
        min_interval = price_avg-price_std
        df['Price_recommendation'] = df['Price'].apply(lambda x: 'Expensive' if x > max_interval
                                                else 'Good price' if x < min_interval
                                                else "In market"
                                                )
        fig = px.scatter_mapbox(df,
        lat="latitude",
        lon="longitude",
        color="Price_recommendation",
        zoom=7.0,
        mapbox_style="carto-positron",
        hover_name='Brand',text='State',
        title='Cars recommendation: Brand selection')
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    print('je lai enregistre')
    return render_template('Accueil1.html',tables=[df.head(5).sort_values(by=['Scoring'],ascending=False).to_html(classes='data')], titles=df.columns.values, graphJSON=graphJSON)
    #else:
    #    return render_template('Accueil.html',tables=[df.head(5).to_html(classes='data')], titles=df.columns.values, graphJSON=graphJSON)

if __name__ == "__main__":
    app.run(debug=True)