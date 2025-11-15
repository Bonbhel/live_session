from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import pickle
import numpy as np
import io

# Charger le modèle
with open('modelChurnPredict.pkl', 'rb') as f:
    model = pickle.load(f)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predictCSV/")
async def predict_csv(file: UploadFile = File(...)):
    # Lire le fichier CSV
    #df = pd.read_csv(await file.read())

    #ATTENTION AU SEPARATEUR DU CSV
    df = pd.read_csv(io.BytesIO(await file.read()), sep=';')

    print('****AVANT**********', df.shape)

    localite = df['Point de vente'].copy().values
    
    # ENCODAGE INTELLIGENT avec la fonction Dummies
    df_encoded = df.apply(lambda x: object_to_int(x))
    
    predictions = []

    print('****APRES**********', df.shape)
    #On parcours chaque ligne du tableau Excel et on fait la prediction.
    for index, row in df_encoded.iterrows():
        # Supposer que toutes les colonnes sont des features
        # Je transforme chaque ligne en Matrice 2D pour la prediction : 

        #Décomposition de reshape(1, -1) :
        # 1 : première dimension → 1 ligne
        # -1 : deuxième dimension → "calcule automatiquement le nombre de colonnes"
        # Avant reshape : [25, 30000, 2]  (shape: (3,))
        # Après reshape : [[25, 30000, 2]] (shape: (1, 3))
        
        features = row.values.reshape(1, -1)

        #La methode predict retourne toujours un tableau numpay contenant la prediction (array([0])). 
        #D'ou la presence du [0] pour récupérer notre prédilection qui est dans le premier élément du tableau 
        prediction = model.predict(features)[0]

        #Calcul de la confidence (le poucentage de chance pour chaque ligne)
        confidence = np.max(model.predict_proba(features))

        #Chaque resutat est stocké dans une liste sous forme de dico : 
        #{Un index, la predictiion, et le Pouecentage}
        predictions.append({
            "id": index + 1,
            "prediction": int(prediction),
            "confidence": float(confidence),
            "localite": localite[index]
        })
    
    return {"predictions": predictions}

"""
#Encodage des valeurs NOMINALES avec la fonction Dummies (recommandé)
#La fonction LabelEncoder est recommandée pour des valeurs ORDINALES ayant clairement une hierarchie (Ex : Petit, Moyen, Grand)
def encode_dataframe(df):
    df_encoded = df.copy()
    
    # Colonnes nominales -> One-Hot Encoding
    nominal_cols = [
        'gender', 'Partner', 'PhoneService', 'MultipleLines', 
        'InternetService', 'TechSupport', 'StreamingTV', 'StreamingMovies',
        'Contract', 'PaymentMethod', 'Point de vente'
    ]
    
    # Vérifier que les colonnes existent
    nominal_cols = [col for col in nominal_cols if col in df_encoded.columns]
    
    # One-Hot Encoding
    df_encoded = pd.get_dummies(df_encoded, columns=nominal_cols, prefix=nominal_cols)
    
    return df_encoded
"""

#Encodeur utilise lors de la prédiction
def object_to_int(df):
    #df_encoded = df.copy()
    if df.dtype=='object':
        df = LabelEncoder().fit_transform(df)
    return df


"""
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8082)
"""