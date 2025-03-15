# Importer nødvendige biblioteker
import pandas as pd  # Pandas bruges til at håndtere og manipulere data i tabelformat (DataFrame)
from sklearn.datasets import fetch_california_housing  # Henter California Housing dataset fra scikit-learn
from sklearn.model_selection import train_test_split  # Funktion til at opdele data i trænings- og testdatasæt
from sklearn.preprocessing import StandardScaler  # Standardiserer data for at sikre, at alle funktioner har samme skala
import tensorflow as tf  # TensorFlow biblioteket til at bygge og træne neurale netværk

# Hent California Housing dataset
data = fetch_california_housing()  # Henter California Housing dataset fra scikit-learn
df = pd.DataFrame(data.data, columns=data.feature_names)  # Opretter en DataFrame med de relevante kolonnenavne
df["MedHouseVal"] = data.target  # Tilføjer målvariablen (huspriser) som en ny kolonne i DataFrame

# Opdel data i features (X) og target (y)
X = df.drop(columns=["MedHouseVal"])  # Features (X) består af alle kolonner undtagen målvariablen
y = df["MedHouseVal"]  # Target (y) er målvariablen (huspriser)

# Del data i trænings- og testdata (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  
# Data opdeles i 80% træningsdata og 20% testdata for at kunne evaluere modelens præstation

# Normaliser inputdata for bedre træning af modellen
scaler = StandardScaler()  # Initialiser StandardScaler til at standardisere dataene
X_train = scaler.fit_transform(X_train)  # Beregn og anvend skaleringsparametre på træningsdataene
X_test = scaler.transform(X_test)  # Brug de samme skaleringsparametre på testdataene (uden fitting)

# Konverter y-data til en TensorFlow-kompatibel datatype (float32)
y_train = y_train.astype("float32")  # Konverterer target (y) til float32 for at være kompatibel med TensorFlow
y_test = y_test.astype("float32")  # Samme konvertering for testdataene

# Udskriv bekræftelse for at sikre, at data er delt og normaliseret korrekt
print("Data er delt og normaliseret til brug i Keras.")  # Bekræftelse af, at forberedelsen af data er færdig