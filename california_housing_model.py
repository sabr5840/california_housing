# Importer nødvendige biblioteker
import tensorflow as tf  # TensorFlow biblioteket, bruges til at bygge og træne modeller
from tensorflow import keras  # Keras API i TensorFlow til modelbygning og træning
import keras_tuner as kt  # Keras Tuner, bruges til at optimere hyperparametre
import pandas as pd  # Pandas til databehandling og oprettelse af DataFrame
from sklearn.datasets import fetch_california_housing  # Import af California Housing dataset fra scikit-learn
from sklearn.model_selection import train_test_split  # Funktion til at opdele data i trænings- og testdatasæt
from sklearn.preprocessing import StandardScaler  # Standardiserer data for at sikre, at funktioner har samme skala

# Definer layers korrekt for nem adgang
layers = keras.layers  # Keras layers bruges til at definere netværkets lag (f.eks. Dense, Dropout)

# Hent California Housing dataset
data = fetch_california_housing()  # Henter California Housing dataset fra scikit-learn
df = pd.DataFrame(data.data, columns=data.feature_names)  # Opretter en DataFrame med funktionerne som kolonnenavne
df["MedHouseVal"] = data.target  # Tilføjer målvariablen (huspriser) som en ekstra kolonne

# Opdel data i features (X) og target (y)
X = df.drop(columns=["MedHouseVal"])  # Fjern målvariablen og gem resten som features (X)
y = df["MedHouseVal"]  # Målvariablen (huspriser) gemmes som target (y)

# Del data i trænings- og testdata (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  
# Data opdeles i 80% træningsdata og 20% testdata for at evaluere modelens præstation

# Normaliser inputdata for bedre modeltræning
scaler = StandardScaler()  # Initialiser StandardScaler til at standardisere dataene
X_train = scaler.fit_transform(X_train)  # Beregn skaleringsparametre og anvend dem på træningsdataene
X_test = scaler.transform(X_test)  # Brug de samme skaleringsparametre på testdataene (ingen fitting)

# Konverter y-data til float32 (TensorFlow anbefaling)
y_train = y_train.astype("float32")  # Konverter target (y) til float32 for kompatibilitet med TensorFlow
y_test = y_test.astype("float32")  # Samme konvertering for testdataene

# Funktion til at bygge en model, som bruges til hyperparameter tuning
def build_model(hp):
    model = keras.Sequential([  # Initialiser en sekventiel model
        keras.Input(shape=(X_train.shape[1],)),  # Input lag defineres med shape baseret på antallet af features
        layers.Dense(hp.Int("units_1", min_value=64, max_value=256, step=32), activation="relu"),  # Første Dense lag med justerbart antal enheder (units)
        layers.Dense(hp.Int("units_2", min_value=32, max_value=128, step=32), activation="relu"),  # Andet Dense lag med justerbart antal enheder
        layers.Dense(1, activation="linear")  # Output lag med én enhed, aktiveringsfunktion linear for regressionsopgave
    ])
    
    # Kompiler modellen
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=hp.Choice("learning_rate", [0.001, 0.01, 0.1])),  # Optimeringsalgoritme og læringsrate som kan justeres
        loss="mse",  # Mean Squared Error som tab for regression
        metrics=["mae"]  # Mean Absolute Error som ekstra metrik til evaluering
    )
    
    return model  # Returner den byggede model

# Definer tuner til at optimere hyperparametre
tuner = kt.RandomSearch(  # Brug RandomSearch fra Keras Tuner til hyperparameter tuning
    build_model,  # Funktion til at bygge model
    objective="val_loss",  # Mål at optimere for er val_loss (valideringstab)
    max_trials=5,  # Maksimalt antal forsøg med forskellige hyperparametre
    executions_per_trial=1,  # Antal træninger pr. forsøg (for at få en bedre estimat)
    directory="tuning",  # Mappen hvor resultaterne gemmes
    project_name="california_housing"  # Projektets navn for organiserede filer
)

# Callbacks for tidlig stopping og model checkpoint
early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)  
# Stop træning tidligt, hvis val_loss ikke forbedres efter 5 epoch, og genopret bedste vægte
model_checkpoint = keras.callbacks.ModelCheckpoint("best_model.keras", save_best_only=True)  
# Gem kun den bedste model (med lavest val_loss) under træning

# Start hyperparameter tuning
tuner.search(X_train, y_train, epochs=50, validation_split=0.2, callbacks=[early_stopping, model_checkpoint])  
# Start tuning af hyperparametre med træningsdata og valideringssplit på 20%

# Hent den bedste model baseret på resultaterne fra hyperparameter tuning
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]  # Hent de bedste hyperparametre
model = tuner.hypermodel.build(best_hps)  # Byg modellen med de bedste hyperparametre

# Træn den bedste model med hele træningssættet
history = model.fit(
    X_train, y_train,  # Træn modellen med træningsdata
    epochs=100,  # Træn i 100 epoker
    validation_data=(X_test, y_test),  # Brug testdata til validering under træning
    callbacks=[early_stopping, model_checkpoint]  # Brug callbacks til tidlig stopping og model checkpoint
)

# Evaluer modellen på testdataene
loss, mae = model.evaluate(X_test, y_test)  # Evaluer modellen på testdataene for at få loss og mae
print(f"Test MAE: {mae:.4f}")  # Udskriv testens Mean Absolute Error (MAE) som resultat
