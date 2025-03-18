import tensorflow as tf
from tensorflow import keras
import keras_tuner as kt
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import numpy as np

# Definer layers for nem adgang
layers = keras.layers

# Hent California Housing dataset
data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
df["MedHouseVal"] = data.target

# Opdel data i features (X) og target (y)
X = df.drop(columns=["MedHouseVal"])
y = np.log1p(df["MedHouseVal"])  

# Del data i trænings- og testdata
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normaliser inputdata
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

y_train = y_train.astype("float32")
y_test = y_test.astype("float32")

def build_model(hp):
    model = keras.Sequential()
    
    # Tune dropout mellem 0.1 og 0.5
    model.add(layers.Dropout(hp.Float("dropout", 0.1, 0.5, step=0.1)))

    # Tune antal skjulte lag og neuroner pr. lag
    for i in range(hp.Int("num_layers", 2, 4)):  # 2 til 4 skjulte lag
        model.add(layers.Dense(hp.Int(f"units_{i}", min_value=32, max_value=256, step=32), activation="relu"))

    model.add(layers.Dense(1, activation="linear"))  # Output lag

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=hp.Float("learning_rate", 1e-4, 1e-2, sampling="LOG")),
        loss="mse",
        metrics=["mae"]
    )
    
    return model

tuner = kt.RandomSearch(
    build_model,
    objective="val_loss",
    max_trials=150,  # Øget antal trials
    directory="tuning",
    project_name="california_housing"
)

# Callbacks
early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.75, patience=5, min_lr=1e-6)
model_checkpoint = keras.callbacks.ModelCheckpoint("best_model.keras", save_best_only=True)

# Learning Rate Scheduler
def lr_scheduler(epoch, lr):
    if epoch < 5:
        return lr * 1.1  # Øger LR i starten
    elif epoch % 10 == 0:
        return lr * 0.9  # Reducerer LR løbende
    return lr


lr_scheduler_callback = keras.callbacks.LearningRateScheduler(lr_scheduler)

# Start tuning med 50 epochs
tuner.search(X_train, y_train, 
             epochs=50, 
             validation_data=(X_test, y_test), 
             callbacks=[early_stopping, reduce_lr, lr_scheduler_callback])  # Tilføjet scheduler i tuning

# Hent den bedste model
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
model = tuner.hypermodel.build(best_hps)

# Træn den bedste model med 200 epochs
history = model.fit(
    X_train, y_train,
    epochs=200,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping, model_checkpoint, reduce_lr, lr_scheduler_callback]
)

# Evaluer modellen på testdata
loss, mae = model.evaluate(X_test, y_test, verbose=0)

print(f"Test Loss (MSE): {loss:.4f}")
print(f"Test MAE: {mae:.4f}")

import matplotlib.pyplot as plt
import numpy as np

# Forudsig på testdata
y_pred = model.predict(X_test).flatten()

# Beregn residualer
residuals = y_test - y_pred

# Plot residualerne
plt.figure(figsize=(8, 5))
plt.scatter(y_test, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel("Faktiske værdier (y_test)")
plt.ylabel("Residualer (y_test - y_pred)")
plt.title("Residual Plot")
plt.show()

# Beregn R²-score
r2 = r2_score(y_test, y_pred)
print(f"R²-score: {r2:.4f}")

plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)  # Perfekt linje
plt.xlabel("Faktiske værdier (y_test)")
plt.ylabel("Forudsagte værdier (y_pred)")
plt.title("Predicted vs. Actual")
plt.show()
