import pandas as pd
df = pd.read_excel('/content/AYF_experiments (1).xlsx')
df.drop(columns=['Exp','Formula','γ C2S','C12A7','C2AS','C$'],inplace = True)

targets = ['β C2S',"α' C2S",'C3S','C3A','C4A3$','C4AF','C']
features = []
for col in df.columns:
  if col not in targets:
    features.append(col)

X = df[features]
y = df[targets]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state=42)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU, PReLU,Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau
from tensorflow.keras.losses import Huber
import tensorflow.keras.backend as K
import tensorflow as tf

model = Sequential([
    Dense(40,activation = 'relu',  input_shape=(X_train.shape[1],)),
    # Dense(28, activation='relu'),
    Dense(18, activation='relu'),
    Dense(y_train.shape[1],activation='softplus'),

])


def custom(y_true,y_pred):
    weights = tf.constant([1.0,1.0,5.0,1.0,1.0,1.0,1.0])
    sq_diff = tf.square(y_true-y_pred)
    weight_sq = sq_diff*weights
    return tf.reduce_mean(weight_sq)



model.compile(optimizer=Adam(learning_rate = 0.05), loss=custom, metrics=['mse'])
lr_schedule = ReduceLROnPlateau(monitor='val_loss', patience=10, factor=0.5, verbose=1)
early_stop = EarlyStopping(monitor='val_loss',patience=40,restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    epochs=200,
    batch_size=2,
    validation_data = (X_test,y_test),
   callbacks=[early_stop,lr_schedule],
    verbose=1
)

y_pred = model.predict(X_test)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

for i, col in enumerate(targets):
    actual = y_test.iloc[:, i]
    prediction = y_pred[:, i]

    r2 = r2_score(actual, prediction)
    mse = mean_squared_error(actual, prediction)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual, prediction)

    print(f"\n{col}:")
    print(f"R² Score: {r2:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")

    plt.scatter(actual,prediction,alpha=0.6,color="purple")
    plt.plot([actual.min(),actual.max()],[actual.min(),actual.max()],"k--")
    plt.xlabel("actual values")
    plt.ylabel("prediction values ")
    plt.title(f"{col} for actual v/s prediction values")
    plt.grid(True)
    plt.show()
