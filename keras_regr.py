import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from keras import layers

def sepfunction(XX):
	x = XX[0]
	return x*x*x

X_train = np.zeros([101, 2])

X_train.T[0] = np.linspace(0.0, 1.0, 101)
X_train.T[1] = X_train.T[0] * X_train.T[0]
Y_train = np.apply_along_axis(sepfunction, 1, X_train) + \
	np.random.normal(0.0, 0.10, size=(101))

X_test = np.zeros([100, 2])
X_test.T[0] = np.linspace(0, 1, 100)
Y_test = np.apply_along_axis(sepfunction, 1, X_test)

model = keras.Sequential()
model.add(keras.Input(shape=(1,)))
model.add(layers.Dense(units=30, activation=keras.activations.sigmoid))
model.add(layers.Dense(units=30, activation=keras.activations.sigmoid))
model.add(layers.Dense(units=1, activation=keras.activations.linear))

model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(0.3))

history = model.fit(X_train.T[0], Y_train, epochs=1000, verbose=False)
epoch = history.epoch[-1]
print(f"РћС€РёР±РєР° РЅР° РІС‹Р±РѕСЂРєРµ: {history.history['loss'][epoch]:.13}.")


nw_res = model.predict(X_test.T[0])

plt.subplot(1, 2, 1)
plt.scatter(X_train.T[0], Y_train, label="A")
plt.plot(X_test.T[0], Y_test, label="B") 	# draw separation line
plt.plot(X_test.T[0], nw_res.T[0], label="C")
plt.xlim([-0.2, 1.2])
plt.ylim([-0.2, 1.2])
plt.title("Р РµРіСЂРµСЃСЃРёСЏ")
plt.grid(True)


# РѕС€РёР±РєРё РѕС‚СЂРёСЃРѕРІС‹РІР°РµРј
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.grid(True)
plt.axis('tight')
plt.title("РР·РјРµРЅРµРЅРёРµ РѕС€РёР±РєРё РІ РїСЂРѕС†РµСЃСЃРµ РѕР±СѓС‡РµРЅРёСЏ")
plt.xlim(0, epoch)
plt.show()