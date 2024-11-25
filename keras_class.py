import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from keras import layers

def sepfunction(x):
	return x*x*x
def generate_classified_data(sz):
    XY = np.random.rand(sz, 2)
    Ans = np.apply_along_axis(classifier, 1, XY)
    return (XY, Ans)
def classifier(X):
    f = sepfunction(X[0])
    if (f - X[1] > 0):
        return 1
    return 0
def sort_points(Points, Point_classes):
    # РљР»Р°СЃСЃРёС„РёС†РёСЂСѓРµРј РґР°РЅРЅС‹Рµ РІСЂСѓС‡РЅСѓСЋ
    ai = np.where(Point_classes > .5)
    bi = np.where(Point_classes <= .5)

    Ax = Points.T[0][ai]
    Ay = Points.T[1][ai]
    Bx = Points.T[0][bi]
    By = Points.T[1][bi]
    return (Ax, Ay, Bx, By)

(XY_train, C_train) = generate_classified_data(100)
(XY_test, C_test) = generate_classified_data(100)
X_sepline = np.linspace(0, 1, 200)
Y_sepline = np.apply_along_axis(sepfunction, 0, X_sepline)


model = keras.Sequential()
model.add(keras.Input(shape=(2,)))
model.add(layers.Dense(units=30, activation=keras.activations.sigmoid))
model.add(layers.Dense(units=30, activation=keras.activations.sigmoid))
model.add(layers.Dense(units=1, activation=keras.activations.linear))

model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(0.3))

history = model.fit(XY_train, C_train, epochs=1000, verbose=False)
epoch = history.epoch[-1]
print(f"РћС€РёР±РєР° РЅР° РІС‹Р±РѕСЂРєРµ {history.history['loss'][epoch]:.13}.")

nw_res = model.predict(XY_test)

plt.subplot(1, 3, 1)
(Ax, Ay, Bx, By) = sort_points(XY_train, C_train)
plt.scatter(Ax, Ay, label='A')
plt.scatter(Bx, By, label='B')
plt.plot(X_sepline, Y_sepline)  # draw separation line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.title("РћР±СѓР°СЋС‰Р°СЏ РІС‹Р±РѕСЂРєР°")


plt.subplot(1, 3, 2)
(Ax, Ay, Bx, By) = sort_points(XY_test, nw_res.T[0])
plt.subplot(1, 3, 2)
plt.scatter(Ax, Ay, label='A')
plt.scatter(Bx, By, label='B')
plt.plot(X_sepline, Y_sepline)  # draw separation line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.title("РўРµСЃС‚РѕРІР°СЏ РІС‹Р±РѕСЂРєР°")

plt.subplot(1, 3, 3)
# РѕС€РёР±РєРё РѕС‚СЂРёСЃРѕРІС‹РІР°РµРј
plt.plot(history.history['loss'])
plt.title("РР·РјРµРЅРµРЅРёРµ РѕС€РёР±РєРё")
plt.xlim(0, epoch)
plt.show()