import pyts.datasets as dt
import matplotlib.pyplot as plt
from pyts.image import GramianAngularField, MarkovTransitionField, RecurrencePlot

x_train, x_test, y_train, y_test  = dt.load_coffee(return_X_y=True)

gaf = GramianAngularField()
X_1gaf = gaf.fit_transform(x_train)

mtf = MarkovTransitionField()
X_1mtf = mtf.fit_transform(x_train)

rp = RecurrencePlot()
X_1rp = rp.fit_transform(x_train)

plt.plot(x_train[0])
plt.savefig("Coffee0traingrafico.pdf")

plt.clf()
plt.imshow(X_1gaf[0], cmap='rainbow', origin='lower', vmin=-1., vmax=1.)
plt.savefig("Coffee0traingaf.pdf")

plt.clf()
plt.imshow(X_1mtf[0], cmap='rainbow', origin='lower', vmin=-1., vmax=1.)
plt.savefig("Coffee0trainmtf.pdf")

plt.clf()
plt.imshow(X_1rp[0], cmap='rainbow', origin='lower', vmin=-1., vmax=1.)
plt.savefig("Coffee0trainrp.pdf")