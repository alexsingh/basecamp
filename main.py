__author__ = 'asingh'

import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pylab
import seaborn as sns
from scipy.optimize import curve_fit


def sigmoid(x, k, c):
	x0 = 8;
	a = 1610;
	y = a / (1 + np.exp(-k*(x-x0)))
	return y

# read in data
filename = '/Users/asingh/work/matt_chandra/powerturbinedata/Workbook1.csv'
data = pd.read_csv(filename)
timestamp = pd.to_datetime(data.Timestamp)
data.Timestamp = timestamp

data.rename(columns={'Wind speed(m/s)': 'windspeed', 'Power(kW)': 'power'}, inplace=True)


# g = sns.jointplot(x="windspeed", y="power", data=data, kind="kde")
# g.plot_joint(plt.scatter, c="w", s=30, linewidth=1, marker="+")
# g.ax_joint.collections[0].set_alpha(0)
# g.set_axis_labels("$X$", "$Y$");

g = sns.jointplot(x="windspeed", y="power", data=data, kind="kde")
g.ax_joint.collections[0].set_alpha(0)
g.set_axis_labels("$X$", "$Y$");
plt.savefig("jointpdf.png")
# Model 1:
# fit the sigmoid function
xdata = data['windspeed']
ydata = data['power']

popt, pcov = curve_fit(sigmoid, xdata, ydata)

fit_x = data['windspeed']
fit_y = sigmoid(fit_x, *popt)

plt.clf()
plt.plot(xdata,ydata,'or')
plt.plot(fit_x,fit_y,'.b')
plt.savefig("sigmoid_fit.png")

# overlay prediction based on 
plt.clf()
p1 = plt.plot(ydata,'r')
p2 = plt.plot(fit_y,'b')

plt.legend(['power', 'Predicted Power'])
plt.xlabel('Sample Index')
plt.ylabel('Power')
plt.savefig("prediction_sigmoid_wind.png")

# xdata = np.linspace(0, 4, 50)
# y = func(xdata, 2.5, 1.3, 0.5)
# ydata = y + 0.2 * np.random.normal(size=len(xdata))
# popt, pcov = curve_fit(func, xdata, ydata)
# plt.plot(xdata,func(xdata, *popt))
# plt.plot(xdata,ydata,'or')

# def func(x, a, b, c):
# 	return a * np.exp(-b * x) + c


