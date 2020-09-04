# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 01:33:04 2020

@author: Shaila Sarker
"""

import pandas as pd
car = pd.read_csv("D:/DS/2. CardiacDisease & CarMilage using SLR-MLR/Cars.csv")

car.describe()

# Multiple Linear Regression or MLR [as output, Y = MPG is continuous data and we've multiple inputs X1 = Waist]
import statsmodels.formula.api as smf

model = smf.ols('MPG ~ HP + VOL + SP + WT', data = car).fit() #ols = ordinary least square | Y ~ X1 + X2 + X3 + X4
model.summary()

#acquired model: MPG = 30.6773 - 0.2054*HP - 0.3361*VOL + 0.3956*SP + 0.4006*WT 
# i.e. Y = b0 + b1X1 + b2X2 + b3X3 + b4X4 