# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 20:55:24 2020

@author: Shaila Sarker
"""

import pandas as pd
wcat = pd.read_csv("D:/DS/2. CardiacDisease & CarMilage using SLR-MLR/wc-at.csv")

wcat.describe()

#Graphical Representation
import matplotlib.pyplot as plt

# Scatter plot
plt.scatter(x=wcat.Waist, y=wcat.AT, color='green') #finds correlation between 2 variables [bi-variate]

#after finding the positive correlation, now it's time to check the strength
#Correlation Coefficient, r
# if 0.4 < |r| < 0.85 then Moderate, if |r| > 0.85 then Strong, if |r| < 0.4 then Weak 
import numpy as np
np.corrcoef(wcat.Waist, wcat.AT) # |r| = 0.82, so moderate positive correlation


# Simple Linear Regression or SLR [as output, Y = AT is continuous data and we've single input X = Waist]
import statsmodels.formula.api as smf

model = smf.ols('AT ~ Waist', data = wcat).fit() # ordinary least square | 'Y ~ X'  
model.summary()

#acquired model: AT = -215.9815 + 3.4589*Waist
# i.e. Y = B0 + B1X 
