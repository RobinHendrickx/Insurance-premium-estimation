from scipy.stats import ks_2samp
import numpy as np
np.random.seed(12345678)
x = np.random.normal(0, 1, 10000)
y = np.random.normal(0, 1, 1000)
z = np.random.normal(300, 20, 1000)

print(ks_2samp(x, y))
print(ks_2samp(x, z))

# http://sparky.rice.edu/astr360/kstest.pdf

"""
if statistic is above threshold stated in there and p-value is very low
then you can be pretty sure taht the two distributions are from different 
populations
"""