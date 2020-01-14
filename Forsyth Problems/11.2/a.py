import pandas as pd
import numpy as np


df = pd.read_csv('student-mat.csv', sep=';')
df.drop(['absences', 'G1', 'G2'], axis=1, inplace=True)

# stores P(class = c) for each c
prior = {
    -1 : df[df['G3'] <= 12].shape[0] / df.shape[0],
    1 : df[df['G3'] > 12].shape[0] / df.shape[0]
}


# returns P(feature = val | class = c) with Laplace smoothing
def likelihood(val, feature, c, SMOOTH_CONST=0.1):
    if c == -1:
        col = df[feature][df['G3'] <= 12]
    else:
        col = df[feature][df['G3'] > 12]

    try:
        count = col.value_counts()[val]
    except KeyError:
        count = 0

    return (count + SMOOTH_CONST) / (col.shape[0] + 2*SMOOTH_CONST)


# predicts whether student['G3'] <= 12 or > 12
# returns -1 or 1 respectively
def classify(student):
    prediction = -1
    max_p = float('-inf')

    for c in [-1, 1]:
        p = sum([np.log(likelihood(student[feature], feature, c)) for feature in df.columns])
        p += np.log(prior[c])
        
        if p > max_p:
            max_p = p
            prediction = c

    return prediction


student = df.iloc[3]
print(classify(student), -1 if student['G3'] <= 12 else 1)

# out:
# 1 1