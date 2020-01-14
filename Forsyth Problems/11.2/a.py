import pandas as pd
import numpy as np


df = pd.read_csv('student-mat.csv', sep=';')
df.drop(['absences', 'G1', 'G2'], axis=1, inplace=True)

FEATURES = df.columns

# returns P(class = c) for each c
def priors(train):
    return {
        -1 : len(train[train['G3'] <= 12]) / len(train),
        1 : len(train[train['G3'] > 12]) / len(train)
    }


# returns P(feature = val | class = c) with Laplace smoothing
def likelihood(val, feature, c, train, SMOOTH_CONST=0.1):
    if c == -1:
        col = train[feature][train['G3'] <= 12]
    else:
        col = train[feature][train['G3'] > 12]

    try:
        count = col.value_counts()[val]
    except KeyError:
        count = 0

    return (count + SMOOTH_CONST) / (len(col) + 2*SMOOTH_CONST)


# predicts whether student['G3'] <= 12 or > 12
# returns -1 or 1 respectively
def classify(student, train):
    prediction = -1
    max_p = float('-inf')

    for c in [-1, 1]:
        p = sum([np.log(likelihood(student[feature], feature, c, train)) for feature in FEATURES])
        p += np.log(priors(train)[c])
        
        if p > max_p:
            max_p = p
            prediction = c

    return prediction


NUM_FOLDS = 10
TEST_RATIO = 0.15
correct_ratios = []

for i in range(NUM_FOLDS):
    msk = np.random.rand(len(df)) < TEST_RATIO
    trainset = df[~msk]
    testset = df[msk]

    correct_count = 0

    for idx, student in testset.iterrows():
        if classify(student, trainset) == (-1 if student['G3'] <= 12 else 1):
            correct_count += 1

    correct_ratio = correct_count / len(testset)
    correct_ratios.append(correct_ratio)

    # print('Fold ', i, ' complete. Correct ratio: ', correct_ratio)

print('mean: ', np.mean(correct_ratios))
print('std: ', np.std(correct_ratios))

# out:
# mean: 0.9886703642313741
# std: 0.009925734528849355