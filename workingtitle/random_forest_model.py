import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier

from sklearn import metrics
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import Imputer
from sklearn.naive_bayes import MultinomialNB, GaussianNB

from utilities.training_functions import create_holdout_set
from utilities.data_quality_functions import drop_pct_missing, output_metrics
from utilities.data_type_functions import convert_to_categorical

'''
A quick script to combine ptid to breast cancer outcome
'''
import dataframe_construction

lidc_data = dataframe_construction.main()

target = np.array(lidc_data['malignancy'])
#write lambda function here to Assign Malignancy to Buckets: {1,2} ->0, {4,5} -> 1

observations = lidc_data.drop(['malignancy'], axis=1)


'''
The below will ONLY give you 75% of total data to use for model training/testing. The other 25% is unavailable
'''
observations, target, _, _, = create_holdout_set(observations, target, .75, seed=42)

'''
Categorical keys are previously called out
Continuous keys are the inversion of that.
'''

CATEGORICAL_KEYS = ['calcification',
                    'internalStructure',
                    ]
CONTINUOUS_KEYS = ['subtlety',
                   'lobulation',
                   'margin',
                   'sphericity',
                   'spiculation',
                   'texture'
                   ]

continuous_vars = observations.loc[:, observations.columns.isin(list(CONTINUOUS_KEYS))]
categorical_vars = observations.loc[:, observations.columns.isin(list(CATEGORICAL_KEYS))]
categorical_vars_imputed = convert_to_categorical(observations, CATEGORICAL_KEYS, cat_only=True, key_var='ID')

#combine continuous and categorical variables

#add commented-out Random Forest Classifier Code Below Here

#Fit a gaussian NB on continuous vars
cont_imputer = Imputer(strategy='mean', axis=1, copy=False)
imputed_continuous_vars = cont_imputer.fit_transform(continuous_vars)
gauss_nb = GaussianNB()
continuous_predictions = cross_val_predict(gauss_nb, imputed_continuous_vars, target, cv=10)
output_metrics("Continuous NB", target, continuous_predictions)

#Fit multinomial NB on categorical vars

mult_nb = MultinomialNB()
categorical_predictions = cross_val_predict(mult_nb, categorical_vars, target, cv=10)
output_metrics("Categorical NB", target, categorical_predictions)


'''
clf = RandomForestClassifier(max_depth=2, random_state=0)

RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=2, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
            oob_score=False, random_state=0, verbose=0, warm_start=False)

rf_predictions = cross_val_predict(clf, observations_after_expansion, target, cv=10)
output_metrics("Random Forest Estimator", target, rf_predictions)

'''