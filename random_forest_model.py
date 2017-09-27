import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier

from sklearn import metrics
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import Imputer
from sklearn.naive_bayes import MultinomialNB, GaussianNB

from utilities.training_functions import create_holdout_set
from utilities.data_quality_functions import drop_pct_missing, output_metrics
from utilities.data_type_functions import convert_to_categorical, combine_categorical_keys_and_output_list, combine_continuous_keys_and_output_list

'''
A quick script to combine ptid to breast cancer outcome
'''

config = configparser.ConfigParser()
configFilePath = 'config'
config.read(configFilePath)

usr_config = configparser.ConfigParser()
usrConfigFilePath = 'user_config'
usr_config.read(usrConfigFilePath)

baseline_key = pd.read_csv(os.path.join(usr_config['OUTPUT']['path'], config['KEY DATA']['baseline_key']), sep='\t')
baseline_key = drop_pct_missing(baseline_key, .3, key_col='ID')
print(baseline_key.columns)
breast_cancer_key = pd.read_csv(os.path.join(usr_config['OUTPUT']['path'],
                                             config['KEY DATA']['breast_cancer_key']),
                                sep='\t')
target = np.array(breast_cancer_key['BREAST_CANCER'])
observations = baseline_key.drop(['ID'], axis=1)

observations, target, _, _, = create_holdout_set(observations, target, .75, seed=config.getint('TRAINING VALUES', 'seed'))

'''
Categorical keys are previously called out
Continuous keys are the inversion of that.
'''
cat_keys = combine_categorical_keys_and_output_list(os.path.join(usr_config['OUTPUT']['path'],
                                                                 config['METADATA']['categorical_manifest']))

cont_keys = combine_continuous_keys_and_output_list(os.path.join(usr_config['OUTPUT']['path'],
                                                                 config['METADATA']['continuous_manifest']))

continuous_vars = observations.loc[:, observations.columns.isin(list(cont_keys))]

#Fit a gaussian NB on continuous vars
cont_imputer = Imputer(strategy='mean', axis=1, copy=False)
imputed_continuous_vars = cont_imputer.fit_transform(continuous_vars)
gauss_nb = GaussianNB()
continuous_predictions = cross_val_predict(gauss_nb, imputed_continuous_vars, target, cv=10)
output_metrics("Continuous NB", target, continuous_predictions)

#Fit multinomial NB on categorical vars
categorical_vars = convert_to_categorical(observations, cat_keys, cat_only=True, key_var='ID')
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
'''