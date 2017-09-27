import csv

import pandas as pd

import make_cohort_key
import make_gail_model_breast_risk_score_table
import make_med_history_key
import make_demographics_key
import make_psychosocial_key


def convert_to_categorical(df, keep_vars, **kwargs):
    keep_columns = []

    if 'cat_only' in kwargs.keys() and kwargs['cat_only']:
        df = df.loc[:, df.columns.isin(keep_vars)]
        keep_columns = list(df.columns.values)
    else:
        keep_columns = _get_column_overlap(df, keep_vars)

    return pd.get_dummies(df, columns=keep_columns)


def _get_column_overlap(df, cols):
    keep_columns = []
    avail_columns = set(df.columns.values)
    for col in cols:
        if col in avail_columns:
            keep_columns.append(col)

    return keep_columns


def combine_categorical_keys_and_output_list(output_loc):
    domains = ['COHORT INFO',
               'DEMOGRAPHICS',
               'GAIL BREAST SCORE',
               'MED HISTORY',
               'PSYCHOSOCIAL'
               ]

    domain_to_function = {'COHORT INFO': make_cohort_key.get_cat,
                          'DEMOGRAPHICS': make_demographics_key.get_cat,
                          'GAIL BREAST SCORE': make_gail_model_breast_risk_score_table.get_cat,
                          'MED HISTORY': make_med_history_key.get_cat,
                          'PSYCHOSOCIAL': make_psychosocial_key.get_cat
                          }

    return combine_keys_and_output_list(domains, domain_to_function, output_loc)


def combine_continuous_keys_and_output_list(output_loc):
    domains = ['COHORT INFO',
               'DEMOGRAPHICS',
               'GAIL BREAST SCORE',
               'MED HISTORY',
               'PSYCHOSOCIAL'
               ]

    domain_to_function = {'COHORT INFO': make_cohort_key.get_cont,
                          'DEMOGRAPHICS': make_demographics_key.get_cont,
                          'GAIL BREAST SCORE': make_gail_model_breast_risk_score_table.get_cont,
                          'MED HISTORY': make_med_history_key.get_cont,
                          'PSYCHOSOCIAL': make_psychosocial_key.get_cont
                          }

    return combine_keys_and_output_list(domains, domain_to_function, output_loc)


def combine_keys_and_output_list(domains, domain_to_func_dict, output_loc):
    cat_keys = []
    variables_to_domains = {}
    for domain in domains:
        f = domain_to_func_dict[domain]
        cat_keys.extend(f())
        variables_to_domains.update(_make_list_of_vars_to_domain(f(), domain).items())

    cat_keys = set(cat_keys)

    _print_dict_to_file(variables_to_domains, output_loc)

    return cat_keys


def _make_list_of_vars_to_domain(var_list, domain):
    domain_var_dict = {var : domain for var in var_list}

    return domain_var_dict


def _print_dict_to_file(dictionary, output_loc):
    with open(output_loc, 'w') as fp:
        fieldnames = ['Variable Name', 'Origin Dataset']
        writer = csv.DictWriter(fp, fieldnames)

        writer.writeheader()
        for key, val in dictionary.items():
            writer.writerow({fieldnames[0] : key, fieldnames[1] : val})


