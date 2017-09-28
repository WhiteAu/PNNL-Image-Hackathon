import csv

import pandas as pd



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


