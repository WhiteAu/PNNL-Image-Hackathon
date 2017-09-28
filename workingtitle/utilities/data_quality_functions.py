from sklearn import metrics

def drop_pct_missing(df, pct_missing, **kwargs):
    keep_columns = []
    row_ct = df.shape[0]
    for column in df:
        num_missing = df[column].isnull().sum()
        if num_missing / row_ct <= pct_missing:
            keep_columns.append(column)

    if 'key_col' in kwargs.keys() and not kwargs['key_col'] in keep_columns:
        keep_columns.append(kwargs['key_col'])

    return df[keep_columns]


def keep_pct_complete(df, pct_complete, **kwargs):
    return drop_pct_missing(df, (1-pct_complete), **kwargs)


def output_metrics(model_name, y_true, y_pred):
    print("The p,r,f1 breakdown for {0}:\n {1}".format(model_name,
                                                        metrics.classification_report(y_true, y_pred)))

    print("The Matthews Correlation Coefficient for {0}:\n {1}".format(model_name,
                                                                       metrics.matthews_corrcoef(y_true, y_pred)))
