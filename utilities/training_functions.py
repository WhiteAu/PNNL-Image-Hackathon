import numpy as np


def create_holdout_set(observations, targets, pct, **kwargs):
    train_idx, test_idx = _get_train_test_inds(targets ,pct, **kwargs)

    return observations[train_idx], targets[train_idx], observations[test_idx], targets[test_idx]

def _get_train_test_inds(y,pct=0.7, **kwargs):

    '''Generates indices, making random stratified split into training set and testing sets
    with proportions pct and (1-pct) of initial sample.
    y is any iterable indicating classes of each observation in the sample.
    Initial proportions of classes inside training and 
    testing sets are preserved (stratified sampling).
    '''
    seed = np.random.seed()
    if 'seed' in kwargs.keys():
        seed = np.random.seed(kwargs['seed'])

    rand = np.random.RandomState(seed)

    y=np.array(y)
    train_inds = np.zeros(len(y),dtype=bool)
    test_inds = np.zeros(len(y),dtype=bool)
    values = np.unique(y)
    for value in values:
        value_inds = np.nonzero(y==value)[0]
        rand.shuffle(value_inds)
        n = int(pct*len(value_inds))

        train_inds[value_inds[:n]]=True
        test_inds[value_inds[n:]]=True

    return train_inds,test_inds
