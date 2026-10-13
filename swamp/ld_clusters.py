import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from termcolor import cprint

from swamp.helpers import create_plot


def ld_groups_pca(genos, sample_size = 1000):
    from collections import Counter
    import numpy as np
    import pandas as pd
    import tensorflow as tf
    
    genos = genos.T
    len_test = round(genos.shape[0]/10)*3
    len_train = genos.shape[0] - len_test
    X_train = genos.head(len_train)
    X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
    X_test = genos.tail(len_test)
    X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)
    X = tf.convert_to_tensor(genos, dtype=tf.float32)
    
    """
    do PCA
    """
    from sklearn.decomposition import PCA
    pca = PCA()
    pca.fit(X)
    pca_data = pca.transform(X)
    LDs = pd.DataFrame(index = genos.index, data=pca_data[:,:2].astype('float32'), columns=['dim1', 'dim2'])
    
    """
    How many clusters do you want and how many point minimal per centroids
    """
    num_clus = round(LDs.shape[0]/sample_size)
    size_min = sample_size - round(sample_size/5)
    size_max = sample_size + round(sample_size/5)
    
    from k_means_constrained import KMeansConstrained
    clf = KMeansConstrained(
        n_clusters=num_clus,
        size_min=size_min,
        size_max=size_max,
        random_state=0
    )
    clf.fit_predict(np.array(LDs))

    genos['ordering_unshuffled'] = clf.labels_

    import random
    from math import floor
    count = Counter(clf.labels_)
    empty = []
    sample_size = floor(len(clf.labels_)/sample_size)
    for val in count.values():
        multi = floor(val/sample_size)+1
        list_to_sample = multi*list(range(sample_size))
        random.shuffle(list_to_sample)
        empty = empty + list_to_sample[0:val]
    genos['ordering_shuffled'] = empty
    return genos, LDs
    

def ld_groups_ae(genos, sample_size = 1000, depth = 2, epoch = 5):
    from collections import Counter
    import numpy as np
    import pandas as pd
    import tensorflow as tf

    from swamp.glob import glob
    from swamp.models import create_autoencoder
    genos = genos.T
    len_test = round(genos.shape[0]/10)*3
    len_train = genos.shape[0] - len_test
    X_train = genos.head(len_train)
    X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
    X_test = genos.tail(len_test)
    X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)
    X = tf.convert_to_tensor(genos, dtype=tf.float32)
    input_len = int(tf.shape(X_train)[1])
    
    """
    Create autoencoder for snp clustering
    """
    AE = create_autoencoder(input_len, 2, 0.01, 'elu', depth = depth, optimizer = 'Adam', loss = 'mse', output_activation = 'tanh')
    history = AE.fit(X_train,
            X_train,
            validation_data=(X_test, X_test),
            epochs = epoch,
            verbose = 0)

    from swamp.helpers import metrics_viewer
    metric_plt = metrics_viewer(history)
    metric_plt.savefig('temporary_plots/metrics_LD_groups.png')    
    metric_plt.close()
    """
    Image of snps in LD
    """
    hidden_repr = AE.layers[1](X).numpy()
    LDs = pd.DataFrame(index = genos.index, data=hidden_repr, columns=['dim1', 'dim2'])
    
    """
    How many clusters do you want and how many point minimal per centroids
    """
    num_clus = round(LDs.shape[0]/sample_size)
    
    size_min = sample_size - round(sample_size/5)
    size_max = sample_size + round(sample_size/5)
    
    from k_means_constrained import KMeansConstrained
    clf = KMeansConstrained(
        n_clusters=num_clus,
        size_min=size_min,
        size_max=size_max,
        random_state=0
    )
    clf.fit_predict(np.array(LDs))
    
    create_plot(clf.labels_, LDs, "dim1", "dim2", f"Clusters of SNP'S", 'temporary_plots/LD_groups.png')
    genos['ordering_unshuffled'] = clf.labels_
    
    import random
    from math import floor
    count = Counter(clf.labels_)
    empty = []
    sample_size = floor(len(clf.labels_)/sample_size)
    for val in count.values():
        multi = floor(val/sample_size)+1
        list_to_sample = multi*list(range(sample_size))
        random.shuffle(list_to_sample)
        empty = empty + list_to_sample[0:val]
    genos['ordering_shuffled'] = empty
    return genos, LDs