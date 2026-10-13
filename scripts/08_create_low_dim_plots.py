from swamp.glob import glob
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from os import listdir

path_pheno = glob.PATH_PHENO_ancestry
path_input = glob.PATH_GENO + 'mini_dim/'
onlyfiles_input = [f for f in listdir(path_input)]
for file in onlyfiles_input:
    path_file = path_input + file
    hidden_repr = np.array(pd.read_pickle(path_file))
    humans = pd.read_csv(path_pheno, sep="\t")
    humans = humans.drop(columns=['Unnamed: 5', 'Unnamed: 6'])
    humans = humans.rename({'sample': 'FID', 'sample.1': 'IID'}, axis=1)
    humans['ancestry'] = humans['super_pop']
    humans['ancestry'].mask(humans['super_pop'] == 'AFR', 0, inplace=True)
    humans['ancestry'].mask(humans['super_pop'] == 'AMR', 1, inplace=True)
    humans['ancestry'].mask(humans['super_pop'] == 'EAS', 2, inplace=True)
    humans['ancestry'].mask(humans['super_pop'] == 'EUR', 3, inplace=True)
    humans['ancestry'].mask(humans['super_pop'] == 'SAS', 4, inplace=True)
    for i in range(np.shape(hidden_repr)[1]):
        humans[f"dim{i+1}"] = hidden_repr[:,i]
        palette = 'mako'

    plt.figure(figsize = (15, 10))
    sns.scatterplot(
        data=humans, 
        x="dim1",
        y="dim2",
        hue="super_pop",
        palette = palette).set(title=f"Populations")
    plt.savefig(f"temporary_plots/population_{file.split('.')[0]}.png")
    plt.close()
    
    afr = humans.loc[humans['ancestry'] == 0]
    amr = humans.loc[humans['ancestry'] == 1]
    eas = humans.loc[humans['ancestry'] == 2]
    eur = humans.loc[humans['ancestry'] == 3]
    sas = humans.loc[humans['ancestry'] == 4]

    import warnings
    warnings.filterwarnings('ignore')

    palette_afr = sns.blend_palette(['limegreen', 'black'], len(list(afr['pop'].unique()))) # afr
    palette_amr = sns.blend_palette(['salmon','darkred'], len(list(amr['pop'].unique()))) # amr
    palette_eur = sns.blend_palette(['yellow','darkorange'], len(list(eur['pop'].unique()))) #europe
    palette_eas = sns.blend_palette(['turquoise', 'midnightblue'], len(list(eas['pop'].unique()))) # eas
    palette_sas = sns.blend_palette(['lightpink','darkviolet'], len(list(sas['pop'].unique()))) # sas

    fig, ax = plt.subplots(figsize=(12, 6))
    afr_plot = sns.scatterplot(
        data=afr, 
        x="dim1",
        y="dim2",
        hue="pop",
        palette = palette_afr)
    h1, l1 = afr_plot.get_legend_handles_labels()
    legend1 = ax.legend(h1, l1,
                    loc="lower left", title="Africans")
    amr_plot = sns.scatterplot(
        data=amr, 
        x="dim1",
        y="dim2",
        hue="pop",
        palette = palette_amr)
    h2, l2 = amr_plot.get_legend_handles_labels()
    l2 = [item for item in l2 if item not in l1]
    h2 = [item for item in h2 if item not in h1]

    legend2 = ax.legend(h2, l2,
                        loc="upper left", title="Americans")
    eur_plot = sns.scatterplot(
        data=eur, 
        x="dim1",
        y="dim2",
        hue="pop",
        palette = palette_eur)
    h3, l3 = eur_plot.get_legend_handles_labels()
    l3 = [item for item in l3 if item not in l1]
    l3 = [item for item in l3 if item not in l2]
    h3 = [item for item in h3 if item not in h1]
    h3 = [item for item in h3 if item not in h2]
    legend3 = ax.legend(h3, l3,
                        loc="upper right", title="Europeans")
    eas_plot = sns.scatterplot(
        data=eas, 
        x="dim1",
        y="dim2",
        hue="pop",
        palette = palette_eas)
    h4, l4 = eas_plot.get_legend_handles_labels()
    l4 = [item for item in l4 if item not in l1]
    l4 = [item for item in l4 if item not in l2]
    l4 = [item for item in l4 if item not in l3]

    h4 = [item for item in h4 if item not in h1]
    h4 = [item for item in h4 if item not in h2]
    h4 = [item for item in h4 if item not in h3]
    legend4 = ax.legend(h4, l4,
                        loc="lower right", title="East-asians")
    sas_plot = sns.scatterplot(
        data=sas, 
        x="dim1",
        y="dim2",
        hue="pop",
        palette = palette_sas)

    h5, l5 = sas_plot.get_legend_handles_labels()
    l5 = [item for item in l5 if item not in l1]
    l5 = [item for item in l5 if item not in l2]
    l5 = [item for item in l5 if item not in l3]
    l5 = [item for item in l5 if item not in l4]

    h5 = [item for item in h5 if item not in h1]
    h5 = [item for item in h5 if item not in h2]
    h5 = [item for item in h5 if item not in h3]
    h5 = [item for item in h5 if item not in h4]
    legend5 = ax.legend(h5, l5,
                        loc="upper center", title="South-asians")

    plt.tight_layout()
    ax.add_artist(legend1)
    ax.add_artist(legend2)
    ax.add_artist(legend3)
    ax.add_artist(legend4)
    ax.add_artist(legend5)
    plt.savefig(f"temporary_plots/subpopulation_{file.split('.')[0]}.png")
    plt.close()