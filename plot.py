import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
plt.rcParams["font.family"] = 'arial'
plt.rcParams['text.usetex'] = True
DATAS = {'fico':'FICO', 'german':'German', 'wine':'WineQuality', 'diabetes':'Diabetes'}
MODELS = {'LR':'Logistic Regression', 'RF':'Random Forest', 'MLP':'Multilayer Perceptron'}
alphas = {'fico':1.0, 'german':0.01, 'wine':0.01, 'diabetes':1.0}



def to_latex():
    datas = ['fico','german','wine','diabetes']
    models = ['LR', 'RF', 'MLP']
    costs = ['TLPS', 'MAD', 'PCC', 'DACE']

    for data in datas:
        print(data)
        for cost in costs:
            s = '{} '.format(cost)
            for model in models:
                df = pd.read_csv('./res/paper/{}/{}_{}_{}.csv'.format(model, data, cost, alphas[data]) if cost=='DACE' else './res/paper/{}/{}_{}.csv'.format(model, data, cost))
                df = df[df['Time'] > 0]
                s += '& {:.3} $\pm$ {:.2} '.format(df['Mahalanobis'].mean(), df['Mahalanobis'].std())
                s += '& {:.3} $\pm$ {:.2} '.format(df['10-LOF'].mean(), df['10-LOF'].std())
                # s += '& {:.3} $\pm$ {:.2} '.format(df['Time'].mean(), df['Time'].std())
            s += ' \\\\'
            print(s)
        print()
    print()

to_latex()


def scatter_act():
    datas = ['fico','german','wine','diabetes']
    models = ['LR', 'RF', 'MLP']
    costs = ['TLPS', 'MAD', 'PCC', 'DACE']
    markers = {'TLPS':'^', 'MAD':'o', 'PCC':'d', 'DACE':'s'}
    colors = {'TLPS':'yellow', 'MAD':'red', 'PCC':'blue', 'DACE':'lime'}
    x = {}; y = {};
    for key1 in datas:
        x[key1] = {}; y[key1] = {};
        for key2 in models:
            x[key1][key2] = {}; y[key1][key2] = {};
            for key3 in costs:
                df = pd.read_csv('./res/paper/{}/{}_{}_{}.csv'.format(key2, key1, key3, alphas[key1]) if key3=='DACE' else './res/paper/{}/{}_{}.csv'.format(key2, key1, key3))
                df = df[df['Time'] > 0]
                df = df[df['Mahalanobis'] < 20]
                df = df[df['10-LOF'] < 8]
                # df = df.iloc[:20]
                x[key1][key2][key3] = df['Mahalanobis']
                y[key1][key2][key3] = df['10-LOF']
    
    fig = plt.figure(figsize=[12,7.5])
    for i, model in enumerate(models):
        for j, data in enumerate(datas):
            plt.subplot(len(models), len(datas), i*(len(datas)) + j + 1)
            for cost in costs:
                plt.scatter(x[data][model][cost], y[data][model][cost], marker=markers[cost], color=colors[cost], label=cost, s=26, edgecolor='black', linewidth=0.2)
            plt.xlabel('MD', fontsize=10)
            plt.ylabel('10-LOF', fontsize=10)
            plt.title('{} ({})'.format(DATAS[data], MODELS[model]), fontsize=12)
            if(i==0 and j==0): plt.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig('./res/scat_all.pdf', bbox_inches='tight', pad_inches=0.05)

# scatter_act()


def to_latex_time():
    datas = ['fico','german','wine','diabetes']
    models = ['LR', 'RF', 'MLP']
    costs = ['TLPS', 'MAD', 'PCC', 'DACE']

    for model in models:
        print(model)
        for cost in costs:
            s = '{} '.format(cost)
            for data in datas:
                df = pd.read_csv('./res/paper/{}/{}_{}_{}.csv'.format(model, data, cost, alphas[data]) if cost=='DACE' else './res/paper/{}/{}_{}.csv'.format(model, data, cost))
                df = df[df['Time'] > 0]
                s += '& {:.3} $\pm$ {:.2} '.format(df['Time'].mean(), df['Time'].std())
            s += ' \\\\'
            print(s)
        print()
    print()

# to_latex_time()


def plot_sens():
    datas = ['fico','german','wine','diabetes']
    
    mahal = {}; lof = {}
    for key in datas:
        mahal[key] = pd.read_csv('./res/paper/sens/{}_Mahalanobis.csv'.format(key)).mean()
        lof[key] = pd.read_csv('./res/paper/sens/{}_10-LOF.csv'.format(key)).mean()
    gammas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    keys = [str(g) for g in gammas]
    fig = plt.figure(figsize=[7.5,9])

    for i, data in enumerate(datas):
        ax1 = fig.add_subplot(4, 1, i+1)
        ln1 = ax1.plot(gammas, mahal[data][keys], marker='o', color='blue', label='MD')
        ax2 = ax1.twinx()
        ln2 = ax2.plot(gammas, lof[data][keys], marker='^', linestyle='dashed', color='red', label='10-LOF')
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        if(i==0): ax1.legend(h1+h2, l1+l2, loc='upper left', borderaxespad=1.5, fontsize=12)
        ax1.set_xlabel(r'$\lambda$', fontsize=16, labelpad=-0.3)
        ax1.set_xscale('log')
        ax1.tick_params(labelsize=12)
        ax2.tick_params(labelsize=12)
        ax1.set_ylabel('MD', fontsize=16)
        ax2.set_ylabel('10-LOF', fontsize=16)
        ax1.set_title(DATAS[data], fontsize=16)

    fig.align_labels()
    plt.tight_layout()
    plt.savefig('./res/sens_all.pdf', bbox_inches='tight', pad_inches=0.05)

# plot_sens()


