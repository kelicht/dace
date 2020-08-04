import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.spatial.distance import mahalanobis
from scipy.stats import gaussian_kde as kde
from scipy.interpolate import interp1d
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.neighbors import NearestNeighbors
from mmd import select_criticism_regularized, greedy_select_protos

def mahalanobis_cost(x, action, X):
    vi = np.linalg.inv(np.cov(X.T) + np.eye(len(x))*1e-16)
    return mahalanobis(x, action, vi)
# Func mahalanobis_cost

def CumulativeDistributionFunction(A_d, X_d, l_buff=1e-6, r_buff=1e-6):
    kde_estimator = kde(X_d)
    pdf = kde_estimator(A_d)
    cdf_raw = np.cumsum(pdf)
    total = cdf_raw[-1] + l_buff + r_buff
    cdf = (l_buff + cdf_raw) / total
    percentile_ = interp1d(x=A_d, y=cdf, copy=False,fill_value=(l_buff,1.0-r_buff), bounds_error=False, assume_sorted=False)
    return percentile_
# Func CumulativeDistributionFunction

# Modified verstion of the code from https://www.dskomei.com/entry/2018/03/04/125249
def plot_decision_regions(X, y, mdl, x, actions, x_range='auto', resolution=0.01,
                          x1_minmax=[], x2_minmax=[], offset1=0.1, offset2=0.1, arrow_width=0.005, x_size=200, aspect=False,
                          colors=['yellow','lime','cyan','magenta'], markers=['^', 's', 'v', '*'],
                          labels = [ 'TLPS','DACE (ours)', 'MAD', 'PCC'],
                          x1='$x_1$', x2='$x_2$', title=r'', filename=''):

    plt.rcParams["font.family"] = 'arial'
    plt.rcParams['text.usetex'] = True

    ## 2変数の入力データの最小値から最大値まで引数resolutionの幅でメッシュを描く
    if(len(x1_minmax)==2 and len(x2_minmax)==2):
        x1_min, x1_max = x1_minmax[0], x1_minmax[1]
        x2_min, x2_max = x2_minmax[0], x2_minmax[1]
    elif(x_range=='auto'):
        x1_min, x1_max = np.vstack([x,X])[:, 0].min()-offset1, np.vstack([x,X])[:, 0].max()+offset1
        x2_min, x2_max = np.vstack([x,X])[:, 1].min()-offset2, np.vstack([x,X])[:, 1].max()+offset2
    elif(x_range=='fixed'):
        x1_min, x1_max = 0.0, 1.0
        x2_min, x2_max = 0.0, 1.0
    else:
        x1_min, x1_max = X.min()-offset, X.max()+offset
        x2_min, x2_max = X.min()-offset, X.max()+offset
    x1_mesh, x2_mesh = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))

    ## メッシュデータ全部を学習モデルで分類
    z = mdl.predict(np.array([x1_mesh.ravel(), x2_mesh.ravel()]).T)
    z = z.reshape(x1_mesh.shape)

    ## メッシュデータと分離クラスを使って決定境界を描いている
    cmap = ListedColormap(('blue','red'))
    plt.contourf(x1_mesh, x2_mesh, z, alpha=0.3, cmap=cmap)
    plt.xlim(x1_mesh.min(), x1_mesh.max())
    plt.ylim(x2_mesh.min(), x2_mesh.max())

    # markers = ('o','x')
    # for i, c in enumerate(np.unique(y)):
    #     plt.scatter(x=X[y==c,0], y=X[y==c,1], alpha=0.6, c=cmap(i), edgecolors='black', marker=markers[i], label=c)
    plt.scatter(x=X[y==0,0], y=X[y==0,1], alpha=0.6, color='blue', edgecolors='black', marker='o')
    plt.scatter(x=X[y==1,0], y=X[y==1,1], alpha=0.6, color='red', edgecolors='black', marker='x')

    # plot actions
    for i, action in enumerate(actions):
        plt.scatter([action[0]], [action[1]], marker=markers[i], s=x_size, color=colors[i], label=labels[i], edgecolor='black', linewidth=1.2)
        plt.annotate('', xytext=(x[0], x[1]), xy=(action[0], action[1]), arrowprops=dict(width=2.0, shrink=0.075, color=colors[i]))
    plt.scatter(x[0], x[1], color='r', edgecolor='black', marker='D', s=x_size, linewidth=1.0)

    if(aspect): plt.axes().set_aspect('equal')
    plt.xlabel(r'{}'.format(x1),fontsize=24,labelpad=10.0)
    plt.ylabel(r'{}'.format(x2),fontsize=24)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    # plt.title(title)
    plt.legend(fontsize=22,frameon=True,facecolor='white',edgecolor='black',fancybox=False,framealpha=1.0,markerscale=1.2,loc='lower left')
    plt.tight_layout()
    # plt.show()
    plt.show() if len(filename)==0 else plt.savefig('./res/' + filename + '.pdf')
# Func plot_decision_regions



FILE_DICT = {
    'g':'german_credit',
    'b':'bindata/german_credit',
    'd':'diabetes',
    'w':'wine',
    'h':'heloc',
    'f':'heloc_categorical',
    's':'synthetic',
    # 'm':'meps',
}

OUT_DICT = {
    'g':'Default',
    'b':'Default',
    'd':'Outcome',
    'w':'Quality<=5',
    'h':'RiskPerformance',
    'f':'RiskPerformance',
    's':'Output',
    # 'm':'HealthExpenditure',
}

OUT_MEAN = {1:'BAD', 0:'GOOD'}

class Database():
    def __init__(self, filename='g', binarized=False):
        if(filename not in FILE_DICT.keys()):
            print('Error: No file exists.')
            print('Available files: ', list(FILE_DICT.keys()))
        else:
            self.dataset_ = FILE_DICT[filename]
            if(filename=='s'):
                self.filename_ = 'Synthetic'
            elif(filename=='r'):
                self.filename_ = 'Real'
            else:
                self.data_initial_ = filename
                path = './data/bindata/{}'.format(self.dataset_) if binarized else './data/{}'.format(self.dataset_)
                self.featurefilename_ = path + '.feature.txt'
                self.filename_ = path  + '.csv'
                self.binarized_ = binarized

    def get_dataset(self, N=100, D=2, split=False, test_size=0.3, file_num=-1, seed=0):
        if(self.filename_=='Synthetic'):
            self.X, self.y = self.__getToy(N, D, seed)
            self.features, self.types, self.categories = ['x_1', 'x_2'], ['C', 'C'], []
        else:
            df = pd.read_csv(self.filename_, dtype='float')
            self.X = self.__getX(df)
            self.y = self.__getY(df)
            self.features, self.types, self.categories = self.__getFeatures()
        if(split):
            from sklearn.model_selection import train_test_split
            self.X_tr, self.X_ts, self.y_tr, self.y_ts = train_test_split(self.X, self.y, stratify=self.y, test_size=test_size, random_state=seed)
            return self.X_tr, self.X_ts, self.y_tr, self.y_ts, self.features, self.types, self.categories
        else:
            return self.X, self.y, self.features, self.types, self.categories

    def __getX(self, df):
        return df.drop([OUT_DICT[self.data_initial_]], axis=1).values

    def __getY(self, df):
        return df[OUT_DICT[self.data_initial_]].values

    def __getFeatures(self):
        features, types, categories = [], [], []
        f = open(self.featurefilename_)
        dctf = f.readline()[1:].replace('\n','').split(' ')
        while(len(dctf)==4):
            d = int(dctf[0])
            c = int(dctf[1])
            if(c != -1):
                if(c==len(categories)):
                    categories.append([d])
                else:
                    categories[c].append(d)
            types.append(dctf[2])
            features.append(dctf[3])
            dctf = f.readline()[1:].replace('\n','').split(' ')
        f.close()
        return features, types, categories

    def __getToy(self, N, D, seed):
        np.random.seed(seed)
        x1 = np.random.uniform(0,1,N) + np.random.uniform(-0.1,0.1,N)
        x2 = x1 + np.random.uniform(-0.3,0.3,N)
        x = [np.random.randn(N) for d in range(D-2)]
        X = np.vstack([x1, x2]+x).T
        w = np.array([1.0] + [0]*(D-1))
        y = np.array([1 if np.dot(w,x)-0.5>0.1*np.random.randn() else 0 for x in X])
        return X, y
# class Database

