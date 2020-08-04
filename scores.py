
import numpy as np
from scipy.spatial.distance import mahalanobis
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EmpiricalCovariance, MinCovDet
from sklearn.metrics.pairwise import pairwise_kernels
from mmd import greedy_select_protos
from collections import Counter
from sklearn.cluster import DBSCAN
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import pairwise_distances_argmin, pairwise_distances_argmin_min, pairwise_distances


def prototype_selection(X, subsample=20, kernel='rbf'):
    return greedy_select_protos(pairwise_kernels(X, metric=kernel), np.array(range(X.shape[0])), subsample) if subsample>1 else np.array(range(X.shape[0]))

class myLocalOutlierFactor():
    def __init__(self, n_neighbors=20, p=2):
        self.n_neighbors_ = n_neighbors
        metric = 'manhattan' if p==1 else 'sqeuclidean'
        self.lof_ = LocalOutlierFactor(n_neighbors=n_neighbors, metric=metric, novelty=False)
        self.lof_novel_ = LocalOutlierFactor(n_neighbors=n_neighbors, metric=metric, novelty=True)

    def fit(self, X_tr):
        # self.weight_ = X_tr.std(axis=0) ** (-1)
        self.weight_ = np.ones(X_tr.shape[1])
        self.X_tr_ = X_tr * self.weight_
        self.lof_ = self.lof_.fit(X_tr)
        self.lof_novel_ = self.lof_novel_.fit(X_tr)
        return self

    def k_distance(self, ind_tr):
        return self.lof_._distances_fit_X_[ind_tr, self.n_neighbors_-1]

    def local_reachability_density(self, ind_tr):
        return self.lof_._lrd[ind_tr]

    def get_params(self, subsample=-1, kernel='rbf'):
        if(subsample>1):
            prototypes = prototype_selection(self.X_tr_, subsample=subsample, kernel=kernel)
        else:
            prototypes = np.array(range(self.X_tr_.shape[0]))
        return prototypes, self.k_distance(prototypes), self.local_reachability_density(prototypes)

    def local_outlier_factor(self, X_ts):
        return -self.lof_novel_.score_samples(X_ts * self.weight_)

# class myLocalOutlierFactor

class myMahalanobisDistance():
    def __init__(self, estimator='ML', tol=1e-6):
        if(estimator=='ML'):
            self.estimator_ = EmpiricalCovariance(store_precision=True, assume_centered=False)
        elif(estimator=='MCD'):
            self.estimator_ = MinCovDet(store_precision=True, assume_centered=False, support_fraction=None, random_state=0)
        else:
            self.estimator_ = None
        self.tol_ = tol

    def fit(self, X_tr):
        self.D_ = X_tr.shape[1]
        if(self.estimator_==None):
            self.cov_ = np.cov(X_tr.T)
            if(np.linalg.matrix_rank(self.cov_)!=self.D_): self.cov_ += self.tol_ * np.eye(self.D_)
        else:
            self.estimator_ = self.estimator_.fit(X_tr)
            self.cov_ = self.estimator_.covariance_
            if(np.linalg.matrix_rank(self.cov_)!=self.D_): self.cov_ += self.tol_ * np.eye(self.D_)
            # self.inv_ = self.estimator_.precision_
        self.inv_ = np.linalg.inv(self.cov_)
        self = self.__setEig()
        return self

    def __setEig(self):
        self.lams_, self.U_ = np.linalg.eig(self.inv_)
        self.U_ = self.U_.T
        # self.lams_, self.U_ = self.lams_.astype(float), self.U_.T.astype(float)
        self.Lam_ = np.diag(np.sqrt(self.lams_))
        self.L_ = self.U_.T.dot(self.Lam_).T
        # self.M_ = self.U_.T.dot(self.Lam_, self.U_)
        return self

    def mahalanobis_dist(self, x, y, p=2):
        if(p==1):
            return np.linalg.norm(self.L_.dot(x-y), ord=1)
        else:
            return mahalanobis(x, y, self.inv_)

# class myMahalanobisDistance

def evaluation_test(action, clf, target, X_tr, y_tr, N_BALL=1000):
    def generate_inside_ball_new(center, segment, n):
        def norm(v):
            return np.linalg.norm(v, ord=2, axis=1)
        d = center.shape[0]
        z = np.random.normal(0, 1, (n, d))
        u = np.random.uniform(segment[0]**d, segment[1]**d, n)
        r = u**(1/float(d))
        z = np.array([a * b / c for a, b, c in zip(z, r,  norm(z))])
        z = z + center
        return z
    X_train_ennemies = X_tr[np.where((y_tr == target) & (clf.predict(X_tr) == y_tr) )]
    idx_ally_adv, delta = pairwise_distances_argmin_min(action.reshape(1,-1), X_train_ennemies, metric='euclidean')
    CF_adv = X_train_ennemies[idx_ally_adv][0]
    delta_adv = pairwise_distances(action.reshape(1, -1), X_train_ennemies).min()
    RADIUS = delta_adv + 0.0
    ball = generate_inside_ball_new(action, (0, RADIUS), n=N_BALL)
    ball_pred = clf.predict(ball)
    ball_allies_adv = ball[np.where(ball_pred == target)]
    ball_allies_adv = np.insert(ball_allies_adv, 0, CF_adv, axis=0)
    ball_allies_adv = np.insert(ball_allies_adv, 0, action, axis=0)
    kg = kneighbors_graph(ball_allies_adv, n_neighbors=1, mode='distance', metric='euclidean', include_self=False, n_jobs=-1)
    closest_distances = kg.toarray()[np.where(kg.toarray()>0)]
    eps = closest_distances.max()
    clustering = DBSCAN(eps=eps, min_samples=2, leaf_size=30, n_jobs=-1).fit(ball_allies_adv)
    labels = clustering.labels_
    # justif = int(labels[0] == labels[1])
    # return justif
    return int(labels[0] == labels[1])


if(__name__ == '__main__'):
    from sklearn.datasets import load_diabetes
    from sklearn.ensemble import RandomForestClassifier
    X, y = load_diabetes().data, load_diabetes().target
    y = 2 * ((y>y.mean()).astype(int)) - 1
    clf = RandomForestClassifier(n_estimators=10)
    clf = clf.fit(X[:100],y[:100])
    print(evaluation_test(X[0], clf, y[0], X[1:], y[1:]))
    # from sklearn.model_selection import train_test_split
    # X = load_diabetes().data
    # X_tr, X_ts = train_test_split(X, test_size=0.1, random_state=0)
    # lof = myLocalOutlierFactor(n_neighbors=1, metric='minkowski', p=2)
    # lof = lof.fit(X_tr)
    # print('LocalReachabilityDensity: \n {}'.format(lof.local_reachability_density(X_tr[:10])))
    # print('LocalOutlierFactor: \n {}'.format(lof.local_outlier_factor(X_ts)))
