import numpy as np
import pandas as pd
from scipy.spatial.distance import mahalanobis
from scipy.stats import median_abs_deviation as mad
from scipy.stats import gaussian_kde as kde
from scipy.interpolate import interp1d
from sklearn.covariance import EmpiricalCovariance, MinCovDet
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.model_selection import train_test_split
from lingam import DirectLiNGAM


def flatten(x): return sum(x, [])

def supp(a, tol=1e-8): return np.where(abs(a)>tol)[0]

def greedy_select_protos(K, candidate_indices, m, is_K_sparse=False):
    # From https://github.com/BeenKim/MMD-critic/blob/master/mmd.py
    import sys
    if len(candidate_indices) != np.shape(K)[0]:
        K = K[:,candidate_indices][candidate_indices,:]
    n = len(candidate_indices)
    if is_K_sparse:
        colsum = 2*np.array(K.sum(0)).ravel() / n
    else:
        colsum = 2*np.sum(K, axis=0) / n
    selected = np.array([], dtype=int)
    value = np.array([])
    for i in range(m):
        maxx = -sys.float_info.max
        argmax = -1
        candidates = np.setdiff1d(range(n), selected)
        s1array = colsum[candidates]
        if len(selected) > 0:
            temp = K[selected, :][:, candidates]
            if is_K_sparse:
                s2array = temp.sum(0) * 2 + K.diagonal()[candidates]
            else:
                s2array = np.sum(temp, axis=0) *2 + np.diagonal(K)[candidates]
            s2array = s2array/(len(selected) + 1)
            s1array = s1array - s2array
        else:
            if is_K_sparse:
                s1array = s1array - (np.abs(K.diagonal()[candidates]))
            else:
                s1array = s1array - (np.abs(np.diagonal(K)[candidates]))
        argmax = candidates[np.argmax(s1array)]
        selected = np.append(selected, argmax)
        KK = K[selected, :][:, selected]
        if is_K_sparse:
            KK = KK.todense()
    return candidate_indices[selected]

def prototype_selection(X, subsample=20, kernel='rbf'):
    return greedy_select_protos(pairwise_kernels(X, metric=kernel), np.array(range(X.shape[0])), subsample) if subsample>1 else np.array(range(X.shape[0]))

def CumulativeDistributionFunction(x_d, X_d, l_buff=1e-6, r_buff=1e-6):
    kde_estimator = kde(X_d)
    pdf = kde_estimator(x_d)
    cdf_raw = np.cumsum(pdf)
    total = cdf_raw[-1] + l_buff + r_buff
    cdf = (l_buff + cdf_raw) / total
    percentile_ = interp1d(x=x_d, y=cdf, copy=False,fill_value=(l_buff,1.0-r_buff), bounds_error=False, assume_sorted=False)
    return percentile_

def interaction_matrix(X, interaction_type='causal', prior_knowledge=None, measure='pwling', estimator='ML', file_name=''):
    if(interaction_type=='causal'):
        lingam = DirectLiNGAM(prior_knowledge=prior_knowledge, measure=measure).fit(X)
        B = lingam.adjacency_matrix_
        C = np.zeros([X.shape[1], X.shape[1]])
        for d in range(1, X.shape[1]): 
            C += np.linalg.matrix_power(B, d)
        return B, C
    elif(interaction_type=='correlation'):
        return np.corrcoef(X.T) - np.eye(X.shape[1])
    elif(interaction_type=='covariance'):
        if(estimator=='ML'):
            est = EmpiricalCovariance(store_precision=True, assume_centered=False).fit(X)
        elif(estimator=='MCD'):
            est = MinCovDet(store_precision=True, assume_centered=False, support_fraction=None).fit(X)
        cov = est.covariance_
        if(np.linalg.matrix_rank(cov)!=X.shape[1]): cov += 1e-6 * np.eye(X.shape[1])
        l_, P_ = np.linalg.eig(np.linalg.inv(cov))
        l = np.diag(np.sqrt(l_))
        P = P_.T
        U = P.T.dot(l).T
        return cov, U
    elif(interaction_type=='precomputed'):
        df = pd.read_csv(file_name)
        return df.values


class Action():
    def __init__(self, x, a, scores={},
                 target_name='Output', target_labels=['Good', 'Bad'], label_before=1, label_after=0, 
                 feature_names=[], feature_types=[], feature_categories=[]):
        self.x_ = x
        self.a_ = a
        self.scores_ = scores
        self.target_name_ = target_name
        self.labels_ = [target_labels[label_before], target_labels[label_after]]
        self.feature_names_ = feature_names if len(feature_names)==len(x) else ['x_{}'.format(d) for d in range(len(x))]
        self.feature_types_ = feature_types if len(feature_types)==len(x) else ['C' for d in range(len(x))]
        self.feature_categories_ = feature_categories

        self.feature_categories_inv_ = []
        for d in range(len(x)):
            g = -1
            if(self.feature_types_[d]=='B'):
                for i, cat in enumerate(self.feature_categories_):
                    if(d in cat): 
                        g = i
                        break
            self.feature_categories_inv_.append(g)            


    def __str__(self):
        s = '* '
        s += 'Action ({}: {} -> {}):\n'.format(self.target_name_, self.labels_[0], self.labels_[1])
        i = 0
        for i,d in enumerate(np.where(abs(self.a_)>1e-8)[0]):
            num = '*'
            g = self.feature_categories_inv_[d]
            if(g==-1):
                if(self.feature_types_[d]=='C'):
                    s += '\t{} {}: {:.4f} -> {:.4f} ({:+.4f})\n'.format(num, self.feature_names_[d], self.x_[d], self.x_[d]+self.a_[d], self.a_[d])
                else:
                    s += '\t{} {}: {} -> {} ({:+})\n'.format(num, self.feature_names_[d], self.x_[d].astype(int), (self.x_[d]+self.a_[d]).astype(int), self.a_[d].astype(int))
            else:
                if(self.x_[d]==1): continue
                cat_name, nxt = self.feature_names_[d].split(':')
                cat = self.feature_categories_[g]
                prv = self.feature_names_[cat[np.where(self.x_[cat])[0][0]]].split(':')[1]
                s += '\t{} {}: {} -> {}\n'.format(num, cat_name, prv, nxt)

        if(len(self.scores_)>0):
            s += '* Scores: \n'
            for i in self.scores_.items():
                s += '\t* {0}: {1:.8f}\n'.format(i[0], i[1])
        return s
    
    def a(self):
        return self.a_

# class Action



ACTION_TYPES = ['B', 'I', 'C']
ACTION_CONSTRAINTS = ['', 'FIX', 'INC', 'DEC']
class ActionCandidates():
    def __init__(self, X, Y=[], feature_names=[], feature_types=[], feature_categories=[], feature_constraints=[], max_candidates=50, tol=1e-6):
        self.X_ = X
        self.Y_ = Y
        self.N_, self.D_ = X.shape
        self.feature_names_ = feature_names if len(feature_names)==self.D_ else ['x_{}'.format(d) for d in range(self.D_)]
        self.feature_types_ = feature_types if len(feature_types)==self.D_ else ['C' for d in range(self.D_)]
        self.feature_categories_ = feature_categories
        self.feature_constraints_ = feature_constraints if len(feature_constraints)==self.D_ else ['' for d in range(self.D_)]
        self.max_candidates = max_candidates
        self.tol_ = tol

        self.X_lb_, self.X_ub_ = X.min(axis=0), X.max(axis=0)
        self.steps_ = [(self.X_ub_[d]-self.X_lb_[d])/max_candidates if self.feature_types_[d]=='C' else 1 for d in range(self.D_)]
        self.grids_ = [np.arange(self.X_lb_[d], self.X_ub_[d]+self.steps_[d], self.steps_[d]) for d in range(self.D_)]

        self.actions_ = None
        self.costs_ = None
        self.Q_ = None
        self.cov_ = None

    def getFeatureWeight(self, cost_type='uniform'):
        weights = np.ones(self.D_)
        if(cost_type=='MAD'):
            for d in range(self.D_):
                weight =  mad(self.X_[:,d], scale='normal')
                if(self.feature_types_[d]=='B' or abs(weight)<self.tol_):
                    weights[d] = (self.X_[:,d]*1.4826).std()
                else:
                    weights[d] = weight ** -1
        elif(cost_type=='PCC' and len(self.Y_)==self.N_):
            for d in range(self.D_):
                weights[d] = abs(np.corrcoef(self.X_[:, d], self.Y_)[0,1])
        elif(cost_type=='standard'):
            weights = np.std(self.X_, axis=0) ** -1
        elif(cost_type=='normalize'):
            weights = (self.X_.max(axis=0) - self.X_.min(axis=0)) ** -1
        elif(cost_type=='robust'):
            q25, q75 = np.percentile(self.X_, [0.25, 0.75], axis=0)
            for d in range(self.D_):
                if(q75[d]-q25[d]==0):
                    weights[d] = self.tol_ ** -1
                else:
                    weights = (q75[d]-q25) ** -1
        return weights

    def setActionSet(self, x):
        self.actions_ = []
        for d in range(self.D_):
            if(self.feature_constraints_[d]=='FIX' or self.steps_[d] < self.tol_):
                self.actions_.append(np.array([ 0 ]))
            elif(self.feature_types_[d]=='B'):
                if((self.feature_constraints_[d]=='INC' and x[d]==1) or (self.feature_constraints_[d]=='DEC' and x[d]==0)):
                    self.actions_.append(np.array([ 0 ]))
                else:
                    self.actions_.append(np.array([ 1-2*x[d], 0 ]))
            else:
                if(self.feature_constraints_[d]=='INC'):
                    start = x[d] + self.steps_[d]
                    stop = self.X_ub_[d] + self.steps_[d]
                elif(self.feature_constraints_[d]=='DEC'):
                    start = self.X_lb_[d]
                    stop = x[d]
                else:
                    start = self.X_lb_[d]
                    stop = self.X_ub_[d] + self.steps_[d]
                A_d = np.arange(start, stop, self.steps_[d]) - x[d]
                A_d = np.extract(abs(A_d)>self.tol_, A_d)
                if(len(A_d) > self.max_candidates): A_d = A_d[np.linspace(0, len(A_d)-1, self.max_candidates, dtype=int)]
                A_d = np.append(A_d, 0)
                self.actions_.append(A_d)
        return self

    def setActionAndCost(self, x, y, cost_type='TLPS', p=1):
        self.costs_ = []
        self = self.setActionSet(x)
        if(cost_type=='TLPS'):
            if(self.Q_==None): self.Q_ = [None if self.feature_constraints_[d]=='FIX' else CumulativeDistributionFunction(self.grids_[d], self.X_[:, d]) for d in range(self.D_)]
            for d in range(self.D_):
                if(self.Q_[d]==None):
                    self.costs_.append([])
                else:
                    Q_d = self.Q_[d]
                    Q_0 = Q_d(x[d])
                    self.costs_.append( [ abs(np.log2( (1-Q_d(x[d]+a)) / (1-Q_0) )) for a in self.actions_[d] ] )
        elif(cost_type=='SCM' or cost_type=='DACE'):
            if(cost_type=='SCM'): 
                B_, _ = interaction_matrix(self.X_, interaction_type='causal') 
                B = np.eye(self.D_) - B_
                C = self.getFeatureWeight(cost_type='standard')
            else:
                self.cov_, B = interaction_matrix(self.X_[self.Y_==y] if len(self.Y_)==self.N_ else self.X_, interaction_type='covariance') 
                C = self.getFeatureWeight(cost_type='uniform')
            for d in range(self.D_):
                cost_d = []
                for d_ in range(self.D_): cost_d.append( [ C[d] * B[d][d_] * a  for a in self.actions_[d_] ] )
                self.costs_.append(cost_d)
        else:
            weights = self.getFeatureWeight(cost_type=cost_type)
            if(cost_type=='PCC'): p=2
            for d in range(self.D_):
                self.costs_.append( list(weights[d] * abs(self.actions_[d])**p) )
        return self

    def generateActions(self, x, y, cost_type='TLPS', p=1):
        self = self.setActionAndCost(x, y, cost_type=cost_type, p=p)
        return self.actions_, self.costs_

    def generateLOFParams(self, y, k=10, p=2, subsample=20, kernel='rbf'):
        lof = LocalOutlierFactor(n_neighbors=k, metric='manhattan' if p==1 else 'sqeuclidean', novelty=False)
        X_lof = self.X_[self.Y_==y]
        lof = lof.fit(X_lof)
        def k_distance(prototypes):
            return lof._distances_fit_X_[prototypes, k-1]
        def local_reachability_density(prototypes):
            return lof._lrd[prototypes]
        prototypes = prototype_selection(X_lof, subsample=subsample, kernel=kernel)
        return X_lof[prototypes], k_distance(prototypes), local_reachability_density(prototypes)

    def mahalanobis_dist(self, x_1, x_2, y):
        if(self.cov_ is None):
            self.cov_, _ = interaction_matrix(self.X_[self.Y_==y] if len(self.Y_)==self.N_ else self.X_, interaction_type='covariance') 
        return mahalanobis(x_1, x_2, np.linalg.inv(self.cov_))

    def local_outlier_factor(self, x, y, k=10, p=2):
        lof = LocalOutlierFactor(n_neighbors=k, metric='manhattan' if p==1 else 'sqeuclidean', novelty=True)
        lof = lof.fit(self.X_[self.Y_==y])
        return -lof.score_samples(x.reshape(1, -1))[0]

# class ActionCandidates


class ForestActionCandidates():
    def __init__(self, X, forest, Y=[], feature_names=[], feature_types=[], feature_categories=[], feature_constraints=[], max_candidates=50, tol=1e-6):
        self.X_ = X
        self.Y_ = Y
        self.N_, self.D_ = X.shape
        self.feature_names_ = feature_names if len(feature_names)==self.D_ else ['x_{}'.format(d) for d in range(self.D_)]
        self.feature_types_ = feature_types if len(feature_types)==self.D_ else ['C' for d in range(self.D_)]
        self.feature_categories_ = feature_categories
        self.feature_constraints_ = feature_constraints if len(feature_constraints)==self.D_ else ['' for d in range(self.D_)]
        self.tol_ = tol

        self.forest_ = forest
        self.T_ = forest.n_estimators
        self.trees_ = [t.tree_ for t in forest.estimators_]
        self.leaves_ = [np.where(tree.feature==-2)[0]  for tree in self.trees_]
        self.L_ = [len(l) for l in self.leaves_]

        self.H_ = self.getForestLabels()
        self.ancestors_, self.regions_ = self.getForestRegions()        
        self.thresholds_ = self.getForestThresholds()
        # self.M_ = [len(self.thresholds_[d])+1 for d in range(self.D_)]
        # self.partitions_ = self.getForestPartitions()

        self.X_lb_, self.X_ub_ = X.min(axis=0), X.max(axis=0)
        self.max_candidates = max_candidates
        self.steps_ = [(self.X_ub_[d]-self.X_lb_[d])/max_candidates if self.feature_types_[d]=='C' else 1 for d in range(self.D_)]
        self.grids_ = [np.arange(self.X_lb_[d], self.X_ub_[d]+self.steps_[d], self.steps_[d]) for d in range(self.D_)]

        self.x_ = None
        self.actions_ = None
        self.costs_ = None
        self.Q_ = None
        self.cov_ = None
        self.I_ = None

    def getForestLabels(self):
        H = []
        for tree, leaves, l_t in zip(self.trees_, self.leaves_, self.L_):
            falls = tree.value[leaves].reshape(l_t, 2)
            h_t = [val[0] if val.shape[0]==1 else val[1]/(val[0]+val[1]) for val in falls]
            H.append(h_t)
        return H

    def getForestRegions(self):
        As, Rs = [], []
        for tree, leaves in zip(self.trees_, self.leaves_):
            A, R = [], []
            stack = [[]]
            L, U = [[-np.inf]*self.D_], [[np.inf]*self.D_]
            for n in range(tree.node_count):
                a, l, u = stack.pop(), L.pop(), U.pop()
                if(n in leaves):
                    A.append(a)
                    R.append([ (l[d], u[d]) for d in range(self.D_)])
                else:
                    d = tree.feature[n]
                    if(d not in a): a_ = list(a) + [d]
                    stack.append(a_); stack.append(a_)
                    # b = int(tree.threshold[n]) if self.feature_types_[d]=='I' else tree.threshold[n]
                    b = tree.threshold[n]
                    l_ = list(l); u_ = list(u)
                    l[d] = b; u[d] = b
                    U.append(u_); L.append(l)
                    U.append(u); L.append(l_)
            As.append(A); Rs.append(R)
        return As, Rs

    def getForestThresholds(self):
        B = []
        for d in range(self.D_):
            b_d = []
            for tree in self.trees_: 
                # b_d += list(tree.threshold[tree.feature==d].astype(int)) if self.feature_types_[d]=='I' else list(tree.threshold[tree.feature==d])
                b_d += list(tree.threshold[tree.feature==d])
            b_d = list(set(b_d))
            b_d.sort()
            B.append(np.array(b_d))
        return B

    def getForestPartitions(self):
        I = []
        for t in range(self.T_):
            I_t = []
            for l in range(self.L_[t]):
                I_t_l = []
                for d in range(self.D_):
                    if(self.regions_[t][l][d][0]==-np.inf):
                        start = 0
                    else:
                        start = self.thresholds_[d].index(self.regions_[t][l][d][0]) + 1
                    if(self.regions_[t][l][d][1]== np.inf):
                        end = self.M_[d]
                    else:
                        end = self.thresholds_[d].index(self.regions_[t][l][d][1]) + 1
                    tmp = list(range(start, end))
                    I_t_l.append(tmp)
                I_t.append(I_t_l)
            I.append(I_t)
        return I

    def getFeatureWeight(self, cost_type='uniform'):
        weights = np.ones(self.D_)
        if(cost_type=='MAD'):
            from scipy.stats import median_abs_deviation as mad
            for d in range(self.D_):
                weight =  mad(self.X_[:,d], scale='normal')
                if(self.feature_types_[d]=='B' or abs(weight)<self.tol_):
                    weights[d] = (self.X_[:,d]*1.4826).std()
                else:
                    weights[d] = weight ** -1
        elif(cost_type=='PCC' and len(self.Y_)==self.N_):
            for d in range(self.D_):
                weights[d] = abs(np.corrcoef(self.X_[:, d], self.Y_)[0,1])
        elif(cost_type=='standard'):
            weights = np.std(self.X_, axis=0) ** -1
        elif(cost_type=='normalize'):
            weights = (self.X_.max(axis=0) - self.X_.min(axis=0)) ** -1
        elif(cost_type=='robust'):
            weights = (np.quantile(self.X_, 0.75, axis=0) - np.quantile(self.X_, 0.25, axis=0)) ** -1
        return weights

    def setActionSet(self, x, use_threshold=True):
        if((x == self.x_).all()): return self
        self.x_ = x
        self.actions_ = []
        for d in range(self.D_):
            if(self.feature_constraints_[d]=='FIX' or self.steps_[d] < self.tol_):
                self.actions_.append(np.array([ 0 ]))
            elif(self.feature_types_[d]=='B'):
                if((self.feature_constraints_[d]=='INC' and x[d]==1) or (self.feature_constraints_[d]=='DEC' and x[d]==0)):
                    self.actions_.append(np.array([ 0 ]))
                else:
                    self.actions_.append(np.array([ 1-2*x[d], 0 ]))
            else:
                if(use_threshold):
                    A_d = self.thresholds_[d].astype(int) - x[d] if self.feature_types_[d]=='I' else self.thresholds_[d] - x[d]
                    A_d[A_d>=0] += self.tol_ if self.feature_types_[d]=='C' else 1
                    if(0 not in A_d): A_d = np.append(A_d, 0)
                    if(self.feature_constraints_[d]=='INC'): 
                        A_d = np.extract(A_d>=0, A_d)
                    elif(self.feature_constraints_[d]=='DEC'): 
                        A_d = np.extract(A_d<=0, A_d)
                else:
                    if(self.feature_constraints_[d]=='INC'):
                        start = x[d] + self.steps_[d]
                        stop = self.X_ub_[d] + self.steps_[d]
                    elif(self.feature_constraints_[d]=='DEC'):
                        start = self.X_lb_[d]
                        stop = x[d]
                    else:
                        start = self.X_lb_[d]
                        stop = self.X_ub_[d] + self.steps_[d]
                    A_d = np.arange(start, stop, self.steps_[d]) - x[d]
                    A_d = np.extract(abs(A_d)>self.tol_, A_d)
                    if(len(A_d) > self.max_candidates): A_d = A_d[np.linspace(0, len(A_d)-1, self.max_candidates, dtype=int)]
                    A_d = np.append(A_d, 0)
                self.actions_.append(A_d)
        self = self.setForestIntervals(x)
        return self

    def setActionAndCost(self, x, y, cost_type='TLPS', p=1, use_threshold=True):
        self.costs_ = []
        self = self.setActionSet(x, use_threshold=use_threshold)
        if(cost_type=='TLPS'):
            if(self.Q_==None): self.Q_ = [None if self.feature_constraints_[d]=='FIX' else CumulativeDistributionFunction(self.grids_[d], self.X_[:, d]) for d in range(self.D_)]
            for d in range(self.D_):
                if(self.Q_[d]==None):
                    self.costs_.append([ 0 ])
                else:
                    Q_d = self.Q_[d]
                    Q_0 = Q_d(x[d])
                    self.costs_.append( [ abs(np.log2( (1-Q_d(x[d]+a)) / (1-Q_0) )) for a in self.actions_[d] ] )
        elif(cost_type=='SCM' or cost_type=='DACE'):
            if(cost_type=='SCM'): 
                B_, _ = interaction_matrix(self.X_, interaction_type='causal') 
                B = np.eye(self.D_) - B_
                C = self.getFeatureWeight(cost_type='standard')
            else:
                self.cov_, B = interaction_matrix(self.X_[self.Y_==y] if len(self.Y_)==self.N_ else self.X_, interaction_type='covariance') 
                C = self.getFeatureWeight(cost_type='uniform')
            for d in range(self.D_):
                cost_d = []
                for d_ in range(self.D_): cost_d.append( [ C[d] * B[d][d_] * a  for a in self.actions_[d_] ] )
                self.costs_.append(cost_d)
        else:
            weights = self.getFeatureWeight(cost_type=cost_type)
            for d in range(self.D_):
                self.costs_.append( list(weights[d] * abs(self.actions_[d])**p) )
        return self

    def setForestIntervals(self, x):
        Is = [np.arange(len(a)) for a in self.actions_]
        I = []
        for t in range(self.T_):
            I_t = []
            for l in range(self.L_[t]):
                I_t_l = []
                for d in range(self.D_):
                    xa = x[d] + self.actions_[d]
                    I_t_l.append( list(((xa > self.regions_[t][l][d][0]) & (xa <= self.regions_[t][l][d][1])).astype(int)) )
                    # I_t_l.append( (Is[d][ (xa > self.regions_[t][l][d][0]) & (xa <= self.regions_[t][l][d][1]) ]) )
                I_t.append(I_t_l)
            I.append(I_t)
        self.I_ = I
        return self

    def generateActions(self, x, y, cost_type='TLPS', p=1, use_threshold=True):
        self = self.setActionAndCost(x, y, cost_type=cost_type, p=p, use_threshold=use_threshold)
        return self.actions_, self.costs_, self.I_

    def generateLOFParams(self, y, k=10, p=2, subsample=20, kernel='rbf'):
        lof = LocalOutlierFactor(n_neighbors=k, metric='manhattan' if p==1 else 'sqeuclidean', novelty=False)
        X_lof = self.X_[self.Y_==y]
        lof = lof.fit(X_lof)
        def k_distance(prototypes):
            return lof._distances_fit_X_[prototypes, k-1]
        def local_reachability_density(prototypes):
            return lof._lrd[prototypes]
        prototypes = prototype_selection(X_lof, subsample=subsample, kernel=kernel)
        return X_lof[prototypes], k_distance(prototypes), local_reachability_density(prototypes)

    def mahalanobis_dist(self, x_1, x_2, y):
        if(self.cov_ is None):
            self.cov_, _ = interaction_matrix(self.X_[self.Y_==y] if len(self.Y_)==self.N_ else self.X_, interaction_type='covariance') 
        return mahalanobis(x_1, x_2, np.linalg.inv(self.cov_))

    def local_outlier_factor(self, x, y, k=10, p=2):
        lof = LocalOutlierFactor(n_neighbors=k, metric='manhattan' if p==1 else 'sqeuclidean', novelty=True)
        lof = lof.fit(self.X_[self.Y_==y])
        return -lof.score_samples(x.reshape(1, -1))[0]

# class ForestActionCandidates


DATASETS = ['g', 'w', 'h', 'd']
DATASETS_NAME = {
    'g':'german',
    'w':'wine',
    'h':'fico',
    'd':'diabetes',
}
DATASETS_PATH = {
    'g':'data/german_credit.csv',
    'w':'data/wine.csv',
    'h':'data/heloc.csv',
    'd':'data/diabetes.csv',
}
TARGET_NAME = {
    'g':'Default',
    'w':'Quality',
    'h':'RiskPerformance',
    'd':'DiseaseProgression',
}
TARGET_LABELS = {
    'g':['No', 'Yes'],
    'w':['>5', '<=5'],
    'h':['Good', 'Bad'],
    'd':['Good', 'Bad'],
}
FEATURE_NUMS = {
    'g':61,
    'w':12,
    'h':23,
    'd':10,
}
FEATURE_TYPES = {
    'g':['I']*7 + ['B']*54,
    'w':['C']*5 + ['I']*2 + ['C']*4 + ['B'],
    'h':['I']*23,
    'd':['I'] + ['B'] + ['C']*2 + ['I'] + ['C']*4 + ['I'],
}
FEATURE_CATEGORIES = {
    'g':[list(range(7,11)),list(range(11,16)),list(range(16,26)),list(range(26,31)),list(range(31,36)),list(range(36,40)),list(range(40,43)),list(range(43,47)),list(range(47,50)),list(range(50,53)),list(range(53,57)),list(range(57,59)),list(range(59,60))],
    'w':[],
    'h':[],
    'd':[],
}
FEATURE_CONSTRAINTS = {
    'g':['']*50 + ['FIX']*3 + ['']*6 + ['FIX']*2,
    'w':['']*11 + ['FIX'],
    'h':['']*23,
    'd':['INC'] + ['FIX'] + ['']*8,
}
class DatasetHelper():
    def __init__(self, dataset='h', feature_prefix_index=False):
        self.dataset_ = dataset
        self.df_ = pd.read_csv(DATASETS_PATH[dataset], dtype='float')
        self.y = self.df_[TARGET_NAME[dataset]].values
        self.X = self.df_.drop([TARGET_NAME[dataset]], axis=1).values
        self.feature_names = list(self.df_.drop([TARGET_NAME[dataset]], axis=1).columns)
        if(feature_prefix_index): self.feature_names = ['[x_{}]'.format(d) + feat for d,feat in enumerate(self.feature_names)]
        self.feature_types = FEATURE_TYPES[dataset]
        self.feature_categories = FEATURE_CATEGORIES[dataset]
        self.feature_constraints = FEATURE_CONSTRAINTS[dataset]
        self.target_name = TARGET_NAME[dataset]
        self.target_labels = TARGET_LABELS[dataset]
        self.dataset_name = DATASETS_NAME[dataset]

    def train_test_split(self, test_size=0.25):
        return train_test_split(self.X, self.y, test_size=test_size)

# class DatasetHelper


def _check_action_and_cost(dataset='h'):
    DH = DatasetHelper(dataset=dataset)
    AC = ActionCandidates(DH.X[1:], feature_names=DH.feature_names, feature_types=DH.feature_types, feature_categories=DH.feature_categories, feature_constraints=DH.feature_constraints)
    print(DH.X[0])
    A,C = AC.generateActions(DH.X[0], cost_type='TLPS')
    for d,a in zip(AC.feature_names_, A): print(d, a)
    for d,c in zip(AC.feature_names_, C): print(d, c)

def _check_forest_action(dataset='h'):
    from sklearn.ensemble import RandomForestClassifier
    DH = DatasetHelper(dataset=dataset)
    X,y = DH.X, DH.y
    forest = RandomForestClassifier(n_estimators=2, max_depth=4)
    forest = forest.fit(X[1:],y[1:])
    AC = ForestActionCandidates(DH.X[1:], forest, feature_names=DH.feature_names, feature_types=DH.feature_types, feature_categories=DH.feature_categories, feature_constraints=DH.feature_constraints)
    A, C, I, = AC.generateActions(X[0], cost_type='unifrom', use_threshold=True)
    # for d in range(X.shape[1]): print(d, A[d])
    # for d in range(X.shape[1]): print(d, C[d])
    for I_t in I:
        for I_t_l in I_t:
            print(I_t_l)
            print(flatten(I_t_l))

if(__name__ == '__main__'):
    # _check_action_and_cost(dataset='h')
    _check_forest_action(dataset='d')
