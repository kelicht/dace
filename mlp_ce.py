import numpy as np
import time
import pulp
from utils import flatten, Action, ActionCandidates
# np.set_printoptions(suppress=True, precision=3)


class MLPActionExtractor():
    def __init__(self, mdl, X, Y=[],
                 feature_names=[], feature_types=[], feature_categories=[], feature_constraints=[], max_cancidates=100, tol=1e-6,
                 target_name='Output', target_labels = ['Good','Bad'],  
                 ):
        self.mdl_ = mdl
        self.hidden_coef_ = mdl.coefs_[0]
        self.coef_ = mdl.coefs_[1]
        self.hidden_intercept_ = mdl.intercepts_[0]
        self.intercept_ = mdl.intercepts_[1][0]
        self.T_ = mdl.intercepts_[0].shape[0]
        self.X_ = X
        self.Y_ = Y
        self.N_, self.D_ = X.shape

        self.feature_names_ = feature_names if len(feature_names)==self.D_ else ['x_{}'.format(d) for d in range(self.D_)]
        self.feature_types_ = feature_types if len(feature_types)==self.D_ else ['C' for d in range(self.D_)]
        self.feature_categories_ = feature_categories
        self.feature_categories_flatten_ = flatten(feature_categories)
        self.feature_constraints_ = feature_constraints if len(feature_constraints)==self.D_ else ['' for d in range(self.D_)]
        self.target_name_ = target_name
        self.target_labels_ = target_labels
        self.AC_ = ActionCandidates(X, Y=Y, feature_names=feature_names, feature_types=feature_types, feature_categories=feature_categories, feature_constraints=feature_constraints, max_candidates=max_cancidates, tol=tol)
        self.tol_ = tol


    def getNeuronBounds(self):
        M_bar, M = self.x_.dot(self.hidden_coef_)+self.hidden_intercept_, self.x_.dot(self.hidden_coef_)+self.hidden_intercept_
        for t, w in enumerate(self.hidden_coef_.T):
            M_bar[t] += np.sum([min(w[d]*self.ub_[d], w[d]*self.lb_[d]) for d in range(self.D_)])
            M[t] += np.sum([max(w[d]*self.ub_[d], w[d]*self.lb_[d]) for d in range(self.D_)])
        M_bar = -1 * M_bar
        M_bar[M_bar<0] = 0.0
        M[M<0] = 0.0
        M_bar[M_bar>0] += self.tol_
        M[M>0] += self.tol_
        return M_bar, M

    def extract(self, x, 
                W=[], max_change_num=4, cost_type='uniform', alpha=0.0, n_neighbors=10, p_neighbors=2, subsample=20, 
                solver='CPLEX_PY', time_limit=180, log_stream=False, mdl_name='', log_name='', init_sols={}, verbose=False):
        self.x_ = x
        self.y_ = self.mdl_.predict(x.reshape(1,-1))[0]

        self.W_ = W if len(W)!=0 else list(range(self.D_))
        self.K_ = min(max_change_num, self.D_)

        self.A_, self.C_ = self.AC_.generateActions(x, self.y_, cost_type=cost_type)
        self.lb_ = [np.min(A_d) for A_d in self.A_]
        self.ub_ = [np.max(A_d) for A_d in self.A_]
        if(alpha > 0):
            X_lof, k_dists, lrds = self.AC_.generateLOFParams(int(1-self.y_), k=n_neighbors, p=p_neighbors, subsample=subsample)
            N_lof = X_lof.shape[0]
            C_lof = [[[(abs(x_lof[d] - (self.x_[d] + a)))**p_neighbors for a in self.A_[d]] for d in range(self.D_)] for x_lof in X_lof]
            U_lof = [np.sum([np.max(c[d]) for d in range(self.D_)]) for c in C_lof]
        non_zeros = []
        for d in range(self.D_):
            non_zeros_d = [1]*len(self.A_[d])
            for i in range(len(self.A_[d])):
                if(d in flatten(self.feature_categories_)):
                    if(self.A_[d][i]<=0):
                        non_zeros_d[i] = 0
                elif(self.A_[d][i]==0):
                    non_zeros_d[i] = 0
            non_zeros.append(non_zeros_d)
        prob = pulp.LpProblem(mdl_name)

        # variables
        act = [pulp.LpVariable('act_{}'.format(d), cat='Continuous', lowBound=self.lb_[d], upBound=self.ub_[d]) for d in range(self.D_)]
        pi = [[pulp.LpVariable('pi_{}_{}'.format(d,i), cat='Binary') for i in range(len(self.A_[d]))] for d in range(self.D_)]
        xi  = [pulp.LpVariable('xi_{}'.format(t), cat='Continuous', lowBound=0) for t in range(self.T_)]
        bxi  = [pulp.LpVariable('bxi_{}'.format(t), cat='Continuous', lowBound=0) for t in range(self.T_)]
        nu  = [pulp.LpVariable('nu_{}'.format(t), cat='Binary') for t in range(self.T_)]
        if(cost_type=='SCM' or cost_type=='DACE'): dist = [pulp.LpVariable('dist_{}'.format(d), cat='Continuous', lowBound=0) for d in range(self.D_)]
        if(alpha > 0):
            rho = [pulp.LpVariable('rho_{}'.format(n), cat='Continuous', lowBound=0, upBound=U_lof[n]) for n in range(N_lof)]
            mu = [pulp.LpVariable('mu_{}'.format(n), cat='Binary') for n in range(N_lof)]

        # set initial values {val: [val_1, val_2, ...], ...}
        u_obj = -1
        if(len(init_sols)!=0):
            for val, sols in init_sols.items():
                if(val=='act'):
                    for d,v in enumerate(sols): act[d].setInitialValue(v)
                elif(val=='pi'):
                    for d,vs in enumerate(sols):
                        for i,v in enumerate(vs):
                            pi[d][i].setInitialValue(v)
                elif(val=='bxi'):
                    for t,v in enumerate(sols): bxi[t].setInitialValue(v)
                elif(val=='nu'):
                    for t,v in enumerate(sols): nu[t].setInitialValue(v)
                elif(val=='obj'):
                    u_obj = sols
                        
        # objective function
        if(cost_type=='SCM' or cost_type=='DACE'):
            if(alpha > 0):
                prob += pulp.lpSum(dist) + alpha * pulp.lpDot(lrds, rho)
                prob.addConstraint(pulp.lpSum(dist) + alpha * pulp.lpDot(lrds, rho) >= 0, name='C_basic_cost')
            else:
                prob += pulp.lpSum(dist)
                prob.addConstraint(pulp.lpSum(dist) >= 0, name='C_basic_cost')
            for d in range(self.D_):
                prob.addConstraint(dist[d] - pulp.lpDot(flatten(self.C_[d]), flatten(pi)) >= 0, name='C_basic_cost_ub_{:04d}'.format(d))
                prob.addConstraint(dist[d] + pulp.lpDot(flatten(self.C_[d]), flatten(pi)) >= 0, name='C_basic_cost_lb_{:04d}'.format(d))
        else:
            if(alpha > 0):
                prob += pulp.lpDot(flatten(self.C_), flatten(pi)) + alpha * pulp.lpDot(lrds, rho)
                prob.addConstraint(pulp.lpDot(flatten(self.C_), flatten(pi)) + alpha * pulp.lpDot(lrds, rho) >= 0, name='C_basic_cost')
            else:
                prob += pulp.lpDot(flatten(self.C_), flatten(pi))
                prob.addConstraint(pulp.lpDot(flatten(self.C_), flatten(pi)) >= 0, name='C_basic_cost')

        # constraint: sum_{i} pi_{d,i} == 1
        for d in range(self.D_): prob.addConstraint(pulp.lpSum(pi[d]) == 1, name='C_basic_pi_{:04d}'.format(d))

        # constraint: sum_{d} sum_{i} pis_{d,i} <= K
        if(self.K_>=1): prob.addConstraint(pulp.lpDot(flatten(non_zeros), flatten(pi)) <= self.K_, name='C_basic_sparsity')

        # constraint: sum_{d in G} (x_d + a_d) = 1
        for i, G in enumerate(self.feature_categories_): prob.addConstraint(pulp.lpSum([act[d] for d in G]) == 0, name='C_basic_category_{:04d}'.format(i))

        # constraint: sum_{d} w_d xi_d + b >= 0
        if(self.y_ == 0):
            prob.addConstraint(pulp.lpDot(self.coef_, xi) >= - self.intercept_ + 1e-8, name='C_basic_alter')
        else:
            prob.addConstraint(pulp.lpDot(self.coef_, xi) <= - self.intercept_ - 1e-8, name='C_basic_alter')

        # constraint: a_d = sum_{i} a_{d,i} pi_{d,i}
        for d in range(self.D_): prob.addConstraint(act[d] - pulp.lpDot(self.A_[d], pi[d]) == 0, name='C_basic_act_{:04d}'.format(d))

        # constraints (Multi-Layer Perceptoron):
        M_bar, M = self.getNeuronBounds()
        for t in range(self.T_): 
            ## constraint: xi_t <= M_t nu_t
            ## constraint: bxi_t <= M_bar_t (1-nu_t)
            prob.addConstraint(xi[t] - M[t] * nu[t] <= 0, name='C_mlp_pos_{:04d}'.format(t))
            prob.addConstraint(bxi[t] + M_bar[t] * nu[t] <= M_bar[t], name='C_mlp_neg_{:04d}'.format(t))

            ## constraint: xi_t = bxi_t + sum_{d} w_{t,d} (x_d + a_d) + b_t
            prob.addConstraint(xi[t] - bxi[t] - pulp.lpDot(self.hidden_coef_.T[t], act) == self.x_.dot(self.hidden_coef_.T[t]) + self.hidden_intercept_[t], name='C_mlp_{}'.format(t))

        # constraint (1-LOF):
        if(alpha > 0):
            # constraint: sum_{n} mu_n = 1
            prob.addConstraint(pulp.lpSum(mu) == 1, name='C_lof_1nn')
            for n in range(N_lof):
                # constraint: rho_n >= d^(n) mu_n
                prob.addConstraint(rho[n] - k_dists[n] * mu[n] >= 0, name='C_lof_kdist_{:04d}'.format(n))
                # constraint: rho_n >= sum_{d} sum_{i} c^(n)_{d,i} pi_{d,i} - U_n (1-mu_n)
                prob.addConstraint(rho[n] - U_lof[n] * mu[n] - pulp.lpDot(flatten(C_lof[n]), flatten(pi))  >= - U_lof[n], name='C_lof_dist_{:04d}'.format(n))
                # constraint: sum_{d} sum_{i} (c^(n)_{d,i} - c^(n')_{d,i} pi_{d,i}) <= U_n (1-mu_n)
                for m in range(N_lof):
                    if(m==n): continue
                    tmp = [[c_n - c_m for c_n, c_m in zip(C_lof[n][d], C_lof[m][d])] for d in range(self.D_)]
                    prob.addConstraint(U_lof[n] * mu[n] + pulp.lpDot(flatten(tmp), flatten(pi)) <= U_lof[n], name='C_lof_compare_{:04d}_{:04d}'.format(n,m))


        if(len(log_name)!=0): prob.writeLP(log_name+'.lp')
        s = time.perf_counter()
        prob.solve(solver=pulp.getSolver(solver, msg=log_stream, warmStart=(len(init_sols)!=0), timeLimit=time_limit))
        t = time.perf_counter() - s
        if(prob.status!=1):
            print('**Warning**: There is No Feasible Solution. Below Sovle the Problem Again with Logging.')
            prob.writeLP('infeasible.lp')
            prob.solve(solver=pulp.getSolver(solver, msg=True, warmStart=(len(init_sols)!=0), timeLimit=time_limit))
            return -1
        obj = prob.objective.value()
        if(cost_type=='SCM' or cost_type=='DACE'):
            act_cost = np.sum([d.value() for d in dist])
        else:
            act_cost = np.sum([c*round(p.value()) for c, p in zip(flatten(self.C_), flatten(pi))])
        if(alpha>0):
            lof_cost = np.sum([l * r.value() for l, r in zip(lrds, rho)])

        a = np.array([ np.sum([ self.A_[d][i] * round(pi[d][i].value())  for i in range(len(self.A_[d])) ]) for d in range(self.D_) ])
        scores = {}
        scores['Time']=t; scores['alpha']=alpha; scores['Objective']=obj; scores['Cost ({})'.format('l1-Mahal.' if cost_type=='DACE' else cost_type)]=act_cost;
        if(alpha>0): scores['Cost (LOF)'] = lof_cost
        scores['Mahalanobis']=self.AC_.mahalanobis_dist(self.x_, self.x_+a, int(self.y_)); scores['10-LOF']=self.AC_.local_outlier_factor(self.x_+a, int(1-self.y_), k=10)
        ret = Action(x, a, scores=scores,
                     target_name=self.target_name_, target_labels=self.target_labels_, label_before=int(self.y_), label_after=int(1-self.y_), 
                     feature_names=self.feature_names_, feature_types=self.feature_types_, feature_categories=self.feature_categories_)

        # save initial values
        self.init_sols_ = {}
        self.init_sols_['act'] = [a_ for a_ in a]
        self.init_sols_['pi'] = []
        for pi_d in pi: self.init_sols_['pi'].append([round(p.value()) for p in pi_d])
        self.init_sols_['bxi'] = [max(0, x.value()) for x in bxi]
        self.init_sols_['nu'] = [round(n.value()) for n in nu]

        return ret



def _check_mlp_ce(N=1):
    import pandas as pd
    from sklearn.neural_network import MLPClassifier
    from utils import DatasetHelper
    np.random.seed(1)

    D = DatasetHelper(dataset='d', feature_prefix_index=False)
    X_tr, X_ts, y_tr, y_ts = D.train_test_split()
    mdl = MLPClassifier(hidden_layer_sizes=(20,), max_iter=500, activation='relu', alpha=0.0001)
    mdl = mdl.fit(X_tr, y_tr)
    X_undesirable = X_ts[mdl.predict(X_ts)==1]
    ce = MLPActionExtractor(mdl, X_tr, Y=y_tr, feature_names=D.feature_names, feature_types=D.feature_types, feature_categories=D.feature_categories, 
                                       feature_constraints=D.feature_constraints, target_name=D.target_name, target_labels=D.target_labels)

    print()
    print('# Demonstration: mlp_ce.py')
    print('- Dataset: Diabetes Dataset')
    print('- Classifier: Multi-Layer Perceptron Classifier')
    print()
    for n,x in enumerate(X_undesirable[:N]):
        print('# {}{} Individual with high risk of diabetes'.format(n+1, 'st' if (n+1)%10==1 else ('nd' if (n+1)%10==2 else ('rd' if (n+1)%10==3 else  'th'))))
        a = ce.extract(x, max_change_num=3, cost_type='DACE', alpha=0.01)
        if(a!=-1): print(a)


if(__name__ == '__main__'):
    from sys import argv
    if(len(argv)<2):
        print('$ python mlp_ce.py N')
        print('[Ex.] $ python mlp_ce.py 3')
    else:
        _check_mlp_ce(N=int(argv[1]))

