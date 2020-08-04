import numpy as np
import time
from cplex import Cplex
from scipy.stats import median_absolute_deviation as mad
from scores import myLocalOutlierFactor, myMahalanobisDistance, prototype_selection
from utils import CumulativeDistributionFunction
import warnings
warnings.simplefilter('ignore', np.ComplexWarning)

ACTION_DICT = {'binary':'B', 'integer':'I', 'natural':'N', 'continuous':'C', 'semi-continuous':'S'}
ACTION_TYPES = ['B', 'I', 'N', 'C', 'S']

class LinearCEExtractor():
    def __init__(self, mdl, X=[], Y=[], cov_estimator='ML', n_neighbors=10, p_neighbors=2,
                 FeatureNames=[], FeatureTypes=[], Categories=[], resolution=50, tol=1e-6, verbose=False):
        self.mdl_ = mdl
        self.coef_ = mdl.coef_[0]
        self.intercept_ = mdl.intercept_[0]
        self.D_ = self.coef_.shape[0]

        self.X_ = X
        self.Y_ = Y
        self.n_neighbors_ = n_neighbors
        self.p_neighbors_ = p_neighbors
        self.mahal_ = myMahalanobisDistance(estimator=cov_estimator, tol=tol)
        self.lof_ = myLocalOutlierFactor(n_neighbors=n_neighbors, p=p_neighbors)

        self.features_ = [fname for fname in FeatureNames] if len(FeatureNames)==self.D_ else ['x_{0}'.format(d) for d in range(self.D_)]
        self.types_ = FeatureTypes if len(FeatureTypes)==self.D_ else ['C']*self.D_
        self.categories_ = Categories
        self.categories_flatten_ = sum(Categories, [])

        self.tol_ = tol
        self.resolution_ = [resolution]*self.D_ if type(resolution)==int else resolution
        self.sig_uncalc_ = True
        self.verbose_ = verbose


    def __getProblem(self, K, ActionSet, WorkingSet=[], theta=0.0, log_stream=False, log_write=False):
        prob = Cplex()
        if(log_stream==False):
            prob.set_log_stream(None)
            prob.set_results_stream(None)
        prob.objective.set_sense(prob.objective.sense.minimize)
        prob.parameters.timelimit.set(self.time_limit_)
        # prob.parameters.timelimit.set(600)
        # prob.parameters.emphasis.mip.set(4)

        pi = [['pi_{}_{}'.format(d,m) for m in range(self.M_[d])] for d in range(self.D_)]
        if(self.interaction_):
            for d in range(self.D_): prob.variables.add(names=pi[d], types=['B']*(self.M_[d]))
        else:
            for d in range(self.D_): prob.variables.add(obj=self.C_[d], names=pi[d], types=['B']*(self.M_[d]))

        if(self.interaction_):
            psi = ['psi_{0}'.format(d) for d in range(self.D_)]
            prob.variables.add(obj=np.sqrt(self.eigenvalues_), names=psi, lb=[0]*(self.D_), ub=self.ubs_, types=['S']*(self.D_))
            prob.linear_constraints.add(lin_expr=[[psi, np.sqrt(self.eigenvalues_)]], senses=['G'], rhs=[0])

        # constraint: decision function value
        coef_const = [[self.coef_[d] * a for a in self.A_[d]] for d in range(self.D_)]
        coef = [ [sum(pi, []), sum(coef_const, [])] ]
        const = [theta + self.tol_ - self.intercept_] if self.y_==0 else [theta + self.tol_ - self.intercept_]
        sense = ['G'] if self.y_==0 else ['L']
        prob.linear_constraints.add(lin_expr=coef, senses=sense, rhs=const)

        # constraint: only one action is selected
        coef = [ [pi[d], [1]*self.M_[d]] for d in range(self.D_)]
        # prob.linear_constraints.add(lin_expr=coef, senses=['E']*self.D_, rhs=[1]*self.D_)
        coef += [[[pi[d][self.m_[d]]], [1]] for d in range(self.D_) if 'FIX' in self.types_[d]]
        if(len(WorkingSet)!=0): coef += [[[pi[d][self.m_[d]]], [1]] for d in range(self.D_) if d not in WorkingSet]
        const = [1]*len(coef)
        prob.linear_constraints.add(lin_expr=coef, senses=['E']*len(coef), rhs=const)

        # constraint: categorical features
        coef = [[[pi[d][1] for d in cat], [1]*len(cat)] for cat in self.categories_]
        prob.linear_constraints.add(lin_expr=coef, senses=['E']*len(self.categories_), rhs=[1]*len(self.categories_))

        # constraint: max modification
        coef_const = [[0,0] if d in self.categories_flatten_ and self.x_[d]==1 else [0 if m==self.m_[d] else 1 for m in range(self.M_[d])] for d in range(self.D_)]
        coef = [ [sum(pi, []), sum(coef_const, [])] ]
        prob.linear_constraints.add(lin_expr=coef, senses=['L'], rhs=[K])

        if(self.interaction_):
            # constraint: absolute values of psi
            prob.linear_constraints.add(lin_expr=[ [ sum(pi, [])+[psi[d]], self.uu_[d]+[1] ] for d in range(self.D_)] , senses=['G']*self.D_, rhs=[0]*self.D_)
            prob.linear_constraints.add(lin_expr=[ [ sum(pi, [])+[psi[d]], self.uu_[d]+[-1] ] for d in range(self.D_)] , senses=['L']*self.D_, rhs=[0]*self.D_)

        if(self.alpha_>0):
            def cost_dev(n,m):
                R_n = self.R_[n]
                R_m = self.R_[m]
                return sum([[R_n[d][i] - R_m[d][i] for i in range(self.M_[d])] for d in range(self.D_)], [])

            nu = ['nu_{}'.format(n) for n in range(self.N_reg_)]
            prob.variables.add(names=nu, types=['B']*self.N_reg_)
            prob.linear_constraints.add(lin_expr=[[nu, [1]*self.N_reg_]], senses=['E'], rhs=[self.n_neighbors_])
            pi_flatten = sum(pi, [])

            if(self.n_neighbors_ == 1):
                rho = ['rho_{}'.format(n) for n in range(self.N_reg_)]
                prob.variables.add(obj=[(self.alpha_ * self.lrd_[n])/self.n_neighbors_ for n in range(self.N_reg_)], names=rho, lb=[0]*self.N_reg_, ub=self.R_ubs_, types=['S']*self.N_reg_)
                prob.linear_constraints.add(lin_expr=[[[nu[n],rho[n]], [self.k_dists_[n],-1]] for n in range(self.N_reg_)], senses=['L']*self.N_reg_, rhs=[0]*self.N_reg_)
                prob.linear_constraints.add(lin_expr=[[pi_flatten+[nu[n],rho[n]], sum(self.R_[n], [])+[self.R_ubs_[n],-1]] for n in range(self.N_reg_)], senses=['L']*self.N_reg_, rhs=self.R_ubs_)
                for n in range(self.N_reg_):
                    prob.linear_constraints.add(lin_expr=[[ pi_flatten+[nu[n]], cost_dev(n,m)+[self.R_ubs_[n]] ] for m in range(self.N_reg_) if m!=n], senses=['L']*(self.N_reg_-1), rhs=[self.R_ubs_[n]]*(self.N_reg_-1))
                # prob.variables.add(obj=[self.alpha_/(self.dists_[n]*(self.n_neighbors_)) for n in range(self.N_reg_)], names=rho, lb=[0]*self.N_reg_, ub=self.R_ubs_, types=['S']*self.N_reg_)
                # prob.linear_constraints.add(lin_expr=[[[nu[n],rho[n]], [self.dists_[n],-1]] for n in range(self.N_reg_)], senses=['L']*self.N_reg_, rhs=[0]*self.N_reg_)
                # prob.linear_constraints.add(lin_expr=[[pi_flatten+[nu[n],rho[n]], sum(self.R_[n], [])+[self.R_ubs_[n],-1]] for n in range(self.N_reg_)], senses=['L']*self.N_reg_, rhs=self.R_ubs_)
                # for n in range(self.N_reg_):
                #     prob.linear_constraints.add(lin_expr=[[ pi_flatten+[nu[n]], cost_dev(n,m)+[self.R_ubs_[n]] ] for m in range(self.N_reg_) if m!=n], senses=['L']*(self.N_reg_-1), rhs=[self.R_ubs_[n]]*(self.N_reg_-1))
                    # indicator
                    # for m in range(self.N_reg_):
                        # if(m!=n): prob.indicator_constraints.add(lin_expr=[pi_flatten, cost_dev(n,m)], sense=['L'], rhs=0, indvar=nu[n], complemented=0)
            else:
                mu = [['mu_{}_{}'.format(n,m) for m in range(self.N_reg_)] for n in range(self.N_reg_)]
                for n in range(self.N_reg_): prob.variables.add(names=[mu[n][m] for m in range(self.N_reg_) if m!=n], types=['B']*(self.N_reg_-1))
                for n in range(self.N_reg_):
                    coef = []
                    const = []
                    for m in range(self.N_reg_):
                        if(m==n): continue
                        coef += [[[nu[n],nu[m],mu[n][m]], [-1,1,-1]], [[nu[n],mu[n][m]], [1,1]], [[nu[m],mu[n][m]], [-1,1]]]
                        const += [0,1,0]
                    prob.linear_constraints.add(lin_expr=coef, senses=['L']*3*(self.N_reg_-1), rhs=const)
                for n in range(self.N_reg_):
                    coef = []
                    const = []
                    for m in range(self.N_reg_):
                        if(m==n): continue
                        coef += [[pi_flatten+[mu[n][m]], cost_dev(n,m)+[-1*self.R_ubs_[m]]]]
                        const += [-1*self.R_ubs_[m]]
                    prob.linear_constraints.add(lin_expr=coef, senses=['G']*(self.N_reg_-1), rhs=const)
                rho = [['rho_{}_{}'.format(n,m) for m in range(self.N_reg_)] for n in range(self.N_reg_)]
                for n in range(self.N_reg_): prob.variables.add(obj=[self.alpha_/(self.dists_[m]*(self.n_neighbors_**2)) for m in range(self.N_reg_)], names=rho[n], lb=[0]*(self.N_reg_), ub=[self.R_ubs_[n]]*(self.N_reg_), types=['S']*(self.N_reg_))
                eta = [['eta_{}_{}'.format(n,m) for m in range(self.N_reg_)] for n in range(self.N_reg_)]
                for n in range(self.N_reg_): prob.variables.add(names=eta[n], types=['B']*(self.N_reg_))
                for n in range(self.N_reg_):
                    coef = []
                    const = []
                    for m in range(self.N_reg_):
                        coef += [[pi_flatten+[eta[n][m],rho[n][m]], sum(self.R_[n],[])+[self.R_ubs_[n], -1]], [[eta[n][m], rho[n][m]], [self.dists_[n], -1]]]
                        const += [self.R_ubs_[n], 0]
                    prob.linear_constraints.add(lin_expr=coef, senses=['L']*(2*self.N_reg_), rhs=const)
                coef = []
                for n in range(self.N_reg_): coef += [[[eta[n][m], eta[m][n]], [1,-1]] for m in range(self.N_reg_) if m!=n]
                coef += [[[eta[n][n], nu[n]], [1,-1]] for n in range(self.N_reg_)]
                prob.linear_constraints.add(lin_expr=coef, senses=['E']*len(coef), rhs=[0]*len(coef))
                for n in range(self.N_reg_):
                    coef = []
                    const = []
                    for m in range(n+1, self.N_reg_):
                        coef += [[[nu[n],nu[m],eta[n][m]], [1,1,-1]], [[nu[n],eta[n][m]], [-1,1]], [[nu[m],eta[n][m]], [-1,1]]]
                        const += [1,0,0]
                    prob.linear_constraints.add(lin_expr=coef, senses=['L']*len(const), rhs=const)

        if(log_write): prob.write('mylog.lp')

        return prob

    def __getActionSet(self, X):
        ActionSet = []
        for d in range(self.D_):
            actionset_d = [ self.x_[d] ]
            # if('FIX' in self.types_[d]):
            #     actionset_d = actionset_d
            # elif(self.types_[d][0]=='B'):
            if(self.types_[d][0]=='B'):
                actionset_d += [0,1]
            elif(self.types_[d][0]=='I'):
                lb = int(self.x_[d]) if 'INC' in self.types_[d] else int(np.min(X[:,d])) - 1
                ub = int(np.max(X[:,d])) + 2
                actionset_d += list([i for i in np.linspace(lb, ub, self.resolution_[d], dtype='int')])
            elif(self.types_[d][0]=='N'):
                lb = int(np.min(X[:,d])) - 1
                lb = int(self.x_[d]) if 'INC' in self.types_[d] else np.max([lb, 0])
                ub = int(np.max(X[:,d])) + 2
                actionset_d += list([i for i in np.linspace(lb, ub, self.resolution_[d], dtype='int')])
            elif(self.types_[d][0]=='S'):
                lb = np.min(X[:,d]) - self.tol_
                lb = np.max([lb, 0.0])
                ub = np.max(X[:,d]) + self.tol_
                actionset_d += list(np.linspace(lb, ub, self.resolution_[d]))
            else:
                lb = np.min(X[:,d]) - self.tol_
                ub = np.max(X[:,d]) + self.tol_
                actionset_d += list(np.linspace(lb, ub, self.resolution_[d]))
            # print(d,list(set(actionset_d)))
            ActionSet.append(list(set(actionset_d)))
        return ActionSet

    def __getWeight(self, d, weight_type):
        if(weight_type=='uniform'):
            return 1.0
        elif(weight_type=='PCC'):
            return abs(np.corrcoef(self.X_[:,d], self.Y_)[0,1])
        elif(weight_type=='MAD'):
            if(self.types_[d][0]=='B'):
                return (self.X_[:,d]*1.4826).std()
            else:
                weight =  mad(self.X_[:,d])
                if(weight!=0.0):
                    return weight**-1
                else:
                    return (self.X_[:,d]*1.4826).std()

    def __getCost(self, weight_type):
        CostSet = []
        for d in range(self.D_):
            if(self.interaction_):
                costset_d = [(self.x_[d] - a) for a in self.A_[d]]
            else:
                if(weight_type=='MPS' or weight_type=='TLPS'):
                    Q_d = CumulativeDistributionFunction(self.A_[d], self.X_[:, d])
                    Q = Q_d(self.x_[d])
                    if(weight_type=='MPS'):
                        costset_d = [abs(Q - Q_d(a)) if a!=self.x_[d] else 0.0 for a in self.A_[d]]
                    elif(weight_type=='TLPS'):
                        costset_d = [abs(np.log2((1-Q_d(a))/(1-Q))) if a!=self.x_[d] else 0.0 for a in self.A_[d]]
                else:
                    weight = self.__getWeight(d, weight_type)
                    if(d in self.categories_flatten_ and self.x_[d]==1):
                        costset_d = [0.0] * self.M_[d]
                    else:
                        costset_d = [weight * abs(self.x_[d] - a)**self.p_ for a in self.A_[d]]
            CostSet.append(costset_d)
        return CostSet

    def __getRegs(self, X, weight):
        Regset = []
        for x in X:
            regset_n = [[(weight[d] * abs(x[d] - a))**self.p_neighbors_ for a in self.A_[d]] for d in range(self.D_)]
            Regset.append(regset_n)
        return Regset

    def extract(self, x, theta=0.0, ActionSet=[], WorkingSet=[], max_mod=-1,
                weight_type='uniform', p=1, interaction=False, alpha=-1, subsample=20, pre_selection=True, n_neighbors_selection=1,
                iterative=False, time_limit=60, log_stream=False, log_write=False, init_sol=[], ret_init_sol=False):

        self.x_ = x
        self.y_ = self.mdl_.predict([x])[0]
        self.p_ = p
        self.interaction_ = interaction
        self.alpha_ = alpha
        self.time_limit_ = time_limit

        if(len(self.Y_)!=0):
            X_cost, X_reg = self.X_[self.Y_==self.y_], self.X_[self.Y_!=self.y_]
        else:
            X_cost, X_reg = X, X

        self.mahal_ = self.mahal_.fit(X_cost)
        self.lof_ = self.lof_.fit(X_reg)

        K = max_mod if max_mod in list(range(1,self.D_)) else self.D_
        self.A_ = self.__getActionSet(self.X_) if len(ActionSet)!=self.D_ else ActionSet
        self.M_ = [len(A) for A in self.A_]
        self.C_ = self.__getCost(weight_type)
        self.m_ = [self.A_[d].index(self.x_[d]) for d in range(self.D_)]
        # print(self.C_)

        # self.weights_ = np.array([self.__getWeight(X_reg[:,d], d, weight_type_outlier) for d in range(self.D_)])
        # self.pns_ = self.pns_.fit(X_reg, weights=self.weights_, pre_selection=selection, subsample=subsample)
        # X_reg = self.pns_.X_
        #
        # if(self.alpha_>0):
        #     self.N_reg_ = X_reg.shape[0]
        #     self.dists_ = self.pns_.k_dists_
        #     self.R_ = self.__getRegs(X_reg, weight_type)        # R[n][d][i] = |x^(n)_{d} - a_{d,i}|
        #     self.R_ubs_ = [np.sum([ np.max(r[d]) for d in range(self.D_)]) for r in self.R_]
        #     # for d in range(self.D_): print(self.A_[d])
        #     # for r in self.R_: print([ np.max(r[d]) for d in range(self.D_)])
        # if(self.interaction_ and (len(X_cost)!=0 or self.sig_uncalc_==False)):
        #     if(self.sig_uncalc_):
        #         self.sig_ = np.cov(X_cost.T) + np.eye(self.D_)*self.tol_
        #         self.eigenvalues_, self.eigenmatrix_ = np.linalg.eig(self.sig_)
        #         self.eigenmatrix_ = (self.eigenmatrix_.T).astype('float')
        #         self.eigenvalues_ = self.eigenvalues_.astype('float')
        #         self.sig_uncalc_ = False
        #
        #     if(self.p_==1 or self.p_=='infty'):
        #         # calculate upper bounds for L_1 and L_infty Mahalanobis cost
        #         self.uu_ = [sum([[self.eigenmatrix_[d][d_] * self.C_[d_][m] for m in range(self.M_[d_])] for d_ in range(self.D_)], []) for d in range(self.D_)]
        #         ub1 = [np.sum([np.max([self.eigenmatrix_[d][d_] * self.C_[d_][m] for m in range(self.M_[d_])]) for d_ in range(self.D_)]) for d in range(self.D_)]
        #         ub2 = [np.sum([-np.min([self.eigenmatrix_[d][d_] * self.C_[d_][m] for m in range(self.M_[d_])]) for d_ in range(self.D_)]) for d in range(self.D_)]
        #         self.ubs_ = [np.max([ub1[d], ub2[d]]) for d in range(self.D_)]

        if(self.alpha_>0):
            if(pre_selection):
                if(n_neighbors_selection>0): self.n_neighbors_ = n_neighbors_selection
                self.proto_ = prototype_selection(X_reg, subsample=subsample)
                self.lof_sub_ = myLocalOutlierFactor(n_neighbors=self.n_neighbors_, p=self.p_neighbors_)
                self.lof_sub_ = self.lof_sub_.fit(X_reg[self.proto_])
                _, self.k_dists_, self.lrd_ = self.lof_sub_.get_params()
            else:
                self.proto_, self.k_dists_, self.lrd_ = self.lof_.get_params(subsample=subsample)
            X_reg = X_reg[self.proto_]
            self.N_reg_ = X_reg.shape[0]
            self.R_ = self.__getRegs(X_reg, self.lof_.weight_)        # R[n][d][i] = |x^(n)_{d} - a_{d,i}|
            self.R_ubs_ = [np.sum([ np.max(r[d]) for d in range(self.D_)]) for r in self.R_]

        if(self.interaction_):
            self.eigenvalues_, self.eigenmatrix_ = self.mahal_.lams_, self.mahal_.U_
            self.uu_ = [sum([[self.eigenmatrix_[d][d_] * self.C_[d_][m] for m in range(self.M_[d_])] for d_ in range(self.D_)], []) for d in range(self.D_)]
            ub1 = [np.sum([np.max([self.eigenmatrix_[d][d_] * self.C_[d_][m] for m in range(self.M_[d_])]) for d_ in range(self.D_)]) for d in range(self.D_)]
            ub2 = [np.sum([-np.min([self.eigenmatrix_[d][d_] * self.C_[d_][m] for m in range(self.M_[d_])]) for d_ in range(self.D_)]) for d in range(self.D_)]
            self.ubs_ = [np.max([ub1[d], ub2[d]]) for d in range(self.D_)]

        s = time.time()
        if(iterative):
            init_sol = []
            self.iterate_results_ = {'time':[], 'obj':[], 'action':[], 'explanation':[]}
            for k in range(1,K):
                prob = self.__getProblem(k, ActionSet, WorkingSet=WorkingSet, theta=theta, log_stream=log_stream, log_write=log_write)
                if(len(init_sol)!=0): prob.MIP_starts.add([list(range(len(init_sol))), init_sol], prob.MIP_starts.effort_level.auto)
                s_k = time.time()
                prob.solve()
                if(prob.solution.get_status()!=103):
                    self.iterate_results_['time'].append(time.time() - s_k)
                    self.iterate_results_['obj'].append(prob.solution.get_objective_value())
                    pi_k = self.__getPi(prob)
                    self.iterate_results_['action'].append(self.PiToAction(pi_k))
                    self.iterate_results_['explanation'].append(self.__getExplanation(self.iterate_results_['obj'][-1], self.iterate_results_['action'][-1], pi_k, self.iterate_results_['time'][-1], X_cost=X_cost, X_reg=X_reg))
                    init_sol = prob.solution.get_values()

        self.prob_ = self.__getProblem(K, ActionSet, WorkingSet=WorkingSet, theta=theta, log_stream=log_stream, log_write=log_write)
        if(iterative or len(init_sol)!=0):
            self.prob_.MIP_starts.add([list(range(len(init_sol))), init_sol], self.prob_.MIP_starts.effort_level.auto)
        self.prob_.solve()
        if(self.prob_.solution.get_status()==103):
            return None
        if(ret_init_sol):
            sol_for_init = self.prob_.solution.get_values()

        self.time_ = time.time() - s
        self.obj_ = self.prob_.solution.get_objective_value()
        pi = self.__getPi(self.prob_)
        action = self.PiToAction(pi)

        self.mahal_dist_ = self.mahal_.mahalanobis_dist(self.x_, action)
        self.lof_score_ = self.lof_.local_outlier_factor(action.reshape([1,self.D_]))[0]
        self.delta_ = action - self.x_
        self.explanation_ = self.__getExplanation(self.obj_, action, pi, self.time_)

        return action, sol_for_init if ret_init_sol else action

    def __getPi(self, prob):
        return [list(map(round, prob.solution.get_values(['pi_{}_{}'.format(d,m) for m in range(self.M_[d])]))) for d in range(self.D_)]

    def PiToAction(self, pi):
        return np.array([self.A_[d][pi[d].index(1)] for d in range(self.D_)])

    def __getNu(self, prob):
        return list(map(round, prob.solution.get_values(['nu_{}'.format(n) for n in range(self.N_reg_)])))

    def NuToNeighbor(self, nu):
        return np.where(nu==1)[0]

    def WeightedNorm(self, x, y, p=2):
        return np.linalg.norm((x-y)*self.weights_, ord=p)**p

    def __getExplanation(self, obj, action, pi, time, k=0):
        m_new = [p.index(1) for p in pi]
        supp = [d for d in range(self.D_) if self.m_[d]!=m_new[d]]
        flag = True
        explanation_ = '' if k==0 else 'Solution K = {0}. '.format(k)
        explanation_ += 'Predicted Class: {0} => {1} (Obj.: {2:.4}, Deci.Func.: {3:.4}, Time: {4:.4}'.format(self.y_, self.mdl_.predict([action])[0], obj, self.mdl_.decision_function([action])[0], time)
        explanation_ += ', Mahal.: {:.4}, {}-LOF: {:.4})\n'.format(self.mahal_dist_, self.lof_.n_neighbors_, self.lof_score_)
        i = 0
        while(i < len(supp)):
            d = supp[i]
            if(d in self.categories_flatten_):
                idx = self.features_[d].find(':')
                cat = self.features_[d][:idx]
                cat1 = self.features_[d][idx+1:]
                cat2 = self.features_[supp[i+1]][idx+1:]
                explanation_ += '\t * \'{}\': \'{}\' => \'{}\'\n'.format(cat, cat1, cat2) if self.x_[d]==1 else '\t * \'{0}\': \'{1}\' => \'{2}\'\n'.format(cat, cat2, cat1)
                i += 1
            else:
                explanation_ += '\t * \'{}\': {:.5} => {:.5} ({:+.5})\n'.format(self.features_[d], self.x_[d], action[d], action[d]-self.x_[d])
            i += 1
        return explanation_
# class LinearCEExtractor


