import numpy as np
import gurobipy as gp
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

class OSCP:
    def __init__(self, pred_Y, real_Y, norm=2, tolerance=0.05, split_ratio=0.5, optTimeLimit=1000):
        '''
        Original dataset is a set of multi-dimensional time series. For each data at each time step, it is a vector of length d.
        Args:
            pred_Y (np.array, shape = (n_samples, time_steps, d)):
                Predicted Y values
            real_Y (np.array, shape = (n_samples, time_steps, d)):
                Real Y values
            norm (int, float, 'ellipsoid'):
                The norm to be used for calculating the distance between predicted and real Y values at each time step.
                Common options include 1, 2, np.inf, etc, with usage consistent with numpy.linalg.norm. For ellipsoid norm, set norm = 'ellipsoid'.
            tolerance (float):
                The error tolerance for the Conformal Prediction ("epsilon" in the paper).
            split_ratio (float):
                Ratio of the data to be used for first calibration dataset.
            optTimeLimit (int):
                Time limit (in seconds) for the optimization problem in stage 1. As the optimization problem is Mixed-Integer Linear Programming, it may take a long time to solve when the problem is huge. By setting a time limit, the solver will return the best solution found within the time limit.
        ---


        self.get_cp_radii(): 
            Returns the radius of CP region at each time step.
        self.visualize_2D_circular_conformal_region(pred_Y, real_Y, t):
            Visualization of CP region at specific time step, centered at pred_Y[t] (moved to origin).
        self.is_in_cp_region(pred_Y, real_Y):
            Returns True if real_Y lies within the CP region of pred_Y, False otherwise.
        self.emp_coverage(pred_Y, real_Y):
            Returns the empirical coverage of the CP region on the given dataset.
        '''
        self.pred_Y = pred_Y
        self.real_Y = real_Y
        self.tolerance = tolerance
        self.split_ratio = split_ratio 
        self.norm = norm  
        if len(self.pred_Y.shape) != 3:
            raise ValueError("pred_Y must have shape = (n_samples, time_steps, d). If d = 1, please reshape it to (n_samples, time_steps, 1)")
        elif self.pred_Y.shape != self.real_Y.shape:
            raise ValueError("pred_Y and real_Y must have the same shape")
        
        self.N, self.T, self.d = self.pred_Y.shape
        self.n1 = int(self.N * self.split_ratio)
        self.n2 = self.N - self.n1
        self.residual = np.zeros((self.N, self.T))  # residual form of data, i.e. the form is [[R_t^(i), R_t+1^(i), ...], ...]
        if self.norm == 'ellipsoid':
            self.covariance_mat_t = np.zeros((self.T, self.d, self.d))
            for t in range(self.T):
                self.covariance_mat_t[t] = self.fit_covariance_matrix(self.pred_Y[:self.n1, t] - self.real_Y[:self.n1, t])
            for i in range(self.N):
                for t in range(self.T):
                    # note that we do not sqrt the norm here
                    self.residual[i, t] = np.dot(np.dot((self.pred_Y[i, t] - self.real_Y[i, t]), np.linalg.inv(self.covariance_mat_t[t])), (self.pred_Y[i, t] - self.real_Y[i, t]))
        else:
            for i in range(self.N):
                for t in range(self.T):
                    self.residual[i, t] = np.linalg.norm(self.pred_Y[i, t] - self.real_Y[i, t], ord=self.norm)


        self.cp_radii = self.get_cp_radii(optTimeLimit=optTimeLimit)
    
    def fit_covariance_matrix(self, residuals):
        '''Fit the covariance matrix of the residuals using the sample covariance matrix'''
        covariance_matrix = np.cov(residuals, rowvar=False)
        return covariance_matrix
    
    def calc_area_1d(self, radii):
        '''this function currently only accepts norm = 'ellipsoid' or norm = 2'''
        if self.norm == 'ellipsoid':
            return self.calc_area(radii)

        area = sum(radii)

        return area
    def calc_area(self, radii):
        '''this function currently only accepts norm = 'ellipsoid' or norm = 2'''
        if self.norm == 'ellipsoid':
            return sum([self.ellipsoid_volume(self.covariance_mat_t[t],np.sqrt(radii[t])) for t in range(self.T)])
        area = sum([np.pi * r**2 for r in radii])


        return area

    def calc_area_3d(self, radii):
        '''this function currently only accepts norm = 'ellipsoid' or norm = 2'''
        if self.norm == 'ellipsoid':
            return self.calc_area(radii)
        area = sum([4 / 3.0 * np.pi * r**3 for r in radii])

        return area

    def ellipsoid_volume(self, cov, r):
        eigenvalues = np.linalg.eigvals(cov)
        eigenvalues = np.real(np.sort(eigenvalues)[::-1])
        eps = 1e-5

        num_r = np.sum(eigenvalues > eps)
        det_sigma = np.prod(eigenvalues[:num_r])
        constant_cd = np.pi**(num_r/2) / np.math.gamma(num_r/2 + 1)  # Volume constant for d-dimensional sphere
        volume = constant_cd * r**num_r * np.sqrt(det_sigma)
        return volume
    
    def get_computing_time(self):
        '''
        Returns:
            computing_time (float): The time taken to run the optimization problem in stage 1
        '''
        return self.computing_time
    
    def get_opt_radius_parameter(self, optTimeLimit, warm_start=True):
        '''
        Returns:
            radius_parameter (numpy.ndarray): The parameters r_t's in the scoring function
        '''
        p = int(np.ceil((self.n1 + 1)*(1-self.tolerance)))
        sum_of_residuals = np.sum(self.residual[:self.n1, :], axis=1) 
        sorted_sum_of_residuals = np.argsort(sum_of_residuals) 
        heuristic_b = sorted_sum_of_residuals[:p]  
        heuristic_r = np.zeros(self.T) 
        for t in range(self.T):
            heuristic_r[t] = np.max(self.residual[heuristic_b, t])  
        R_t_p= np.zeros(self.T) # the R_t^[p] at each t
        for t1 in range(self.T):
            residual_t1 = self.residual[:self.n1, t1]  # residual of the first n1 samples
            sorted_indices_t1 = np.argsort(residual_t1) 
            
            R_t_p[t1] = self.residual[sorted_indices_t1[p-1]][t1]

        # identify the redundant constraints sets s1, s2
        s1, s2 = [], [] # contain indices
        for ind in range(self.n1):
            if len(s1) >= p:
                return R_t_p, 0
            elif all([self.residual[ind][t] <= R_t_p[t] for t in range(self.T)]):
                s1.append(ind)
            elif all([self.residual[ind][t] > heuristic_r[t] for t in range(self.T)]):
                s2.append(ind)
        print(f"Total Indices: {self.n1} Redundant: Type-I: {len(s1)} Type-II: {len(s2)}")
        
        s = [i for i in range(self.n1) if i not in s1 and i not in s2]
        
        # set-up the optimization problem
        m = gp.Model()
        m.setParam('OutputFlag', 0) # disable Gurobi output; set to 1 to show the running log
        m.setParam('timelimit', optTimeLimit) 

        # set up the lower bound for r
        lb_r = np.zeros(self.T)
        if len(s1) != 0:
            for t in range(self.T):
                lb_r[t] = np.max(self.residual[s1, t])  
        r = m.addVars(self.T, lb=lb_r, vtype=gp.GRB.CONTINUOUS, name="r")

        b = m.addVars(len(s), vtype=gp.GRB.BINARY, name="b")
        m.update()

        # set up the objective function
        m.setObjective(gp.quicksum(r[t] for t in range(self.T)), gp.GRB.MINIMIZE)
        # set up the constraints
        m.addConstrs(b[i]*(self.residual[s[i]][t] - r[t])<=0 for t in range(self.T) for i in range(len(s)))
        m.addConstr(gp.quicksum(b[i] for i in range(len(s))) ==  p - len(s1))

        # add the heuristic solution as a warm start to speed up the optimization
        
        if warm_start:

            for t in range(self.T):
                r[t].start = heuristic_r[t]
            count = 0
            for i in range(len(s)):
                if s[i] in heuristic_b and count < p - len(s1):
                    count += 1
                    b[i].start = 1
                else:
                    b[i].start = 0

        m.optimize()
        
        computing_time = m.Runtime
        opt_r = np.zeros(self.T)
        for t in range(self.T):
            opt_r[t] = r[t].X
        
        return opt_r, computing_time


        

    def get_cp_radii(self, optTimeLimit):
        '''
        Returns:
            cp_radii (numpy.ndarray, shape = (time_steps, )): 
            The radius of CP region at each time step

        '''
        self.r, self.computing_time = self.get_opt_radius_parameter(optTimeLimit=optTimeLimit)
        
        p = int(np.ceil((self.n2 + 1)*(1-self.tolerance)))

        cp_radii = np.zeros(self.T)
        nonconformity_scores = np.zeros(self.n2)
        for ind in range(self.n1, self.N):
            nonconformity_scores[ind-self.n1] = np.max([(self.residual[ind][t] - self.r[t]) for t in range(self.T)])
        R_p = np.sort(nonconformity_scores)[p-1]
        
        for t in range(self.T):
            cp_radii[t] = self.r[t] + R_p
        return cp_radii
    


    def visualize_2D_circular_conformal_region(self, pred_Y, real_Y, t):
        '''
        Args:
            Pred_Y (numpy.ndarray, shape = (n_samples, time_steps, d) or (time_steps, d)):
                Predicted Y values
            Real_Y (numpy.ndarray, shape = (n_samples, time_steps, d) or (time_steps, d)):
                Real Y values
            t (int): time step
        ---   

        Visualization of CP region at specific time step, where the pred_Y are moved to the origin, and real_Y are adjusted accordingly
        to reflect the relative position between pred_Y and real_Y.
        \\
        REMARK: Currently this function can only visualize 2-D circle-shaped CP region, i.e. d = 2 and norm = 2
        '''
        if self.d != 2 or self.norm != 2:
            raise ValueError("Currently this function can only visualize 2-D circle-shaped CP region, i.e. d = 2 and norm = 2")
        if len(pred_Y.shape) != len(real_Y.shape):
            raise ValueError("Pred_Y and Real_Y must have the same shape")
        if len(pred_Y.shape) > 3 or len(pred_Y.shape) < 2:
            raise ValueError("Pred_Y and Real_Y must have shape = (time_steps, d) or (n_samples, time_steps, d)")
        # draw a circle with center at Pred_Y[t] and radius = cp_radii[t]
        # also plot the real_Y[t] on the same plot
        
        lower_lim_x, upper_lim_x, lower_lim_y, upper_lim_y = -self.cp_radii[t], self.cp_radii[t], -self.cp_radii[t], self.cp_radii[t]
        
        if len(real_Y.shape) == 3:
            for i in range(pred_Y.shape[0]):
                plt.scatter(real_Y[i, t, 0]-pred_Y[i, t, 0], real_Y[i, t, 1]-pred_Y[i, t, 1], color='red', label='Real Y' if i == 0 else "")
                lower_lim_x, upper_lim_x, lower_lim_y, upper_lim_y = \
                min(lower_lim_x, real_Y[i, t, 0]-pred_Y[i, t, 0]), max(upper_lim_x, real_Y[i, t, 0]-pred_Y[i, t, 0]), \
                min(lower_lim_y, real_Y[i, t, 1]-pred_Y[i, t, 1]), max(upper_lim_y, real_Y[i, t, 1]-pred_Y[i, t, 1])
        else:
            plt.scatter(real_Y[t, 0]-pred_Y[t, 0], real_Y[t, 1]-pred_Y[t, 1], color='red', label='Real Y')
            lower_lim_x, upper_lim_x, lower_lim_y, upper_lim_y = \
            min(lower_lim_x, real_Y[t, 0]-pred_Y[t, 0]), max(upper_lim_x, real_Y[t, 0]-pred_Y[t, 0]), \
            min(lower_lim_y, real_Y[t, 1]-pred_Y[t, 1]), max(upper_lim_y, real_Y[t, 1]-pred_Y[t, 1])
        circle = Circle((0, 0), self.cp_radii[t], color='green', alpha=0.5)
        plt.gca().add_artist(circle)
        plt.scatter(0, 0, color='blue', label='Predicted Y')
       
        
        plt.xlim(lower_lim_x, upper_lim_x)
        plt.ylim(lower_lim_y, upper_lim_y)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.title(f'CP Region at time step {t}, epsilon={self.tolerance}')
        plt.xlabel('Axis-1')
        plt.ylabel('Axis-2')
        plt.legend()
        plt.grid()
        plt.show()
        # plt.savefig(f'example.svg', dpi=300)

        

        
    def is_in_cp_region(self, pred_Y, real_Y):
        '''
        Args:
            Pred_Y (numpy.ndarray, shape = (time_steps, d) or (n_samples, time_steps, d)):
                Predicted Y values
            Real_Y (numpy.ndarray, shape = (time_steps, d) or (n_samples, time_steps, d)):
                Real Y values, must have the same shape as Pred_Y
        ---   
        Returns True if Real_Y lies within the CP region of Pred_Y, False otherwise.
        
        '''
        if len(pred_Y.shape) != len(real_Y.shape):
            raise ValueError("pred_Y and real_Y must have the same shape")
        if len(pred_Y.shape) == 2:
            if self.norm == 'ellipsoid':
                return all(self.ellipsoid_norm(pred_Y[t], real_Y[t], t) <= self.cp_radii[t] for t in range(self.T))
            return all(np.linalg.norm(pred_Y[t] - real_Y[t], ord=self.norm) <= self.cp_radii[t] for t in range(self.T))

        elif len(pred_Y.shape) == 3:
            results = np.zeros(pred_Y.shape[0], dtype=bool)
            for i in range(pred_Y.shape[0]):
                results[i] = self.is_in_cp_region(pred_Y[i], real_Y[i])
            return results

    def emp_coverage(self, pred_Y, real_Y):
        '''
        Args:
            pred_Y (numpy.ndarray, shape = (n_samples, time_steps, d) or (time_steps, d)):
                Predicted Y values
            real_Y (numpy.ndarray, shape = (n_samples, time_steps, d) or (time_steps, d)):
                Real Y values, must have the same shape as pred_Y
        ---   
        Returns:
            emp_coverage (float): Empirical coverage of CP region
        '''
        if len(pred_Y.shape) != len(real_Y.shape):
            raise ValueError("pred_Y and real_Y must have the same shape")
        if len(pred_Y.shape) != 3 and len(pred_Y.shape) != 2:
            raise ValueError("pred_Y and real_Y must have shape = (n_samples, time_steps, d) or (time_steps, d)")
        
        else:
            results = self.is_in_cp_region(pred_Y, real_Y)
            # each element in results is True if the corresponding sample trajectory is in all CP regions, False otherwise
            return np.mean(results)
    
    def average_radius(self):
        '''
        Returns:
            average_radius (float): Average radius of CP regions
        '''
        return np.mean(self.cp_radii)

if __name__ == "__main__":
    # calibration dataset (pred_Y is predicted result of some model)
    pred_Y = np.random.noncentral_chisquare(2, 555, (1000, 20, 2))  # shape = (n_samples, time_steps, d)
    real_Y = np.random.noncentral_chisquare(2, 555, (1000, 20, 2))  # shape = (n_samples, time_steps, d)

    # test set
    pred_Y_test = np.random.noncentral_chisquare(2, 555, (1000, 20, 2))  # shape = (n_samples, time_steps, d)
    real_Y_test = np.random.noncentral_chisquare(2, 555, (1000, 20, 2))  # shape = (n_samples, time_steps, d)

    # OSCP usage
    oscp = OSCP(pred_Y, real_Y)
    print("Radii of the CP-regions: ", oscp.cp_radii) # CP regions are in norm-ball shapes
    print("Prediction errors at all time-steps lie inside CP-regions: ", oscp.is_in_cp_region(pred_Y_test, real_Y_test))
    print("Empirical coverage on test-set: ", oscp.emp_coverage(pred_Y_test, real_Y_test))

    # Visualization (currently only support d=2 and norm=2)
    oscp.visualize_2D_circular_conformal_region(pred_Y_test, real_Y_test, 5) # visualize CP region at t = 5 (starts at 0)

            
