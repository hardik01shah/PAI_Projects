"""Solution."""
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
# import additional ...
from sklearn.gaussian_process.kernels import *
from sklearn.gaussian_process import GaussianProcessRegressor
from scipy.stats import norm

# global variables
DOMAIN = np.array([[0, 10]])  # restrict \theta in [0, 10]
SAFETY_THRESHOLD = 4  # threshold, upper bound of SA


# Added Paramters
KERNEL_F = Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=0.15**2)
KERNEL_V = Matern(length_scale=0.5, nu=2.5) + DotProduct(sigma_0=0, sigma_0_bounds='fixed') + WhiteKernel(noise_level=0.0001**2) + ConstantKernel(constant_value=4)
ACQUISITION_PENATLY_SCALE = 1.0
OPTIMIZE_GP = None # "fmin_l_bfgs_b"
BETA_V = 1.96
N_RESTARTS = 1


# TODO: implement a self-contained solution in the BO_algo class.
# NOTE: main() is not called by the checker.
class BO_algo():
    def __init__(self):
        """Initializes the algorithm with a parameter configuration."""
        # TODO: Define all relevant class members for your BO algorithm here.

        self.f_approx = GaussianProcessRegressor(kernel=KERNEL_F, optimizer=None)
        self.v_approx = GaussianProcessRegressor(kernel=KERNEL_V, optimizer=None)
        self.points = []
        self.f_obs = []
        self.v_obs = []

        pass

    def next_recommendation(self):
        """
        Recommend the next input to sample.

        Returns
        -------
        recommendation: float
            the next point to evaluate
        """
        # TODO: Implement the function which recommends the next point to query
        # using functions f and v.
        # In implementing this function, you may use
        # optimize_acquisition_function() defined below.

        return np.array([self.optimize_acquisition_function()]).reshape(1, -1)

    def optimize_acquisition_function(self):
        """Optimizes the acquisition function defined below (DO NOT MODIFY).

        Returns
        -------
        x_opt: float
            the point that maximizes the acquisition function, where
            x_opt in range of DOMAIN
        """

        def objective(x):
            return -self.acquisition_function(x)

        f_values = []
        x_values = []

        # Restarts the optimization 20 times and pick best solution
        for _ in range(20):
            x0 = DOMAIN[:, 0] + (DOMAIN[:, 1] - DOMAIN[:, 0]) * \
                 np.random.rand(DOMAIN.shape[0])
            result = fmin_l_bfgs_b(objective, x0=x0, bounds=DOMAIN,
                                   approx_grad=True)
            x_values.append(np.clip(result[0], *DOMAIN[0]))
            f_values.append(-result[1])

        ind = np.argmax(f_values)
        x_opt = x_values[ind].item()

        return x_opt

    def acquisition_function(self, x: np.ndarray):
        """Compute the acquisition function for x.

        Parameters
        ----------
        x: np.ndarray
            x in domain of f, has shape (N, 1)

        Returns
        ------
        af_value: np.ndarray
            shape (N, 1)
            Value of the acquisition function at x
        """
        x = np.atleast_2d(x)
        # TODO: Implement the acquisition function you want to optimize.

        f_pred, f_std = self.f_approx.predict(X = x, return_std = True)
        v_pred, v_std = self.v_approx.predict(X = x, return_std = True)

        vals = np.array(self.f_obs)
        max_point = np.max(vals)
        z = (f_pred-max_point)/f_std

        EI = f_std*(z*norm.cdf(z) + norm.pdf(z))
        score = EI - ACQUISITION_PENATLY_SCALE*max(0,(v_pred+(BETA_V*v_std)) - SAFETY_THRESHOLD)
        return score
    
        PF = norm.cdf((SAFETY_THRESHOLD-v_pred)/v_std)
        score = PF*EI
        
        return score

    def add_data_point(self, x: float, f: float, v: float):
        """
        Add data points to the model.

        Parameters
        ----------
        x: float
            structural features
        f: float
            logP obj func
        v: float
            SA constraint func
        """
        # TODO: Add the observed data {x, f, v} to your model.
        x = np.array([x]).reshape(DOMAIN.shape[0])
        self.points.append(x)
        self.f_obs.append(f)
        self.v_obs.append(v)

        self.f_approx.fit(X = np.array(self.points).reshape(-1, 1), y = np.array(self.f_obs))
        self.v_approx.fit(X = np.array(self.points).reshape(-1, 1), y = np.array(self.v_obs))

        return

    def get_solution(self):
        """
        Return x_opt that is believed to be the maximizer of f.

        Returns
        -------
        solution: float
            the optimal solution of the problem
        """
        # TODO: Return your predicted safe optimum of f.
        vals = np.array(self.f_obs)
        # constraints = np.array(self.v_obs) < SAFETY_THRESHOLD
        argmax_point = np.argmax(vals[np.array(self.v_obs) < SAFETY_THRESHOLD])

        return self.points[argmax_point]

    def plot(self, plot_recommendation: bool = True):
        """Plot objective and constraint posterior for debugging (OPTIONAL).

        Parameters
        ----------
        plot_recommendation: bool
            Plots the recommended point if True.
        """
        pass


# ---
# TOY PROBLEM. To check your code works as expected (ignored by checker).
# ---

def check_in_domain(x: float):
    """Validate input"""
    x = np.atleast_2d(x)
    return np.all(x >= DOMAIN[None, :, 0]) and np.all(x <= DOMAIN[None, :, 1])


def f(x: float):
    """Dummy logP objective"""
    mid_point = DOMAIN[:, 0] + 0.5 * (DOMAIN[:, 1] - DOMAIN[:, 0])
    return - np.linalg.norm(x - mid_point, 2)


def v(x: float):
    """Dummy SA"""
    return 2.0


def get_initial_safe_point():
    """Return initial safe point"""
    x_domain = np.linspace(*DOMAIN[0], 4000)[:, None]
    c_val = np.vectorize(v)(x_domain)
    x_valid = x_domain[c_val < SAFETY_THRESHOLD]
    np.random.seed(0)
    np.random.shuffle(x_valid)
    x_init = x_valid[0]

    return x_init


def main():
    """FOR ILLUSTRATION / TESTING ONLY (NOT CALLED BY CHECKER)."""
    # Init problem
    agent = BO_algo()

    # Add initial safe point
    x_init = get_initial_safe_point()
    obj_val = f(x_init)
    cost_val = v(x_init)
    agent.add_data_point(x_init, obj_val, cost_val)

    # Loop until budget is exhausted
    for j in range(20):
        # Get next recommendation
        x = agent.next_recommendation()

        # Check for valid shape
        assert x.shape == (1, DOMAIN.shape[0]), \
            f"The function next recommendation must return a numpy array of " \
            f"shape (1, {DOMAIN.shape[0]})"

        # Obtain objective and constraint observation
        obj_val = f(x) + np.random.randn()   # NOTE THERE WAS NO RANDOM, possible template error
        cost_val = v(x) + np.random.randn()   # NOTE THERE WAS NO RANDOM, possible template error
        agent.add_data_point(x, obj_val, cost_val)

    # Validate solution
    solution = agent.get_solution()
    assert check_in_domain(solution), \
        f'The function get solution must return a point within the' \
        f'DOMAIN, {solution} returned instead'

    # Compute regret
    regret = (0 - f(solution))

    print(f'Optimal value: 0\nProposed solution {solution}\nSolution value '
          f'{f(solution)}\nRegret {regret}\nUnsafe-evals TODO\n')


if __name__ == "__main__":
    main()
