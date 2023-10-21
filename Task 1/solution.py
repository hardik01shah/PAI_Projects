import os
import typing
from sklearn.gaussian_process.kernels import *
from sklearn.cluster import KMeans
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.model_selection import train_test_split


# Set `EXTENDED_EVALUATION` to `True` in order to visualize your predictions.
EXTENDED_EVALUATION = False
EVALUATION_GRID_POINTS = 300  # Number of grid points used in extended evaluation

# Cost function constants
COST_W_UNDERPREDICT = 50.0
COST_W_NORMAL = 1.0

# GENERAL PARAMS
TRAIN_VAL_SPLIT = 0.1
PREDICTION_METHOD = 'local_gp' # one of ['single_gp', 'nystrom', 'local_gp']

# SINGLE_GP
SINGLE_GP_KERNEL = 3.0 * RBF(length_scale=0.5, length_scale_bounds=[0.001, 1.]) + WhiteKernel()
OPTIMIZER_RESTARTS = 3
EVAL_KERNEL = 18.1**2 * RBF(length_scale=0.0251) + WhiteKernel(noise_level=4.67) # do not optimize hyperparams while running eval to make it faster
EVAL_MODE = True

# NYSTROM APPROXIMATION
DIM_Q = 3000
SIGMA_N = np.sqrt(4.67)
RBF_INPUT_SCALE = 0.025
RBF_OUTPUT_SCALE = 3.0**2

# Clustering and Sampling for SingleGP and Nystrom
NUM_CLUSTERS = 7000
NUM_SAMPLES_PER_CLUSTER = 1
PLOT_SELECTED_CLUSTERS = False  # plot cluster centers
PLOT_SELECTED_SAMPLES = False  # plot selected samples
CLUSTER_OVER_Y = False  # if labels should be considered during clustering

# LOCAL GP
NUM_GP = 50
WEIGHTING_FACTOR = -12
LOCAL_GP_OPTIMIZER_RESTARTS = 3
# LOCAL_GP_KERNEL = ConstantKernel() * (RBF(length_scale=1., length_scale_bounds=[1e-2, 1e2]) + DotProduct() + WhiteKernel())
LOCAL_GP_KERNEL = 3.0 * RBF(length_scale=0.5, length_scale_bounds=[0.001, 100.]) + WhiteKernel()

# Asymmetric cost params
RESIDENTIAL_DEVIATION = 1.1

class Model(object):
    """
    Model for this task.
    You need to implement the fit_model and predict methods
    without changing their signatures, but are allowed to create additional methods.
    """

    def __init__(self):
        """
        Initialize your model here.
        We already provide a random number generator for reproducibility.
        """
        self.rng = np.random.default_rng(seed=0)

        # TODO: Add custom initialization for your model here if necessary
        self.n_gp = 50
        self.Km = KMeans(n_clusters=self.n_gp, init='random', random_state=0)
    
    def local_gp_fitting(self, train_y: np.ndarray, train_x_2D: np.ndarray):
        self.gpr = []
        self.n_gp = NUM_GP

        train_features, val_features, train_labels, val_labels = train_test_split(train_x_2D, train_y, test_size = TRAIN_VAL_SPLIT, random_state=0)

        self.Km = KMeans(n_clusters=self.n_gp, init='random', random_state=0, n_init='auto').fit(train_features)
        train_cls = self.Km.labels_

        train_mse = 0
        for i in range(self.n_gp):
            print(f'Fitting model: {i + 1}/ {self.n_gp}')
            
            model =  GaussianProcessRegressor(kernel=LOCAL_GP_KERNEL, random_state=0, n_restarts_optimizer=LOCAL_GP_OPTIMIZER_RESTARTS)
            model.fit( train_features[np.where(train_cls == i)[0]], train_labels[np.where(train_cls == i)[0]])
            
            train_predictions = model.predict(train_features[np.where(train_cls == i)[0]])
            train_mse += np.sum(np.square(train_predictions- train_labels[np.where(train_cls == i)[0]]))
            
            ll = model.log_marginal_likelihood()
            print(f"Log Likelihood: {ll}")
            self.gpr.append(model)

        train_mse /= train_features.shape[0]
        print(f'Train MSE:{train_mse}')

        val_predictions = self.local_gp_prediction(val_features)[0]
        val_mse = np.mean(np.square(val_predictions - val_labels))
        print(f'Val MSE:{val_mse}')

    def local_gp_prediction(self, test_x_2D: np.ndarray):

        gp_mean = np.zeros((test_x_2D.shape[0]))
        gp_std = np.zeros((test_x_2D.shape[0]))

        if WEIGHTING_FACTOR != 'INF':
            test_cls_whts = np.power(self.Km.transform(test_x_2D), WEIGHTING_FACTOR)
            for i in range(self.n_gp):
                print(f'Evaluating model: {i + 1}/ {self.n_gp}')
                
                test_mean_i, test_std_i = self.gpr[i].predict(test_x_2D, return_std=True)
                gp_mean += test_mean_i*test_cls_whts[:, i]
                gp_std += test_std_i*test_cls_whts[:, i]
            gp_mean /= np.sum(test_cls_whts, axis = 1)
            gp_std /= np.sum(test_cls_whts, axis = 1)
        
        else:
            for i in range(len(test_x_2D)):
                cur_sample = test_x_2D[i].reshape(1,-1)
                closest = np.argmin(np.linalg.norm(cur_sample - self.Km.cluster_centers_, axis=1))
                test_mean_i, test_std_i = self.gpr[closest].predict(cur_sample, return_std=True)
                gp_mean[i] = test_mean_i
                gp_std[i] = test_std_i
        return gp_mean, gp_std

    def clustering_and_sampling(self, train_y: np.ndarray, train_x_2D:np.ndarray):
        """
        Clustering of input data using K Means clustering.
        Choose NUM_SAMPLES_PER_CLUSTER nearest samples for each cluster centroid.
        """
        if CLUSTER_OVER_Y:
            datas = np.column_stack((train_x_2D, train_y))
        else:
            datas = train_x_2D

        print(f"Clustering input data using K-Means.")
        Km = KMeans(n_clusters=NUM_CLUSTERS, init='random', random_state=0, max_iter=600, n_init='auto').fit(datas)
        labels = Km.labels_
        centers = Km.cluster_centers_

        if PLOT_SELECTED_CLUSTERS:
            plt.scatter(train_x_2D[:,0], train_x_2D[:,1], label='train')
            plt.scatter(centers[:,0], centers[:,1], label='kmeans')
            plt.legend()
            plt.show()

        orig_indx = np.arange(len(datas)).reshape(-1,1)
        points = []
        val_points = []
        for i in range(np.max(labels)):
            indx = np.where(labels== i)[0]
            center = centers[i]
            
            if NUM_SAMPLES_PER_CLUSTER < len(datas[indx]):
                sel_indx = np.argpartition(np.linalg.norm(datas[indx] - center, axis = 1), NUM_SAMPLES_PER_CLUSTER)[:NUM_SAMPLES_PER_CLUSTER]
                n_sel_indx = np.argpartition(np.linalg.norm(datas[indx] - center, axis = 1), NUM_SAMPLES_PER_CLUSTER)[NUM_SAMPLES_PER_CLUSTER:]
                points.append(orig_indx[indx][sel_indx])
                val_points.append(orig_indx[indx][n_sel_indx])
            else:
                points.append(indx.reshape(-1,1))

        points = np.concatenate(points).reshape(-1)
        val_points = np.concatenate(val_points).reshape(-1)

        if PLOT_SELECTED_SAMPLES:
            plt.scatter(train_x_2D[:,0], train_x_2D[:,1], label='train')
            plt.scatter(train_x_2D[points,0], train_x_2D[points,1], label='selected train')
            plt.legend()
            plt.show()
            exit()

        return points, val_points

    def single_gp_fitting(self, train_y: np.ndarray, train_x_2D: np.ndarray):

        train_features, val_features, train_labels, val_labels = train_test_split(train_x_2D, train_y, test_size = TRAIN_VAL_SPLIT, random_state=0)
        sel_indx, non_sel_indx = self.clustering_and_sampling(train_labels, train_features)

        rem_train_features = train_features[non_sel_indx]
        rem_train_labels = train_labels[non_sel_indx]
        train_features = train_features[sel_indx]
        train_labels = train_labels[sel_indx]

        if EVAL_MODE:
            self.gpr = GaussianProcessRegressor(
                kernel=EVAL_KERNEL,
                random_state=0,
                optimizer=None).fit(train_features, train_labels)
        else:
            self.gpr = GaussianProcessRegressor(
                kernel=SINGLE_GP_KERNEL,
                random_state=0,
                n_restarts_optimizer=OPTIMIZER_RESTARTS).fit(train_features, train_labels)

        print(f"Log Likelihood: {self.gpr.log_marginal_likelihood()}")
        print(f"Kernel Params after Hyperparameter Optimization: {self.gpr.kernel_}")

        train_predictions = self.single_gp_prediction(train_features)[0]
        train_mse = np.mean(np.square(train_predictions - train_labels))
        print(f'Train MSE:{train_mse}')
        
        val_predictions = self.single_gp_prediction(val_features)[0]
        val_mse = np.mean(np.square(val_predictions - val_labels))
        print(f'Val MSE:{val_mse}')
    
    def single_gp_prediction(self, test_x_2D: np.ndarray):
        gp_mean, gp_std = self.gpr.predict(test_x_2D, return_std=True)
        return gp_mean, gp_std
    
    def get_RBF_kernel(self, x_i, x_j):
        indx_i, indx_j = np.meshgrid(
            np.arange(len(x_i)), np.arange(len(x_j)), indexing='ij')
        
        return RBF_OUTPUT_SCALE*np.exp(
            -0.5*(np.power(
                np.linalg.norm(x_i[indx_i] - x_j[indx_j], axis=2), 2)/(RBF_INPUT_SCALE**2)))

    def nystrom_fitting(self, train_y: np.ndarray, train_x_2D: np.ndarray):

        train_features, val_features, train_labels, val_labels = train_test_split(train_x_2D, train_y, test_size = TRAIN_VAL_SPLIT, random_state=0)
        sel_indx, non_sel_indx = self.clustering_and_sampling(train_labels, train_features)

        rem_train_features = train_features[non_sel_indx]
        rem_train_labels = train_labels[non_sel_indx]
        train_features = train_features[sel_indx]
        train_labels = train_labels[sel_indx]

        train_x_subset = train_features[np.random.randint(0,len(train_features),DIM_Q), :]
        print(f"Subset shape for Nystrom Approximation: {train_x_subset.shape}")

        K_nq = self.get_RBF_kernel(train_features, train_x_subset)
        K_q = self.get_RBF_kernel(train_x_subset, train_x_subset)
        K_qn = K_nq.T
        print(f"Estimating Kernel matrix")

        # We don't need K_hat, but (K_hat + sigma_n^2I)^-1 in closed form predictions for GP
        # K_hat = K_nq@np.linalg.inv(K_q)@K_qn
        # K_hat = K_nq . K_q^-1 . K_qn

        # ref: http://gpss.cc/gpss19/slides/Dai2019.pdf (slide 17)
        # Applying the Woodbury formula:
        # (A+UCV)^-1 = A^-1 + A^-1U(C^-1+VA^-1U)^-1VA^-1
        # A=sigma_n^2I, U=K_nq, C=K_q^-1, V=K_qn

        sigma2_i = (1/SIGMA_N**2)*np.eye(len(train_features))
        mid_term = (1/SIGMA_N**4)*np.linalg.pinv(K_q+((1/SIGMA_N**2)*(K_qn@K_nq)))
        self.K_hat_inv = sigma2_i - K_nq@mid_term@K_qn
        print(f"Finished estimating (K_hat+sigma_n^2)^-1")

        self.train_features = train_features
        self.K_hat_inv_y = self.K_hat_inv@train_labels

        train_predictions = self.nystrom_prediction(train_features)[0]
        train_mse = np.mean(np.square(train_predictions - train_labels))
        print(f'Train MSE:{train_mse}')
        
        val_predictions = self.nystrom_prediction(val_features)[0]
        val_mse = np.mean(np.square(val_predictions - val_labels))
        print(f'Val MSE:{val_mse}')

    def nystrom_prediction(self, test_x_2D: np.ndarray):
        k_test_x = self.get_RBF_kernel(test_x_2D, self.train_features)

        gp_mean = k_test_x@self.K_hat_inv_y
        gp_std = np.sqrt(np.diag(RBF_OUTPUT_SCALE - (k_test_x@self.K_hat_inv@(k_test_x.T))))

        return gp_mean, gp_std
    
    def make_predictions(self, test_x_2D: np.ndarray, test_x_AREA: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict the pollution concentration for a given set of city_areas.
        :param test_x_2D: city_areas as a 2d NumPy float array of shape (NUM_SAMPLES, 2)
        :param test_x_AREA: city_area info for every sample in a form of a bool array (NUM_SAMPLES,)
        :return:
            Tuple of three 1d NumPy float arrays, each of shape (NUM_SAMPLES,),
            containing your predictions, the GP posterior mean, and the GP posterior stddev (in that order)
        """

        # TODO: Use your GP to estimate the posterior mean and stddev for each city_area here
        gp_mean_init = np.zeros(test_x_2D.shape[0], dtype=float)
        gp_std_init = np.zeros(test_x_2D.shape[0], dtype=float)

        # TODO: Use the GP posterior to form your predictions here
        if PREDICTION_METHOD == 'single_gp':
            gp_mean, gp_std = self.single_gp_prediction(test_x_2D)
        elif PREDICTION_METHOD == 'nystrom':
            gp_mean, gp_std = self.nystrom_prediction(test_x_2D)
        elif PREDICTION_METHOD == 'local_gp':
            gp_mean, gp_std = self.local_gp_prediction(test_x_2D)
        else:
            raise ValueError(f"PREDICTION_METHOD should be in ['single_gp', 'nystrom', 'local_gp'].")

        assert gp_mean.shape == gp_mean_init.shape
        assert gp_std.shape == gp_std_init.shape

        predictions = gp_mean.copy()
        res_indx = np.where(test_x_AREA==1.)[0]
        predictions[res_indx] = gp_mean[res_indx] + RESIDENTIAL_DEVIATION*gp_std[res_indx]

        return predictions, gp_mean, gp_std

    def fitting_model(self, train_y: np.ndarray,train_x_2D: np.ndarray):
        """
        Fit your model on the given training data.
        :param train_x_2D: Training features as a 2d NumPy float array of shape (NUM_SAMPLES, 2)
        :param train_y: Training pollution concentrations as a 1d NumPy float array of shape (NUM_SAMPLES,)
        """

        # TODO: Fit your model here
        if PREDICTION_METHOD == 'single_gp':
            self.single_gp_fitting(train_y, train_x_2D)
        elif PREDICTION_METHOD == 'nystrom':
            self.nystrom_fitting(train_y, train_x_2D)
        elif PREDICTION_METHOD == 'local_gp':
            self.local_gp_fitting(train_y, train_x_2D)
        else:
            raise ValueError(f"PREDICTION_METHOD should be in ['single_gp', 'nystrom', 'local_gp'].")


# You don't have to change this function
def cost_function(ground_truth: np.ndarray, predictions: np.ndarray, AREA_idxs: np.ndarray) -> float:
    """
    Calculates the cost of a set of predictions.

    :param ground_truth: Ground truth pollution levels as a 1d NumPy float array
    :param predictions: Predicted pollution levels as a 1d NumPy float array
    :param AREA_idxs: city_area info for every sample in a form of a bool array (NUM_SAMPLES,)
    :return: Total cost of all predictions as a single float
    """
    assert ground_truth.ndim == 1 and predictions.ndim == 1 and ground_truth.shape == predictions.shape

    # Unweighted cost
    cost = (ground_truth - predictions) ** 2
    weights = np.ones_like(cost) * COST_W_NORMAL

    # Case i): underprediction
    mask = (predictions < ground_truth) & [bool(AREA_idx) for AREA_idx in AREA_idxs]
    weights[mask] = COST_W_UNDERPREDICT

    # Weigh the cost and return the average
    return np.mean(cost * weights)


# You don't have to change this function
def is_in_circle(coor, circle_coor):
    """
    Checks if a coordinate is inside a circle.
    :param coor: 2D coordinate
    :param circle_coor: 3D coordinate of the circle center and its radius
    :return: True if the coordinate is inside the circle, False otherwise
    """
    return (coor[0] - circle_coor[0])**2 + (coor[1] - circle_coor[1])**2 < circle_coor[2]**2

# You don't have to change this function 
def determine_city_area_idx(visualization_xs_2D):
    """
    Determines the city_area index for each coordinate in the visualization grid.
    :param visualization_xs_2D: 2D coordinates of the visualization grid
    :return: 1D array of city_area indexes
    """
    # Circles coordinates
    circles = np.array([[0.5488135, 0.71518937, 0.17167342],
                    [0.79915856, 0.46147936, 0.1567626 ],
                    [0.26455561, 0.77423369, 0.10298338],
                    [0.6976312,  0.06022547, 0.04015634],
                    [0.31542835, 0.36371077, 0.17985623],
                    [0.15896958, 0.11037514, 0.07244247],
                    [0.82099323, 0.09710128, 0.08136552],
                    [0.41426299, 0.0641475,  0.04442035],
                    [0.09394051, 0.5759465,  0.08729856],
                    [0.84640867, 0.69947928, 0.04568374],
                    [0.23789282, 0.934214,   0.04039037],
                    [0.82076712, 0.90884372, 0.07434012],
                    [0.09961493, 0.94530153, 0.04755969],
                    [0.88172021, 0.2724369,  0.04483477],
                    [0.9425836,  0.6339977,  0.04979664]])
    
    visualization_xs_AREA = np.zeros((visualization_xs_2D.shape[0],))

    for i,coor in enumerate(visualization_xs_2D):
        visualization_xs_AREA[i] = any([is_in_circle(coor, circ) for circ in circles])

    return visualization_xs_AREA

# You don't have to change this function
def perform_extended_evaluation(model: Model, output_dir: str = '/results'):
    """
    Visualizes the predictions of a fitted model.
    :param model: Fitted model to be visualized
    :param output_dir: Directory in which the visualizations will be stored
    """
    print('Performing extended evaluation')

    # Visualize on a uniform grid over the entire coordinate system
    grid_lat, grid_lon = np.meshgrid(
        np.linspace(0, EVALUATION_GRID_POINTS - 1, num=EVALUATION_GRID_POINTS) / EVALUATION_GRID_POINTS,
        np.linspace(0, EVALUATION_GRID_POINTS - 1, num=EVALUATION_GRID_POINTS) / EVALUATION_GRID_POINTS,
    )
    visualization_xs_2D = np.stack((grid_lon.flatten(), grid_lat.flatten()), axis=1)
    visualization_xs_AREA = determine_city_area_idx(visualization_xs_2D)
    
    # Obtain predictions, means, and stddevs over the entire map
    predictions, gp_mean, gp_stddev = model.make_predictions(visualization_xs_2D, visualization_xs_AREA)
    predictions = np.reshape(predictions, (EVALUATION_GRID_POINTS, EVALUATION_GRID_POINTS))
    gp_mean = np.reshape(gp_mean, (EVALUATION_GRID_POINTS, EVALUATION_GRID_POINTS))

    vmin, vmax = 0.0, 65.0

    # Plot the actual predictions
    fig, ax = plt.subplots()
    ax.set_title('Extended visualization of task 1')
    im = ax.imshow(predictions, vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(im, ax = ax)

    # Save figure to pdf
    figure_path = os.path.join(output_dir, 'extended_evaluation.pdf')
    fig.savefig(figure_path)
    print(f'Saved extended evaluation to {figure_path}')

    plt.show()


def extract_city_area_information(train_x: np.ndarray, test_x: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extracts the city_area information from the training and test features.
    :param train_x: Training features
    :param test_x: Test features
    :return: Tuple of (training features' 2D coordinates, training features' city_area information,
        test features' 2D coordinates, test features' city_area information)
    """
    train_x_2D = np.zeros((train_x.shape[0], 2), dtype=float)
    train_x_AREA = np.zeros((train_x.shape[0],), dtype=bool)
    test_x_2D = np.zeros((test_x.shape[0], 2), dtype=float)
    test_x_AREA = np.zeros((test_x.shape[0],), dtype=bool)

    #TODO: Extract the city_area information from the training and test features
    train_x_2D = train_x[:,:2]
    train_x_AREA = train_x[:,2]

    test_x_2D = test_x[: ,:2]
    test_x_AREA = test_x[: ,2]

    assert train_x_2D.shape[0] == train_x_AREA.shape[0] and test_x_2D.shape[0] == test_x_AREA.shape[0]
    assert train_x_2D.shape[1] == 2 and test_x_2D.shape[1] == 2
    assert train_x_AREA.ndim == 1 and test_x_AREA.ndim == 1

    return train_x_2D, train_x_AREA, test_x_2D, test_x_AREA

# you don't have to change this function
def main():
    # Load the training dateset and test features
    train_x = np.loadtxt('train_x.csv', delimiter=',', skiprows=1)
    train_y = np.loadtxt('train_y.csv', delimiter=',', skiprows=1)
    test_x = np.loadtxt('test_x.csv', delimiter=',', skiprows=1)

    # Extract the city_area information
    train_x_2D, train_x_AREA, test_x_2D, test_x_AREA = extract_city_area_information(train_x, test_x)
    # Fit the model
    print('Fitting model')
    model = Model()
    model.fitting_model(train_y,train_x_2D)

    # Predict on the test features
    print('Predicting on test features')
    predictions = model.make_predictions(test_x_2D, test_x_AREA)
    print(predictions)

    if EXTENDED_EVALUATION:
        perform_extended_evaluation(model, output_dir='.')


if __name__ == "__main__":
    main()
