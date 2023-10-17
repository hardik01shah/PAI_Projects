import os
import typing
from sklearn.gaussian_process.kernels import *
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from matplotlib import cm
from sklearn.model_selection import train_test_split


# Set `EXTENDED_EVALUATION` to `True` in order to visualize your predictions.
EXTENDED_EVALUATION = False
EVALUATION_GRID_POINTS = 300  # Number of grid points used in extended evaluation

# Cost function constants
COST_W_UNDERPREDICT = 50.0
COST_W_NORMAL = 1.0


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
        gp_mean = np.zeros(test_x_2D.shape[0], dtype=float)
        gp_std = np.zeros(test_x_2D.shape[0], dtype=float)

        # TODO: Use the GP posterior to form your predictions here
        predictions, gp_std = self.gpr.predict(test_x_2D, return_std=True)
        params = self.gpr.get_params()
        # print(params)
        gp_mean = predictions
        return predictions, gp_mean, gp_std

    def fitting_model(self, train_y: np.ndarray,train_x_2D: np.ndarray):
        """
        Fit your model on the given training data.
        :param train_x_2D: Training features as a 2d NumPy float array of shape (NUM_SAMPLES, 2)
        :param train_y: Training pollution concentrations as a 1d NumPy float array of shape (NUM_SAMPLES,)
        """

        # TODO: Fit your model here
        # RBF_kernel = 30.0 * RBF(length_scale=0.5)
        RBF_kernel = 1 * RBF(length_scale=0.1, length_scale_bounds=[1e-2, 1e2])
        Dot_kernel = DotProduct()
        Combibed_kernal = ConstantKernel() * (RBF(length_scale=1., length_scale_bounds=[1e-2, 1e2]) + DotProduct() + WhiteKernel())
        Combibed_kernal_v2 = DotProduct() + 1*RationalQuadratic(length_scale=0.1, alpha=1e-2)
        # self.gpr = GaussianProcessRegressor(kernel=Combibed_kernal_v2, random_state=0).fit(train_x_2D, train_y)
        self.gpr = GaussianProcessRegressor(kernel=Combibed_kernal, random_state=0).fit(train_x_2D, train_y)
        # self.gpr = GaussianProcessRegressor(kernel=RBF_kernel, random_state=0).fit(train_x_2D, train_y)
        # self.gpr = GaussianProcessRegressor(kernel=Dot_kernel, random_state=0).fit(train_x_2D, train_y)

        ll = self.gpr.log_marginal_likelihood()
        print(f"Log Likelihood: {ll}")
        # print(f"Kernel params: {RBF_kernel.get_params()}")
        pass

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
    train_x_2D = train_x[: ,:2]
    train_x_AREA = train_x[: ,2]

    
    test_x_2D = test_x[: ,:2]
    test_x_AREA = test_x[: ,2]

    assert train_x_2D.shape[0] == train_x_AREA.shape[0] and test_x_2D.shape[0] == test_x_AREA.shape[0]
    assert train_x_2D.shape[1] == 2 and test_x_2D.shape[1] == 2
    assert train_x_AREA.ndim == 1 and test_x_AREA.ndim == 1

    return train_x_2D, train_x_AREA, test_x_2D, test_x_AREA




def cluster_points(features, labels, n_clusters = 300):

    Km = KMeans(n_clusters=n_clusters, init='random', random_state=0).fit(np.column_stack((features, labels)))
    labels = Km.labels_
    points = []
    test_points = []
    for i in range(np.max(labels)):
        points.append(np.where(labels== i)[0][:10])
        test_points.append(np.where(labels== i)[0][10:])
    points = np.concatenate(points).reshape(-1)
    test_points = np.concatenate(test_points)

    return points, test_points

def induced_points(features, labels, num_points = 1000):

    Km = KMeans(n_clusters=num_points, init='random', random_state=0).fit(np.column_stack((features, labels)))
    labels = Km.labels_
    points = np.array(Km.cluster_centers_)
    features = points[:, :-1]
    
    labels = points[:, -1]
    
    return features, labels



def local_gp():
    # Load the training dateset and test features
    train_x = np.loadtxt('train_x.csv', delimiter=',', skiprows=1)
    train_y = np.loadtxt('train_y.csv', delimiter=',', skiprows=1)
    test_x = np.loadtxt('test_x.csv', delimiter=',', skiprows=1)

    # Extract the city_area information
    train_x_2D, train_x_AREA, test_x_2D, test_x_AREA = extract_city_area_information(train_x, test_x)

    # Local Gaussians
    n_gp = 15
    train_features, test_features, train_labels, test_labels = train_test_split(train_x_2D, train_y, test_size = 0.2, random_state=0)

    Km = KMeans(n_clusters=n_gp, init='random', random_state=0).fit(train_features)
    train_cls = Km.labels_
    test_cls_whts = Km.transform(test_features)

    train_mse = 0
    test_predictions_total = np.zeros((test_cls_whts.shape[0]))

    for i in range(n_gp):
        print(f'Fitting model: {i + 1}')
        
        model = Model()
        model.fitting_model(train_labels[np.where(train_cls == i)[0]], train_features[np.where(train_cls == i)[0]])
        
        train_predictions = model.make_predictions(train_features[np.where(train_cls == i)[0]], test_x_AREA)[0]
        train_mse += np.sum(np.square(train_predictions- train_labels[np.where(train_cls == i)[0]]))

        test_predictions_i = model.make_predictions(test_features, test_x_AREA)[0]
        test_predictions_total += test_predictions_i*test_cls_whts[:, i]

    train_mse /= train_features.shape[0]
    test_predictions_total /= np.sum(test_cls_whts, axis = 1)
    test_mse = np.mean(np.square(test_predictions_total - test_labels))

    print(f'Train MSE:{train_mse}')
    print(f'Test MSE: {test_mse}')


# you don't have to change this function
def main():
    # Load the training dateset and test features
    train_x = np.loadtxt('train_x.csv', delimiter=',', skiprows=1)
    train_y = np.loadtxt('train_y.csv', delimiter=',', skiprows=1)
    test_x = np.loadtxt('test_x.csv', delimiter=',', skiprows=1)

    # Extract the city_area information
    train_x_2D, train_x_AREA, test_x_2D, test_x_AREA = extract_city_area_information(train_x, test_x)


    # Clustering and extracting points
    points, test_points = cluster_points(train_x_2D, train_y)


    # Induced Points
    train_features, train_labels = induced_points(train_x_2D, train_y, num_points = 4000)

    # Fit the model
    print('Fitting model')
    model = Model()
    # model.fitting_model(train_labels, train_features)
    model.fitting_model(train_y[points], train_x_2D[points])

    # Predict on the test features
    print('Predicting on train features')
    # predictions = model.make_predictions(train_x_2D[points], test_x_AREA)
    predictions = model.make_predictions(train_x_2D[points], test_x_AREA)
    print(f'MSE {np.mean(np.square(predictions[0]- train_y[points]))}')
    
    print('Predicting on test features')
    predictions = model.make_predictions(train_x_2D[test_points], test_x_AREA)
    print(f'MSE {np.mean(np.square(predictions[0]- train_y[test_points]))}')

    if EXTENDED_EVALUATION:
        perform_extended_evaluation(model, output_dir='.')


if __name__ == "__main__":
    local_gp()
    # main()
