import warnings
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA
from numpy import random
from numpy.linalg import inv
from pomegranate import (
    DiscreteDistribution,
    GammaDistribution,
    GeneralMixtureModel,
    IndependentComponentsDistribution,
    MultivariateGaussianDistribution,
)
from scipy.stats import invgamma, multivariate_normal, norm

warnings.simplefilter(action="ignore", category=FutureWarning)
np.random.seed(30)

# __all__ = ["SSDetector", "main_online_ss_detection"]
__all__ = ["SSDetector"]


class SSDetector:
    """
    To use:
        (1) Build Model Object - model = SSDetector() # passing parmas is optional
        (2) First call Train on your model - model.train(data) # pass in your time series of interest
        (3) Visualize the results using any of the plotting functionality
        (4) If recieving new arguments (i.e. new observations) you can call the model on the new data point as so
            model(data) # pass in complete data history including the current observation, and it will return the result
            as a tuple
            return p_steady, y_predicted

    Some Notes:
        * Increasing the number of particles increases the computational cost
        * state_transition_probability should be in range [0.1, 0.5]
        * you can change the priors but there is really no need given how robust this model is
        * n_timeless determines for how many particles do we move the discrete component at time t.
        Keep as number_particles / 100.
        * The detection threashold is up to the user
        * slope_threashold should stay around the default value. Ex. the user could go as low as 0.001 or as high as
        0.1.
    """

    def __init__(
        self,
        slope_threashold: float = 0.003,
        detection_threashold: float = 0.9,
        state_transition_probability: float = 0.4,
        number_particles: int = 2000,
        n_timeless: int = 40,
    ):

        # Init model attributes
        self.__saved_args = locals()
        self.pt_threashold = detection_threashold  # if Pt is above this then we are in steady state
        self.__s0 = slope_threashold  # if abs(regression line slope) < this for particle i we are in steady state
        self.__p = state_transition_probability  # mixing coeffcient for mixture model f(x)
        self.__n_particles = number_particles  # number of particles to use for the filter
        self.__n_prime = n_timeless  # number of __n1 sampled particles to move tau to time t
        self.__weights_t = np.ones(self.__n_particles) / self.__n_particles  # particle weights at current time
        self.__weights_tm1 = None  # previous particle weightings
        self.__mu_0 = np.array(
            [0, 0]
        )  # a prior on the mean of the regression coeffcients beta = (at/slope, bt/intercept)
        self.__sigma_0 = prior_covariance = 100 * np.identity(
            2
        )  # a prior on the variance of the regression coeffcients beta = (at/slope, bt/intercept)
        self.alpha0 = 10  # a prior on the observation noise
        self.alpha1 = 0.1

        self.__n0 = int(
            self.__n_particles * self.__p
        )  # calcualte deterministic sample sizes for stratified resampling method
        self.__n1 = self.__n_particles - self.__n0
        self.__C = self.__n_particles * 10  # To create the subset for Gibbs move
        self.__xt_group0 = np.zeros(
            (self.__n0, 4)
        )  # init 1st deterministic sample group. This comes from the mixture distribution f(x)
        self.xt = np.zeros((self.__n_particles, 4))  # init state
        self.__xt_group1 = np.zeros((self.__n1, 4))
        self.__state_posterior = None
        self.__alpha0_posterior = None
        self.__alpha1_posterior = None
        self.__history: List
        self.__history = []

    @property
    def prev_particle_states(self):
        return self.__xtm1

    @property
    def prev_particle_weights(self):
        return self.__weights_tm1

    @property
    def current_particle_states(self):
        return self.xt

    @property
    def current_particle_weights(self):
        return self.__weights_t

    @property
    def regression_priors(self):
        return self.__mu_0, self.__sigma_0, self.alpha0, self.alpha1

    @property
    def train_hist(self):
        return self.__history

    @property
    def model_params(self):
        return self.__saved_args

    def train(self, data: np.ndarray, show_results: bool = False):
        """
        The model training loop. Just need to pass the historical data and the model handles the rest. Assuming we are
        starting from t = 1 i.e. the first observation. The model does not require a uniform or filtered time series to
        perform. However, it is expected that some treatment can improve results.
        """
        self.p_steady = np.zeros(len(data))
        self.y_hat = np.zeros(len(data))
        self.__init_state()
        weights = self.__likelihood(self.xt, data[0], 1)
        weights_norm = weights / weights.sum()
        self.__resample_basic(weights_norm)
        self.__estimation(1)
        self.__xtm1 = self.xt
        self.__weights_tm1 = self.__weights_t
        if show_results:
            print("t: ", 1, "\ny_hat: ", self.y_hat[1], "\nPt: ", self.p_steady[1])
        for t1 in range(2, len(data)):
            # (1 - a) Sample particles from the proposal distribution f(x)
            self.__sample_importance_density(t1)
            self.__resample_stratified_deterministic()  # resample weights deterministic stratified sampling algorithm
            # (1 - b) Timeliness Improvement Strategy | Randomly select n particles from the __n1 particles resampled
            # from the posterior
            random_particles = np.random.choice(self.__n1, self.__n_prime, replace=False)
            self.__xt_group1[random_particles, 3] = random.randint(1, t1, len(random_particles))  # move tau
            # (2) Compute normalizwed weights and state according to Algorithm 2 in source paper
            weights_norm = self.__normalize(data[t1 - 1], t1)
            self.xt = np.concatenate(
                (self.__xt_group0, self.__xt_group1), axis=0
            )  # concatenate both particle groups into one state space
            # (3) Generic Particle Filter Resampling step
            self.__resample_basic(weights_norm)
            # (4) Compute Targets | Pt and y_hat
            self.__estimation(t1)
            if show_results:
                print("\rt:", t1, "y_hat:", self.y_hat[t1], "Pt: ", self.p_steady[t1], end="")
            self.__xtm1 = self.xt  # set particle posterior
            self.__weights_tm1 = self.__weights_t  # set posterior weights
            # (5) Gibbs Move | Generate the Epsilon_t posterior
            self.__gibbs_move(data, t1)
            # (6) Update Priors | prior_t = posterior_tm1
            self.__update()
        self.__t = len(data)

    def __call__(self, data: np.ndarray, store: bool = False):
        """
        If we recieve a data point and do not want to retrain from the beggining use this function. Must pass the entire
        data history for now b/c we do not know when the latest change point was.
        """
        t = self.__t + 1  # t is the next time step after the final training time
        # (1 - a) Sample particles from the proposal distribution f(x)
        self.__sample_importance_density(t)
        self.__resample_stratified_deterministic()  # resample weights deterministic stratified sampling algorithm
        # (1 - b) Timeliness Improvement Strategy | Randomly select n particles from the __n1 particles resampled
        # from the posterior
        random_particles = np.random.choice(self.__n1, self.__n_prime, replace=False)
        self.__xt_group1[random_particles, 3] = random.randint(1, t, len(random_particles))  # move tau
        # (2) Compute normalizwed weights and state according to Algorithm 2 in source paper
        weights_norm = self.__normalize(data[t - 1], t)
        self.xt = np.concatenate(
            (self.__xt_group0, self.__xt_group1), axis=0
        )  # concatenate both particle groups into one state space
        # (3) Generic Particle Filter Resampling step
        self.__resample_basic(weights_norm)
        # (4) Compute Targets | Pt and y_hat
        pt, yhat = self.__results(t)
        self.__xtm1 = self.xt  # set particle posterior
        self.__weights_tm1 = self.__weights_t  # set posterior weights
        if store:
            self.__history.append(self.xt)  # Save state history
        # (5) Gibbs Move | Generate the Epsilon_t posterior
        self.__gibbs_move(data, t)
        # (6) Update Priors | prior_t = posterior_tm1
        self.__update()
        return pt, yhat

    def plot_detection_proba(self, data: np.ndarray, y_label="Flow Rate"):
        """
        Plot the detection probabilities determined in training vs the time series
        """
        plt.rcParams["figure.figsize"] = (20, 10)
        z = np.arange(0, self.__t, 1)  # time scale
        fig, ax1 = plt.subplots()
        color = "tab:red"
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Pt", color=color)
        ax1.plot(z, self.p_steady[0 : len(z)], color=color)  # noqa: E203
        ax1.tick_params(axis="y", labelcolor=color)

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

        color = "tab:blue"
        ax2.set_ylabel(y_label, color=color)  # we already handled the x-label with ax1
        ax2.plot(z, data[: self.__t], color=color)
        ax2.tick_params(axis="y", labelcolor=color)

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.title("Steady State Detection Over the Time Series ")
        plt.show()

    def plot_y_hat(self, data, y_label="Flow Rate"):
        """
        Plot the predicted regression values vs the true time series
        """
        plt.plot(self.y_hat[3:], "--", linewidth=2)
        plt.plot(data[3:])
        plt.ylabel(y_label)
        plt.xlabel("Time (s)")
        plt.title("Regression Fit to Time Series")
        plt.show()

    def __init_state(self):
        """
        Function to take the first particle samples from the importance density function f(x)
        """
        model = self.__importance_density_init()  # Step 1 | Sample Particles
        samples = model.sample(self.__n_particles)
        self.xt[:, [0, 1]] = np.row_stack(samples[:, 0])  # set slope and bias components
        self.xt[:, 2] = np.clip(1 / samples[:, 1], 0, 100)  # set variance component
        self.xt[:, 3] = samples[:, 2]  # set tau component

    def __estimation(self, t):
        """
        Estimate the steady state probability and the predicted observation value from the regression.
        Used during training.
        """
        self.p_steady[t] = self.__P_steady()
        self.y_hat[t] = self.__estimated_observation(t)

    def __results(self, t):
        """
        Estimate the steady state probability and the predicted observation value from the regression.
        Used during __call__ method returns the values at time t not a vector
        """
        return self.__P_steady(), self.__estimated_observation(t)

    def __importance_density_init(self):
        """
        Samples the initial importance density for time t = 1 during training.
        """
        xtm1 = IndependentComponentsDistribution(
            [
                MultivariateGaussianDistribution(self.__mu_0, self.__sigma_0),
                GammaDistribution(self.alpha0, self.alpha1),
                DiscreteDistribution.from_samples([1]),
            ]
        )

        epsilon = IndependentComponentsDistribution(
            [
                MultivariateGaussianDistribution(self.__mu_0, self.__sigma_0),
                GammaDistribution(self.alpha0, self.alpha1),
                DiscreteDistribution.from_samples([1]),
            ]
        )

        return GeneralMixtureModel([xtm1, epsilon], weights=[1 - self.__p, self.__p])

    def __sample_importance_density(self, t):
        """
        Sample the importance density for time t > 1. This is a mixture of the previous state of the resampled particles
        and the resampled particles updated in Gibbs move.
        """
        xtm1 = IndependentComponentsDistribution(
            [
                MultivariateGaussianDistribution.from_samples(self.__xtm1[:, 0:2]),
                GammaDistribution(self.alpha0, self.alpha1),
                DiscreteDistribution.from_samples(self.__xtm1[:, 3]),
            ]
        )

        epsilon = IndependentComponentsDistribution(
            [
                MultivariateGaussianDistribution(self.__mu_0, self.__sigma_0),
                GammaDistribution(self.alpha0, self.alpha1),
                DiscreteDistribution.from_samples([t]),
            ]
        )

        model = GeneralMixtureModel(
            [xtm1, epsilon], weights=[1 - self.__p, self.__p]
        )  # weigh each distribution accordingly

        samples = model.sample(len(self.__xt_group0))  # sample from f(x) our mixture model
        self.__xt_group0[:, [0, 1]] = np.row_stack(samples[:, 0])
        self.__xt_group0[:, 2] = np.clip(1 / samples[:, 1], 0, 100)
        self.__xt_group0[:, 3] = samples[:, 2]

    def __P_steady(self):
        """
        Calculates the steady state detection probability given the particle state and weights at time t.
        If the absolute value of the slope parameter corresponding to particle i < the preset threashold then add else
        dont add.
        """
        idx = np.where(abs(self.xt[:, 0]) < self.__s0)  # check against slope threashold
        return (self.__weights_t[idx]).sum()

    def __estimated_observation(self, time):  # predicted observation value
        """
        Estimate the observation given the regression line.
        """
        return (self.__weights_t.squeeze(axis=1) * (self.xt[:, 0] * time + self.xt[:, 1])).sum()

    def __lemma1(self, state_t, observations, var_t, time, LCP):
        """
        Function to update generate the posterior distributions of the state variables. Only updates slope, bias, and
        variance components.
        """
        mu_t = inv((state_t.T @ state_t) / var_t + inv(self.__sigma_0)) @ (
            (state_t.T @ observations) / var_t + (inv(self.__sigma_0) @ self.__mu_0)
        )
        cov_t = inv((state_t.T @ state_t) / var_t + inv(self.__sigma_0))
        beta_new = multivariate_normal(mu_t, cov_t, allow_singular=True).rvs()
        new_alpha0 = self.alpha0 + ((time - LCP + 1) / 2)
        new_alpha1 = self.alpha1 + (LA.norm(observations - (state_t @ beta_new)) / 2)
        var_t_new = invgamma(new_alpha0, new_alpha1).rvs()
        return beta_new, var_t_new, new_alpha0, new_alpha1

    def __resample_basic(self, weights_normalized):
        """
        The basic resampling step. Throw away particles with really low weights and replace them with samples of the
        particles with high weights.
        """
        N = len(self.xt)
        cumulative_sum = np.cumsum(weights_normalized)
        cumulative_sum[-1] = 1.0  # avoid round-off error
        indexes = np.clip(np.searchsorted(cumulative_sum, np.random.randn(N)) - 1, 0, N - 1)
        indexes.dtype = np.int
        self.__resample_from_index(indexes, weights_normalized)

    def __resample_from_index(self, indexes, weights_normalized):
        """
        Reset state and weights at time t given the resampled indicies.
        """
        particles = self.xt[indexes]
        weights = weights_normalized[indexes]
        weights.fill(1 / len(weights))
        self.__weights_t = weights
        self.xt = particles

    def __resample_stratified_deterministic(self):
        """
        Resample step at time t > 1. Useful for preventing particle impoverisment.
        """
        idx = self.__stratified_resample()
        self.__xt_group1 = np.copy(self.__xtm1[idx, :])

    def __stratified_resample(self):
        """
        Perform the stratified resample algorithm.
        """
        N = self.__n1
        weights__ = np.copy(self.__weights_tm1)
        positions = (np.random.randn(N) + range(N)) / N
        id_ = np.zeros(self.__n1, "i")
        cumulative_sum = np.cumsum(weights__)
        i, j = 0, 0
        while i < N - 1 and j < len(cumulative_sum):
            if positions[i] < cumulative_sum[j]:
                id_[i] = j
                i += 1
            else:
                j += 1
        return id_

    def __normalize(self, observation, time_t):
        """
        Normalize the particle samples according to the weighting schema presented in the source paper.
        Compute the weights for each particle grouping at normalize.
        """
        weights_g0 = self.__likelihood(self.__xt_group0, observation, time_t)
        weights_g1 = self.__likelihood(self.__xt_group1, observation, time_t)
        nume_g0 = self.__n1 * weights_g0 * self.__p
        nume_g1 = self.__n0 * (1 - self.__p) * weights_g1
        denom = self.__n1 * self.__p * weights_g0.sum() + self.__n0 * (1 - self.__p) * weights_g1.sum()
        if denom <= 0:
            denom = 1
        group0 = nume_g0 / denom
        group1 = nume_g1 / denom
        return np.concatenate((group0, group1), axis=0)

    def __likelihood(self, state, observation, time):
        """
        Function to determine the weights of the particles at time t. Basically, given the parameters for each particle
        how likely is it that its params predict the true yt (observation) value.
        """
        y_hat = state[:, [0, 1]] @ [time, 1]
        weights = norm(abs(y_hat - observation), np.sqrt(state[:, 2])).rvs()
        return np.expand_dims(weights, axis=1)

    def __gibbs_move(self, data, t):
        """
        Step (4) of the particle filtering algorithm. Take a subset of particles such according to the d_t condition
        otherwise this function can get quite computationally expensive. This updates the posterior distributions for
        the 2nd mixing component to sample from.
        """
        state = np.copy(self.xt)
        sample_order = np.random.choice(
            self.__n_particles, self.__n_particles, replace=False
        )  # Create a random order of particles to sample
        counter = 0
        d_t = 0
        alpha_0 = []
        alpha_1 = []

        while d_t <= self.__C and counter < self.__n_particles:  # This is how all the variables are connected
            particle = sample_order[counter]
            lcp_particle = int(state[particle, 3])
            vec = np.arange(lcp_particle, t + 1)
            vec = vec.reshape((len(vec), 1))
            state_ = np.concatenate((vec, np.ones((len(vec), 1))), axis=1)
            beta, var, alpha0_, alpha1_ = self.__lemma1(
                state_, data[lcp_particle - 1 : t], state[particle, 2], t, lcp_particle  # noqa: E203
            )  # move posteriors, return invgamma params
            state[particle, 0] = beta[0]
            state[particle, 1] = beta[1]
            state[particle, 2] = var  # set
            d_t += t - lcp_particle + 1
            counter += 1  # So we dont sample more than all the particles
            alpha_0.append(alpha0_)
            alpha_1.append(alpha1_)

        self.__state_posterior = [state[:, 0:2].mean(axis=0), np.cov(state[:, 0:2].T)]
        self.__alpha0_posterior = np.mean(alpha_0)
        self.__alpha1_posterior = np.mean(alpha_1)

    def __update(self):
        """
        Posteriors become the priors for the next iteration.
        """
        self.__mu_0 = self.__state_posterior[0]
        self.__sigma_0 = self.__state_posterior[1]
        self.alpha0 = self.__alpha0_posterior
        self.alpha1 = self.__alpha1_posterior
