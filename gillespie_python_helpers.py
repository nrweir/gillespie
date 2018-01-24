"""Helper functions for Gillespie simulations in the Jupyter Notebook."""
# import dependencies
import numpy as np
import pandas as pd


def one_state_simulation(
        ksynth, kdecay, sim_arr=np.empty(shape=(1,)), t_final=240,
        checkpoint_freq=1, return_sim=False
        ):
    """Perform a one-state Gillespie simulation.

    Arguments:
    ----------
    ksynth : float
        a floating point number representing the synthesis rate constant for
        the simulation in units of minutes.
    kdecay : float
        a floating point number representing the decay rate constant for the
        simulation in units of minutes.
    sim_arr : 1D NumPy array, optional
        a NumPy array representing the starting state of the simulation to
        run. Each value corresponds to the age of one molecule. Defaults to a
        single "empty vessel" initialization, where no molecules exist at t=0.
    t_final : int, optional
        The simulated time in minutes at which point the simulation should end.
        Defaults to 240.
    checkpoint_freq : int, optional
        The amount of time, in minutes, between sampling values for output.
        Defaults to 1.
    return_sim : bool, optional
        Should the entire simulation series be returned upon completion?
        Defaults to no (False).

    Returns:
    --------
    A tuple (ct_checkpoints, age_checkpoints[, sim_df]).
    ct_checkpoints : 2D NumPy array
        An array where the first axis corresponds to a unique simulation and
        the second axis corresponds to simulation checkpoints. Each value
        corresponds to the number of molecules present in the simulation at the
        checkpoint.
    age_checkpoints : 2D NumPy array
        An array where the first axis corresponds to a unique simulation and
        the second axis corresponds to simulation checkpoints. Each value
        corresponds to the number of molecules present in the simulation at the
        checkpoint.
    sim_series : pandas Series
        The simulation Series where each value is a 1D NumPy array
        corresponding to the state of the simulation at a given checkpoint.
        Useful for plotting the distribution of molecule ages over the course
        of the simulation.

    """
    sim_series = pd.Series([np.copy(sim_arr)], index=[0])
    next_checkpt = checkpoint_freq
    t = 0
    while next_checkpt < t_final:  # until you reach the desired sim end
        decay_prob = sim_arr.size*kdecay  # N*kdecay
        tau = np.random.exponential(1/(ksynth + decay_prob))  # exp draw
        # the following function updates next_checkpt and the output data if
        # t+tau > than its curr value. it keeps doing so until t+tau <
        # next_checkpt.
        sim_series, next_checkpt = _recursive_update_output(
                sim_series, sim_arr, t, tau, next_checkpt, checkpoint_freq)
        sim_arr = sim_arr + tau  # age the molecules in the model
        # determine which event took place by drawing from binomial.
        # "positive" outcome (binomial returns 1) = add a new particle,
        # "negative" outcome (binomial returns 0) = remove a particle
        event = np.random.binomial(n=1, p=ksynth/(ksynth + decay_prob))
        if event == 1:  # if this random draw chose to add a particle
            sim_arr = np.append(sim_arr, 0)  # add a new molecule with age 0
        else:  # if the simulation chose to remove a particle
            # randomly select one index of sim_arr and remove it
            sim_arr = np.delete(sim_arr, np.random.randint(0, sim_arr.size))
        t = t + tau  # increment time
    ct_checkpoints = sim_series.apply(np.size).values
    age_checkpoints = sim_series.apply(np.mean).values
    if return_sim:
        return(ct_checkpoints, age_checkpoints, sim_series)
    else:
        return(ct_checkpoints, age_checkpoints)


def _recursive_update_output(sim_series, sim_arr, t, tau, next_checkpt,
                             checkpoint_freq):
    if t + tau > next_checkpt:  # if the next step will pass the checkpt
        sim_series = sim_series.append(pd.Series([np.copy(sim_arr)],
                                                 index=[next_checkpt]))
        next_checkpt += checkpoint_freq
        sim_series, next_checkpt = _recursive_update_output(
                sim_series, sim_arr, t, tau, next_checkpt, checkpoint_freq)
    return(sim_series, next_checkpt)
