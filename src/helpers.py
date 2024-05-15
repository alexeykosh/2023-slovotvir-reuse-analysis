import arviz as az
from tabulate import tabulate
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import lines

from bayesflow.amortizers import AmortizedPosterior
from bayesflow.networks import (InvertibleNetwork, 
                                DeepSet)
from bayesflow.trainers import (Trainer, 
                                SimulationDataset)
import tensorflow as tf

def generate_latex_table(ps, post_samples):
    '''
    Generate a latex table from the posterior samples of the parameters.
    '''
    table_data = []
    for i, p in enumerate(ps):
        row = [p, np.mean(post_samples[:, i]).round(3), 
               az.hdi(post_samples[:, i], hdi_prob=0.95).round(3)]
        table_data.append(row)

    table_headers = ['Parameter', 'Mean', 'HDI']
    latex_table = tabulate(table_data, headers=table_headers, tablefmt='latex')

    return latex_table

def configure_input(forward_dict, prior_means, prior_stds):
    """
    Function to configure the simulated quantities (i.e., simulator outputs)
    into a neural network-friendly (BayesFlow) format.
    """

    # Prepare placeholder dict
    out_dict = {}

    # Convert data to logscale
    logdata = np.log1p(forward_dict["sim_data"]).astype(np.float32)

    # Extract prior draws and z-standardize with previously computed means
    params = forward_dict["prior_draws"].astype(np.float32)
    # compute prior means and stds
    # prior_means = np.mean(params, axis=0)
    # prior_stds = np.std(params, axis=0)
    # z-standardize
    params = (params - prior_means) / prior_stds

    # Add to keys
    out_dict["summary_conditions"] = logdata
    out_dict["parameters"] = params

    out_dict["summary_conditions"] = np.expand_dims(out_dict["summary_conditions"], axis=2)

    return out_dict

def binning(likes):
    '''
    Binning of the likes distribution on a log scale.
    '''
    # Determine the number of bins based on the length of the likes array
    num_bins = int(np.ceil(np.log2(len(likes))))
    
    # Define bin edges at powers of 2 based on the indexes of likes
    bin_edges = 2 ** np.arange(num_bins + 1)
    
    # Assign values to bins
    bin_indices = np.digitize(np.arange(len(likes)), bin_edges)
    
    # Calculate the sum of values in each bin
    bin_sums = np.bincount(bin_indices, weights=likes, minlength=num_bins + 1)
    
    return bin_sums


def plot_posterior(post_samples_, param_names, true_values, save):
    '''
    Plot the posterior distribution of the parameters. 

    Parameters

    post_samples_: np.array
        The posterior samples of the parameters.
    
    param_names: list
        The names of the parameters for the figure. 
    
    true_values: list
        The true values of the parameters.
    
    save: str
        The name of the file to save the figure.
    '''
    num_params = len(param_names)
    fig, axs = plt.subplots(1, num_params, figsize=(10, 3.5))

    for i, param_name in enumerate(param_names):
        # Plot posterior density
        sns.kdeplot(post_samples_[:, i], ax=axs[i], color='blue')

        # Get KDE curve for each subplot
        x_values, y_values = sns.kdeplot(post_samples_[:, i], ax=axs[i], color='blue').lines[0].get_data()

        # Fill between specified points for KDE plot
        post_hdi = az.hdi(post_samples_[:, i], hdi_prob=0.95)
        axs[i].fill_between(x_values, y_values, where=(x_values >= post_hdi[0]) & (x_values <= post_hdi[1]), color='blue', alpha=0.3)

        # Plot prior density (assuming uniform for first two parameters and lognormal for the third)
        if i < 2:
            sns.kdeplot(np.random.uniform(0, 10, 10000), ax=axs[i], color='black', linestyle='--')
        else:
            sns.kdeplot(np.random.lognormal(0, sigma=0.5, size=10000), ax=axs[i], color='black', linestyle='--')

        # Plot true values
        axs[i].axvline(true_values[i], color='red', linestyle='--')

        if i == 0:
            axs[i].set_ylabel('Density')
        else:
            axs[i].set_ylabel('')

        # Set titles
        axs[i].set_title(param_name)

    # Custom legend handles
    posterior_line = lines.Line2D([], [], color='blue', linestyle='-')
    prior_line = lines.Line2D([], [], color='black', linestyle='--')

    # Add legend below the subplots
    plt.legend(handles=[posterior_line, prior_line],
               labels=['Posterior', 'Prior'],
               bbox_to_anchor=(-0.7, -0.2),
               loc='lower center', ncol=num_params,
               frameon=False)

    # Save to pdf
    if save:
        plt.savefig(f'figures/{save}.pdf', bbox_inches='tight', pad_inches=0)

    # Show the plot
    plt.show()


def train_and_amortize(train_data, batch_size, test_data, epochs, num_params, summary_dim, learning_rate, prior_means, prior_stds):
    '''
    Train and amortize given the simulation data using the BayesFlow framework.
    '''
    summary_net = DeepSet(summary_dim=summary_dim)
    inference_net = InvertibleNetwork(num_params=num_params, num_coupling_layers=4)
    amortizer = AmortizedPosterior(inference_net, summary_net, name="slovotvir_amortizer")
    
    # Define trainer
    trainer = Trainer(amortizer=amortizer, configurator=lambda x: configure_input(x, prior_means, prior_stds), memory=True)
    
    # Define learning rate schedule
    schedule = tf.keras.optimizers.schedules.CosineDecay(learning_rate, 
                                                         epochs * SimulationDataset(train_data, 
                                                                                    batch_size).num_batches, 
                                                        name = "lr_decay")
    optimizer = tf.keras.optimizers.legacy.Adam(schedule, global_clipnorm = 1)

    # Run training
    history = trainer.train_offline(simulations_dict=train_data, epochs=epochs, 
                                    batch_size=batch_size, optimizer=optimizer, 
                                    validation_sims=test_data,
                                    early_stopping=True, patience=20)
    
    # Save the trainer
    trainer._save_trainer("model")
    
    return history, trainer, amortizer