import numpy as np
from multiprocessing import Pool
import pickle

# Load data
with open('data/n_words.pkl', 'rb') as f:
    n_words = pickle.load(f)

with open('data/translation_len.pkl', 'rb') as f:
    translation_len = pickle.load(f)

with open('data/votes.pkl', 'rb') as f:
    votes = pickle.load(f)
    votes = np.array([sum(v) for v in votes.values()])

with open('data/n_translations.pkl', 'rb') as f:
    n_translations = pickle.load(f)

with open('data/true_likes.pkl', 'rb') as f:
    true_likes = pickle.load(f)

# usefull functions
def safelog(x):
    '''Safe log function.'''
    with np.errstate(divide='ignore'):
        return np.log(x)


class SlovotvirModel:
    '''Slovotvir model.'''
    def __init__(self, n_words, translation_len, votes, n_translations, a, b, t):
        self.cum_words = np.cumsum(n_words)
        self.translation_len = translation_len
        self.votes = votes
        self.n_translations = n_translations
        self.a = a # length bias
        self.b = b # frequency bias (translation selection)
        self.t = t # frequency bias (word selection)
        # Initialize words as a dictionary of numpy arrays for likes and lengths
        self.words = {i: [np.array([], dtype=int), np.array([], dtype=int)] 
                      for i in range(self.cum_words[-1])}

    def like_prob(self, lengths, likes):
        '''Calculates the probability of liking each translation.'''
        probs = np.exp(self.a * safelog(lengths) + self.b * safelog(likes))
        return probs / probs.sum()

    def run_epoch(self, epoch):
        '''Runs a single epoch of the model.'''
        n_translations = self.n_translations[epoch]
        votes = self.votes[epoch]
        cum_words = self.cum_words[epoch]

        # Distribution of translations across words
        translation_num = np.random.multinomial(n_translations, np.ones(cum_words) / cum_words)

        # Update the words with new translations
        indices = np.nonzero(translation_num)[0]
        for i in indices:
            n = translation_num[i]
            new_lengths = np.random.choice(self.translation_len, size=n)
            new_likes = np.zeros(n, dtype=int)
            self.words[i][0] = np.concatenate([self.words[i][0], new_likes])
            self.words[i][1] = np.concatenate([self.words[i][1], new_lengths])

        # Voting mechanism
        indices = np.array([i for i in range(cum_words) if len(self.words[i][0]) > 0])
        if len(indices) > 0:
            cum_likes = np.array([np.max(self.words[i][0]) + 1 for i in indices])
            cum_likes = np.exp(self.t * safelog(cum_likes))
            cum_likes = cum_likes / cum_likes.sum()
            chosen_indices = np.random.choice(indices,
                                              size=votes,
                                              p=cum_likes,
                                              replace=True)
            for idx in chosen_indices:
                lengths = self.words[idx][1]
                # laplace smoothing of likes:
                likes = self.words[idx][0] + 1
                probs = self.like_prob(lengths, likes)
                choice = np.random.choice(len(probs),
                                          p=probs)
                self.words[idx][0][choice] += 1 # adding a like

    def run(self, n_epochs):
        '''Runs the model for n_epochs.'''
        for epoch in range(n_epochs):
            self.run_epoch(epoch)

def run_model_instance(args):
    '''Run a single model instance.'''
    model, n_epochs = args
    model.run(n_epochs)
    return model

def refactor(model):
    '''Refactor the model results.'''
    # Extract values from the model
    word_values = list(model.words.values())

    # Concatenate likes
    likes = np.concatenate([vals[0] for vals in word_values])

    # Concatenate lengths
    lengths = np.concatenate([vals[1] for vals in word_values])

    # Sort and return results
    sorted_likes = likes[np.argsort(likes)[::-1]]
    sorted_lengths = lengths[np.argsort(likes)[::-1]]

    return sorted_likes, sorted_likes * sorted_lengths

def run_model_parallel(a, b, t, num_runs, batch_size=1000):
    '''Run the model in parallel for multiple instances.'''
    num_processes = 32  # Number of processes

    # Calculate number of batches
    num_batches = (num_runs + batch_size - 1) // batch_size

    results = []

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, num_runs)
  
        # Create model instances for the current batch
        models = [SlovotvirModel(n_words, translation_len, votes, n_translations, 
                                 a[i], b[i], t[i]) for i in range(start_idx, end_idx)]

        # Use multiprocessing to run models for the current batch
        with Pool(num_processes) as pool:
            batch_results = pool.map(run_model_instance, 
                                     [(model, len(n_words)) for model in models])

        results.extend([refactor(model) for model in batch_results])
        
    return results
