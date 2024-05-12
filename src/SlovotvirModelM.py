import numpy as np
from multiprocessing import Pool
import pickle

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


class SlovotvirModel:
    '''Slovotvir model.'''
    def __init__(self, n_words, translation_len, votes, n_translations, a, b, t):
        self.cum_words = np.cumsum(n_words)
        self.translation_len = translation_len
        self.votes = votes
        self.n_translations = n_translations
        self.a = a
        self.b = b
        self.t = t
        # Initialize words as a dictionary of numpy arrays for likes and lengths
        self.words = {i: [np.array([], dtype=int), np.array([], dtype=int)] 
                      for i in range(self.cum_words[-1])}

    @staticmethod
    def min_max_scaling_numpy(data, new_min=1, new_max=2):
        '''Scales data to a new range.'''
        if data.size == 0:
            return data
        original_min = np.min(data)
        original_max = np.max(data)

        if original_min == original_max or new_min == new_max:
            return data
        scaled_data = ((data - original_min) / (original_max - original_min)) \
            * (new_max - new_min) + new_min
        return scaled_data
    
    @staticmethod
    def adjust_entropy(prob_distribution, temperature):
        '''Adjusts entropy of a probability distribution.'''
        temperature = max(1e-3, temperature)
        scaled_probs = np.power(prob_distribution, 1 / temperature)
        adjusted_distribution = scaled_probs / np.sum(scaled_probs)

        return adjusted_distribution

    def like_prob(self, lengths, likes):
        """Calculates the probability of liking each translation."""
        probs = lengths ** self.a + likes ** self.b
        return probs / probs.sum()

    def run_epoch(self, epoch):
        """Runs an epoch."""
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
            chosen_indices = np.random.choice(indices, size=votes, 
                                              p=self.adjust_entropy(cum_likes / cum_likes.sum(), 
                                                                    temperature=self.t), 
                                              replace=True)

            for idx in chosen_indices:
                lengths = self.min_max_scaling_numpy(1 / self.words[idx][1])  # Simplified inverse lengths
                likes = self.min_max_scaling_numpy(self.words[idx][0])
                probs = self.like_prob(lengths, likes)
                choice = np.random.choice(len(probs), p=probs)
                self.words[idx][0][choice] += 1

    def run(self, n_epochs):
        """Runs the model for a number of epochs."""
        for epoch in range(n_epochs):
            self.run_epoch(epoch)

def run_model_instance(args):
    """Function to run a model instance, used for multiprocessing."""
    model, n_epochs = args
    model.run(n_epochs)
    return model

def refactor(model):
    """Refactor the model results for comparison."""
    likes = np.concatenate(list(model.words.values()), axis=1)[0]
    return likes[np.argsort(likes)[::-1]]

# def run_model_parallel(a, b, t, num_runs):
#     """Run the model in parallel."""

#     num_processes = 32  # Number of processes

#     # Create model instances to run
#     models = [SlovotvirModel(n_words, translation_len, votes, n_translations, 
#                              a[_], b[_], t[_]) for _ in range(num_runs)]

#     # Use multiprocessing to run models
#     with Pool(num_processes) as pool:
#         results = pool.map(run_model_instance, [(model, len(n_words)) for model in models])

#     return [refactor(model) for model in results]

def run_model_parallel(a, b, t, num_runs, batch_size=1000):
    """Run the model in parallel with batch processing."""
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
            batch_results = pool.map(run_model_instance, [(model, len(n_words)) for model in models])
        
        results.extend([refactor(model) for model in batch_results])
        
    return results