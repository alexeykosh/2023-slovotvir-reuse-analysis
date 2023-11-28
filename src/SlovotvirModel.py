from re import T
import numpy as np
import pickle

# unpickle n_words, translation_len, votes, n_translations from data folder
with open('../data/n_words.pkl', 'rb') as f:
    n_words = pickle.load(f)

with open('../data/translation_len.pkl', 'rb') as f:
    translation_len = pickle.load(f)

with open('../data/votes.pkl', 'rb') as f:
    votes = pickle.load(f)
    votes = np.array([sum(v) for v in votes.values()])

with open('../data/n_translations.pkl', 'rb') as f:
    n_translations = pickle.load(f)

with open('../data/true_likes.pkl', 'rb') as f:
    true_likes = pickle.load(f)


def shannon_diversity_index(likes):
    '''Calculates the Shannon diversity index.'''
    likes_sorted = np.sort(likes)[::-1]
    _, like_counts = np.unique(likes_sorted, return_counts=True)
    like_proportions = like_counts / np.sum(like_counts)
    return -np.sum(like_proportions * np.log(like_proportions))


def simpson_diversity_index(likes):
    '''Calculates the Simpson diversity index.'''
    likes_sorted = np.sort(likes)[::-1]
    _, like_counts = np.unique(likes_sorted, return_counts=True)
    like_proportions = like_counts / np.sum(like_counts)
    return np.sum(like_proportions ** 2)


def summary_func(likes):
    '''Calculates the summary statistics of a list of likes.'''
    return np.mean(likes == 1), \
        likes.max() / np.sum(likes), \
            shannon_diversity_index(likes), \
                simpson_diversity_index(likes), \
                    np.mean(likes), \
                        np.median(likes)


def min_max_scaling_numpy(data, new_min=1, new_max=10):
    '''Scales data to a new range.'''
    if data.size == 0:
        return data
    original_min = np.min(data)
    original_max = np.max(data)

    if original_min == original_max or new_min == new_max:
        return data
    scaled_data = ((data - original_min) / (original_max - original_min)) * (new_max - new_min) + new_min
    return scaled_data


def adjust_entropy(prob_distribution, temperature):
    '''Adjusts entropy of a probability distribution.'''
    temperature = max(1e-3, temperature)
    scaled_probs = np.power(prob_distribution, 1 / temperature)
    adjusted_distribution = scaled_probs / np.sum(scaled_probs)

    return adjusted_distribution


class SlovotvirModel:
    '''Slovotvir model.'''

    def __init__(self, n_words, 
                 translation_len, 
                 votes, 
                 n_translations, 
                 a, b, t) -> None:
        # epoch states 
        self.n_words = n_words
        self.translation_len = translation_len
        self.votes = votes
        self.n_translations = n_translations

        # cumulative words
        self.cum_words = np.cumsum(n_words)

        # hyperparameters
        self.a = a
        self.b = b
        self.t = t

        self.words = dict()
        for i in range(sum(n_words)):
            self.words[i] = [[], []] # [[likes], [lengths]]

    
    def like_prob(self, 
                  lengths, 
                  likes) -> np.ndarray:
        '''Calculates the probability of liking each translation.'''
        probs = lengths ** self.a + likes ** self.b
        return probs / probs.sum()
    
    def run_epoch(self, epoch) -> None:
        '''Runs an epoch.'''
        # n_words = self.n_words[epoch]
        n_translations = self.n_translations[epoch]
        votes = self.votes[epoch]
        cum_words = self.cum_words[epoch]

        # distribute n_translation into an array of length cum_words
        translation_num = np.random.multinomial(n_translations, np.ones(cum_words)/cum_words)

        # get the indices of non-zero elements in translation_num
        non_zero_indices = np.nonzero(translation_num)[0]

        # update the words list only for non-zero indices
        for i in non_zero_indices:
            n = translation_num[i]
            self.words[i][0] += [0] * n # initialize likes to 0
            self.words[i][1] += np.random.choice(self.translation_len, n).tolist()

        indices = np.array([i for i in range(cum_words) if len(self.words[i][0]) > 0])
        cum_likes = np.array([sum(self.words[i][1]) for i in indices])
        indices = np.random.choice(indices,
                                   size=votes,
                                   p=adjust_entropy(cum_likes / cum_likes.sum(), temperature=self.t),
                                   replace=True)
        
        for _ in indices:
            # get the word
            likes = min_max_scaling_numpy(np.array(self.words[_][0]))
            lengths = min_max_scaling_numpy(np.array(self.words[_][1]))

            # get the probability of liking each translation
            probs = self.like_prob(lengths, likes)

            # choose a translation
            translation = np.random.choice(likes, p=probs)

            # update the number of likes for the translation
            self.words[_][0][likes.tolist().index(translation)] += 1

    def run(self) -> None:
        '''Runs the model.'''
        n_epochs = len(self.n_words)
        for epoch in range(n_epochs):
            self.run_epoch(epoch)

def run_model(params):
    '''Runs the model with given hyperparameters.'''
    a, b, t = params
    model = SlovotvirModel(n_words, translation_len, votes, n_translations, a, b, t)
    model.run()
    # combine all likes list from the model
    likes = []
    for i in range(len(model.words)):
        likes += model.words[i][0]

    likes = np.array(likes)

    return np.histogram(likes, bins=100)[0]
