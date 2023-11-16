import numpy as np
import uuid
import sys
from collections import defaultdict
import pickle
import tqdm.auto as tqdm
import multiprocessing

class SlovotvirModel:
    def __init__(self, words_series, translation_series, length_distr, votes_user, a, b):
        self.words_series = words_series
        self.translation_series = translation_series
        self.length_distr = length_distr
        self.votes_user = votes_user
        self.a = a
        self.b = b
        self.n_epochs = len(self.words_series)
        self.word_pool = defaultdict(list)
        self.translation_pool = defaultdict(dict)
                    
    def like_prob(self, lengths, likes):
        probs = (1 / (lengths ** self.a)) + likes ** self.b
        return probs

    def process_epoch(self, epoch, shared_model):  # Pass shared_model as an argument
        n_words = self.words_series[epoch]
        n_translations = self.translation_series[epoch]
        n_votes = self.votes_user[epoch]

        for _ in range(n_words):
            self.word_pool[np.random.randint(0, 1000000)]
        
        

        unique_words = list(self.word_pool.keys())

        for _ in range(n_translations):
            word_id = np.random.choice(unique_words)
            translation_id = np.random.randint(0, 1000000)
            self.word_pool[word_id].append(translation_id)
            self.translation_pool[translation_id] = {'length': np.random.choice(self.length_distr), 
                                                     'likes': 1}

        unique_words = list(self.word_pool.keys())

        likes_w = np.array([sum([self.translation_pool[translation_id]['likes'] for translation_id in 
                                 self.word_pool[word_id]]) for word_id in unique_words])

        likes_w = likes_w / likes_w.sum()

        for n in n_votes:
            for _ in range(n):
                translations = self.word_pool[np.random.choice(unique_words, p=likes_w)]

                while not translations:
                    translations = self.word_pool[np.random.choice(unique_words, p=likes_w)]

                lengths = np.array([self.translation_pool[translation_id]['length'] 
                                    for translation_id in translations]).astype(int)
                likes = np.array([self.translation_pool[translation_id]['likes'] 
                                  for translation_id in translations]).astype(int)

                # ranks = np.argsort(np.argsort(likes)[::-1]) + 1

                probs = self.like_prob(lengths, likes)
                probs = probs / probs.sum()

                tr = np.random.choice(translations, p=probs)
                self.translation_pool[tr]['likes'] += 1
    
        # After each epoch, update the shared model
        shared_model['model'] = self  # Update the shared model with the modified model state

def process_epoch_parallel(args):
    epoch, shared_model = args
    model = shared_model['model']
    model.process_epoch(epoch, shared_model)  # Pass shared_model as an argument
    shared_model['model'] = model

if __name__ == '__main__':
    # collect a and b values from command line
    a = float(sys.argv[1]) # lengths
    b = float(sys.argv[2]) # rank 

    # unpickle data
    with open('../data/n_words.pkl', 'rb') as f:
        words_series = pickle.load(f)
    
    with open('../data/n_translations.pkl', 'rb') as f:
        translation_series = pickle.load(f)
    
    with open('../data/votes.pkl', 'rb') as f:
        votes_user = pickle.load(f)

    with open('../data/translation_len.pkl', 'rb') as f:
        length_distr = pickle.load(f)

    # run model
    model = SlovotvirModel(words_series,
                           translation_series,
                           length_distr,
                           votes_user,
                           a,
                           b)

    # Create a shared dictionary to hold the model object
    manager = multiprocessing.Manager()
    shared_model = manager.dict()
    shared_model['model'] = model

    # Determine the number of CPU cores available
    num_cores = multiprocessing.cpu_count()

    # Create a pool of workers
    pool = multiprocessing.Pool(processes=num_cores)

    # Create a list of epochs to be processed in parallel
    epochs_to_process = list(range(model.n_epochs))

    # Run the model in parallel using the pool of workers
    with tqdm.tqdm(total=model.n_epochs, 
                   bar_format='{l_bar}{bar:30}{r_bar}{bar:-10b}',
                   desc='Running the model', 
                   position=0, 
                   leave=True) as progress_bar:
        for _ in pool.imap_unordered(process_epoch_parallel, [(epoch, shared_model) for epoch in epochs_to_process]):
            progress_bar.update()

    # Close the pool of workers
    pool.close()
    pool.join()

    model = shared_model['model']

    # print total number of words
    print(f'Total number of words: model -- {len(model.word_pool.keys())} data -- {sum(words_series)}')

    likes = np.array([model.translation_pool[tr]['likes'] for tr in model.translation_pool.keys()])
    ranked = np.argsort(likes)[::-1]

    # save results
    with open(f'../data/likes-a{a}-b{b}.pkl', 'wb') as f:
        pickle.dump(likes[ranked], f)
