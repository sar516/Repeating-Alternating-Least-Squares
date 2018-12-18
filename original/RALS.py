import numpy as np
import pandas as pd
import sparse_bag_tools as spt
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from sklearn.metrics import mean_squared_error
import seaborn as sns
sns.set(color_codes=True)


class RALS:
    def __init__(self, k = 1, l = 20):
        self.k = k
        self.l = l
        
    def iniatilize_fit(self, train_set, beta = 1):
        self.num_user, self.num_item = train_set.shape
        self.W = (train_set > 0) * 1.0
        self.R = np.ones(train_set.shape)*train_set
        self.U_dict = {number : np.random.rand(self.num_user, self.k)
                       for number in range(self.l)}
        self.I_dict = {number : np.random.rand(self.k, self.num_item)
                       for number in range(self.l)}
        self.beta = beta          
        self.Beta = self.beta*np.eye(self.k)
        self.mean = np.sum(self.R)/np.sum(self.W)
        self.errs = []
        
    def fit(self, train_set, steps = 10, beta = 1):
        self.iniatilize_fit(train_set, beta)

        for level in range(self.l):
            (self.U_dict[level])[:,0] = (self.R.sum(1)/self.W.sum(1))
            (self.I_dict[level])[0,:] = (self.R.sum(0)/self.W.sum(0))
            self.level = level
            step = 0
            while step < steps:
                self.als_step()
                step += 1
            else:
                self.R -= (self.W*np.dot(self.U_dict[level],self.I_dict[level]))
                    
        return self
    
    def als_step(self):
        U = self.U_dict[self.level]
        I = self.I_dict[self.level]
        for u in range(self.num_user):
            I_tild = I*(self.W[u].reshape(1, -1))
            inv_for_U = np.linalg.inv(np.dot(I_tild, I_tild.T) + (self.W[u].sum()*self.Beta))
            U[u, :] = np.dot(np.dot(self.R[u,:].reshape(1,-1), I.T), inv_for_U).reshape(self.k)
         
        for i in range(self.num_item):
            U_tild = U*(self.W[:,i].reshape(-1, 1))
            inv_for_I = np.linalg.inv(np.dot(U_tild.T, U_tild) + (self.W[:,i].sum()*self.Beta))
            I[:,i] = np.dot(np.dot(self.R[:, i].reshape(1,-1), U), inv_for_I).reshape(self.k)
            
        self.U_dict[self.level] = U
        self.I_dict[self.level] = I   
            
        err = get_mse(self.R, self.W*np.dot(U,I))
        self.errs.append(err)
    
    def predict(self,test):
        W_test = 1*(test > 0)
        self.predictions = sum([self.U_dict[number].dot(self.I_dict[number])
                                for number in range(self.level+1)])
        return W_test*self.predictions
    
    def learning_curve(self, train_set, test_set, steps = 10, beta = .1):
        
        self.iniatilize_fit(train_set, beta)
        self.pred_err = []
        for level in range(self.l):
            (self.U_dict[level])[:,0] = (self.R.sum(1)/self.W.sum(1))
            (self.I_dict[level])[0,:] = (self.R.sum(0)/self.W.sum(0))
            self.level = level
            step = 0
            while step < steps:
                self.als_step()
                predictions = self.predict(test_set)
                W = 1*(test_set > 0)
                MSE = get_mse(predictions, test_set)
                self.pred_err.append(MSE)
                step += 1
            else:
                self.R -= (self.W*np.dot(self.U_dict[level],self.I_dict[level]))
            if self.errs[-1] < 0.001:
                self.stopping_point = level
                break     
        return self


def bag_to_array(bag):
    item_bag = spt.sparse_bag_transpose(bag)
    user_list = list(bag.keys())
    item_list = list(item_bag.keys())
    R = np.zeros((len(user_list), len(item_list)))
    U_index = {user_list[u]: u for u in range(len(user_list))}
    I_index = {item_list[i]: i for i in range(len(item_list))}
    
    for user in bag.keys():
        u = U_index[user]
        for item in bag[user].keys():
            i = I_index[item]
            R[u,i] = bag[user][item]
            
    return R

def train_test_split(ratings, test_size = 0.01):
    test = np.zeros(ratings.shape)
    num_rows, num_cols = ratings.shape
    train = ratings.copy()
    W = (1*(ratings > 0))
    non_zeros = int(W.sum()*(1-test_size))
    num_test = int((test_size)*W.sum())
    num_train = int((1-test_size)*W.sum())
    W_train = 1*W
    for k in range(num_test):
        valid  = False
        valid_rows =(W_train.sum(1)-1).nonzero()[0]
        valid_cols = (W_train.sum(0)-1).nonzero()[0]
        j = 0
        while not valid:
            u = np.random.choice(valid_rows, 1)
            i = np.random.choice(W[u].nonzero()[0])
            j+=1
            if i in valid_cols:
                valid = True
            else:
                valid = False
                if j > 300:
                    print("might be stuck")
                    print((k/W.sum()))
                    raise ValueError
        else:
            W_train[u,i] = 0
    else:
        W_test = W - W_train
        train = ratings*W_train
        test = ratings*W_test
    # Test and training are truly disjoint
    assert(np.all((train * test) == 0)) 
    return train, test


def sparse_trainer(shape, sparsity, split_size):
    num_rows, num_cols = shape
    num_entries = num_rows * num_cols
    non_zeros = int(num_entries*sparsity)
    valid = False
    multiplier = 1-split_size
    while not valid:
        random_mat = np.random.rand(num_rows, num_cols)
        W_train = 1*(random_mat < sparsity*multiplier)
        W_test = 1*(random_mat < sparsity) - W_train
        valid_rows = (W_train.sum(1) > 0).all()
        valid_cols = (W_train.sum(0) > 0).all()
        valid_num_of_entries = (W_train + W_test).sum() in range(non_zeros-10, non_zeros+11)
        valid = valid_rows and valid_cols and valid_num_of_entries
    else:
        assert(np.all((W_train*W_test) == 0)) 
        return W_train, W_test

    

def discrete_sim(shape, n, p, sparsity, split_size):
    W_train, W_test = sparse_trainer(shape, sparsity, split_size)
    binom_mat = np.random.binomial(n,p, shape)
    train = binom_mat*W_train
    test = binom_mat*W_test
    return train, test

def continuous_sim(shape, mu, std, sparsity, split_size):
    W_train, W_test = sparse_trainer(shape, sparsity, split_size)
    norm_mat = np.abs(np.random.normal(mu, std, shape))
    train = norm_mat*W_train
    test = norm_mat*W_test
    return train, test

def get_mse(pred, actual):
    pred = pred[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()
    return mean_squared_error(pred, actual)

def read_game_data():
    raw_games = pd.read_csv("steam-200k.csv", names =["user_id", "game", "behavior", "value", "misc."])
    playtime = raw_games[raw_games["behavior"] == "play" ]
    purchased = raw_games[raw_games["behavior"] == "purchase"]
    bag = spt.bag_of_games(playtime)
    proc_bag = spt.preprocessing(bag)
    log_bag = spt.norm(proc_bag)
    return bag_to_array(log_bag)


def read_movie_data():
    names = ['user_id', 'item_id', 'rating', 'timestamp']
    df = pd.read_csv('ml-100k/u.data', sep='\t', names=names)
    n_users = df.user_id.unique().shape[0]
    n_items = df.item_id.unique().shape[0]
    ratings = np.zeros((n_users, n_items))
    for row in df.itertuples():
        ratings[row[1]-1, row[2]-1] = row[3]
    return ratings

def Comparitive_Learning_Curve(train, test, k, beta, Name):
    models = {"ALS": RALS(k=k, l=1).learning_curve(train, test, steps = 100, beta = .1), #When we set l = 1 we get ALS back
              "RALS": RALS(k=1, l=k).learning_curve(train, test, steps = int(100/k), beta = .1)}
    
    for model in models.keys():
        plt.plot(np.arange(1,101), models[model].errs, label=model)
    else:
        plt.title("Training Error for " + Name)
        plt.xlabel("Number of Iterations")
        plt.ylabel("Training Error (MSE)")
        plt.legend()
        plt.savefig(Name.replace(" ", "_")+"_training_err.png")
        plt.close()
    
    for model in models.keys():
        plt.plot(np.arange(1,101), models[model].pred_err, label=model)
    else:
        plt.title("Testing Error for " + Name)
        plt.xlabel("Number of Iterations")
        plt.ylabel("Testing Error (MSE)")
        plt.legend()
        plt.savefig(Name.replace(" ", "_")+"_testing_err.png")
        plt.close()
    print(Name + " done")

def main():
    game_data = read_game_data()
    movie_data = read_movie_data()
    game_train, game_test = train_test_split(game_data, test_size = 0.25)
    movie_split_done = False
    while not movie_split_done:
        try:
            movie_train, movie_test = train_test_split(movie_data, test_size = 0.05)
            movie_split_done = True
        except ValueError:
            print("Movie split failed; restarting splitting process")
    Comparitive_Learning_Curve(game_train, game_test, 20, 0.1, "Video Game Hours Played")
    Comparitive_Learning_Curve(movie_train, movie_test, 20, 0.1, "Movie Ratings")
    flat_game_data = game_data[game_data.nonzero()].flatten()
    flat_movie_data = movie_data[movie_data.nonzero()].flatten()
    mu = flat_game_data.mean()
    sigma = flat_game_data.std()
    n = 4
    p = (flat_movie_data.mean() - 1)/n
    shape = movie_data.shape
    sparsity = 0.07
    continuous_sim_train, continuous_sim_test = continuous_sim(shape, mu, sigma, sparsity, 0.25)
    discrete_sim_train, discrete_sim_test = discrete_sim(shape, n, p, sparsity, 0.25)
    Comparitive_Learning_Curve(continuous_sim_train, continuous_sim_test, 20, 0.1, "Continuous Simulated Data")
    Comparitive_Learning_Curve(discrete_sim_train, discrete_sim_test, 20, 0.1, "Discrete Simulated Data")
    
    
if __name__ == "__main__" :
    main()