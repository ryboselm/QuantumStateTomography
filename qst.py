import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import pickle
import argparse

import torch
from torch.distributions import constraints
import torch.nn as nn
import torch.nn.functional as F
from scipy.linalg import sqrtm

import pyro
import pyro.distributions as dist
from pyro.infer import Predictive, SVI, Trace_ELBO
from pyro.optim import Adam, SGD

assert pyro.__version__.startswith('1.8.6')
pyro.distributions.enable_validation(False)
pyro.set_rng_seed(1)

"""
the probabilistic model for QST class
"""
class QuantumStateTomography(nn.Module):
    def __init__(self, N=3, alpha = 1, K=3, POVM=None, sigma = 1):
        super().__init__()
        #define hyperparameters
        self.N = 2**N        # size of system 2^(num qubits)
        self.alpha = alpha   # sparsity of the density matrix
        self.K = K           # rank of the density matrix
        if POVM is not None:
            self.POVM = torch.from_numpy(POVM).type(torch.complex64) # measurement set used
        else:
            self.POVM = None
        self.sigma = sigma   # variance of the complex vectors

    def model(self, data):
        # draw a probability distribution over K pure states
        theta = pyro.sample("theta", dist.Dirichlet(torch.ones(self.K) * self.alpha))
    
        # generate K pure states
        with pyro.plate("pure states", self.K):
            real_part = pyro.sample("real_part", dist.MultivariateNormal(torch.zeros(self.N), torch.eye(self.N)))
            imag_part = pyro.sample("imag_part", dist.MultivariateNormal(torch.zeros(self.N), torch.eye(self.N)))
            complex_vector = torch.complex(real_part, imag_part)
            norm_vector = complex_vector / torch.norm(complex_vector, dim=-1, keepdim=True)
        norm_vector = torch.transpose(norm_vector,0,1)
        
        # compute the density matrix (rho)
        if self.POVM is None:
            # standard basis measurements
            squared = torch.conj(norm_vector) * norm_vector
            probabilities = (squared @ theta.type(torch.complex64))
            with pyro.plate("data", len(data)):
                return pyro.sample("obs", dist.Categorical(probabilities.real), obs=data)
        else:
            inner_prods = torch.matmul(torch.conj(self.POVM), norm_vector)
            squared = torch.conj(inner_prods) * inner_prods
            probabilities = (squared @ theta.type(torch.complex64))
            with pyro.plate("data", len(data)):
                return pyro.sample("obs", dist.Categorical(probabilities.real), obs=data)
        
    def guide(self, data):
        constrained_vector = pyro.param("constrained_vector", torch.ones(self.K) / self.K, constraint=constraints.simplex)
        theta = pyro.sample("theta", dist.Delta(constrained_vector).to_event(1))
    
        with pyro.plate("pure states", self.K):
            real_mean = pyro.param("real_mean", torch.randn(self.K, self.N))
            imag_mean = pyro.param("imag_mean", torch.randn(self.K, self.N))
            sigma_real = pyro.param("sigma_real", 0.1 * torch.ones(self.K, self.N), constraint=constraints.positive)
            sigma_imag = pyro.param("sigma_imag", 0.1 * torch.ones(self.K, self.N), constraint=constraints.positive)
            
            cov_matrix_real = torch.zeros(self.K, self.N, self.N)
            cov_matrix_imag = torch.zeros(self.K, self.N, self.N)
            for i in range(self.K):
                cov_matrix_real[i] = torch.diag(sigma_real[i])
                cov_matrix_imag[i] = torch.diag(sigma_imag[i])
    
            real_part = pyro.sample("real_part", dist.MultivariateNormal(real_mean, covariance_matrix=cov_matrix_real))
            imag_part = pyro.sample("imag_part", dist.MultivariateNormal(imag_mean, covariance_matrix=cov_matrix_imag))

            
"""
trains on the data
"""
def train(svi, train_loader, use_cuda=False):
    # initialize loss accumulator
    epoch_loss = 0.
    # do a training epoch over each mini-batch x returned
    # by the data loader
    for x in train_loader:
        # if on GPU put mini-batch into CUDA memory
        if use_cuda:
            x = x.cuda()
        # do ELBO gradient and accumulate loss
        epoch_loss += svi.step(x)

    # return epoch loss
    normalizer_train = len(train_loader.dataset)
    total_epoch_loss_train = epoch_loss / normalizer_train
    return total_epoch_loss_train

"""
runs on the test data
"""
def evaluate(svi, test_loader, use_cuda=False):
    # initialize loss accumulator
    test_loss = 0.
    # compute the loss over the entire test set
    for x in test_loader:
        # if on GPU put mini-batch into CUDA memory
        if use_cuda:
            x = x.cuda()
        # compute ELBO estimate and accumulate loss
        test_loss += svi.evaluate_loss(x)
    normalizer_test = len(test_loader.dataset)
    total_epoch_loss_test = test_loss / normalizer_test
    return total_epoch_loss_test

"""
prepares dataloaders
"""
def setup_data_loaders(data, batch_size=128, use_cuda=False):
    root = './data'
    download = True
    random.shuffle(data)
    n = len(data)
    cutoff = int(0.9*n)
    train_set = torch.Tensor(data[:cutoff])
    test_set = torch.Tensor(data[cutoff:])

    kwargs = {'num_workers': 4, 'pin_memory': use_cuda}
    train_loader = torch.utils.data.DataLoader(dataset=train_set,
        batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(dataset=test_set,
        batch_size=batch_size, shuffle=False, **kwargs)
    return train_loader, test_loader


"""
calculates trace distance between two density matrices a and b
"""
def trace_dist(a, b):
    k = a-b
    M = np.matmul(k.conj().T,k)
    B = sqrtm(M)
    return np.trace(B).real/2
"""
calculates fidelity between two density matrices
"""
def fidelity(a,b):
    k = sqrtm(a)
    M = k @ b @ k
    return (np.trace(sqrtm(M)).real)**2

"""
loads data
"""
def load_data(path, n_samples=10000):
    probabilities = []
    with open(path, "r") as f:
        lines = f.read().splitlines()
    for line in lines:
        _, freq = line.split(" ")
        probabilities.append(float(freq))
    options = range(len(probabilities))
    data = np.random.choice(options, n_samples, p=probabilities)
    return data

def load_povm(path):
    with open(path, "rb") as f:
        return pickle.load(f)

"""
Runs the experiment
"""
def run_tomography_model(name, n_qubits=2, n_states=1, batch_size = 128, n_samples=10000, n_epochs=100, lr=1e-3):
    povmpath = "data/" + name + "_POVM"
    povmsize = 2**n_qubits
    try:
        povm = load_povm(povmpath)
        povmsize = povm.shape[0]
    except:
        print("The POVM could not be found. using standard basis instead.")
        povm = None

    statspath = "data/" + name + "_statistics"
    try:
        data = load_data(statspath, n_samples)
    except:
        print(f"An exception occurred with loading the data from {statspath}!")
        return

    # Run options
    LEARNING_RATE = lr
    USE_CUDA = False
    smoke_test = False

    # Run only for a single iteration for testing
    NUM_EPOCHS = 1 if smoke_test else n_epochs
    TEST_FREQUENCY = 5

    train_loader, test_loader = setup_data_loaders(data,batch_size=batch_size, use_cuda=USE_CUDA)
    # clear param store
    pyro.clear_param_store()

    qst = QuantumStateTomography(N=n_qubits, K=n_states, POVM=povm)
    optimizer = Adam({"lr": LEARNING_RATE})
    svi = SVI(qst.model, qst.guide, optimizer, loss=Trace_ELBO())

    train_elbo = []
    test_elbo = []
    # training loop
    pbar = tqdm(range(NUM_EPOCHS))
    for epoch in pbar:
        total_epoch_loss_train = train(svi, train_loader, use_cuda=USE_CUDA)
        train_elbo.append(-total_epoch_loss_train)
        pbar.set_description("[epoch %03d]  average training loss: %.4f" % (epoch, total_epoch_loss_train))

        if epoch % TEST_FREQUENCY == 0:
            # report test diagnostics
            total_epoch_loss_test = evaluate(svi, test_loader, use_cuda=USE_CUDA)
            test_elbo.append(-total_epoch_loss_test)
            pbar.set_description("[epoch %03d] average test loss: %.4f" % (epoch, total_epoch_loss_test))

    epochs = np.arange(len(train_elbo))
    plt.plot(epochs, train_elbo, label='Train ELBO')
    epochs = np.arange(len(test_elbo))
    plt.plot(epochs*TEST_FREQUENCY, test_elbo, label='Test ELBO')
    plt.xlabel('Epochs')
    plt.ylabel('ELBO')
    plt.legend()
    plt.title(f"Experiment: {name}, Qubits: {n_qubits}, Pure States: {n_states}")
    plt.savefig("output/"+name+"_ELBO.png",dpi=600)

    real_part = pyro.param('real_mean').data
    imag_part = pyro.param('imag_mean').data
    theta = pyro.param('constrained_vector').data.numpy()
    vecs = F.normalize(torch.complex(real_part, imag_part), p=2, dim=1).numpy()
    approx_DM = (np.conj(vecs.T) @ np.diag(theta) @ vecs)

    densitypath = "data/" + name + "_density"
    with open(densitypath, "rb") as file:
        rho = pickle.load(file)
    true_DM = rho
    print(name)
    print('Trace Distance:',trace_dist(approx_DM, true_DM))
    print('Fidelity:', fidelity(approx_DM, true_DM))
    approx_prob = np.round(np.real(np.diag(approx_DM)),3)
    real_prob = np.round(np.real(np.diag(true_DM)),3)
    outputpath = "output/"+name+"_results.txt"
    with open(outputpath, "w") as f:
        f.write(f"name: {name}, num_qubits: {n_qubits}, num_states: {n_states}, POVM size: {povmsize}, batchsize: {batch_size}, epochs: {NUM_EPOCHS}, Learning Rate: {LEARNING_RATE} \n")
        f.write(f"Trace Distance to true Density Matrix: {trace_dist(approx_DM, true_DM)}\n")
        f.write(f"Fidelity to true Density Matrix {fidelity(approx_DM, true_DM)}\n")
        f.write(f"Model Probabilities: {approx_prob}\n")
        f.write(f"True Probabilities: {real_prob}\n")
        f.write("\n")
        f.write(f"theta: {theta}\n")
        f.write(f"Approx Density Matrix:\n {np.round(approx_DM,3)}\n")
        f.write(f"True Density Matrix:\n {np.round(true_DM,3)}\n")




"""
How to run this:
1. make sure that the desired statistics, POVM, and density data are present inside the ./data folder. These are generated by GenerateMeasurements.ipynb
2. run a command such as `python ./qst.py 5 3 -n "random_5"` where the parameters 5, 3, and "random_5" are changed depending on what experiment you want to perform
3. it will run in the command line, collect data from the ./data folder and then save the plot and results in the ./output folder
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('n', type=int, help='number of qubits')
    parser.add_argument('k', type=int, help='rank of density matrix')
    parser.add_argument("--name", "-n", type=str, default = "MISSING", help="Choose name of experiment")
    parser.add_argument("--batchsize", "-b", type=int, default=128, help="batch size of the training")
    parser.add_argument("--samplenum", "-s", type=int, default=10000, help="how many samples in the dataset")
    parser.add_argument("--epochs", "-e", type=int, default=100, help="how many epochs run for")
    parser.add_argument("--learn", "-l", type=int, default=.001, help="learning rate")
    args = parser.parse_args()
    run_tomography_model(args.name, args.n, args.k, batch_size=args.batchsize, n_samples=args.samplenum, n_epochs=args.epochs, lr=args.learn)
