import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

class py_data(Dataset):
    def __init__(self, sampler, train_length, device="cuda", cond=False):
        self.device = device #("cuda" if torch.cuda.is_available() else "cpu")
        self.cond = cond
        self.sampler = sampler
        self.train_length = train_length

    def __len__(self):
        return self.train_length

    def __getitem__(self, item):
        """
        # getitem uses the diffusion object to sample a time step and noisy version of the next data item
        # it concatenates the dimensions of the data with a time indicator so that prediction can depend on time
        # returns the epsilon parameter as the target value
        """
        t, x, e = self.sampler.reverse_train(self.sampler.data[:, item])
        t = torch.tensor(t * self.sampler.dt, device=self.device)
        x = torch.cat((x, t.unsqueeze(dim=-1)), dim=-1)
        if self.cond is True:
            # if it's a conditional model, append the class to the data as well
            c = self.sampler.cls[item].type(torch.int)
            x = torch.cat((x, c.unsqueeze(dim=-1)), dim=-1)
        return x, e


class py_model(nn.Module):
    def __init__(self, dims, cond=False, device="cuda"):
        super(py_model, self).__init__()
        self.device = device #("cuda" if torch.cuda.is_available() else "cpu")
        self.cond = cond
        if self.cond is True:
            # if it's a conditional model, do embedding, mixing, and adjust dims for the class input
            #self.embed = nn.Embedding(num_embeddings=5, embedding_dim=8, device=self.device)
            #self.mix = nn.Linear(11, 8, device=self.device)
            self.embed = nn.Embedding(num_embeddings=5, embedding_dim=4, device=self.device)
            self.mix = nn.Linear(7, 8, device=self.device)
            dims[0] = 8
        self.layers = nn.ModuleList()
        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i + 1], device=self.device))
            if i < len(dims) - 2:
                self.layers.append(nn.BatchNorm1d(dims[i + 1], device=self.device))
                self.layers.append(nn.LeakyReLU())
        self.apply(self.init_weights)
        

    def init_weights(self, m):
        if type(m) == nn.Linear:
            input_dim = m.in_features
            stdev = 1/np.sqrt(input_dim)
            nn.init.normal_(m.weight, mean=0, std=stdev)

    def forward(self, x):
        if self.cond is True:
            c = x[:, -1].type(torch.int)
            c = self.embed(c)
            x = self.mix(torch.concat((x[:, :3], c), dim=1))
        for i in range(len(self.layers)):
            x = self.layers[i](x)
        return x


def train_model(datapoints, net, lr=1e-3, epochs=3000, save="diffusion", cond=False, device="cuda"):
    """
    Function to load a fusion object and train a pytorch model on it
    :param datapoints: number of points to generate
    :param lr: learning rate
    :param epochs: training epochs
    :param save: filename to save for model.pt and results.pkl
    param cond: whether the model should be conditioned on a class
    :return: trained model
    """
    print("Training diffusion model")
    dif = Diffusion(datapoints=datapoints, device=device)
    if cond is True:
        dif.set_classes()
    data = py_data(dif, dif.datapoints, cond=cond, device=device)
    loader = DataLoader(data, batch_size=dif.datapoints)
    model = py_model(net, cond=cond, device=device)
    # if torch version 2.0
    #torch.compile(model)
    model.train()
    loss_fn = torch.nn.MSELoss()
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    loss_track = []
    for epoch in tqdm(range(epochs)):
        for data, target in loader:
            optim.zero_grad()
            predict = model(data)
            loss = loss_fn(predict, target)
            loss_track.append(loss.item())
            loss.backward()
            optim.step()
    model.eval()
    state_dict = model.state_dict()
    torch.save(state_dict, f"{save}.pt")
    #with open(f'{save}-train-loss.pkl', 'wb') as f:
    #    pickle.dump(loss_track, f)
    return model


class Diffusion:
    def __init__(self, T=1000, schedule_param="exp", datapoints=1000, device="cuda"):
        self.device = device #("cuda" if torch.cuda.is_available() else "cpu")
        self.T = T
        self.dt = 1 / self.T
        self.datapoints = datapoints
        self.data = self.create_data(datapoints)
        self.schedule_param = schedule_param
        self.sigmas = self.scheduler()

    def create_data(self, numpoints):
        return torch.rand(2, numpoints, device=self.device) * 2 - 1

    def scheduler(self):
        """
        function to create noise scheduler
        :return: 1D tensor of size T with sigmas
        """
        if self.schedule_param == "t":
            schedule = torch.linspace(0.0, 1.0, self.T, device=self.device)
        elif self.schedule_param == "sqrt":
            schedule = torch.sqrt(torch.linspace(0.0, 1.0, self.T, device=self.device))
        else:
            schedule = torch.exp(5 * (torch.linspace(0.0, 1.0, self.T, device=self.device) - 1))
        return schedule

    def forward_noise(self, x, t):
        """
        Takes a 2D point, a noise parameter epsilon, and a timestep as input
        :param x: datapoint before adding noise
        :param: t: timestep sampled uniformly from T steps, to extract sigma from scheduler
        :return: a noised data sample and the epsilon parameter sampled from normal distribution
        """
        epsilon = torch.distributions.MultivariateNormal(torch.zeros(2, device=self.device),
                                                         torch.eye(2, device=self.device)).sample()
        return x + (epsilon * self.sigmas[t]), epsilon

    def reverse_train(self, x):
        """
        function to obtain sample and noise pair as training data
        it chooses a random time t
        :param x: data point at start without noise
        :return: noised data point at random time t, and the noise parameter epsilon to predict
        """
        t = np.random.randint(1, self.T)
        x_t, epsilon = self.forward_noise(x, t)
        return t, x_t, epsilon

    @torch.no_grad()
    def generate(self, model, from_point=None, stochastic=False, cls=False):
        """
        Uses denoising model to estimate original points from noisy points in reverse process.
        :param model: trained denoiser, takes coordinates, timestep and class as inputs and estimates epsilon noise parameter.
        :param from_point: 2D tensor specifying start points. If supplied, denoise it, otherwise generate from rondom points.
        :param stochastic: boolean, defines whether to add stochasticity to reverse step.
        :param cls: boolean, if true then generate random classes too (unless from_point supplied, in which case get classes from there).
        :return: trajectory of points over denoising process, as dictionary of coordinates (listed by time step) and point classes.
        """
        model.eval()
        dt = - self.dt
        # first sample starting points
        if from_point is None:
            z = torch.distributions.MultivariateNormal(torch.zeros(2, device=self.device),
                                                       torch.eye(2, device=self.device)).sample()
            z = z.unsqueeze(dim=0)
            for _ in range(self.datapoints - 1):
                z = torch.cat((z, torch.distributions.MultivariateNormal(torch.zeros(2, device=self.device),
                                                           torch.eye(2, device=self.device)).sample().unsqueeze(dim=0)), dim=0)
        else:
            z = torch.tensor(from_point, device=device)
            # control for event of single datapoint
            if len(z.size()) == 1:
                z = z.unsqueeze(dim=-1)
                z = z.transpose(0, 1)
        # get batch size
        m = z.size()[0]
        store = {'coordinates': [z.detach().clone()]}
        # generate classes for conditional case
        if cls is True:
            cond = torch.tensor([random.randint(0, 4) for _ in range(self.datapoints)],
                                device=self.device).unsqueeze(dim=-1)
            store['class'] = cond.detach().clone()
        # begin loop for reverse process
        for t in tqdm(range(self.T - 1, 0, -1)):
            # put together input data: z, t, c
            step = torch.tensor([1 + (t * dt)], device=self.device)
            step = step.repeat(m).unsqueeze(dim=-1)
            inp = torch.cat((z, step), dim=1)
            if cls is True:
                inp.unsqueeze(dim=-1)
                inp = torch.cat((inp, cond), dim=1)
            # get estimated noise from model
            epsilon = model(inp)
            # update according to scheduler type (exponential is default) - see readme file for derivations
            if self.schedule_param == "t":
                dz = epsilon * dt
            elif self.schedule_param == "sqrt":
                dz = (epsilon * dt) / (2 * self.sigmas[t].item())
            else:
                if stochastic is False:
                    dz = self.sigmas[t].item() * 5 * dt * epsilon
                else:
                    # if we want to introduce variation, we have a stochastic version of update, coded only for exp scheduler
                    w = torch.distributions.MultivariateNormal(torch.zeros(2, device=self.device),
                                                               torch.eye(2, device=self.device)).sample()
                    # control for event of single datapoint
                    if m == 1:
                        w = w.unsqueeze(dim=0)
                    else:
                        w = w.unsqueeze(dim=-1)
                    dz = (0.9 * (5 * self.sigmas[t].item() * epsilon * dt)) + (0.1 * (self.sigmas[t].item() * w))
            # update and loop back
            z += dz
            copy = z.detach().clone()
            store['coordinates'].append(copy)
        return store

    def get_prob(self, x, model, cls=None, iters=10000):
        """
        Method takes x as input and calculates estimated point probability p(x)
        It does so by sampling many noisy versions of x, denoising them with the model and use estimator
        :param x: the input we wish to give a point estimate for
        :param model: the trained denoiser
        :param cls: the class of the point (for the conditional case)
        :return: estimated lower bound (ELBO) for p(x) as logprob
        """
        L = np.zeros(iters)
        for i in tqdm(range(iters)):
            t, x_t, _ = self.reverse_train(x)
            x_t = x_t.unsqueeze(dim=-1)
            step = torch.tensor([t * self.dt], device=self.device).unsqueeze(dim=-1)
            inp = torch.cat((x_t, step), dim=0).transpose(0, 1)
            if cls is not None:
                cond = torch.tensor([cls], device=self.device).unsqueeze(dim=-1)
                inp = torch.cat((inp, cond), dim=1)
            x_0 = x_t - (self.sigmas[t] * model(inp))
            SNR = (1/(self.sigmas[t - 1]**2)) - (1/(self.sigmas[t]**2))
            norm = torch.linalg.norm((x - x_0)).squeeze()
            L[i] = SNR.item() * norm.item()
        print("")
        return -(self.T/2) * np.mean(L)

    def set_classes(self):
        """
        Function uses generated data points (size (2, M)), and classifies them
        :return: list of size M with class index for each point in self.data
        """
        self.cls = torch.zeros(self.data.size()[1], device=self.device, dtype=torch.int)
        for i in range(self.data.size()[1]):
            self.cls[i] = classify(self.data[:2, i])
        return


def classify(x):
    """
    Function receives point, returns class
    :param x: 2D point
    :return: Class in concentric squares (one of five)
    """
    if torch.max(torch.abs(x)) > 0.8:
        cls = 4
    elif torch.max(torch.abs(x)) > 0.6 and torch.max(torch.abs(x)) <= 0.8:
        cls = 3
    elif torch.max(torch.abs(x)) > 0.4 and torch.max(torch.abs(x)) <= 0.6:
        cls = 2
    elif torch.max(torch.abs(x)) > 0.2 and torch.max(torch.abs(x)) <= 0.4:
        cls = 1
    elif torch.max(torch.abs(x)) <= 0.2:
        cls = 0
    return cls


def accuracy_classes(x, c):
    """
    function takes set of sample points and their conditioned classes
    classifies them and returns accuracy
    :param x: torch tensor, set of points size (m, 2)
    :param c: corresponding classes, tensor of ints, indices align with sample indices in x
    :return: accuracy score
    """
    m = x.size()[0]
    c = c.detach().squeeze().tolist()
    acc = [0, 0, 0, 0, 0]
    for i in range(m):
        truth = classify(x[i, :])
        if truth == c[i]:
            acc[truth] += 1
    for i in range(5):
        total = c.count(i)
        print(f"total points in class {i}: {total}")
        if total != 0:
            acc[i] = acc[i]/total
            print(f"accuracy for class {i}: {acc[i]}")
    return acc


def visualize_classes(data, cls):
    """
    function to visualize a classified set of points
    coloration automatically decided according to range of class
    :param data: set of 2D points as tensor of shape [2, M] and corresponding 1D tensor of class labels
    Displays a matplotlib figure.
    """
    x = data[0, :].cpu().detach().numpy()
    y = data[1, :].cpu().detach().numpy()
    c = cls[:].cpu().detach().numpy()
    colors = {0: "tab:blue", 1: "tab:orange", 2: "tab:green", 3: "tab:red", 4: "tab:purple"}
    plt.axis([-2.0, 2.0, -2.0, 2.0])
    plt.scatter(x, y, c=[colors[v] for v in c])
    plt.show()
    return


def sample(model, dif=None, points=1000, cond=False):
    """
    function to sample with a trained denoiser. It sets random seed and generates 1000 random points.
    It then runs a reverse process using the model.
    :param: model - a trained denoiser
    :param: points - int, number of points to generate
    :param: cond - bool, whether to condition on some class
    :return: lists of points before and after each run, divided into x and y coordinates for easy plotting.
    If conditional then returns corresponding list of classes.
    """
    print("Sampling")
    if dif is None:
        dif = Diffusion(datapoints=points)
    store = dif.generate(model, cls=cond)
    x_start = store['coordinates'][0][:, 0].tolist()
    x_denoised = store['coordinates'][-1][:, 0].tolist()
    y_start = store['coordinates'][0][:, 1].tolist()
    y_denoised = store['coordinates'][-1][:, 1].tolist()

    if cond is True:
        colors = {0: "tab:blue", 1: "tab:orange", 2: "tab:green", 3: "tab:red", 4: "tab:purple"}
        cls = store['class'].detach().squeeze().tolist()
        c = [colors[v] for v in cls]
    else:
        c = 'blue'

    figure, [axis1, axis2] = plt.subplots(ncols=2)
    axis1.axis([-2.0, 2.0, -2.0, 2.0])
    axis1.set_title('Before')
    axis1.scatter(x_start, y_start, c=c, s=4)
    axis2.axis([-2.0, 2.0, -2.0, 2.0])
    axis2.set_title('After')
    axis2.scatter(x_denoised, y_denoised, c=c, s=4)
    plt.show()
    return store


def sample9(model, dif=None, points=1000, cond=False):
    """
    like sample function but produces 9 samples with 9 different random seeds and plots on 3X3 figure
    """
    if dif is None:
        dif = Diffusion(datapoints=points)
    colors = {0: "tab:blue", 1: "tab:orange", 2: "tab:green", 3: "tab:red", 4: "tab:purple"}
    figure, axis = plt.subplots(3, 3)
    for row in range(3):
        for column in range(3):
            axis[row, column].axis([-2.0, 2.0, -2.0, 2.0])
            # set random seed
            torch.manual_seed(row + column)
            random.seed(row + column)
            store = dif.generate(model, cls=cond)
            print("frames: ", len(store['coordinates']))
            #x_start = store['coordinates'][0][:, 0].tolist()
            x_denoised = store['coordinates'][-1][:, 0].tolist()
            #y_start = store['coordinates'][0][:, 1].tolist()
            y_denoised = store['coordinates'][-1][:, 1].tolist()
            if cond is True:
                cls = store['class'].detach().tolist()
                c = [colors[v] for v in cls]
            else:
                c = 'blue'
            axis[row, column].scatter(x_denoised, y_denoised, c=c, s=4)
    plt.show()
    return



if __name__ == '__main__':

    device = ("cuda" if torch.cuda.is_available() else "cpu")

    # Training
    """
    net = [3, 256, 256, 256, 2]
    version = "diffusion_unconditional"
    train_model(1000, net=net, epochs=1000, save=version)

    # present losses
    with open(f'{version}-train-loss.pkl', 'rb') as f:
        losses = pickle.load(f)
    plt.plot(range(len(losses)), losses)
    plt.suptitle("Losses During Training")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.show()
    """

    #load model
    net = [3, 256, 256, 256, 2]
    version = "diffusion_unconditional"
    model = py_model(net)
    loaded_dict = torch.load(f'{version}.pt')
    model.load_state_dict(loaded_dict)
    model.eval()

    # Evaluate with 9 random seeds
    sample9(model)

    # Try different time-step lengths
    dif = Diffusion(T=1)
    sample(model, dif)
    dif = Diffusion(T=3)
    sample(model, dif)
    dif = Diffusion(T=5)
    sample(model, dif)

    # Try different noising schedules
    schedules = []
    dif = Diffusion(schedule_param="t")
    schedules.append([dif.sigmas[x].item() for x in range(0, 1000, 20)])
    sample(model, dif)
    dif = Diffusion(schedule_param="sqrt")
    schedules.append([dif.sigmas[x].item() for x in range(0, 1000, 20)])
    sample(model, dif)
    dif = Diffusion(schedule_param="exp")
    schedules.append([dif.sigmas[x].item() for x in range(0, 1000, 20)])
    sample(model, dif)
    t = np.linspace(0, 1000)
    plt.plot(t, schedules[0], 'r', label='sigma(t)=t')
    plt.plot(t, schedules[1], 'b', label='sigma(t)=sqrt(t)')
    plt.plot(t, schedules[2], 'g', label='sigma(t)=exp(5(t-1))')
    plt.legend(loc='upper left')
    plt.show()

    # Train conditional model
    """
    net = [3, 256, 256, 256, 2]
    version = "diffusion_conditional"
    train_model(1000, save=version, net=net, epochs=1000, cond=True, device=device)
    """

    # load conditional model
    net = [3, 256, 256, 256, 2]
    version = "diffusion_conditional"
    model = py_model(net, cond=True, device=device)
    loaded_dict = torch.load(f'{version}.pt')
    model.load_state_dict(loaded_dict)
    model.eval()

    # Evaluate
    sample(model, points=1000, cond=True)

    # Point estimation
    dif = Diffusion()
    point1 = dif.data[:, np.random.randint(0, dif.datapoints)]
    point2 = dif.data[:, np.random.randint(0, dif.datapoints)]
    cls1 = classify(point1)
    cls2 = classify(point2)
    sign1 = random.choice([-1, 1])
    sign2 = random.choice([-1, 1])
    point3 = point1 + torch.tensor([np.random.randint(2, 3)*sign1, np.random.randint(2, 3)*sign2], device=device)
    point4 = point2 + torch.tensor([np.random.randint(4, 5)*sign2, np.random.randint(4, 5)*sign1], device=device)
    print(f"Point estimation for {point1} with class {cls1}: ", dif.get_prob(point1, model, cls=cls1))
    cls_wrong = (cls1 + 1) % 5
    print(f"Point estimation for {point1} with wrong class {cls_wrong}: ", dif.get_prob(point1, model, cls=cls_wrong))
    cls_wrong = (cls_wrong + 1) % 5
    print(f"Point estimation for {point1} with wrong class {cls_wrong}: ", dif.get_prob(point1, model, cls=cls_wrong))
    print(f"Point estimation for {point2} with class {cls2}: ", dif.get_prob(point2, model, cls=cls2))
    cls_wrong = (cls2 + 1) % 5
    print(f"Point estimation for {point2} with wrong class {cls_wrong}: ", dif.get_prob(point2, model, cls=cls_wrong))
    cls_wrong = (cls_wrong + 1) % 5
    print(f"Point estimation for {point2} with wrong class {cls_wrong}: ", dif.get_prob(point2, model, cls=cls_wrong))
    print(f"Point estimation for out of bounds {point3}: ", dif.get_prob(point3, model, cls=4))
    print(f"Point estimation for further out of bounds {point4}: ", dif.get_prob(point4, model, cls=4))

