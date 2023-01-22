import time
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.optim import lr_scheduler
from torchvision import datasets, transforms, utils
from tensorboardX import SummaryWriter
from utils import * 
from model import * 
from PIL import Image
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser()
# data I/O
parser.add_argument('-i', '--data_dir', type=str,
                    default='data', help='Location for the dataset')
parser.add_argument('-o', '--save_dir', type=str, default='models',
                    help='Location for parameter checkpoints and samples')
parser.add_argument('-d', '--dataset', type=str,
                    default='cifar', help='Can be either cifar|mnist')
parser.add_argument('-p', '--print_every', type=int, default=50,
                    help='how many iterations between print statements')
parser.add_argument('-t', '--save_interval', type=int, default=10,
                    help='Every how many epochs to write checkpoint/samples?')
parser.add_argument('-r', '--load_params', type=str, default=None,
                    help='Restore training from previous model checkpoint?')
# model
parser.add_argument('-q', '--nr_resnet', type=int, default=5,
                    help='Number of residual blocks per stage of the model')
parser.add_argument('-n', '--nr_filters', type=int, default=160,
                    help='Number of filters to use across the model. Higher = larger model.')
parser.add_argument('-m', '--nr_logistic_mix', type=int, default=10,
                    help='Number of logistic components in the mixture. Higher = more flexible model')
parser.add_argument('-l', '--lr', type=float,
                    default=0.0002, help='Base learning rate')
parser.add_argument('-e', '--lr_decay', type=float, default=0.999995,
                    help='Learning rate decay, applied every step of the optimization')
parser.add_argument('-b', '--batch_size', type=int, default=1000,
                    help='Batch size during training per GPU')
parser.add_argument('-x', '--max_epochs', type=int,
                    default=5000, help='How many epochs to run in total?')
parser.add_argument('-s', '--seed', type=int, default=1,
                    help='Random seed to use')
parser.add_argument('-z', '--block_dim', type=int,
                    default=1, help='What is the block size?')

args = parser.parse_args()

# reproducibility
torch.manual_seed(args.seed)
np.random.seed(args.seed)

model_name = 'pcnn_lr{:.5f}_nr-resnet{}_nr-filters{}'.format(args.lr, args.nr_resnet, args.nr_filters)

sample_batch_size = 100
obs = (1, 28, 28) if 'mnist' in args.dataset else (3, 32, 32)
input_channels = obs[0]*args.block_dim * args.block_dim
rescaling     = lambda x : (x - .5) * 2.
rescaling_inv = lambda x : .5 * x  + .5
class Block(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, block_x, block_y):
        self.n_block_x_dim = block_x
        self.n_block_y_dim = block_y

    def __call__(self, sample):
        image = sample

        h, w = image.shape[1:]
        n_blocks = int((h*w) / (self.n_block_x_dim * self.n_block_y_dim))
        n_x_blocks = int(w / self.n_block_x_dim)
        n_y_blocks = int(h / self.n_block_y_dim)
        n_block_dim = 3*self.n_block_x_dim * self.n_block_y_dim
        x_rnn = np.zeros([n_block_dim, n_x_blocks, n_y_blocks], dtype = np.float32)

        for xi in range(n_x_blocks):
            for yi in range(n_y_blocks):
                x_rnn[:, xi,yi] = image[
                          :, (xi*self.n_block_x_dim):((xi+1)*self.n_block_x_dim),(yi*self.n_block_y_dim):((yi+1)*self.n_block_y_dim)
                ].flatten()

        return torch.tensor(x_rnn)
kwargs = {'num_workers':1, 'pin_memory':True, 'drop_last':True}
ds_transforms = transforms.Compose([transforms.ToTensor(), Block(args.block_dim, args.block_dim), rescaling])

if 'mnist' in args.dataset : 
    train_loader = torch.utils.data.DataLoader(datasets.MNIST(args.data_dir, download=True, 
                        train=True, transform=ds_transforms), batch_size=args.batch_size, 
                            shuffle=True, **kwargs)
    
    test_loader  = torch.utils.data.DataLoader(datasets.MNIST(args.data_dir, train=False, 
                    transform=ds_transforms), batch_size=args.batch_size, shuffle=True, **kwargs)
    
    loss_op   = lambda real, fake : discretized_mix_logistic_loss_1d(real, fake)
    sample_op = lambda x : sample_from_discretized_mix_logistic_1d(x, args.nr_logistic_mix)

elif 'cifar' in args.dataset : 
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=ds_transforms)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                          shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=ds_transforms)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                         shuffle=False, num_workers=2)
    loss_op   = lambda real, fake : energy_distance(real, fake)#discretized_mix_logistic_loss(real, fake)
    sample_op = lambda x : x[0]#sample_from_discretized_mix_logistic(x, args.nr_logistic_mix)
else :
    raise Exception('{} dataset not in {mnist, cifar10}'.format(args.dataset))

model = PixelCNN(nr_resnet=args.nr_resnet, nr_filters=args.nr_filters, 
            input_channels=input_channels, nr_logistic_mix=args.nr_logistic_mix)
model = model.cuda()

if args.load_params:
    load_part_of_model(model, args.load_params)
    model.load_state_dict(torch.load(args.load_params))
    print('model parameters loaded')

optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.lr_decay)

def sample(model):
    model.train(False)
    data = torch.zeros(sample_batch_size, obs[0]*args.block_dim * args.block_dim, obs[1]//args.block_dim, obs[2]//args.block_dim)
    data = data.cuda()
    for i in range(obs[1]//args.block_dim):
        for j in range(obs[2]//args.block_dim):
            data_v = Variable(data, volatile=True)
            out   = model(data_v, sample=True)
            out_sample = sample_op(out)
            data[:, :, i, j] = out_sample.data[:, :, i, j]
    data = data.cpu().detach().numpy()
    pixels2 = np.zeros(shape=(sample_batch_size, obs[0], obs[1], obs[2]))
    print(obs[0])
    for i in range(obs[1]//args.block_dim):
      for j in range(obs[2]//args.block_dim):
          digit = data[:, :, i, j].reshape([sample_batch_size, obs[0], args.block_dim, args.block_dim])
          pixels2[:, :, i * args.block_dim: (i + 1) * args.block_dim,
               j * args.block_dim: (j + 1) * args.block_dim] = digit

    return torch.from_numpy(pixels2).cuda()

print('starting training')
writes = 0
model.eval()
with torch.no_grad():
    sample_list = []
    start_time = time.time()
    for i in range(1):
        print(i, flush = True)
        sample_t = sample(model)
        sample_t = rescaling_inv(sample_t)
        sample_list.append(sample_t.cpu().detach().numpy().transpose(0, 2, 3, 1))
    print(time.time() - start_time, flush = True)
    final_sample = np.concatenate(sample_list, axis = 0)

    np.save("block1_alt.npy", final_sample)
#print(sample_t.size(), flush = True)
#utils.save_image(sample_t,'images/samples.png',            nrow=5, padding=0)
