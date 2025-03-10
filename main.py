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
from tqdm import tqdm
from model_conditional import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())
print(torch.cuda.device_count())
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
                    default=3e-4, help='Base learning rate')
parser.add_argument('-e', '--lr_decay', type=float, default=1,
                    help='Learning rate decay, applied every step of the optimization')
parser.add_argument('-b', '--batch_size', type=int, default=50,
                    help='Batch size during training per GPU')
parser.add_argument('-x', '--max_epochs', type=int,
                    default=5000, help='How many epochs to run in total?')
parser.add_argument('-s', '--seed', type=int, default=1,
                    help='Random seed to use')
parser.add_argument('-z', '--block_dim', type=int,
                    default=1, help='What is the block size?')
parser.add_argument('-c', '--conditional', action = 'store_true')
parser.add_argument('--name', type = str, default = "pcnn", help = "name of model")
args = parser.parse_args()

# reproducibility
torch.manual_seed(args.seed)
np.random.seed(args.seed)

model_name = args.name+('{}'.format(args.block_dim))
assert not os.path.exists(os.path.join('runs', model_name)), '{} already exists!'.format(model_name)
writer = SummaryWriter(log_dir=os.path.join('runs', model_name))

sample_batch_size = 25
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
    loss_op   = lambda real, fake : energy_distance(real, fake)#kernelized_energy_distance(real, fake, device = device, reduction = "sum")#discretized_mix_logistic_loss(real, fake)
    sample_op = lambda x : x[0]#sample_from_discretized_mix_logistic(x, args.nr_logistic_mix)
else :
    raise Exception('{} dataset not in {mnist, cifar10}'.format(args.dataset))
if args.conditional:
    model = PixelCNN_Conditional(nr_resnet=args.nr_resnet, nr_filters=args.nr_filters*args.block_dim, 
            input_channels=input_channels, nr_logistic_mix=args.nr_logistic_mix)
else:
    model = PixelCNN(nr_resnet=args.nr_resnet, nr_filters=args.nr_filters*args.block_dim,
            input_channels=input_channels, nr_logistic_mix=args.nr_logistic_mix)
model = model.cuda()

if args.load_params:
    load_part_of_model(model, args.load_params)
    # model.load_state_dict(torch.load(args.load_params))
    print('model parameters loaded')

optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.lr_decay)

def sample(model):
    model.train(False)
    data = torch.zeros(sample_batch_size, obs[0], obs[1], obs[2])
    data = data.cuda()
    for i in range(obs[1]):
        for j in range(obs[2]):
            data_v = Variable(data, no_grad=True)
            out   = model(data_v, sample=True)
            out_sample = sample_op(out)
            data[:, :, i, j] = out_sample.data[:, :, i, j]
    return data

print('starting training')
writes = 0
for epoch in range(args.max_epochs):
    model.train(True)
    torch.cuda.synchronize()
    train_loss = 0.
    time_ = time.time()
    model.train()
    with tqdm(enumerate(train_loader)) as pbar:
        for batch_idx, (input,_) in pbar:
            input = input.cuda(non_blocking=True)
            input = Variable(input)
            

            fs = [int(y) for y in input.shape]
            if not args.conditional:
                fs[0] = fs[0]*10
            alpha = torch.rand(fs).cuda()
            alpha2 = torch.rand(fs).cuda()
            if args.conditional:
                output = model(input, alpha)
                output2 = model(input, alpha2)
                output = (output, output2)
            else:
                output = model(input, alpha)
            loss = loss_op(input, output)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.data.item()
            if (batch_idx +1) % args.print_every == 0 : 
                deno = args.print_every * args.batch_size * np.prod(obs) * np.log(2.)
                writer.add_scalar('train/bpd', (train_loss / deno), writes)
                print(('loss : {:.4f}, time : {:.4f}'.format(
                    (train_loss / deno), 
                    (time.time() - time_))))
                train_loss = 0.
                writes += 1
                time_ = time.time()
            pbar.set_description(
                        f"Loss: {loss:.5f}"
                        )
            

    # decrease learning rate
    scheduler.step()
    
    torch.cuda.synchronize()
    model.eval()
    test_loss = 0.
    for batch_idx, (input,_) in enumerate(test_loader):
        input = input.cuda(non_blocking=True)
        input_var = Variable(input)
        fs = [int(y) for y in input.shape]
        if not args.conditional:
            fs[0] = fs[0]*10
        alpha = torch.rand(fs).cuda()
        alpha2 = torch.rand(fs).cuda()
        if args.conditional:
            output = model(input, alpha)
            output2 = model(input, alpha2)
            output = (output, output2)
        else:
            output = model(input, alpha)

        loss = loss_op(input_var, output)
        test_loss += loss.data.item()
        del loss, output

    deno = batch_idx * args.batch_size * np.prod(obs) * np.log(2.)
    writer.add_scalar('test/bpd', (test_loss / deno), writes)
    print(('test loss : %s' % (test_loss / deno)))
    
    if (epoch + 1) % args.save_interval == 0: 
        torch.save(model.state_dict(), 'models/{}_{}.pth'.format(model_name, epoch))
        #print('sampling...')
        #sample_t = sample(model)
        #sample_t = rescaling_inv(sample_t)
        #utils.save_image(sample_t,'images/{}_{}.png'.format(model_name, epoch),            nrow=5, padding=0)
