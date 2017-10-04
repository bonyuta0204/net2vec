import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

import torchvision.transforms as transforms
import torchvision.models as models

import numpy as np

import os

def alpha_prior(x, alpha=2.):
    #return torch.sum(torch.pow(torch.abs(x),alpha))
    return torch.abs(x.view(-1)**alpha).sum()


def tv_norm(x, beta=2.):
    assert(x.size(0) == 1)
    img = x[0]
    dy = torch.mean(torch.abs(img[:,:-1,:] - img[:,1:,:]).pow(beta))
    dx = torch.mean(torch.abs(img[:,:,:-1] - img[:,:,1:]).pow(beta))
    return dy + dx


def norm_loss(input, target):
    return torch.div(alpha_prior(input - target, alpha=2.), alpha_prior(target, alpha=2.))


class Alpha_Norm(nn.Module):
    def __init__(self, alpha):
        super(Alpha_Norm, self).__init__()
        self.alpha = alpha
        
    def forward(self, x):
        a_norm = (torch.abs(x.view(-1))**self.alpha).sum()
        return a_norm


class TV_Norm(nn.Module):
    def __init__(self, beta, cuda=False):
        super(TV_Norm, self).__init__()
        self.beta = beta
        self.diff_x = nn.Conv2d(3, 3, kernel_size=(1,2), groups=3, padding=0, bias=False)
        self.diff_y = nn.Conv2d(3, 3, kernel_size=(2,1), groups=3, padding=0, bias=False)
        if cuda:
            self.diff_x.cuda()
            self.diff_y.cuda()
        #initialise weights appropriately
        diff_weight = np.array([-1,1], dtype=np.float32)
        diff_weight = np.tile(diff_weight[None,None,None,:],(3,1,1,1))
        diff_weight = torch.from_numpy(diff_weight)
        list(self.diff_x.parameters())[0].data = diff_weight.clone()
        list(self.diff_y.parameters())[0].data = diff_weight.transpose(2,3).clone()
        
    def forward(self, x):
        dx = self.diff_x(F.pad(x, (0,1,0,0), mode='replicate'))#.add_(1e-5)
        dy = self.diff_y(F.pad(x, (0,0,0,1), mode='replicate'))#.add_(1e-5)
        tv = ((dx**2 + dy**2)**(self.beta/2.)).sum()
        return tv


class Feature_Distance(nn.Module):
    def __init__(self, target):
        super(Feature_Distance, self).__init__()
        self.target = target
        self.target_norm = Alpha_Norm(2)(target)
        
    def forward(self, x):
        feature_distance = Alpha_Norm(2)(x-self.target) / self.target_norm
        return feature_distance


class Denormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    
    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


class Clip(object):
    def __init__(self):
        return

    def __call__(self, tensor):
        t = tensor.clone()
        t[t>1] = 1
        t[t<0] = 0
        return t


#alexnet definition that conveniently let's you grab the outputs from any layer. 
#Also we ignore dropout here
class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        #convolutional layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
        self.conv2 = nn.Conv2d(64, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        self.conv3 = nn.Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv4 = nn.Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv5 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #pooling layers
        self.pool1 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), dilation=(1, 1))
        self.pool2 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), dilation=(1, 1))
        self.pool5 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), dilation=(1, 1))
        #fully connected layers
        self.fc1 = nn.Linear(9216, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 1000)
        
    def forward(self, x, out_keys):
        out = {}
        out['c1'] = self.conv1(x)
        out['r1'] = F.relu(out['c1'])
        out['p1'] = self.pool1(out['r1'])
        out['r2'] = F.relu(self.conv2(out['p1']))
        out['p2'] = self.pool2(out['r2'])
        out['r3'] = F.relu(self.conv3(out['p2']))
        out['r4'] = F.relu(self.conv4(out['r3']))
        out['r5'] = F.relu(self.conv5(out['r4']))
        out['p5'] = self.pool5(out['r5'])
        out['fc1'] = F.relu(self.fc1(out['p5'].view(1, -1)))
        out['fc2'] = F.relu(self.fc2(out['fc1']))
        out['fc3'] = self.fc3(out['fc2'])
        res = {}
        for k in out_keys:
            res[k] = out[k]
        #return [out[key] for key in out_keys]
        return res

def alexnet(pretrained=False, **kwargs):
    alexnet = AlexNet(**kwargs)
    if pretrained:
        state_dict = model_zoo.load_url('https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth')
        keys = alexnet.state_dict().keys()
        weights = {}
        for k, key in enumerate(state_dict.keys()):
            weights[keys[k]] = state_dict[key]
        alexnet.load_state_dict(weights)
    return alexnet


#function to decay the learning rate
def decay_lr(optimizer, factor):
    for param_group in optimizer.param_groups:
        param_group['lr'] *= factor 


def get_pytorch_module(net, blob):
    modules = blob.split('.')
    if len(modules) == 1:
        return net._modules.get(blob)
    else:
        curr_m = net
        for m in modules:
            curr_m = curr_m._modules.get(m)
        return curr_m


def main():
    gpu = 0

    cuda = True if gpu is not None else False
    use_mult_gpu = isinstance(gpu, list)
    if cuda:
        if use_mult_gpu:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu).strip('[').strip(']')
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = '%d' % gpu
    print(torch.cuda.device(), use_mult_gpu, cuda)

    mu = [0.485, 0.456, 0.406]
    sigma = [0.229, 0.224, 0.225]

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mu, sigma),
    ])

    detransform = transforms.Compose([
        Denormalize(mu, sigma),
        #transforms.Normalize(mean=[0, 0, 0], std=np.true_divide([1,1,1], sigma)),
        #transforms.Normalize(mean=-1*mu, std=[1,1,1]),
        #Clip(),
        transforms.ToPILImage(),
    ])

    model = alexnet(pretrained=True)
    model.eval()
    if cuda:
        model.cuda()

    img_path = '/home/ruthfong/NetDissect/dataset/broden1_227/images/pascal/2008_004017.jpg'
    img_ = transform(Image.open(img_path)).unsqueeze(0)

    blob = 'p5' # pool5
    input_var = Variable(img_.cuda() if cuda else img_)
    ref_acts = model(input_var, [blob])[blob].detach()

    #features_blobs = []
    #def hook_feature(module, input, output):
    #    features_blobs.append(output)
    #hook = get_pytorch_module(model, blob).register_forward_hook(hook_feature)
    #input = Variable(img_.cuda() if cuda else img_)
    #output = model.forward(input, 'p5')

    x_ = Variable((1e-3 * torch.randn(*img_.size()).cuda() if cuda else 
        1e-3 * torch.randn(*img_.size())))

    optimizer = torch.optim.SGD([x_], lr=1e3, momentum=0.9)

    alpha_lambda = 1e-5
    tv_lambda = 1e-5

    num_epochs = 200 

    for i in range(num_epochs):
        acts = model(x_, [blob])[blob]

        alpha_term = alpha_prior(x_, alpha=6)
        tv_term = tv_norm(x_, beta=2)
        loss_term = norm_loss(acts, ref_acts)

        tot_loss = alpha_lambda*alpha_term + tv_lambda*tv_term + loss_term

        if i % 25 == 0:
            print('Epoch %d:\tAlpha: %f\tTV: %f\tLoss: %f\tTot Loss: %f' % (i+1, i
                alpha_term.data.cpu().numpy()[0], tv_term.data.cpu().numpy()[0],
                loss_term.data.cpu().numpy()[0], tot_loss.data.cpu().numpy()[0]))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            decay_lr(optimizer, 1e-1)

        if i % 10 == 0:
            f, ax = plt.subplots(1,1)
            ax.imshow(detransform(x_[0].data.cpu()))
            plt.show()


if __name__ == '__main__':
    main()
