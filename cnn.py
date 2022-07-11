# Author: Haoguang Liu
# Date: 2022.07.11 19:02
# E-mail: Liu.gravity@gmail.com

import torch
from torch.utils.data import DataLoader
from torchvision.datasets.mnist import MNIST
from torchvision import transforms
from cnn_model import Classifier

class Eval():
    def __init__(self, path):
        self.path = path

        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

        self.dataset = MNIST(root=self.path, train=False, transform=self.transform, download=False)

        self.test_loader = DataLoader(dataset=self.dataset, batch_size=64, shuffle=False)

        # load model
        self.net = Classifier()
        self.net.load_state_dict(torch.load('mnistCls_net_param.pkl', map_location=torch.device('cpu')))
        
        self.eval_acc = 0

    def eval(self):
        with torch.no_grad():
            for imgs, labels in self.test_loader:
                outputs = self.net(imgs)

                pred_cat = torch.max(outputs, 1)[1]
                eval_correct = (pred_cat == labels).sum()
                self.eval_acc += eval_correct.item()

        return self.eval_acc / len(self.dataset)

    def evalImg(self, input):
        img_t = self.transform(input)
        img_t = torch.unsqueeze(img_t, 0)   # [1, 28, 28] --> [1, 1, 28, 28]
        with torch.no_grad():
            output = self.net(img_t)
            pred_cat = torch.max(output, 1)[1]
        
        return pred_cat.item()

if __name__=='__main__':
    eval = Eval('/Users/lhg/work/pytorch-learn/mnist_data/')