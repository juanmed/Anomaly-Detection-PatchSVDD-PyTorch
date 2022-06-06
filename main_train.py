import argparse
import torch
from codes import mvtecad
from functools import reduce
from torch.utils.data import DataLoader
from codes.datasets import *
from codes.networks import *
from codes.inspection import eval_encoder_NN_multiK
from codes.utils import *
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()

parser.add_argument('--obj', default='hazelnut', type=str)
parser.add_argument('--lambda_value', default=1, type=float)
parser.add_argument('--D', default=64, type=int)

parser.add_argument('--epochs', default=300, type=int)
parser.add_argument('--lr', default=1e-4, type=float)

args = parser.parse_args()


def train():
    obj = args.obj
    D = args.D
    lr = args.lr
        
    with task('Networks'):
        enc = EncoderHier(64, D).cuda()
        cls_64 = PositionClassifier(64, D).cuda()
        cls_32 = PositionClassifier(32, D).cuda()

        modules = [enc, cls_64, cls_32]
        params = [list(module.parameters()) for module in modules]
        params = reduce(lambda x, y: x + y, params)

        opt = torch.optim.Adam(params=params, lr=lr)

    with task('Datasets'):
        train_x = mvtecad.get_x_standardized(obj, mode='train')
        train_x = NHWC2NCHW(train_x)

        rep = 100
        datasets = dict()
        datasets[f'pos_64'] = PositionDataset(train_x, K=64, repeat=rep)
        datasets[f'pos_32'] = PositionDataset(train_x, K=32, repeat=rep)
        
        datasets[f'svdd_64'] = SVDD_Dataset(train_x, K=64, repeat=rep)
        datasets[f'svdd_32'] = SVDD_Dataset(train_x, K=32, repeat=rep)

        dataset = DictionaryConcatDataset(datasets)
        loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=5, pin_memory=True)

    print('Start training')
    fig0 = plt.figure(figsize=(10,10))
    ax1 = fig0.add_subplot(2,3,1)
    ax2 = fig0.add_subplot(2,3,2)
    ax3 = fig0.add_subplot(2,3,3)
    ax4 = fig0.add_subplot(2,3,4)
    ax5 = fig0.add_subplot(2,3,5)
    loss_list = []
    loss_p64 = []
    loss_p32 = []
    loss_s64 = []
    loss_s32 = []
    save_frequency = 1
    for i_epoch in range(args.epochs):
        print("Epoch {}".format(i_epoch))
        if i_epoch != 0:
            for module in modules:
                module.train()

            for d in loader:
                d = to_device(d, 'cuda', non_blocking=True)
                opt.zero_grad()

                loss_pos_64 = PositionClassifier.infer(cls_64, enc, d['pos_64'])
                loss_pos_32 = PositionClassifier.infer(cls_32, enc.enc, d['pos_32'])
                loss_svdd_64 = SVDD_Dataset.infer(enc, d['svdd_64'])
                loss_svdd_32 = SVDD_Dataset.infer(enc.enc, d['svdd_32'])

                loss = loss_pos_64 + loss_pos_32 + args.lambda_value * (loss_svdd_64 + loss_svdd_32)

                loss_list.append(loss.item())
                loss_p64.append(loss_pos_64.item())
                loss_p32.append(loss_pos_32.item())
                loss_s64.append(loss_svdd_64.item())
                loss_s32.append(loss_svdd_32.item())

                loss.backward()
                opt.step()

            # graph losses
            if (i_epoch % save_frequency) == 0 and epoch != 0:
                ax1.plot(train_losses, label="loss", color = "red")
                ax1.set_xlabel('Epoch')
                ax1.set_ylabel('Loss')
                ax1.set_title('loss')
                ax1.legend(loc='best')

                ax2.plot(loss_p64, label="loss_pos_64", color = "red")
                ax2.set_xlabel('Epoch')
                ax2.set_ylabel('Loss')
                ax2.set_title('loss_pos_64')
                ax2.legend(loc='best')

                ax3.plot(loss_p32, label="loss_pos_32", color = "red")
                ax3.set_xlabel('Epoch')
                ax3.set_ylabel('Loss')
                ax3.set_title('loss_pos_32')
                ax3.legend(loc='best')

                ax4.plot(loss_s64, label="loss_svdd_64 ", color = "red")
                ax4.set_xlabel('Epoch')
                ax4.set_ylabel('Loss')
                ax4.set_title('loss_svdd_64')
                ax4.legend(loc='best')

                ax5.plot(loss_s32, label="loss_svdd_32", color = "red")
                ax5.set_xlabel('Epoch')
                ax5.set_ylabel('Loss')
                ax5.set_title('loss_svdd_32')
                ax5.legend(loc='best')

                fig0.savefig("PatchSVDD_Training_Graphs.png")

        aurocs = eval_encoder_NN_multiK(enc, obj)
        log_result(obj, aurocs)
        enc.save(obj)


def log_result(obj, aurocs):
    det_64 = aurocs['det_64'] * 100
    seg_64 = aurocs['seg_64'] * 100

    det_32 = aurocs['det_32'] * 100
    seg_32 = aurocs['seg_32'] * 100

    det_sum = aurocs['det_sum'] * 100
    seg_sum = aurocs['seg_sum'] * 100

    det_mult = aurocs['det_mult'] * 100
    seg_mult = aurocs['seg_mult'] * 100

    print(f'|K64| Det: {det_64:4.1f} Seg: {seg_64:4.1f} |K32| Det: {det_32:4.1f} Seg: {seg_32:4.1f} |mult| Det: {det_sum:4.1f} Seg: {seg_sum:4.1f} |mult| Det: {det_mult:4.1f} Seg: {seg_mult:4.1f} ({obj})')


if __name__ == '__main__':
    train()
