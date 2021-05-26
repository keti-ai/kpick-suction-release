from ketisdk.vision.base.base_objects import BasObj
from .basic_classification_models import Net
import torchvision.transforms as transforms
import torch, torchvision
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
# from ..dataset.datasets import Dataset, get_mean_std
import random
from . import cifar_classification_models as models
import torch.backends.cudnn as cudnn
import shutil
from time import time

from .classification_utils import Logger, AverageMeter, accuracy, mkdir_p, savefig
from .classification_utils.progress.progress.bar import Bar
from PIL import Image
from torch.autograd import Variable
import torch.nn.functional as F
from ketisdk.import_basic_utils import *



class BasCifarClassfier(BasObj):
    def __init__(self,args=None, cfg_path=None, name='unnamed'):
        super().__init__(args=args, cfg_path=cfg_path, name=name)
        self.get_model()

    def load_params(self, args):
        super().load_params(args=args)
        # self.state = args.todict()
        self.get_data()

    def predict_array(self,im, transform):
        # t = time()
        # img = Image.fromarray(im)
        img = self.transform(im)
        if self.use_cuda: img = img.cuda()
        img = img.view(1, *img.size())
        img = Variable(img)
        predicted = self.model(img)
        probs = F.softmax(predicted, dim=1)
        if self.use_cuda: probs = probs.cpu()
        # print('classification elapsed: %.3fs' %(time()-t))
        return probs.detach().numpy().flatten()

    def predict_rgbd(self, rgbd):
        im = rgbd.resize(self.args.net.input_shape[:2]).array(get_rgb=self.args.net.get_rgb,
                                                          get_depth=self.args.net.get_depth,
                                                          depth2norm=self.args.net.depth2norm)
        return self.predict_array(im)

    def predict_arrays_m(self, ims, transform, print_info=True):
        timer = Timer()
        img_concat = self.arrays2tensors(ims=ims, transform=transform)
        timer.pin_time(label='concat')
        predicted = self.model(img_concat)
        probs = F.softmax(predicted, dim=1)
        if self.use_cuda: probs = probs.cpu()
        probs = probs.detach().numpy()
        timer.pin_time(label='predict')
        if print_info: print(timer.pin_times_str())
        return probs

    def arrays2tensors(self, ims, transform):
        img_concat = []
        for im in ims:
            img = transform(im)
            if self.use_cuda: img = img.cuda()
            img_concat.append(img)
        img_concat = torch.stack(img_concat)
        # img_concat= torch.autograd.Variable(img_concat, volatile=True)
        return img_concat

    # def predict_arrays(self, ims, print_info=True):
    #     timer = Timer()
    #     num_im = len(ims)
    #     if num_im <= self.args.test_batch: return self.predict_arrays_m(ims)
    #
    #     part = list(np.arange(0, num_im, self.args.test_batch))
    #     if part[-1] != num_im: part.append(num_im)
    #
    #     probs = []
    #     for i in range(len(part)-1):
    #         select_ims = ims[part[i]: part[i+1]]
    #         partial_probs = self.predict_arrays_m(select_ims, print_info=False)
    #         probs.append(partial_probs)
    #     probs = np.vstack(probs)
    #
    #     if print_info:
    #         print('test batch size: %d' %self.args.test_batch)
    #         print(timer.pin_times_str())
    #
    #
    #     return probs

    def predict_arrays(self, ims, Dataset, transform, print_info=True):
        num_im = len(ims)
        if num_im<self.args.net.test_batch: return self.predict_arrays_m(ims=ims, transform=transform, print_info=print_info)
        timer1 = Timer()
        testset = Dataset(name=self.args.net.dber , transform=transform, train=False, data=ims, im_shape=self.args.net.input_shape, indexing=True)
        testloader = torch.utils.data.DataLoader(testset, batch_size=self.args.net.test_batch,
                                                      shuffle=True, num_workers=self.args.net.num_workers)
        timer1.pin_time('load_data')

        num_batch = int(np.ceil(num_im/self.args.net.test_batch))
        probs = np.zeros((num_im, self.num_classes))

        # probs_tensor = torch.zeros((num_im, self.num_classes), dtype=torch.float,device=self.device)
        for batch_idx, (inputs, targets, indexes) in enumerate(testloader):
            timer = Timer()
            if self.use_cuda:
                inputs = inputs.cuda()
            # inputs = torch.autograd.Variable(inputs, volatile=True)
            outputs = self.model(inputs)
            outputs = F.softmax(outputs, dim=1)
            timer.pin_time('score')

            if self.use_cuda: outputs = outputs.cpu()
            outputs = outputs.detach().numpy()
            timer.pin_time('cvt_cpu')

            probs[(indexes,)] = outputs
            timer.pin_time('save_array')
            # print(timer.pin_times_str())

        timer1.pin_time('scoring_all')
        if print_info:
            print('batch_szie:%d-num_batch:%d-msPerBatch:%0.1f' %(self.args.net.test_batch, num_batch,
                                                                  1000*timer1.get_pinned_time('scoring_all')/num_batch))
            print(timer1.pin_times_str())

        return probs


    def predict_rgbds(self, rgbds):
        ims = []
        for rgbd in rgbds:
            ims.append(rgbd.resize(self.args.net.input_shape[:2]).array(get_rgb=self.args.net.get_rgb,
                                                                    get_depth=self.args.net.get_depth,
                                                                    depth2norm=self.args.net.depth2norm))
        probs = self.predict_arrays(ims)
        # print('classification elapsed: %.3fs' % (time() - t))
        return probs


    def trainval(self):
        # Train and val
        for epoch in range(self.start_epoch, self.args.net.epochs):
            self.adjust_learning_rate(epoch)

            print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, self.args.net.epochs, self.state['lr']))

            train_loss, train_acc = self.train()
            test_loss, test_acc = self.test()

            # append logger file
            self.logger.append([self.state['lr'], train_loss, test_loss, train_acc, test_acc])

            # save model
            is_best = test_acc > self.best_acc
            self.best_acc = max(test_acc, self.best_acc)
            self.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'acc': test_acc,
                'best_acc': self.best_acc,
                'optimizer': self.optimizer.state_dict(),
            }, is_best)

        self.logger.close()
        self.logger.plot()
        savefig(os.path.join(self.args.net.checkpoint_dir, 'log.eps'))

        print('Best acc:')
        print(self.best_acc)


    def train(self):
        # switch to train mode
        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        end = time()

        bar = Bar('Processing', max=len(self.trainloader))
        for batch_idx, (inputs, targets) in enumerate(self.trainloader):
            # measure data loading time
            data_time.update(time() - end)

            if self.use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda(async=True)
            inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

            # compute output
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1,2))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))

            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # measure elapsed time
            batch_time.update(time() - end)
            end = time()

            # plot progress
            bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                batch=batch_idx + 1,
                size=len(self.trainloader),
                data=data_time.avg,
                bt=batch_time.avg,
                total=bar.elapsed_td,
                eta=bar.eta_td,
                loss=losses.avg,
                top1=top1.avg,
                top5=top5.avg,
            )
            bar.next()
        bar.finish()
        return (losses.avg, top1.avg)

    def test(self):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        # switch to evaluate mode
        self.model.eval()

        end = time()
        bar = Bar('Processing', max=len(self.testloader))
        for batch_idx, (inputs, targets) in enumerate(self.testloader):
            # measure data loading time
            data_time.update(time() - end)

            if self.use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = torch.autograd.Variable(inputs, volatile=True), torch.autograd.Variable(targets)

            # compute output
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 2))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))

            # measure elapsed time
            batch_time.update(time() - end)
            end = time()

            # plot progress
            bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                batch=batch_idx + 1,
                size=len(self.testloader),
                data=data_time.avg,
                bt=batch_time.avg,
                total=bar.elapsed_td,
                eta=bar.eta_td,
                loss=losses.avg,
                top1=top1.avg,
                top5=top5.avg,
            )
            bar.next()
        bar.finish()
        return (losses.avg, top1.avg)



    def get_data(self):
        # # Transform
        # db_mean, db_std = get_mean_std(self.args.dber)
        #
        # self.transform_train = transforms.Compose([
        #     # transforms.RandomCrop(32, padding=4),
        #     # transforms.RandomHorizontalFlip(),
        #     transforms.ToTensor(),
        #     transforms.Normalize(db_mean, db_std),
        # ])
        # self.transform_test = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize(db_mean, db_std),
        # ])
        if self.args.flag.train_mode:
            # Data
            if not hasattr(self.args, 'combine_dir'): data_root = self.args.root_dir
            else: data_root = self.args.combine_dir
            testset = Dataset(name=self.args.dber, root=data_root, train=False, download=self.args.db_download, im_shape=self.args.input_shape)
            self.testloader = torch.utils.data.DataLoader(testset, batch_size=self.args.test_batch,
                                                          shuffle=True, num_workers=self.args.num_workers)

            trainset = Dataset(name=self.args.dber, root=data_root, train=True, download=self.args.db_download, im_shape=self.args.input_shape)
            self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.args.train_batch,
                                                           shuffle=True, num_workers=self.args.num_workers)
            self.classes = trainset.classes
        else: self.classes = self.args.net.classes

        self.num_classes = len(self.classes)

    def get_model(self):
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if self.use_cuda else "cpu")

        # Model
        print("==> creating model '{}'".format(self.args.net.arch))
        if self.args.net.arch.startswith('resnext'):
            model = models.__dict__[self.args.net.arch](
                cardinality=self.args.net.cardinality,
                num_classes=self.num_classes,
                depth=self.args.net.model_depth,
                widen_factor=self.args.net.widen_factor,
                dropRate=self.args.net.drop,
            )
        elif self.args.net.arch.startswith('densenet'):
            model = models.__dict__[self.args.net.arch](
                num_classes=self.num_classes,
                depth=self.args.net.depth,
                growthRate=self.args.net.growthRate,
                compressionRate=self.args.net.compressionRate,
                dropRate=self.args.net.drop,
            )
        elif self.args.net.arch.startswith('wrn'):
            model = models.__dict__[self.args.net.arch](
                num_classes=self.num_classes,
                depth=self.args.net.model_depth,
                widen_factor=self.args.net.widen_factor,
                dropRate=self.args.net.drop,
            )
        elif self.args.net.arch.endswith('resnet'):
            model = models.__dict__[self.args.net.arch](
                num_classes=self.num_classes,
                depth=self.args.net.model_depth,
                input_shape=self.args.net.input_shape
                # block_name=self.args.block_name,
            )
        elif self.args.net.arch.endswith('resnet_roi'):
            model = models.__dict__[self.args.net.arch](
                num_classes=self.num_classes,
                depth=self.args.net.model_depth,
                input_shape=self.args.net.input_shape
                # block_name=self.args.block_name,
            )
        else:
            model = models.__dict__[self.args.net.arch](num_classes=self.num_classes)
        self.model = torch.nn.DataParallel(model).cuda()

        cudnn.benchmark = True
        self.title = '%s-%s-' % (self.args.net.dber, self.args.net.arch)
        self.checkpoint_prefix = os.path.join(self.args.net.checkpoint_dir, self.title)
        if self.args.flag.train_mode:
            self.criterion = nn.CrossEntropyLoss()
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.args.net.lr, momentum=self.args.net.momentum,
                                       weight_decay=self.args.net.weight_decay)
            self.resume()
        else: self.load_model()

    def load_model(self):
        checkpoint_path = self.checkpoint_prefix + 'model_best.pth'
        checkpoint_path = checkpoint_path.replace('_roi', '')
        assert os.path.exists(checkpoint_path)
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()

    def resume(self):
        # save
        if not os.path.exists(self.args.checkpoint_dir): os.makedirs(self.args.checkpoint_dir)

        # Resume
        self.resume_checkpoint = self.checkpoint_prefix + 'model_best.pth'
        has_resume = os.path.exists(self.resume_checkpoint)

        if has_resume:
            # Load checkpoint.
            print('==> Resuming from checkpoint..')
            checkpoint = torch.load(self.resume_checkpoint)
            self.best_acc = checkpoint['best_acc']
            self.start_epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.logger = Logger(self.checkpoint_prefix + 'log.txt', title=self.title, resume=True)
        else:
            self.logger = Logger(self.checkpoint_prefix + 'log.txt', title=self.title)
            self.logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])
            self.best_acc = 0
            self.start_epoch = 0

    def save_checkpoint(self, state, is_best):
        filepath = self.checkpoint_prefix + str(state['epoch']).rjust(4, '0') + '.pth'
        temp = self.checkpoint_prefix +  'temp.pth'
        torch.save(state, temp)
        if is_best:
            shutil.copyfile(temp, self.resume_checkpoint)
        if state['epoch'] % self.args.save_every == 0:
            shutil.copyfile(temp, filepath)

    def adjust_learning_rate(self, epoch):
        if epoch in self.args.schedule:
            self.state['lr'] *= self.args.gamma
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.state['lr']


class RGBDLoader():

    def __init__(self, args, rgbds, transform):
        self.transform = transform
        self.ims = []
        for rgbd in rgbds:
            self.ims.append(rgbd.resize(args.input_shape[:2]).array(get_rgb=args.get_rgb,
                                                                    get_depth=args.get_depth))
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: image.
        """
        img = self.ims[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.ims)







