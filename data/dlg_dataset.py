import os.path
import torchvision.transforms as transforms
import torch
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import os


class DLGDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_AB = os.path.join(opt.dataroot, opt.phase + 'AB')
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')

        self.AB_paths = sorted(make_dataset(self.dir_AB))
        self.A_paths = make_dataset(self.dir_A)
        self.B_paths = make_dataset(self.dir_B)

        self.A_paths = sorted(self.A_paths)
        self.B_paths = sorted(self.B_paths)
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        self.AB_size = len(self.AB_paths)
        self.transform = get_transform(opt)

    def __getitem__(self, index):
        # AB paired data
        AB_path = self.AB_paths[index % self.AB_size]
        AB = Image.open(AB_path).convert('RGB')
        w, h = AB.size
        w2 = int(w / 2)
        # A = AB.crop((0, 0, w2, h)).resize((self.opt.loadSize, self.opt.loadSize), Image.BICUBIC)
        # B = AB.crop((w2, 0, w, h)).resize((self.opt.loadSize, self.opt.loadSize), Image.BICUBIC)
        A = self.__scale_width(AB.crop((0, 0, w2, h)), self.opt.loadSize)
        B = self.__scale_width(AB.crop((w2, 0, w, h)), self.opt.loadSize)
        w, h = A.size
        A = transforms.ToTensor()(A)
        B = transforms.ToTensor()(B)
        # w_offset = random.randint(0, max(0, self.opt.loadSize - self.opt.fineSize - 1))
        # h_offset = random.randint(0, max(0, self.opt.loadSize - self.opt.fineSize - 1))
        w_offset = random.randint(0, max(0, w - self.opt.fineSize - 1))
        h_offset = random.randint(0, max(0, h - self.opt.fineSize - 1))

        A = A[:, h_offset:h_offset + self.opt.fineSize, w_offset:w_offset + self.opt.fineSize]
        B = B[:, h_offset:h_offset + self.opt.fineSize, w_offset:w_offset + self.opt.fineSize]

        A = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(A)
        B = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(B)

        if self.opt.which_direction == 'BtoA':
            input_nc = self.opt.output_nc
            output_nc = self.opt.input_nc
        else:
            input_nc = self.opt.input_nc
            output_nc = self.opt.output_nc

        if (not self.opt.no_flip) and random.random() < 0.5:
            idx = [i for i in range(A.size(2) - 1, -1, -1)]
            idx = torch.LongTensor(idx)
            A = A.index_select(2, idx)
            B = B.index_select(2, idx)

        if input_nc == 1:  # RGB to gray
            tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
            A = tmp.unsqueeze(0)

        if output_nc == 1:  # RGB to gray
            tmp = B[0, ...] * 0.299 + B[1, ...] * 0.587 + B[2, ...] * 0.114
            B = tmp.unsqueeze(0)

        # A, B unpaired data
        A_path = self.A_paths[index % self.A_size]
        if self.opt.serial_batches:
            index_B = index % self.B_size
        else:
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        # print('(A, B) = (%d, %d)' % (index_A, index_B))
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')

        A_ = self.transform(A_img)
        B_ = self.transform(B_img)
        if self.opt.which_direction == 'BtoA':
            input_nc = self.opt.output_nc
            output_nc = self.opt.input_nc
        else:
            input_nc = self.opt.input_nc
            output_nc = self.opt.output_nc

        if input_nc == 1:  # RGB to gray
            tmp = A_[0, ...] * 0.299 + A_[1, ...] * 0.587 + A_[2, ...] * 0.114
            A_ = tmp.unsqueeze(0)

        if output_nc == 1:  # RGB to gray
            tmp = B_[0, ...] * 0.299 + B_[1, ...] * 0.587 + B_[2, ...] * 0.114
            B_ = tmp.unsqueeze(0)

        return {'A': A, 'B': B, 'A_': A_, 'B_': B_, 'AB_paths': AB_path, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        return max([self.A_size, self.B_size, self.AB_size])

    def __scale_width(self, img, target_width):
        ow, oh = img.size
        if ow < oh:
            if (ow == target_width):
                return img
            w = target_width
            h = int(target_width * oh / ow)
            return img.resize((w, h), Image.BICUBIC)
        else:
            if (oh == target_width):
                return img
            h = target_width
            w = int(target_width * ow / oh)
            return img.resize((w, h), Image.BICUBIC)

    def name(self):
        return 'DLGDataset'
