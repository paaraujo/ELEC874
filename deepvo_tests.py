# Python packages
import math
import random
import os
import glob
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optims
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch.nn.init import kaiming_normal_
from pykitti import odometry

class KittiDataset(Dataset):
    def __init__(self, seqs, seq_len, path, transform=None, normalizer=None, left=False):
        self.train_sets = []
        self.seq_len = seq_len
        for seq in seqs:
            self.train_sets.append(KittiRandomSequenceDataset(seq,seq_len,path,transform=transform,normalizer=normalizer,left=left))
        self.idxs = []
        self.seps = []
        for name, ts in enumerate(self.train_sets):
            for i in range(len(ts) - self.seq_len - 1):
                self.idxs.append(i)
                self.seps.append(name)

    def __len__(self):
        return len(self.seps)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.to_list()
        return self.train_sets[self.seps[idx]][self.idxs[idx]]

class KittiValidationDataset(Dataset):
    def __init__(self, seqs, seq_len, val_len, path, transform=None, normalizer=None, left=False):
        self.train_sets = []
        self.seq_len = seq_len
        self.val_len = val_len
        for seq in seqs:
            self.train_sets.append(KittiRandomSequenceDataset(seq,seq_len,path,transform=transform,normalizer=normalizer,left=left))
        self.idxs = []
        self.seps = []
        for name, ts in enumerate(self.train_sets):
            for i in range(0, len(ts) - seq_len - 1):
                self.idxs.append(i)
                self.seps.append(name)
        rand_idxs = np.random.choice(range(len(self.idxs)), val_len)
        self.idxs = np.stack(self.idxs)
        self.seps = np.stack(self.seps)
        self.idxs = self.idxs[rand_idxs]
        self.seps = self.seps[rand_idxs]

    def __len__(self):
        return self.val_len 

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.to_list()
        return self.train_sets[self.seps[idx]][self.idxs[idx]]

class KittiRandomSequenceDataset(Dataset):
    def __init__(self, seq, seq_len, path, transform=None, normalizer=None, left=False):
        self.odom = odometry(path, seq, poses=True, calib=False)
        self.transform = transform
        self.left = left
        self.normalizer = normalizer
        self.seq_len = seq_len

    def __len__(self):
        return len(self.odom) - 1 - self.seq_len

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.to_list()
        if self.left:
            reader = self.odom.get_cam2
        else:
            reader = self.odom.get_cam3
        rgb = []
        pos = []
        angle = []
        original_angle = torch.FloatTensor(euler_from_matrix(self.odom.poses[idx]))
        original_pos = torch.FloatTensor(se3_to_position(self.odom.poses[idx]))
        original_rot = se3_to_rot(self.odom.poses[idx]).T
        pos.append(original_pos)
        angle.append(original_angle)
        for i in range(self.seq_len):
            cur_rgb = self.transform(reader(idx + i))
            next_rgb = self.transform(reader(idx + i + 1))
            cur_rgb = cur_rgb - 0.5
            next_rgb = next_rgb - 0.5
            cur_rgb = self.normalizer(cur_rgb)
            next_rgb = self.normalizer(next_rgb)
            next_pos = torch.FloatTensor(se3_to_position(self.odom.poses[idx + i + 1]))
            next_angle = torch.FloatTensor(euler_from_matrix(self.odom.poses[idx + i + 1]))
            rgb.append(torch.cat((cur_rgb, next_rgb), dim=0))
            pos.append(next_pos)
            angle.append(next_angle)
        rgb = torch.stack(rgb)
        pos = torch.stack(pos)
        angle = torch.stack(angle)

        # preprocessing
        pos[1:] = pos[1:] - original_pos
        angle[1:] = angle[1:] - original_angle

        for i in range(1, len(pos)):
            loc = torch.FloatTensor(original_rot.dot(pos[i]))
            pos[i][:] = loc[:]

        pos[2:] = pos[2:] - pos[1:-1]
        angle[2:] = angle[2:] - angle[1:-1]

        for i in range(1, len(angle)):
            angle[i][0] = normalize_angle_delta(angle[i][0])

        return rgb, pos, angle

def eulerAnglesToRotationMatrix(theta):
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(theta[0]), -np.sin(theta[0])],
                    [0, np.sin(theta[0]), np.cos(theta[0])]
                    ])
    R_y = np.array([[np.cos(theta[1]), 0, np.sin(theta[1])],
                    [0, 1, 0],
                    [-np.sin(theta[1]), 0, np.cos(theta[1])]
                    ])
    R_z = np.array([[np.cos(theta[2]), -np.sin(theta[2]), 0],
                    [np.sin(theta[2]), np.cos(theta[2]), 0],
                    [0, 0, 1]
                    ])
    R = np.dot(R_z, np.dot(R_y, R_x))
    return R

def euler_from_matrix(matrix):
    # y-x-z Tait–Bryan angles intrincic
    # the method code is taken from https://github.com/awesomebytes/delta_robot/blob/master/src/transformations.py
    i = 2
    j = 0
    k = 1
    repetition = 0
    frame = 1
    parity = 0

    EPS = np.finfo(float).eps * 4.0

    M = np.array(matrix, dtype=np.float64, copy=False)[:3, :3]
    if repetition:
        sy = math.sqrt(M[i, j] * M[i, j] + M[i, k] * M[i, k])
        if sy > EPS:
            ax = math.atan2(M[i, j], M[i, k])
            ay = math.atan2(sy, M[i, i])
            az = math.atan2(M[j, i], -M[k, i])
        else:
            ax = math.atan2(-M[j, k], M[j, j])
            ay = math.atan2(sy, M[i, i])
            az = 0.0
    else:
        cy = math.sqrt(M[i, i] * M[i, i] + M[j, i] * M[j, i])
        if cy > EPS:
            ax = math.atan2(M[k, j], M[k, k])
            ay = math.atan2(-M[k, i], cy)
            az = math.atan2(M[j, i], M[i, i])
        else:
            ax = math.atan2(-M[j, k], M[j, j])
            ay = math.atan2(-M[k, i], cy)
            az = 0.0

    if parity:
        ax, ay, az = -ax, -ay, -az
    if frame:
        ax, az = az, ax
    return ax, ay, az

def se3_to_rot(mat):
    return mat[:3, :3]

def se3_to_position(mat):
    t = mat[:, -1][:-1]
    return t

# due to -pi to pi discontinuity
def normalize_angle_delta(angle):
    if (angle > np.pi):
        angle = angle - 2 * np.pi
    elif (angle < -np.pi):
        angle = 2 * np.pi + angle
    return angle

def conv(in_channel, out_channel, kernel_size, stride, padding, dropout):
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout)
    )

class DeepVO(nn.Module):
    def __init__(self):
        super(DeepVO, self).__init__()

        self.conv1 = conv(6, 64, 7, 2, 3, 0.2)
        self.conv2 = conv(64, 128, 5, 2, 2, 0.2)
        self.conv3 = conv(128, 256, 5, 2, 2, 0.2)
        self.conv3_1 = conv(256, 256, 3, 1, 1, 0.2)
        self.conv4 = conv(256, 512, 3, 2, 1, 0.2)
        self.conv4_1 = conv(512, 512, 3, 1, 1, 0.2)
        self.conv5 = conv(512, 512, 3, 2, 1, 0.2)
        self.conv5_1 = conv(512, 512, 3, 1, 1, 0.2)
        self.conv6 = conv(512, 1024, 3, 2, 1, 0.2)

        self.rnn = nn.LSTM(
            input_size=3 * 10 * 1024,
            hidden_size=1000,
            num_layers=2,
            dropout=0.2,
            batch_first=True)
        self.rnn_drop = nn.Dropout(0.5)
        self.linear = nn.Linear(1000, 6)

        # initalization from https://github.com/ChiWeiHsiao/DeepVO-pytorch/blob/master/model.py
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.LSTM):
                kaiming_normal_(m.weight_ih_l0)
                kaiming_normal_(m.weight_hh_l0)
                m.bias_ih_l0.data.zero_()
                m.bias_hh_l0.data.zero_()
                n = m.bias_hh_l0.size(0)
                start, end = n // 4, n // 2
                m.bias_hh_l0.data[start:end].fill_(1.)

                kaiming_normal_(m.weight_ih_l1)
                kaiming_normal_(m.weight_hh_l1)
                m.bias_ih_l1.data.zero_()
                m.bias_hh_l1.data.zero_()
                n = m.bias_hh_l1.size(0)
                start, end = n // 4, n // 2
                m.bias_hh_l1.data[start:end].fill_(1.)

    def forward(self, x):
        batch_size = x.size(0)
        seq_len = x.size(1)
        x = x.view(batch_size * seq_len, x.size(2), x.size(3), x.size(4))
        x = self.flow(x)
        x = x.view(batch_size, seq_len, -1)
        x, _ = self.rnn(x)
        x = self.rnn_drop(x)
        x = self.linear(x)
        return x

    def flow(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv3_1(x)
        x = self.conv4(x)
        x = self.conv4_1(x)
        x = self.conv5(x)
        x = self.conv5_1(x)
        x = self.conv6(x)
        return x

    def get_loss(self, seq, pos, ang):
        pos = pos[:, 1:, :]
        ang = ang[:, 1:, :]
        y_hat = self.forward(seq)
        pos_loss = nn.functional.mse_loss(y_hat[:, :, 3:], pos)
        ang_loss = nn.functional.mse_loss(y_hat[:, :, :3], ang)
        return 100 * ang_loss + pos_loss
     
def load_model(device, origin='', path=''):
    model = DeepVO()
    model.to(device)
    optimizer = optims.Adagrad(model.parameters(), lr=0.001)

    if origin == 'FlowNet':
        pretrained_flownet = torch.load(path, map_location=device)
        current_state_dict = model.state_dict()
        update_state_dict = {}
        for k, v in pretrained_flownet['state_dict'].items():
            if k in current_state_dict.keys():
                update_state_dict[k] = v
        current_state_dict.update(update_state_dict)
        print('[===== Loading FlowNet weights: ', model.load_state_dict(current_state_dict),' =====]')
        cur = 0

    elif origin == 'DeepVO':
        pretrained = torch.load(path, map_location=device)
        model_current_state_dict = model.state_dict()
        model_update_state_dict = {}
        optimizer_current_state_dict = optimizer.state_dict()
        optimizer_update_state_dict = {}

        if path.split('/')[-1] == 'pretrained.weights':
            pretrained = {'model_state_dict': pretrained}

        if 'model_state_dict' in pretrained.keys():
            for k, v in pretrained['model_state_dict'].items():
                if k in model_current_state_dict.keys():
                    model_update_state_dict[k] = v
        if 'optimizer_state_dict' in pretrained.keys():
            for k, v in pretrained['optimizer_state_dict'].items():
                if k in optimizer_current_state_dict.keys():
                    optimizer_update_state_dict[k] = v

        model_current_state_dict.update(model_update_state_dict)
        optimizer_current_state_dict.update(optimizer_update_state_dict)
        print('[===== Loading DeepVO weights: ', model.load_state_dict(model_current_state_dict),' =====]')
        optimizer.load_state_dict(optimizer_current_state_dict)
        cur = int(path.split("/")[-1].split(".")[0])

    else:
        raise Exception("Origin not defined [FlowNet|DeepVO].")

    return cur, model, optimizer

def test(model, dataloader, device, path, test_seq):
    model.eval()
    answer = [[0.0] * 6]
    gt = []
    odom = odometry(path, test_seq, poses=True, calib=False)
    for i in range(len(odom)):
        temp = []
        temp += list(euler_from_matrix(odom.poses[i]))
        temp += list(se3_to_position(odom.poses[i]))
        gt.append(temp)
        
    for i, batch in enumerate(dataloader):
        seq, pos, ang = batch
        seq = seq.to(device)
        pos = pos.to(device)
        ang = ang.to(device)
        predicted = model(seq)
        predicted = predicted.data.cpu().numpy()
        if i == 0:
            for pose in predicted[0]:
                for i in range(len(pose)):
                    pose[i] = pose[i] + answer[-1][i]
                answer.append(pose.tolist())
            predicted = predicted[1:]
        for pose in predicted:
            ang = eulerAnglesToRotationMatrix([0, answer[-1][0], 0])
            location = ang.dot(pose[-1][3:])
            pose[-1][3:] = location[:]

            last_pose = pose[-1]
            for j in range(len(last_pose)):
                last_pose[j] = last_pose[j] + answer[-1][j]
            last_pose[0] = (last_pose[0] + np.pi) % (2 * np.pi) - np.pi
            answer.append(last_pose.tolist())
    return gt, answer

def draw_route(y, y_hat, name, weight_folder, c_y="r", c_y_hat="b"):
    plt.figure()
    plt.clf()
    x = [v[3] for v in y]
    y = [v[5] for v in y]
    plt.plot(x, y, color=c_y, label="ground truth")

    x = [v[3] for v in y_hat]
    y = [v[5] for v in y_hat]
    plt.plot(x, y, color=c_y_hat, label="estimated")
    plt.title(f'Seq {name}')
    plt.legend()
    plt.savefig(f"{weight_folder}/" + name)
    plt.gca().set_aspect('equal', adjustable='datalim')
    plt.show()

if __name__ == '__main__':

    # random seeds
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.benchmark = True

    # constants
    SEQ_LEN = 4
    TRAIN_SEQ = ['00','02','08','09']
    TEST_SEQ = ['01','03','04','05','06','07','10','99']
    PREV_WEIGHTS_PATH = 'opt6/100.weights'
    PREV_WEIGHTS_ORIGIN = 'DeepVO'
    KITTI_PATH = 'kitti'
    WEIGHTS_PATH = 'opt6'

    # transforms
    transform = transforms.Compose([
        transforms.Resize((640, 192)),
        transforms.ToTensor()
    ])

    # used with opt1, opt4
    #normalizer = transforms.Compose([
    #    transforms.Normalize(
    #        (-0.1497185379266739, -0.1330135315656662, -0.14354705810546875),
    #        (0.31963783502578735, 0.3207162916660309, 0.3213176131248474))
    #])

    # used with opt2, opt5
    #normalizer = transforms.Compose([
    #    transforms.Normalize(
    #        (-0.15923872590065002, -0.13179142773151398, -0.13611172139644623),
    #        (0.30761271715164185, 0.308977335691452, 0.31186044216156006))
    #])

    # used with opt3, opt6
    normalizer = transforms.Compose([
        transforms.Normalize(
            (-0.18282340466976166, -0.12876379489898682, -0.11769190430641174),
            (0.27557167410850525, 0.27776771783828735, 0.28709277510643005))
    ])

    # building model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cur, model, optimizer = load_model(device, origin=PREV_WEIGHTS_ORIGIN, path=PREV_WEIGHTS_PATH)

    # testing
    for seq in TEST_SEQ:
        print(f'Sequence {seq} initialized.')
        test_dl = DataLoader(KittiDataset([seq], SEQ_LEN, KITTI_PATH, transform=transform, normalizer=normalizer, left=True), batch_size=8, num_workers=8, pin_memory=True, drop_last=True, shuffle=False)
        gt,estimated = test(model, test_dl, device, KITTI_PATH, test_seq=seq)
        gt = np.array(gt)
        estimated = np.array(estimated)
        draw_route(gt,estimated,seq,WEIGHTS_PATH)

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
        ax1.plot(gt[:,0],color='r',label='ground truth')
        ax1.plot(estimated[:,0],color='b',label='estimated')
        ax1.set_ylabel('Azimuth [rad]')
        ax1.set_xticklabels([])
        ax1.grid()
        ax2.plot(gt[:,1],color='r',label='ground truth')
        ax2.plot(estimated[:,1],color='b',label='estimated')
        ax2.set_ylabel('Pitch [rad]')
        ax2.set_xticklabels([])
        ax2.grid()
        ax3.plot(gt[:,2],color='r',label='ground truth')
        ax3.plot(estimated[:,2],color='b',label='estimated')
        ax3.set_ylabel('Roll [rad]')
        ax3.grid()
        plt.savefig(f"{WEIGHTS_PATH}/attitude_angles_{seq}")
        plt.show()     
