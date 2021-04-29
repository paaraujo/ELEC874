# Python packages
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':

    PATH = 'kitti\\sequences\\99/speed.txt'
    speed = np.loadtxt(PATH) * 3.6
    mean = np.mean(speed)

    plt.figure()
    plt.plot(speed,color='b')
    plt.hlines(mean,0,6000,colors=['r'],linestyles=['dashed'])
    plt.title('Speed')
    plt.xlabel('epochs')
    plt.ylabel('km/h')
    plt.xlim([0,6000])
    plt.ylim([0,45*3.6])
    plt.show()
