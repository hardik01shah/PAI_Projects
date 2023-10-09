import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy as np
import pandas as pd
import os

def main():

    train_x = np.loadtxt('train_x.csv',delimiter=',',skiprows=1)  
    train_y = np.loadtxt('train_y.csv',delimiter=',',skiprows=1)  

    train_x_res = train_x[np.where(train_x[:,2]==0)[0],:2]
    train_y_res = train_y[np.where(train_x[:,2]==0)[0]]
    train_x_nonres = train_x[np.where(train_x[:,2]==1)[0],:2]
    train_y_nonres = train_y[np.where(train_x[:,2]==1)[0]]

    assert len(train_x) == len(train_x_res) + len(train_x_nonres)
    assert len(train_y) == len(train_y_res) + len(train_y_nonres)
    assert len(train_y_res) == len(train_x_res)
    assert len(train_y_nonres) == len(train_x_nonres)

    save_pth = os.path.join(os.getcwd(), "Plots")
    if not os.path.isdir(save_pth): os.makedirs(save_pth)

    """
    ax3 = plt.figure().add_subplot()
    ax3.scatter(train_x_res[:,0], train_x_res[:,1], label='residential')
    ax3.scatter(train_x_nonres[:,0], train_x_nonres[:,1], label='non-residential')
    ax3.legend()
    title = 'Latitude, Longitude and Area'
    ax3.set_title(title)
    ax3.set(xlabel='Latitude', ylabel='Longitude')
    plt.savefig(os.path.join(save_pth, f"{title}.png"))

    ax4 = plt.figure().add_subplot()
    ax4.scatter(train_x_res[:,0], train_y_res, label='residential')
    ax4.scatter(train_x_nonres[:,0], train_y_nonres, label='non-residential')
    ax4.legend()
    title = 'Latitude and PM25'
    ax4.set_title(title)
    ax4.set(xlabel='Latitude', ylabel='PM25')
    plt.savefig(os.path.join(save_pth, f"{title}.png"))

    ax5 = plt.figure().add_subplot()
    ax5.scatter(train_x_res[:,1], train_y_res, label='residential')
    ax5.scatter(train_x_nonres[:,1], train_y_nonres, label='non-residential')
    ax5.legend()
    title = 'Longitude and PM25'
    ax5.set_title(title)
    ax5.set(xlabel='Longitude', ylabel='PM25')
    plt.savefig(os.path.join(save_pth, f"{title}.png"))
    """

    # plot Latitiude and PM25 given longitude (and vice versa)
    # clear patterns here (in a range)

    ax6 = plt.figure().add_subplot()
    range = [0.3, 0.4]
    idx = np.where(np.logical_and(train_x[:,0]>=range[0], train_x[:,0]<=range[1]))
    ax6.scatter(train_x[idx,1], train_y[idx])
    title = f'Longitude and PM25 (without area classification) in Lat - {range}'
    ax6.set_title(title)
    ax6.set(xlabel='Longitude', ylabel='PM25')
    plt.savefig(os.path.join(save_pth, f"{title}.png"))

    ax7 = plt.figure().add_subplot()
    range = [0.3, 0.4]
    idx = np.where(np.logical_and(train_x[:,1]>=range[0], train_x[:,1]<=range[1]))
    ax7.scatter(train_x[idx,0], train_y[idx])
    title = f'Latitude and PM25 (without area classification) in Lon - {range}'
    ax7.set_title(title)
    ax7.set(xlabel='Latitude', ylabel='PM25')
    plt.savefig(os.path.join(save_pth, f"{title}.png"))

    """
    ax8 = plt.figure().add_subplot()
    ax8.scatter(train_x_res[:,0], train_y_res)
    title = 'Latitude and PM25 (residential areas)'
    ax8.set_title(title)
    ax8.set(xlabel='Latitude', ylabel='PM25')
    plt.savefig(os.path.join(save_pth, f"{title}.png"))

    ax9 = plt.figure().add_subplot()
    ax9.scatter(train_x_nonres[:,0], train_y_nonres)
    title = 'Latitude and PM25 (non-residential areas)'
    ax9.set_title(title)
    ax9.set(xlabel='Latitude', ylabel='PM25')
    plt.savefig(os.path.join(save_pth, f"{title}.png"))

    ax10 = plt.figure().add_subplot()
    ax10.scatter(train_x_res[:,1], train_y_res)
    title = 'Longitude and PM25 (residential areas)'
    ax10.set_title(title)
    ax10.set(xlabel='Longitude', ylabel='PM25')
    plt.savefig(os.path.join(save_pth, f"{title}.png"))

    ax11 = plt.figure().add_subplot()
    ax11.scatter(train_x_nonres[:,1], train_y_nonres)
    title = 'Longitude and PM25 (non-residential areas)'
    ax11.set_title(title)
    ax11.set(xlabel='Longitude', ylabel='PM25')
    plt.savefig(os.path.join(save_pth, f"{title}.png"))
    """

    plt.show()
    plt.clf()
    plt.close('all')
    
    """
    # 3d plots
    ax = plt.figure().add_subplot(projection='3d')
    ax.scatter(train_x_res[:,0], train_x_res[:,1], zs=train_y_res, zdir='z', label='residential')
    ax.scatter(train_x_nonres[:,0], train_x_nonres[:,1], zs=train_y_nonres, zdir='z', label='non-residential')
    ax.legend()
    title = 'PM25, Latitude, Longitude (with area markers)'
    ax.set_title(title)
    plt.savefig(os.path.join(save_pth, f"{title}.png"))

    ax2 = plt.figure().add_subplot(projection='3d')
    ax2.scatter(train_x[:,0], train_x[:,1], zs=train_y, zdir='z')
    title = 'PM25, Latitude, Longitude'
    ax2.set_title(title)
    plt.savefig(os.path.join(save_pth, f"{title}.png"))

    plt.show()
    """
if __name__=="__main__":
    main()