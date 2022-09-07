import numpy as np
from mplot_thread import Mplot2d, Mplot3d
import cv2

results = np.load('/home/javier/Desktop/Resultados.npy', allow_pickle=True)

idx = 0

time_plt = Mplot2d(xlabel='img id', ylabel='s', title='Execution times')
err_plt = Mplot2d(xlabel='img id', ylabel='m',title='error')
matched_points_plt = Mplot2d(xlabel='img id', ylabel='# matches', title='# matches')
plt3d = Mplot3d(title='3D trajectory')

for img_id, _ in enumerate(results[idx]['track']):
    # Execution times
    stop_time = [img_id, results[idx]['exec_time'][img_id]]
    time_plt.draw(stop_time, 'exec time', color='g')
    time_plt.refresh()

    # Errors
    errx = [img_id,results[idx]['errors'][img_id][0]]
    erry = [img_id,results[idx]['errors'][img_id][1]]
    errz = [img_id,results[idx]['errors'][img_id][2]]
    err_plt.draw(errx, 'err_x', color='g')
    err_plt.draw(erry, 'err_y', color='b')
    err_plt.draw(errz, 'err_z', color='r')
    err_plt.refresh()

    # Matches
    matched_kps_signal = [img_id, results[idx]['matches'][img_id][0]]
    inliers_signal = [img_id, results[idx]['matches'][img_id][1]]
    matched_points_plt.draw(matched_kps_signal, '# matches', color='b')
    matched_points_plt.draw(inliers_signal, '# inliers', color='g')
    matched_points_plt.refresh()

# 3D path
plt3d.drawTraj(results[idx]['3D'][0], 'ground truth', color='r', marker='.')
plt3d.drawTraj(results[idx]['3D'][1], 'estimated', color='g', marker='.')
plt3d.refresh()

input("Press Enter to continue...")

plt3d.quit()
cv2.destroyAllWindows()