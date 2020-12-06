import h5py
import numpy as np
from klampt.io.loader import loadTrajectory
import os, sys
from pdb import set_trace

MIN_OBJECTS = 1
MAX_OBJECTS = 3
NUM_TRAJS_PER_SIZE = 200
DATA_PATH = "pushing_dataset"

def main(out_name):
    # Start out by loading in all data
    DATA = dict()
    for nobj in range(MIN_OBJECTS, MAX_OBJECTS+1):
        for i in range(NUM_TRAJS_PER_SIZE):
            fn = os.path.join(DATA_PATH, str(nobj), 
                "pose_{}.traj".format(i))
            traj = loadTrajectory(fn)
            traj_len = len(traj.milestones)
            states = np.array(traj.milestones)
            inp = np.zeros((traj_len-2, 6*(nobj+1)))
            out = np.zeros((traj_len-2, 3*(nobj)))
            pusher_xy = states[:, :2]
            pusher_delta = pusher_xy[1:, :] - pusher_xy[:-1, :]
            inp[:, -6:-4] = pusher_xy[1:-1, :]
            inp[:, -4:-2] = pusher_delta[:-1, :]
            for j in range(nobj):
                obj_xy = states[:, 2+3*j:4+3*j]
                obj_angle = states[:, 4+3*j]
                obj_delta = obj_xy[1:,:] - obj_xy[:-1, :]
                angle_delta = obj_angle[1:] - obj_angle[:-1]
                fixed_angle_delta = angle_delta
                fixed_angle_delta[angle_delta > 1.0] -= 2*np.pi
                fixed_angle_delta[angle_delta < -1.0] += 2*np.pi
                inp[:, 6*j:6*j+2] = obj_xy[1:-1, :]
                inp[:, 6*j+2:6*j+4] = obj_delta[:-1, :]
                inp[:, 6*j+4] = np.cos(obj_angle[1:-1])
                inp[:, 6*j+5] = np.sin(obj_angle[1:-1])
                out[:, 3*j:3*j+2] = obj_delta[1:, :]
                out[:, 3*j+2] = fixed_angle_delta[1:]
            ori_name = str(nobj) + "_" + str(i)
            DATA[ori_name+"-IN"] = inp
            DATA[ori_name+"-OUT"] = out
    h = h5py.File(out_name)
    for k,v in DATA.items():
        h.create_dataset(k, data=v)

if __name__ == "__main__":
    out_name = sys.argv[1]
    main(out_name)
