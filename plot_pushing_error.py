import numpy as np
import matplotlib.pyplot as plt
import h5py
import torch
import os
from klampt.model.trajectory import Trajectory
from klampt.io.loader import save
from pdb import set_trace

def plot_pushing_error(bg, max_horiz=100):
    test_data = bg.T.MTEST
    test_ans = bg.MTestAnswers
    test_structures = bg.S.TestStructures
    plt.switch_backend("Qt5Agg")
    DATA = h5py.File("pushing.hdf5", "r")

    sse = np.zeros((max_horiz,))
    sse_base = np.zeros((max_horiz,))
    for i, (data, structure) in enumerate(zip(test_data, test_structures)):
        _, _, _, _, _, weights, composer = bg.run_MAML(structure, data, 
                ret_weights=True)
        inp = DATA[data.name + "-IN"]
        out = DATA[data.name + "-OUT"]
        nobj = out.shape[1] // 3
        curr_state = inp
        n_datapoints = 0
        for k in range(max_horiz):
            inp_tensor = bg.D.normalize_input(torch.from_numpy(curr_state[:-1,:])
                    .float().cuda())
            out_tensor  = composer.forward_with_weights(inp_tensor, weights)
            pred_out = bg.D.denormalize_output(out_tensor).cpu().detach().numpy()
            next_state = np.zeros((curr_state.shape[0]-1, curr_state.shape[1]))
            next_state[:, -6:] = curr_state[1:, -6:]
            for j in range(nobj):
                next_state[:, 6*j:6*j+2] = (curr_state[:-1, 6*j:6*j+2] 
                        + pred_out[:, 3*j:3*j+2])
                next_state[:, 6*j+2:6*j+4] = pred_out[:, 3*j:3*j+2]
                next_state[:, 6*j+4] = (curr_state[:-1, 6*j+4] + pred_out[:, 3*j+2])
                next_state[:, 6*j+5] = pred_out[:, 3*j+2]
                sse[k] += np.sum((next_state[:, 6*j:6*j+2] - inp[k+1:, 6*j:6*j+2])**2,
                        axis=None)
                sse_base[k] += np.sum((inp[:-(k+1), 6*j:6*j+2] - inp[k+1:, 6*j:6*j+2])**2,
                        axis=None)
                sse[k] += np.sum((next_state[:, 6*j+4] - inp[k+1:, 6*j+4])**2,
                        axis=None)
                sse_base[k] += np.sum((inp[:-(k+1), 6*j+4] - inp[k+1:, 6*j+4])**2,
                        axis=None)
                n_datapoints += 3*next_state.shape[0]
            curr_state = next_state

    rmses = np.sqrt(sse/n_datapoints)
    rmses_base = np.sqrt(sse_base/n_datapoints)
    fig = plt.figure()
    ax = fig.gca()
    ax.plot(np.linspace(0.01, max_horiz*0.01, max_horiz), rmses)
    ax.plot(np.linspace(0.01, max_horiz*0.01, max_horiz), rmses_base)
    ax.legend(["Ours", "Naive Baseline"])
    ax.set_xlabel("Prediction Horizon (s)")
    ax.set_ylabel("RMSE")
    ax.set_title("Pushing prediction error")
    plt.show()

def save_predicted_trajs(bg, out_dir):
    test_data = bg.T.MTEST
    test_structures = bg.S.TestStructures
    DATA = h5py.File("pushing.hdf5", "r")
    for i, (data, structure) in enumerate(zip(test_data, test_structures)):
        _, _, _, _, _, weights, composer = bg.run_MAML(structure, data, 
                ret_weights=True)
        inp = DATA[data.name + "-IN"]
        out = DATA[data.name + "-OUT"]
        nobj = out.shape[1] // 3
        curr_state = inp
        milestones = np.zeros((299, 2 + 3*nobj))
        for k in range(299):
            milestones[k, :2] = curr_state[0, -6:-4]
            for j in range(nobj):
                milestones[k, 2+3*j:4+3*j] = curr_state[0, 6*j:6*j+2]
                milestones[k, 4+3*j] = curr_state[0, 6*j+4]
            inp_tensor = bg.D.normalize_input(torch.from_numpy(curr_state[:-1,:])
                    .float().cuda())
            out_tensor  = composer.forward_with_weights(inp_tensor, weights)
            pred_out = bg.D.denormalize_output(out_tensor).cpu().detach().numpy()
            next_state = np.zeros((curr_state.shape[0]-1, curr_state.shape[1]))
            next_state[:, -6:] = curr_state[1:, -6:]
            for j in range(nobj):
                next_state[:, 6*j:6*j+2] = (curr_state[:-1, 6*j:6*j+2] 
                        + pred_out[:, 3*j:3*j+2])
                next_state[:, 6*j+2:6*j+4] = pred_out[:, 3*j:3*j+2]
                next_state[:, 6*j+4] = (curr_state[:-1, 6*j+4] + pred_out[:, 3*j+2])
                next_state[:, 6*j+5] = pred_out[:, 3*j+2]
            curr_state = next_state
        traj = Trajectory(times=np.linspace(0.01, 2.99, 299), milestones=milestones)
        save_name = os.path.join(out_dir, "pred_pose_" + data.name +".traj")
        save(traj, 'auto', save_name)
