import numpy as np
import matplotlib.pyplot as plt
import h5py
import torch
from pdb import set_trace

def plot_pushing_error(bg, max_horiz=50):
    test_data = bg.T.MTEST
    test_ans = bg.MTestAnswers
    test_structures = bg.S.TestStructures
    plt.switch_backend("Qt5Agg")
    DATA = h5py.File("pushing.hdf5", "r")

    sse = np.zeros((max_horiz,))
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
                sse[k] += np.sum((next_state[:, 6*j+4] - inp[k+1:, 6*j+4])**2,
                        axis=None)
                n_datapoints += 3*next_state.shape[0]
            curr_state = next_state

    rmses = np.sqrt(sse/n_datapoints)
    fig = plt.figure()
    ax = fig.gca()
    ax.plot(np.linspace(0.01, max_horiz*0.01, max_horiz), rmses)
    ax.set_xlabel("Prediction Horizon (s)")
    ax.set_ylabel("RMSE")
    ax.set_title("Pushing prediction error")
    plt.show()
            
