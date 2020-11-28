import numpy as np
import matplotlib.pyplot as plt
from create_functions_datasets import special_sine
from pdb import set_trace

def plot_ndim_sines(bg):
    test_data = bg.T.MVAL
    test_ans = bg.MTestAnswers
    plt.switch_backend("Qt5Agg")

    for i, data in enumerate(test_data):
        out_dim = data.original_output_shape[1]
        fig, axs = plt.subplots(2, 2)
        freqs_phases = data.name.split("_")
        freqs = freqs_phases[::2]
        phases = freqs_phases[1::2]
        inp = bg.D.denormalize_input(data.ValInput.cpu().numpy())
        ans = np.array(test_ans[i][1])
        for j in range(out_dim):
            ax = axs.flat[j]
            freq = float(freqs[j][1:].replace("d", "."))
            phase = float(phases[j][1:].replace("d", "."))
            x = np.linspace(-1, 1, 100)
            y = special_sine(x, freq, phase)
            out = ans[:, j]
            
            ax.plot(x, y, "r")
            ax.scatter(inp, out)

        set_trace()
        plt.show()
            
