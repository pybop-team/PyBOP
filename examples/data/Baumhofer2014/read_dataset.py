from pybamm import citations
from pybop import Datasets
from scipy.io import loadmat

citations.register("""@article{
    Baumhöfer2014,
    title={{Production caused variation in capacity aging trend and correlation to initial cell performance}},
    author={Baumhöfer, T and Brühl, M and Rothgang, S and Sauer, D},
    journal={Journal of Power Sources},
    volume={247},
    pages={332-338},
    year={2014},
    doi={10.1016/j.jpowsour.2013.08.108}
}""")
citations.register("""@article{
    Attia2022,
    title={{Review—“Knees” in Lithium-Ion Battery Aging Trajectories}},
    author={Attia, P and Bills, A and Brosa Planella, F and Dechent, P and dos Reis, G and Dubarry, M and Gasper, P and Gilchrist, R and Greenbank, S and Howey, D and Liu, O and Khoo, E and Preger, Y and Soni, A and Sripad, S and Stefanopoulou, A and Sulzer, V},
    journal={Journal of The Electrochemical Society},
    volume={169},
    pages={060517},
    year={2022},
    doi={10.1149/1945-7111/ac6d13}
}""")

matlab_degradation_data = loadmat("../../data/Baumhofer2014/baumhofer.mat")['lifetime'][0][0]
degradation_data = Datasets([
    {
        "Time [s]": dataset[0][0][0],
        "Capacity fade": dataset[0][0][1] / dataset[0][0][1][0]
    }
    for dataset in matlab_degradation_data
], domain="Time [s]")
