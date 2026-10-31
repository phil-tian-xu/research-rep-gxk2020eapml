"""NN1-NN5 baselines and the approved deeper, wider NN6-NN10 extensions.

This module has no PyTorch dependency so launchers and CPU reports can inspect
model names without installing the optional NN environment.
"""


ARCHITECTURES = {
    "NN1": (32,),
    "NN2": (32, 16),
    "NN3": (32, 16, 8),
    "NN4": (32, 16, 8, 4),
    "NN5": (32, 16, 8, 4, 2),
    "NN6": (64, 32, 16, 8, 4, 2),
    "NN7": (128, 64, 32, 16, 8, 4, 2),
    "NN8": (256, 128, 64, 32, 16, 8, 4, 2),
    "NN9": (512, 256, 128, 64, 32, 16, 8, 4, 2),
    "NN10": (1024, 512, 256, 128, 64, 32, 16, 8, 4, 2),
}
