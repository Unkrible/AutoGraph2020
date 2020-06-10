from models.nas_azcs import NasAzcs
from models.nas_azpo import NasAzpo
from models.nas_coauthorcs import NasCoauthorcs
from models.nas_coauthorphy import NasCoauthorphy
from models.nas_phy10000 import NasPhy10000

__all__ = [
    "GCN", "SAGE", "GAT", "NasCora", "NasCiteseer", "NasPubmed", "SimpleGCN", "EmbGCN",
    "NasAutoGraphA", "NasAutoGraphB", "NasAutoGraphD", "NasAutoGraphE",
    "NasCoauthorcs", "NasCoauthorphy", "NasPhy10000", "NasAzpo", "NasAzcs"
]


from .gcn import GCN
from .emb_gcn import EmbGCN
from .sage import SAGE
from .gat import GAT
from .nas_cora import NasCora
from .nas_citeseer import NasCiteseer
from .nas_pubmed import NasPubmed
from .simple_gcn import SimpleGCN
from .nas_autograph_a import NasAutoGraphA
from .nas_autograph_b import NasAutoGraphB
from .nas_autograph_c import NasAutoGraphC
from .nas_autograph_d import NasAutoGraphD
from .nas_autograph_e import NasAutoGraphE

MODEL_LIB = {
    'gcn': GCN,
    'emb_gcn': EmbGCN,
    'simple_gcn': SimpleGCN,
    'gat': GAT,
    'nas_cora': NasCora,
    'nas_citeseer': NasCiteseer,
    'nas_pubmed': NasPubmed,
    'nas_autograph_a': NasAutoGraphA,
    'nas_autograph_b': NasAutoGraphB,
    'nas_autograph_c': NasAutoGraphC,
    'nas_autograph_d': NasAutoGraphD,
    'nas_autograph_e': NasAutoGraphE,
    'nas_coauthorcs': NasCoauthorcs,
    'nas_coauthorphy': NasCoauthorphy,
    'nas_phy10000': NasPhy10000,
    'nas_azpo': NasAzpo,
    'nas_azcs': NasAzcs
}

MODEL_PARAMETER_LIB = {
    'default': [0.005, 0.5, 5e-4, 64],
    # 'nas_cora': [0.01, 0.9, 0.0001, 64],
    # 'nas_citeseer': [0.005, 0.8, 1e-05, 128],
    # 'nas_pubmed': [0.01, 0.4, 5e-05, 64],
    'nas_autograph_a': [0.01, 0.9, 0, 128],
    'nas_autograph_b': [0.001, 0.7, 0, 256],
    'nas_autograph_c': [0.0005, 0.8, 1e-05, 256],
    'nas_autograph_d': [0.005, 0.1, 0.001, 8],
    'nas_autograph_e': [0.005, 0.7, 0.0001, 32],
    'nas_coauthorcs': [0.005, 0.5, 1e-05, 64],
    'nas_coauthorphy': [0.01, 0.4, 5e-05, 128],
    'nas_phy10000': [0.001, 0.5, 0.0001, 128],
    'nas_azpo': [0.0005, 0.5, 0.0005, 32],
    'nas_azcs': [0.0005, 0.5, 1e-05, 512]
}
