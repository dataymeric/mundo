from mundo.backbones.arches import ArchesNet
from mundo.backbones.conv import ConvNet
from mundo.backbones.nat import NATNet
from mundo.backbones.simvp import SimVPNet

BACKBONES = {"conv": ConvNet, "arches": ArchesNet, "nat": NATNet, "simvp": SimVPNet}
