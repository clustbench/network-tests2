import argparse
from netgen import SupecomputerNet

parser = argparse.ArgumentParser(description='Generate Node and Switch List in GML Format')
parser.add_argument('--nodes', type=int, help='Number of Nodes in supercomputer')
parser.add_argument('--switches', type=int, help='Number of Switches in supercomputer')
parser.add_argument('--conn_file', type=str, help='Name of file with connections')
parser.add_argument('--filename', type=str, default='net.gml', help='Name of output file')
parser.add_argument('--node_name_mapping', type=str, default=None, help='File node-name mappings')

args = parser.parse_args()

if __name__ == "__main__":
    g = SupecomputerNet(args.nodes, args.switches, args.conn_file)
    g.serialize(args.filename, args.node_name_mapping)

