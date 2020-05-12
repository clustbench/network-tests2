from enum import Enum
import re

class ConnectionTypes(Enum):
    INFINIBAND_56GBIT = 1
    ETHERNET_100MBIT = 2

class DirectionTypes(Enum):
    ONE_DIRECTIONAL = 1
    BI_DIRECTIONAL = 2

ConnectStrings = {ConnectionTypes.INFINIBAND_56GBIT : "IB56GB", ConnectionTypes.ETHERNET_100MBIT : "ETH100MB"}
DirectString = {DirectionTypes.ONE_DIRECTIONAL : "ONE", DirectionTypes.BI_DIRECTIONAL : "BI"}

ConnectTypesMap = {"IB56GB" : ConnectionTypes.INFINIBAND_56GBIT, "ETH100MB" : ConnectionTypes.ETHERNET_100MBIT}
DirectTypesMap = {"ONE" : DirectionTypes.ONE_DIRECTIONAL,  "BI" : DirectionTypes.BI_DIRECTIONAL}

class SupercomputerNet:

    class Edge:
        def __init__(self, source, target, direct_type = DirectionTypes.BI_DIRECTIONAL, conn_type = ConnectionTypes.INFINIBAND_56GBIT):
            self.source = source
            self.target = target
            self.direct_type = direct_type
            self.conn_type = conn_type
            
        def __eq__(self, other):
            if (self.source != other.source) or (self.target != other.target) \
                or (self.conn_type != other.conn_type) or (self.direct_type != other.direct_type):
                return False
            return True

    def __init__(self, nodes, switches, conn_name):
        self.edgelist = []
        self.nodes = nodes 
        self.switches = switches
        self.supercomputername = ""

        self.__read_edge_list(conn_name)

    def __read_edge_list(self, filename):
        with open(filename, "r") as f:
            line = f.readline()
            while line:
                if line.strip():
                    line = line.strip()
                    line.replace(' ', '').replace('\t', '').replace('\r', '')
                    info = line.split('->')

                    if (len(info) < 2):
                        self.supercomputername = info[0]
                        line = f.readline()
                        continue

                    sources = info[0].strip('[]')
                    dests = info[1].strip('[]')

                    direct_type = DirectionTypes.BI_DIRECTIONAL
                    conn_type = ConnectionTypes.INFINIBAND_56GBIT

                    if len(info) > 2:
                        val = DirectTypesMap.get(info[2])
                        if val is not None:
                            direct_type = val
                        else:
                            val = ConnectTypesMap.get(info[2])
                            if val is not None:
                                conn_type = val
                            else:
                                raise RuntimeError('Cannot recognize link parameteter: {:s}'.format(info[2]))

                    if len(info) > 3:
                        conn_type = ConnectTypes[info[3]]
                    
                    src_ids = []
                    dest_ids = []

                    for src_rng in sources.split(','):
                        if 'node' in src_rng:
                            rng_type = 'node'
                            shift = 0
                        elif 'switch' in src_rng:
                            rng_type = 'switch'
                            shift = self.nodes
                        else:
                            raise RuntimeError('Interval can contain only \'nodes\' or \'switches\'.')

                        rng_split = src_rng.split(':')
                        rng_beg = rng_split[0]
                        rng_end = None
                        if len(rng_split) > 1:
                            rng_end = rng_split[1]
                        rng_beg = int(rng_beg.split(rng_type)[1])
                        if rng_end is not None:
                            rng_end = int(rng_end.split(rng_type)[1])
                        else:
                            rng_end = rng_beg
                        
                        if (rng_beg > rng_end):
                            raise RuntimeError('The start of the interval cannot be lesser than the end.')
                        
                        src_ids.extend([x + shift for x in range(rng_beg, rng_end + 1)])
                    
                    for dst_rng in dests.split(','):
                        if 'node' in dst_rng:
                            rng_type = 'node'
                            shift = 0
                        elif 'switch' in dst_rng:
                            rng_type = 'switch'
                            shift = self.nodes
                        else:
                            raise RuntimeError('Interval can contain only \'nodes\' or \'switches\'.')

                        rng_split = dst_rng.split(':')
                        rng_beg = rng_split[0]
                        rng_end = None
                        if len(rng_split) > 1:
                            rng_end = rng_split[1]
                        rng_beg = int(rng_beg.split(rng_type)[1])
                        if rng_end is not None:
                            rng_end = int(rng_end.split(rng_type)[1])
                        else:
                            rng_end = rng_beg

                        if (rng_beg > rng_end):
                            raise RuntimeError('The start of the interval cannot be lesser than the end.')

                        dest_ids.extend([x + shift for x in range(rng_beg, rng_end + 1)])

                    for src in src_ids:
                        for dst in dest_ids:
                            self.__put_edge_in_edgelist(self.Edge(src, dst, direct_type, conn_type))
                line = f.readline()

    def __put_edge_in_edgelist(self, edge):
        for e in self.edgelist:
            if e == edge:
                return

        self.edgelist.append(edge)


    def serialize(self, output_name, name_map):
        with open(output_name, 'w+') as f:
            f.write('graph [\n')
            f.write('\tnodesNum {:d}\n'.format(self.nodes))
            f.write('\tswitchesNum {:d}\n'.format(self.switches))
            f.write('\tdirected 1\n')
            f.write('\tsupercomputerName "{:s}"\n'.format(self.supercomputername))
            for i in range(1, self.nodes + 1):
                f.write('\tnode [\n')
                f.write('\t\tid {:d}\n'.format(i))
                f.write('\t\tlabel "node{:d}"\n'.format(i))
                f.write('\t\tdeviceType "node"\n')
                f.write('\t\tgraphics [\n')
                f.write('\t\t\ttype "circle"\n')
                f.write('\t\t\tfill "#FFCC00"\n')
                f.write('\t\t]\n')
                f.write('\t]\n')
            for i in range(1, self.switches + 1):
                f.write('\tnode [\n')
                f.write('\t\tid {:d}\n'.format(self.nodes + i))
                f.write('\t\tlabel "switch{:d}"\n'.format(i))
                f.write('\t\tdeviceType "switch"\n')
                f.write('\t\tgraphics [\n')
                f.write('\t\t\ttype "rectangle"\n')
                f.write('\t\t\tfill "#0F52BA"\n')
                f.write('\t\t]\n')
                f.write('\t]\n')
            for e in self.edgelist:
                f.write('\tedge [\n')
                f.write('\t\tsource {:d}\n'.format(e.source))
                f.write('\t\ttarget {:d}\n'.format(e.target))
                f.write('\t\tdirected "{:s}"\n'.format(DirectString[e.direct_type]))
                f.write('\t\tconnectionType "{:s}"\n'.format(ConnectStrings[e.conn_type]))
                f.write('\t\tgraphics [\n')
                f.write('\t\t\ttargetArrow "standard"\n')
                if (e.direct_type == DirectionTypes.BI_DIRECTIONAL):
                    f.write('\t\t\tsourceArrow "standard"\n')
                if (e.conn_type == ConnectionTypes.INFINIBAND_56GBIT):
                    f.write('\t\t\ffill "#33CCCC"\n')
                if (e.conn_type == ConnectionTypes.ETHERNET_100MBIT):
                    f.write('\t\t\ffill "#ff6600"\n')
                f.write('\t\t]\n')
                f.write('\t\t')

                f.write('\t]\n')
            f.write(']')
    

