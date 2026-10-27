# Data

The three `.csv` files follow the network format used by the paper's released
code (`From, To, Volume, Cost`).  `Cost` is the mean travel time of a link;
`Volume` is not used by FMA.

| Network | File | Nodes | Links | Mean link cost |
|---|---|---:|---:|---:|
| Anaheim | `Anaheim/Anaheim_network.csv` | 416 | 914 | 0.905 |
| Winnipeg | `Winnipeg/Winnipeg_network.csv` | 1040 (IDs up to 1052) | 2836 | 0.775 |
| Chicago Sketch | `Chicago_Sketch/Chicago_Sketch_network.csv` | 933 | 2950 | 3.666 |

Node identifiers are 1-based and may be non-contiguous (Winnipeg has 12
missing IDs); `src/network.py` remaps them to contiguous internal indices.

## Source

The files were copied from the authors' reference repository:

- https://github.com/scott0793-debug/FMA (`Networks/`)

These networks originate from the widely used Transportation Networks for
Research collection:

- https://github.com/bstabler/TransportationNetworks

If you use them, cite the FMA paper and the Transportation Networks collection.
