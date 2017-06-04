#!/usr/bin/env python
# -*- coding: utf-8 -*-

""""
Usage: toEspresso.py <data> <desc> <out>

Options:
   -h --help   show this message
"""

import os
import numpy as np
import struct
import json

INPUT = 0
DENSE = 1
BNORM = 2
CONV = 3
POOL = 4
NUM = 1<<4
DATA = 2<<4


def read_param(params, n):
    return params['arr_%d' % n].astype(np.float32)


def main(data_file, desc_file, out_file):
    assert(os.path.exists(desc_file) and os.path.exists(data_file))
    desc = json.load(open(desc_file, 'r'))
    params, n = np.load(data_file), 0
    with open(out_file, 'wb') as out:
        for elem in desc:
            tmp = ''

            if elem['type'] == "ndense":
                N = elem['val']
                print 'ndense: %d' % N
                tmp = struct.pack('B', DENSE | NUM) + \
                      struct.pack('i', N)

            elif elem['type'] == "nbnorm":
                N = elem['val']
                print 'nbnorm: %d' % N
                tmp = struct.pack('B', BNORM | NUM) + \
                      struct.pack('i', N)

            elif elem['type'] == "nconv":
                N = elem['val']
                print 'nconv: %d' % N
                tmp = struct.pack('B', CONV | NUM) + \
                      struct.pack('i', N)

            elif elem['type'] == "npool":
                N = elem['val']
                print 'npool: %d' % N
                tmp = struct.pack('B', POOL | NUM) + \
                      struct.pack('i', N)

            elif elem['type'] == 'input':
               tmp = struct.pack('B', INPUT | DATA)
               dim = elem['dim']
               print 'input: %d %d %d' % tuple(dim)
               #tmp += struct.pack('3i', *dim)

            elif elem['type'] == 'dense':
                M, N = elem['dim']
                print 'dense: %d %d' % (M, N)
                tmp = struct.pack('B', DENSE | DATA) + \
                      struct.pack('2i', M, N)
                W = read_param(params, n).T
                b = read_param(params, n + 1)
                tmp += W.tostring('C') + b.tostring('C')
                n += 2

            elif elem['type'] == 'bnorm':
                N = elem['dim']
                print 'bnorm: %d' % N
                tmp = struct.pack('B', BNORM | DATA) + \
                      struct.pack('i', N)
                for i in range(4):
                    tmp += read_param(params, n+i).tostring('C')
                n += 4

            elif elem['type'] == 'conv':
                dim = elem['dim']
                print 'conv: %d %d %d %d %d %d %d' % tuple(dim)
                tmp = struct.pack('B', CONV | DATA) + \
                      struct.pack('7i', *dim)
                H = read_param(params, n)
                b = read_param(params, n + 1)
                tmp += H.tostring('C')
                tmp += b.tostring('C')
                n += 2

            elif elem['type'] == 'pool':
                dim = elem['dim']
                print 'max pool: %d %d %d %d' % tuple(dim)
                tmp = struct.pack('B', POOL | DATA) + \
                      struct.pack('4i', *dim)
            else:
                pass

            print len(tmp)
            out.write(tmp)


if __name__ == '__main__':
    from docopt import docopt

    args = docopt(__doc__)
    data, desc, out = args['<data>'], args['<desc>'], args['<out>']

    main(data, desc, out)
