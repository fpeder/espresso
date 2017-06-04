#!/usr/bin/env python
# -*- coding: utf-8 -*-

""""
Usage: readEspresso.py <src>

Options:
   -h --help
"""

import os
import sys
import struct
import numpy as np

INPUT = 0
DENSE = 1
BNORM = 2
CONV = 3
POOL = 4
NUM = 1<<4
DATA = 2<<4


def pop_string(asd, n):
    return asd[:n], asd[n:]

def uint8_read(string):
    t, c = pop_string(string, 1)
    return struct.unpack('B', t)[0], c

def int32_read(string):
    t, c = pop_string(string, 4)
    return struct.unpack('i', t)[0], c

def float_read(f):
    return struct.unpack('f', string[i:i+4])[0], i+4



def main(infile):
    assert(os.path.exists(infile))
    parameters = []
    with open(infile, 'rb') as f:
        data = f.read()
        while data:
            caz, data = uint8_read(data)

            if caz == DENSE | NUM:
                num, data = int32_read(data)
                print 'ndense %d' % num

            elif caz == BNORM | NUM:
                num, data = int32_read(data)
                print 'nbnorm %d' % num

            elif caz == POOL | NUM:
                num, data = int32_read(data)
                print 'npool %d' % num

            elif caz == CONV | NUM:
                num, data = int32_read(data)
                print 'nconv %d' % num

            elif caz == INPUT | DATA:
                pass
                #caz, data = pop_string(data, 4*3)
                #M, N, nch = struct.unpack('3i', caz)
                #print 'input %d %d %d' % (M, N, nch)

            elif caz == DENSE | DATA:
                caz, data = pop_string(data, 4*2)
                M, N = struct.unpack('2i', caz)
                print 'dense %d %d' % (M, N)
                caz, data = pop_string(data, 4*M*N)
                asd = np.frombuffer(caz, dtype=np.float32)
                asd = asd.reshape((M, N))
                parameters.append(asd)
                caz, data = pop_string(data, 4*M)
                asd = np.frombuffer(caz, dtype=np.float32)
                parameters.append(asd)

            elif caz == BNORM | DATA:
                dim, data = int32_read(data)
                print 'bnorm %d' % dim
                for i in range(4):
                    caz, data = pop_string(data, 4*dim)
                    asd = np.frombuffer(caz, dtype=np.float32)
                    parameters.append(asd)

            elif caz == CONV | DATA:
                caz, data = pop_string(data, 4*7)
                pad, nfil, M, N, nch, Sx, Sy = struct.unpack('7i', caz)
                print 'conv %d %d %d %d %d %d %d' % (pad, nfil, M, N, nch, Sx, Sy)
                caz, data = pop_string(data, 4*M*N*nch*nfil)
                asd = np.frombuffer(caz, dtype=np.float32)
                asd = asd.reshape((nfil, nch, M, N))
                parameters.append(asd)
                caz, data = pop_string(data, 4*nfil)
                asd = np.frombuffer(caz, dtype=np.float32)
                parameters.append(asd)

            elif caz == POOL | DATA:
                dim, data = pop_string(data, 4*4)
                dim = struct.unpack('4i', dim)
                print 'pool %d %d %d %d' % tuple(dim)

            else:
                print 'caz'

    return parameters


if __name__ == '__main__':
    from docopt import docopt

    args = docopt(__doc__)
    params = main(args['<src>'])
    import pdb; pdb.set_trace()
