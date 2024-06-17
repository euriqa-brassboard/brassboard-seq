#

import sys
import re
import os.path

depsfile = sys.argv[1]
basedir = sys.argv[2]

lines = open(depsfile).readlines()

with open(depsfile, 'w') as fh:
    for line in lines:
        m = re.match('^([ \t]*)([^/ \t].*)$', line)
        if m is not None:
            line = m[1] + basedir + '/' + m[2] + '\n'
        fh.write(line)
