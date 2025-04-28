#

import sys
import re
import os.path

def strip_linecont(line):
    if line[-1] != '\\':
        return line, False
    return line[:-1].strip(), True

# The dep file cython generate is not stable,
# we need to actually parse it to see if it really changed
class DepFileParser:
    def __init__(self):
        self.header = None
        self.res = {}
        self.cur_deps = set()

    def parse(self, fh):
        for line in fh.readlines():
            self.parse_line(line)
        return self.finalize()

    def parse_line(self, line):
        line = line.strip()
        if not line:
            return self.end_current()
        line, should_cont = strip_linecont(line)
        if not line:
            assert should_cont
            return

        if self.header is None:
            if line[-1] != ':':
                raise SyntaxError("Invalid deps file syntax")
            self.header = line[:-1]
        else:
            self.cur_deps.add(line)

        if not should_cont:
            self.end_current()

    def end_current(self):
        if self.header is None:
            return
        prev_deps = self.res.get(self.header, set())
        self.res[self.header] = prev_deps | self.cur_deps
        self.header = None
        self.cur_deps = set()

    def finalize(self):
        return {key: sorted(list(value)) for key, value in self.res.items() if value}

def parse_dep_file(f):
    with open(f, "r") as fh:
        parser = DepFileParser()
        return parser.parse(fh)

depsfile = sys.argv[1]
depsfile_out = sys.argv[2]
basedir = sys.argv[3]

dep_in = parse_dep_file(depsfile)
old_dep = parse_dep_file(depsfile_out)

def update_deppath(p):
    if p.startswith('/'):
        return p
    return basedir + '/' + p

new_dep = {update_deppath(key): sorted([update_deppath(dep) for dep in deps])
               for key, deps in dep_in.items()}

if new_dep != old_dep:
    print(f"Update dep file {depsfile_out}")
    keys = sorted(new_dep.keys())
    with open(depsfile_out, 'w') as fh:
        for key in keys:
            deps = new_dep[key]
            fh.write(key + ":")
            for dep in deps:
                fh.write(' \\\n  ' + dep)
            fh.write('\n')
