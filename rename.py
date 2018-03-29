# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import os
import argparse

ignored_dirs = ['./.git', './.idea']
ignored_files = ['./rename.py']

replacements = {'name': 'lab',
                'name_punctuated': 'LAB',
                'author_first_name': 'Wessel',
                'author_last_name': 'Bruinsma',
                'repo': 'wesselb/lab',
                'description': 'A generic linear algebra interface'}

# Prefix keys in `replacements` with `skeleton_`, and convert to list.
replacements = [('skeleton_' + k, v) for k, v in replacements.items()]

# Order based on length, to avoid prefix problems.
replacements = sorted(replacements, cmp=lambda x, y: cmp(len(x), len(y)))


def list_files(base_dir='.'):
    """List all files in a directory.

    Ignored ignored directories and ignored files.
    """
    fs = []
    for path, dirs, files in os.walk(base_dir):
        if any(path.startswith(d) for d in ignored_dirs):
            continue
        fs.extend([os.path.join(path, f)
                   for f in files
                   if f not in ignored_files])
    return fs


parser = argparse.ArgumentParser()
parser.add_argument('-w', '--wet', action='store_true', help='overwrite files')
args = parser.parse_args()

first = True
for file in list_files():
    first = False if first else print('')

    # Print the current file, and read the file.
    print('File: {}'.format(file))
    with open(file, 'r') as f:
        content = f.read()

    # Perform replacements and show an overview.
    ln = 0
    lines = []
    for line in content.splitlines():
        ln += 1
        if any(k in line for k in zip(*replacements)[0]):
            for k, v in replacements:
                line = line.replace(k, v)
            print('{:4d}: {}'.format(ln, line))
        lines.append(line)

    # If wet run, then overwrite.
    if args.wet:
        with open(file, 'w') as f:
            f.write('\n'.join(lines))
        print('File overwritten.')