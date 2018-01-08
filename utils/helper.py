#!/usr/bin/env python

from __future__ import print_function

# make color list
def color_rgb_list():
    """Return rgb list"""
    import random
    from itertools import permutations
    rgb = [51, 102, 153, 204, 255]*2
    color_list = list(permutations(rgb, 3))
    color_list=list((set(color_list)))
    random.seed(123) # for reproductive
    random.shuffle(color_list)
    return color_list


def var_sizes_check(global_dic):
    """Check the size of variables used to prevent from consuming memory
    print out variables which have length or size attribute"""
    import types
    def _print(a, b):
        print("|{:>15}|{:>13}|".format(a, b))

    _print("Variable", "Size")
    print("-"*31)
    for k, v in global_dic.items():
        if not k.startswith('_') and not isinstance(v, types.ModuleType):
            # print size of variable
            if hasattr(v, 'size'):
                try:
                    _print(k, v.size)
                except:
                    continue
            # print length of variable
            elif hasattr(v, '__len__'):
                try:
                    _print(k, len(v))
                except:
                    continue
