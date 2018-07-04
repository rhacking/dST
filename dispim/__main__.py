#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# import regex
import argparse
import logging
import re

# logging.basicConfig(level='DEBUG')

logger = logging.getLogger(__name__)

ops = ['deskew', 'register', 'fuse', 'deconvolve', 'deconvolve_separate', 'center_crop', 'scale', 'discard_a',
       'discard_b', 'show_slice_y_z']
VAR_SPEC = r"((?:\d+(?:\.\d+))|(?:False)|(?:True)"
OPERATION_SPEC = (r"^(" + "|".join(ops) + r")(?::" + VAR_SPEC + r")?(?:,VAR_SPEC)*$")


def process(args):
    print('starting')
    import dispim
    from dispim import process
    if args.pixel_size > args.interval:
        logger.warning('The pixel size is greater than the interval. Arguments may have been swapped. ')
    volumes = dispim.load_volumes([args.spim_a, args.spim_b] if args.spim_b is not None else [args.spim_a],
                                  (args.pixel_size, args.pixel_size, args.interval), args.scale)

    print('still starting')

    # steps = []
    # for op in args.operations:
    #     class_name = 'Process' + op[0].upper() + op[1:]
    #     class_ = getattr(process, class_name)
    #     inst = class_()
    #     steps.append(inst)

    logger.info("Starting data processing...")
    processor = process.Processor(args.operations)

    result = processor.process(tuple(volumes), args.save_inter)

    if args.no_save:
        return

    if len(result) == 2:
        if args.save_rg:
            logging.info('Saving volume a/b...')
            dispim.save_dual_tiff('out_AB', result[0], result[1], path=args.output)
        logger.info('Saving volume a...')
        if args.single_file_out:
            result[0].save_tiff_single('out_A_single', swap_xy=args.swap_xy_a, path=args.output)
        else:
            result[0].save_tiff('out_A', swap_xy=args.swap_xy_a)
        logger.info('Saving volume b...')
        if args.single_file_out:
            result[1].save_tiff_single('out_B_single', swap_xy=args.swap_xy_b, path=args.output)
        else:
            result[1].save_tiff('out_B', swap_xy=args.swap_xy_b)
    else:
        logger.info('Saving volume...')
        if args.single_file_out:
            result[0].save_tiff_single("out_single", path=args.output)
        else:
            result[0].save_tiff("out", path=args.output)


def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def str_to_bool(s):
    if s == 'True':
        return True
    elif s == 'False':
        return False
    else:
        raise ValueError('{} is not a boolean'.format(s))


def image_operation(s: str):
    from dispim import process
    # from inspect import signature
    # m = regex.match(OPERATION_SPEC, s)
    name = s.strip().split(":")[0]
    sargs = []
    if len(s.strip().split(":")) > 1:
        sargs = s.strip().split(":")[1].split(",")

    # if m is None:
    #     raise argparse.ArgumentTypeError("Invalid operation specification")
    class_name = 'Process' + name[0].upper() + name[1:]
    class_name = re.sub(r"_(\w)", lambda m: m.group(1).upper(), class_name)
    class_ = getattr(process, class_name)
    args = [float(sarg) if isfloat(sarg) else str_to_bool(sarg) for sarg in sargs]
    try:
        inst = class_(*args)
    except TypeError:
        raise argparse.ArgumentTypeError("Failed to initialize {}".format(class_name))
    return inst


def extract_psf(args):
    from dispim import extract_psf
    extract_psf(args)


def main():
    p_main = argparse.ArgumentParser()
    sub_parsers = p_main.add_subparsers(dest='cmd')
    sub_parsers.required = True
    p_process = sub_parsers.add_parser('process')
    p_process.add_argument('pixel_size', type=float)
    p_process.add_argument('interval', type=float)
    p_process.add_argument('spim_a', type=str)
    p_process.add_argument('spim_b', type=str, nargs='?')
    p_process.add_argument('-p', '--operations', type=image_operation, nargs='*', required=True)
    p_process.add_argument('-s', '--deconvolve-sigma', type=float)
    p_process.add_argument('-o', '--output', type=str, default='out')
    p_process.add_argument('--swap-xy-a', action='store_true', default=False)
    p_process.add_argument('--swap-xy-b', action='store_true', default=False)
    p_process.add_argument('--scale', type=float)
    p_process.add_argument('--save-rg', action='store_true', default=False)
    p_process.add_argument('--single-file-out', action='store_true', default=False)
    p_process.add_argument('--save-inter', action='store_true', default=False)
    p_process.add_argument('--no-save', action='store_true', default=False)

    p_process.set_defaults(func=process)

    p_extract = sub_parsers.add_parser('extract')
    p_extract.add_argument('volume', type=str)
    p_extract.set_defaults(func=extract_psf)

    args = p_main.parse_args()

    import coloredlogs
    coloredlogs.install(fmt='%(asctime)s %(name)s %(message)s', level='DEBUG')
    args.func(args)


if __name__ == '__main__':
    main()
