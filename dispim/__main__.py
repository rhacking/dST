#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import logging
import re

logger = logging.getLogger(__name__)

ops = ['deskew', 'register', 'fuse', 'deconvolve', 'deconvolve_separate', 'center_crop', 'scale', 'discard_a',
       'discard_b', 'show_slice_y_z', 'rot90', 'make_isotropic', 'show_dual', 'apply_registration', 'register_syn',
       'register2d', 'brighten', 'show_a_iso', 'show_b_iso', 'show_overlay_iso', 'show_seperate_iso', 'deconvolve_diag',
       'extract_psf', 'show_front']
VAR_SPEC = r"(?P<arg>(?:\d+(?:\.\d+))|(?:False)|(?:True)|(?:[a-zA-Z0-9_./\\]+))"
OPERATION_SPEC = (r"^(" + "|".join(ops) + r")(?::" + VAR_SPEC + r")?(?:," + VAR_SPEC + ")*$")


def process(args):
    print('starting')
    # import pydevd
    # pydevd.settrace('localhost', port=8080, stdoutToServer = True, stderrToServer = True)
    import dispim
    if args.debug:
        dispim.debug = True
    from dispim import process
    if args.pixel_size > args.interval:
        logger.warning('The pixel size is greater than the interval. Arguments may have been swapped. ')
    volumes = dispim.load_volumes([args.spim_a, args.spim_b] if args.spim_b is not None else [args.spim_a],
                                  (args.pixel_size, args.pixel_size, args.interval), args.scale, not args.no_skew)

    if args.a_invert:
        logger.debug('A is inverted')
        volumes[0].inverted = True

    if args.b_flipped_x:
        volumes[1].flipped[0] = True

    if args.b_flipped_y:
        volumes[1].flipped[1] = True

    if args.b_flipped_z:
        volumes[1].flipped[2] = True

    if args.swap_xy_a:
        volumes[0].data = volumes[0].data.swapaxes(0, 1)

    if args.swap_xy_b:
        volumes[1].data = volumes[1].data.swapaxes(0, 1)

    print('still starting')

    # steps = []
    # for op in args.operations:
    #     class_name = 'Process' + op[0].upper() + op[1:]
    #     class_ = getattr(process, class_name)
    #     inst = class_()
    #     steps.append(inst)

    logger.info("Starting data processing...")
    processor = process.Processor(args.operations)

    result = processor.process(tuple(volumes), args.output, args.save_inter)

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
            result[0].save_tiff('out_A', swap_xy=args.swap_xy_a, path=args.output)
        logger.info('Saving volume b...')
        if args.single_file_out:
            result[1].save_tiff_single('out_B_single', swap_xy=args.swap_xy_b, path=args.output)
        else:
            result[1].save_tiff('out_B', swap_xy=args.swap_xy_b, path=args.output)
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
    import regex
    m = regex.match(OPERATION_SPEC, s)

    if m is None:
        raise argparse.ArgumentTypeError("Invalid process argument {}".format(s))

    name = m.group(1)

    class_name = 'Process' + name[0].upper() + name[1:]
    class_name = re.sub(r"_(\w)", lambda m: m.group(1).upper(), class_name)
    class_ = getattr(process, class_name)
    args = [float(sarg) if isfloat(sarg) else (str_to_bool(sarg) if sarg in ['True', 'False'] else str(sarg)) for sarg
            in m.captures('arg')]
    try:
        inst = class_(*args)
    except TypeError:
        raise argparse.ArgumentTypeError("Failed to initialize {}".format(class_name))

    return inst


def extract_psf(args):
    from dispim import extract_psf
    extract_psf(args)


def start_gui(args):
    import dispim.gui

    app = dispim.gui.DSTApp(0)
    app.MainLoop()


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
    # TODO: Clean this up
    p_process.add_argument('--no-skew', action='store_true', default=False)
    p_process.add_argument('--a-invert', action='store_true', default=False)
    p_process.add_argument('--b-flipped-x', action='store_true', default=False)
    p_process.add_argument('--b-flipped-y', action='store_true', default=False)
    p_process.add_argument('--b-flipped-z', action='store_true', default=False)

    p_process.add_argument('--debug', action='store_true', default=False)


    p_process.set_defaults(func=process)

    p_extract = sub_parsers.add_parser('extract')
    p_extract.add_argument('volume', type=str)
    p_extract.set_defaults(func=extract_psf)

    p_gui = sub_parsers.add_parser('gui')
    p_gui.set_defaults(func=start_gui)

    args = p_main.parse_args()

    import coloredlogs
    coloredlogs.install(fmt='%(asctime)s %(name)s %(message)s', level='DEBUG')
    args.func(args)


if __name__ == '__main__':
    main()
