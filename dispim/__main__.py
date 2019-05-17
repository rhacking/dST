#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import logging
import re

logger = logging.getLogger(__name__)

ops = ['deskew', 'deskew_diag', 'register', 'fuse', 'deconvolve', 'deconvolve_separate', 'center_crop', 'scale',
       'discard_a',
       'discard_b', 'show_slice_y_z', 'rot90', 'make_isotropic', 'show_dual', 'apply_registration', 'register_syn',
       'register2d', 'brighten', 'show_a_iso', 'show_b_iso', 'show_overlay_iso', 'show_seperate_iso', 'deconvolve_diag',
       'extract_psf', 'show_front', 'deconvolve_chunked', 'register_com', 'save_chunks']
VAR_SPEC = r"(?P<arg>(?:\d+(?:\.\d+))|(?:False)|(?:True)|(?:[a-zA-Z0-9_./\\]+))"
OPERATION_SPEC = (r"^(" + "|".join(ops) + r")(?::" + VAR_SPEC + r")?(?:," + VAR_SPEC + ")*$")


def process(args):
    import dispim
    if args.debug:
        dispim.debug = True
    from dispim import process
    import dispim.io as dio
    if args.pixel_size > args.interval:
        logger.warning('The pixel size is greater than the interval. Arguments may have been swapped. ')

    logging.getLogger("tifffile").setLevel(logging.CRITICAL)

    if args.spim_b is None:
        volumes = dio.load_tiff(args.spim_a, channel=args.channel_index, series=args.series_index,
                                inverted=args.a_invert, pixel_size=args.pixel_size, step_size=args.interval,
                                flipped=(args.b_flipped_x, args.b_flipped_y, args.b_flipped_z))
        # TODO: Make this nicer
        if len(volumes) == 2 and args.b_not_invert:
            volumes[1] = dispim.Volume(volumes[1], inverted=False)
    else:
        vol_a = dio.load_tiff(args.spim_a, channel=args.channel_index, series=args.series_index,
                              inverted=args.a_invert, pixel_size=args.pixel_size, step_size=args.interval)
        vol_b = dio.load_tiff(args.spim_b, channel=args.channel_index, series=args.series_index,
                              inverted=not args.b_not_invert, pixel_size=args.pixel_size, step_size=args.interval,
                              flipped=(args.b_flipped_x, args.b_flipped_y, args.b_flipped_z))
        volumes = (vol_a, vol_b)

    logger.info("Starting data processing...")
    processor = process.Processor(args.operations)

    result = processor.process(tuple(volumes), args.output, args.save_inter)

    if args.no_save:
        return

    if len(result) == 2:
        logger.info('Saving volume A...')
        dio.save_tiff_output(result[0], args.output, f'vol_a_{args.channel_index}_{args.series_index}', args.b_8bit)
        logger.info('Saving volume b...')
        dio.save_tiff_output(result[1], args.output, f'vol_b_{args.channel_index}_{args.series_index}', args.b_8bit)
        if args.save_rg:
            logger.info('Saving volume A/B...')
            dio.save_tiff_output_dual(result[0], result[1], args.output,
                                      f'vol_ab_{args.channel_index}_{args.series_index}', args.b_8bit)
    else:
        logger.info('Saving volume...')
        dio.save_tiff_output(result[0], args.output, f'vol_{args.channel_index}_{args.series_index}', args.b_8bit)

    # if len(result) == 2:
    #     if args.save_rg:
    #         logging.info('Saving volume a/b...')
    #         dispim.save_dual_tiff('out_AB', result[0], result[1], path=args.output)
    #     logger.info('Saving volume a...')
    #     if args.single_file_out:
    #         result[0].save_tiff_single('out_A_single', swap_xy=args.swap_xy_a, path=args.output)
    #     else:
    #         result[0].save_tiff('out_A', swap_xy=args.swap_xy_a, path=args.output)
    #     logger.info('Saving volume b...')
    #     if args.single_file_out:
    #         result[1].save_tiff_single('out_B_single', swap_xy=args.swap_xy_b, path=args.output)
    #     else:
    #         result[1].save_tiff('out_B', swap_xy=args.swap_xy_b, path=args.output)
    # else:
    #     logger.info('Saving volume...')
    #     if args.single_file_out:
    #         result[0].save_tiff_single("out_single", path=args.output)
    #     else:
    #         result[0].save_tiff("out", path=args.output)


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
    from dispim.deconvolution import extract_psf
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
    p_process.add_argument('--series-index', type=int, default=0)
    p_process.add_argument('--channel-index', type=int, default=0)
    p_process.add_argument('--save-rg', action='store_true', default=False)
    p_process.add_argument('--single-file-out', action='store_true', default=False)
    p_process.add_argument('--save-inter', action='store_true', default=False)
    p_process.add_argument('--no-save', action='store_true', default=False)
    # TODO: Clean this up
    p_process.add_argument('--no-skew', action='store_true', default=False)
    p_process.add_argument('--a-invert', action='store_true', default=False)
    p_process.add_argument('--b-not-invert', action='store_true', default=False)
    p_process.add_argument('--b-flipped-x', action='store_true', default=False)
    p_process.add_argument('--b-flipped-y', action='store_true', default=False)
    p_process.add_argument('--b-flipped-z', action='store_true', default=False)
    p_process.add_argument('--b-8bit', action='store_true', default=False)

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
