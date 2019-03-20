import sys
from dispim import __main__


# sys.path.append("pycharm-debug-py3k.egg")
# import pydevd
# pydevd.settrace('137.120.167.98', port=8080, stdoutToServer=True, stderrToServer=True)

# sys.argv = ["dispim", "process", "0.1625", "1.0", "D:\\Roel\\beadscan\\MMStack_Pos0.ome.tif", "--save-rg", "-p", "center_crop:0.6", "deskew", "register"]
# sys.argv = ["dispim", "process", "0.725", "2.0", "D:\\Roel\\dresden\\brain_MG_AO_560_640_Eci_test aquisition dual view deep_MMStack_Pos0.ome.tif", '--output', 'out', "--single-file-out", "-p", "register:0.75", "apply_registration", "extract_psf", "deconvolve:32", "--b-flipped-x", "--b-flipped-y", "--a-invert"]
for i in range(5, 12):
    try:
        sys.argv = ["dispim", "process", "0.725", "2.0",
                    f"D:\\Roel\\dresden\\brain_MG_AO_560_640_Eci_test aquisition dual view deep_MMStack_Pos0.ome.tif",
                    '--output', f'D:\\Roel\\dresden\\out_{i}_0', "-p", "register_com", "register2d:2,translation",
                    "register2d:0,translation",
                    "register2d:2,rigid", "register2d:0,rigid", "register2d:2,rigid",
                    "register:rigid,0.0034,0.9", "apply_registration", "extract_psf:1,15,11",
                    "deconvolve_chunked:80,13,False",
                    "--b-flipped-x", "--channel-index", "0", "--series-index", str(i),
                    "--b-flipped-y", "--a-invert", "--single-file-out"]
        __main__.main()
    except KeyboardInterrupt:
        raise
    except Exception as e:
        print(e)

    try:
        sys.argv = ["dispim", "process", "0.725", "2.0",
                    f"D:\\Roel\\dresden\\brain_MG_AO_560_640_Eci_test aquisition dual view deep_MMStack_Pos0.ome.tif",
                    '--output', f'D:\\Roel\\dresden\\out_{i}_1', "-p", "register_com", "register2d:2,translation",
                    "register2d:0,translation",
                    "register2d:2,rigid", "register2d:0,rigid", "register2d:2,rigid",
                    "register:rigid,0.0034,0.9", "apply_registration", "extract_psf:1,15,11",
                    "deconvolve_chunked:80,13,False",
                    "--b-flipped-x", "--channel-index", "1", "--series-index", str(i),
                    "--b-flipped-y", "--a-invert", "--single-file-out"]
        __main__.main()
    except KeyboardInterrupt:
        raise
    except Exception as e:
        print(e)

# sys.argv = ["dispim", "process", "0.725", "2.0",
#             f"D:\\Roel\\dresden\\brain_MG_AO_560_640_Eci_test aquisition dual view deep_MMStack_Pos0.ome.tif",
#             '--output', f'D:\\Roel\\dresden\\out', "-p", "register_com", "register2d:2,translation", "register2d:0,translation",
#             "register2d:2,rigid", "register2d:0,rigid", "register2d:2,rigid",
#             "register:rigid,0.001,0.9", "apply_registration", "extract_psf:1,8,11", "deconvolve_chunked:100,12,False",
#             "--b-flipped-x", "--channel-index", "0",
#             "--b-flipped-y", "--a-invert", "--save-rg"]
# __main__.main()
# sys.argv = ["dispim", "process", "0.725", "2.0", "D:\\Roel\\dresden\\brain_MG_AO_560_640_Eci_test aquisition dual view deep_MMStack_Pos0.ome.tif", "--no-save", "-p", "show_front"]
