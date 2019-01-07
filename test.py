import sys

import dispim.__main__

# sys.argv = ["dispim", "process", "0.1625", "1.0", "D:\\Roel\\beadscan\\MMStack_Pos0.ome.tif", "--save-rg", "-p", "center_crop:0.6", "deskew", "register"]
# sys.argv = ["dispim", "process", "0.725", "2.0", "D:\\Roel\\dresden\\brain_MG_AO_560_640_Eci_test aquisition dual view deep_MMStack_Pos0.ome.tif", '--output', 'out', "--single-file-out", "-p", "register:0.75", "apply_registration", "extract_psf", "deconvolve:32", "--b-flipped-x", "--b-flipped-y", "--a-invert"]
sys.argv = ["dispim", "process", "0.725", "2.0",
            "D:\\Roel\\dresden\\brain_MG_AO_560_640_Eci_test aquisition dual view deep_MMStack_Pos0.ome.tif",
            '--output', 'out', "--single-file-out", "-p", "register:0.6", "apply_registration", "--b-flipped-x",
            "--b-flipped-y", "--a-invert"]
# sys.argv = ["dispim", "process", "0.725", "2.0", "D:\\Roel\\dresden\\brain_MG_AO_560_640_Eci_test aquisition dual view deep_MMStack_Pos0.ome.tif", "--no-save", "-p", "show_front"]

dispim.__main__.main()
