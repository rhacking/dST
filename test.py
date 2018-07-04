import sys
import dispim.__main__

# sys.argv = ["dispim", "process", "0.1625", "1.0", "D:\\Roel\\beadscan\\MMStack_Pos0.ome.tif", "--save-rg", "-p", "center_crop:0.6", "deskew", "register"]
sys.argv = ["dispim", "process", "0.1625", "1.0", "D:\\Roel\\beadscan\\MMStack_Pos0.ome.tif", '--output', 'D:\\Roel\\beadscan\\out', "--single-file-out", "-p", "deskew:False,True"]

dispim.__main__.main()