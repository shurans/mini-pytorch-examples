f = open("data/datalist", "w+")
for i in range(0,100):
    f.write("train/rgb-imgs/%09d-rgb.jpg,train/combined-edges/%09d-segmentation.png\n"%(i, i))

f.close()
