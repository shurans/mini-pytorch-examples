f = open("data/datalist", "w+")
for i in range(0,100):
    f.write("train/preprocessed-rgb-imgs/%09d-rgb.npy,train/edges-imgs/%09d-segmentation.png\n"%(i, i))

f.close()
