f = open("data/datalist", "w+")
for i in range(0,50):
    f.write("/rgb-imgs/%09d-rgb.jpg,/surface-normals-converted/%09d-normals.exr.npy\n"%(i, i))

f.close()