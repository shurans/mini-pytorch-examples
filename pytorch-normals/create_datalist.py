f = open("data/datalist", "w+")
for i in range(0,100):
    f.write("train/rgb-imgs/%09d-rgb.jpg,train/camera-normals/%09d-cameraNormals.npy\n"%(i, i))

f.close()
