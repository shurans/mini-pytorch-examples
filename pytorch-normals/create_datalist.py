f = open("data/datalist", "w+")
for i in range(0,2300):
    f.write("train/preprocessed-rgb-imgs/%09d-rgb.npy,train/preprocessed-camera-normals/%09d-cameraNormals.npy\n"%(i, i))

f.close()
