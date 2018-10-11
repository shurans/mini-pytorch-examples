f = open("data/datalist", "w+")
for i in range(0,2300):
    f.write("train/rgb-imgs-preprocessed/%09d-rgb.npy,train/surface-normals-converted-preprocessed/%09d-cameraNormals.npy\n"%(i, i))

f.close()
