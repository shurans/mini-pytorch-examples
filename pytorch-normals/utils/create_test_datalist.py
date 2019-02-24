f = open("data/datalist_test", "w+")
for i in range(0,27):
    f.write("test-real/preprocessed-rgb-imgs/%09d-rgb.npy,test-real/preprocessed-camera-normals/%09d-cameraNormals.npy\n"%(i,i))

for i in range(0,100):
    f.write("test/preprocessed-rgb-imgs/%09d-rgb.npy,test/preprocessed-camera-normals/%09d-cameraNormals.npy\n"%(i, i))
f.close()
