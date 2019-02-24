f = open("data/datalist", "w+")
for i in range(0,100):
    f.write("datasets/milk-bottles/resized-files/preprocessed-rgb-imgs/%09d-rgb.png,datasets/milk-bottles/resized-files/preprocessed-camera-normals/%09d-cameraNormals.exr\n"%(i, i))

f.close()
