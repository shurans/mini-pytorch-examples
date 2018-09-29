f = open("data/datalist", "w+")
for i in range(0,50):
    f.write("/rgb_imgs/%09d-rgb.jpg,/normals_masks/%09d-normals.exr\n"%(i, i))

f.close()