f = open("data/datalist", "w+")
for i in range(0, 100):
    f.write("datasets/val/milk-bottles-val/resized-files/preprocessed-rgb-imgs/%09d-rgb.png,\
datasets/val/milk-bottles-val/resized-files/preprocessed-outlines/%09d-outlineSegmentation.png\n"
            % (i, i))
f.close()
