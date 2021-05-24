import os 

path = "D:/NTUT/大二下/多媒體技術與應用/hw07/helmet/"
output_path = "/drive/yolo/helmet/"
images_file = list()
os.chdir(os.path.join(path, "dataset"))

for fname in os.listdir(os.getcwd()):
    if fname.endswith(".jpg"):
        images_file.append("dataset/" + fname)

os.chdir("..")
with open("train.txt", "w") as output:
    for image in images_file:
        output.write(output_path + image)
        output.write("\n")
    output.close()
os.chdir("..")
