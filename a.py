import os

with open("requirements.txt") as rfile:
    list = rfile.readlines()

    for i in list:
        try:
            os.system(f'pip install {i}')
        except:
            pass