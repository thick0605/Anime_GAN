
# coding: utf-8

import matplotlib.pyplot as plt
from util.ACGAN_resnet import ACGAN
import numpy as np
import sys

hair_color = ['orange hair', 'white hair', 'aqua hair', 'gray hair',
              'green hair', 'red hair', 'purple hair', 'pink hair',
              'blue hair', 'black hair', 'brown hair', 'blonde hair']
eyes_color = ['gray eyes', 'black eyes', 'orange eyes',
              'pink eyes', 'yellow eyes', 'aqua eyes', 'purple eyes',
              'green eyes', 'brown eyes', 'red eyes', 'blue eyes']

def save_imgs(samples):
    r, c = 5, 5
    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            sample = np.array(((samples[cnt]*127.5)+127.5), dtype=np.uint8)
            axs[i,j].imshow(sample)
            axs[i,j].axis('off')
            cnt += 1
    fig.savefig("results/sample.png")
    plt.close()

def att2one_hot(hair_index,eyes_index,att_len=23):
    attribute = np.zeros(att_len)
    attribute[hair_index] = 1
    attribute[eyes_index+12] = 1
    return attribute

def load_tags(tags_file):
    file = open(tags_file,'r')
    attribute = []
    content = file.readlines()
    sample_num = len(content)
    for line in content:
        tags = line.split(' ')
        hair = tags[0].split(',')[1]
        hair_index = hair_color.index(hair+' hair')
        eyes_index = eyes_color.index(tags[2]+' eyes')
        attribute.append(att2one_hot(hair_index,eyes_index))

    attribute = np.asarray(attribute)
    return sample_num, attribute
       


model = ACGAN()
model.build_model()
sample_num, tags = load_tags(sys.argv[1])
noise = np.random.uniform(-1., 1., size=(sample_num,100))
samples = model.generate(noise,tags)
save_imgs(samples)
