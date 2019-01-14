# Anime Generator

Anime genertor implemented by ACGAN and based on tensorflow framework.

## Results
![generated anime sample](https://github.com/thick0605/Anime_GAN/blob/master/results/sample.png?raw=true)

## Requirement

  - tensorflow
  - numpy
  - matplotlib
  - PIL
  - csv

## Training
```
python train.py <path/to/training_tags_file.csv> <path/to/training_images_folder/>
```

### Training Tag Format
Each row in csv file records attributes of the training images.
```<img_name>,<hair attribute> <eyes attribute>```
```
img1,blue hair red eyes
img2,red hair green eyes
...
```

## Testing
Result will save in the folder named results.
```
python generate.py <path/to/testing_tag.txt>
```

### Testing Tag Foramt
The default of generated sample is 25 so the tag file should be 25 lines.
```<img_id>,<hair attribute> <eyes attribute>```
```
1,blue hair red eyes
2,red hair green eyes
...
25,green hair blue eyes
```

## Download Pre-trained Model
```
wget https://www.dropbox.com/s/m8g8w4jlb327bu8/cgan.ckpt.data-00000-of-00001 -P model_file/
wget https://www.dropbox.com/s/b9gjs5dh6qfnsjj/cgan.ckpt.index -P model_file/
wget https://www.dropbox.com/s/fucrwc50x44ot2v/cgan.ckpt.meta -P model_file/
```

## Model Parameters Setting
  - Data augmentation: horizontal flip and +/- 5 degree rotation
  - Batch size: 64
  - Iteration: 50,000
  - Learning rate: 2e-4
  - beta1: 0.5
  - beta: 0.9
  - Loss ratio: 25(real/fake):1(attribute)
  - Training iteration ratio: 1(discriminator):3(generator)

