# Install Libraries:
```
pip install tensorflow
pip install scikit-image
pip install numpy
pip install pandas
pip install matplotlib
```

# Training:
-Change root directory for training files equivalent to loaction of "final_dataset (train+val)"
-Highly recommended to run on GPU over CPU/TPU
```
python train.py
```

# Further training previous model:
-Change root directory for training files equivalent to loaction of "final_dataset (train+val)"
-Change curent model path
-Change location of new model to be saved
-Highly recommended to run on GPU over CPU/TPU
```
python train_further.py
```

# Testing:
-Change root directory for validation/testing files equivalent to loaction of "final_dataset (train+val)/val"
-Modified template provided to project 1
-Highly recommended to run on TPU/GPU over CPU
```
python testing_code_template_new.py
```

-You can change the model name which is by default "pix2pix.h5"
-It is not provided in GitHub dur to the size of the model


## Augmentation:
-Training on augmentation is not recommended
-Change directory for training files equivalent to location of "final_dataset (train+val)/train"
```
python augment.py
```

## Note:
For further information, please refer to provided pdf.
