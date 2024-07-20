# SRGAN-Super Resolution GAN



### Train

```python
!python main.py --LR_path Data/train_LR --GT_path Data/train_HR
```

### Test
```python
!python main.py --mode test_only --LR_path test_data --generator_path model/SRGAN.pt
```

at first deactivate current env  using command deactivate and then activate new virtaul environment using command 
myenv\Scripts\activate 
