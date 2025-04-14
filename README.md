# FruitVegNet
This program is an application created using Python and PyQt6 as part of my Master's thesis. It is a software that performs the recognition of selected fruits and vegetables wrapped in a mesh bag. 

## Description
The program can recognize selected pieces of fruit and vegetables wrapped in a net wrapper using neural networks from the PyTorch library. There are 3 pre-trained models implemented in the program: ResNet18, EfficientNet-B0 and VGG16. In addition, a simpler model called Simple CNN is created. However, the program can easily be extended with other neural networks from the PyTorch library.
A custom simple dataset was created for this purpose. But the program will work with any other dataset, it just needs to be put in a ```dataset``` folder and called ```fruitveg-dataset```, and support the following structure:  
- ```dataset/fruitveg-dataset```
   - ```/train```
   - ```/test```
   - ```/valid```


## Dependencies
- OS: Windows/Linux 
- Python 3.13.2 (recommended)
- PyQt6
- torch
- torchvision
- scikit-learn
- Pillow
- matplotlib
- numpy

## How to run
1. Open CMD in the root folder of the project (main folder with the main.py, requirements.txt, ...)

2. Install the required packages (you need to have pip installed):
   ```bash
   pip install -r requirements.txt
   ```

3. Copy the ```fruitveg-dataset``` folder from the following path:
   ```bash
   prilohy/dataset/
   ```
   to the following path:
   ```bash
   prilohy/source_code/fruitvegnet/dataset
   ```

4. Copy the contents of the ```saved_models``` folder from the path:
   ```bash
   prilohy/saved_models
   ```
   to the following folder:
   ```bash
   prilohy/source_code/fruitvegnet/saved_models
   ```

5. To start the application, use the following command:
    ```bash
    python main.py
    ```
