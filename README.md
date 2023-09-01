# mlp
Implementation of a simple multilayer perceptron, which is able to train on emnist dataset and perform recognition of handwritten letters of the English alphabet.
Written in C++. GUI was made with QT by another member of my team. This project is a part of education in School 21.

## Usage

### Download
```
git clone https://github.com/sisthers/mlp.git
```

### Installation
```
make install
```

### Description

#### Training
You can train network yourself with emnist dataset, presented in ```datasets``` folder, or you can use weights from ```src/weights``` folder.
You can choose mini-batch size, number of epoches, number of layers and matrix/graph network implenemtation. 
Also you have an ability to train network using cross-validation. 

#### Tests
You can run tests on dataset, which is also presented in ```datasets``` folder. After testing you will see some statistics, such as accuracy, precision, recall and F-measure.

#### Recognition from picture
In other menu tab you can test network by drowing a letter youself or by downloading a picture. Some test samples are presented in ```src/image_samples```

