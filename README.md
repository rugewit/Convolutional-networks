

## Свёрточные нейронные сети

### Pet Faces

Датасет [Pet Faces](https://www.soshnikov.com/permanent/data/petfaces.tar.gz) представляет собой множество изображений 13 пород кошек и 23 пород собак, по 200 изображений на каждую породу. Изображение центрированы и уменьшены до небольшого размера.

Вам необходимо обучить свёрточные нейронные сети для решения двух задач классификации:

- Определение кошки или собаки (бинарная классификация)
- Определение породы кошки или собаки (мультиклассовая классификация)



Проведём преобразования над датасетом

```python
dataset = torchvision.datasets.ImageFolder(
    dataset_folder,
    transform = tr.Compose([
        tr.Resize(image_size), # Меняем все картинки на картинки с одним размером
        tr.CenterCrop(image_size), # Обрезает изображение по центру
        tr.ToTensor() # Преобразует картинки в тензоры, причём с нормализацией
    ]))

pet_classes_names = dataset.classes
```

Разделим на тренировочные и тестовые

```python
len_dataset = len(dataset)
train_length = int(0.8 * len_dataset)
test_length = len_dataset - train_length
split_size = [train_length, test_length]

train_data, test_data = torch.utils.data.random_split(dataset, split_size)
```

Посмотрим на примеры из датасета

![](https://github.com/rugewit/Convolutional-networks/blob/main/report_images/1.png)

Определим модель и гиперпараметры

```python
model = torch.nn.Sequential(
    torch.nn.Conv2d(3, 16,(3, 3)),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(2),
    torch.nn.Conv2d(16, 32,(3, 3)),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(2),
    torch.nn.Conv2d(32, 64,(3, 3)),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(2),
    torch.nn.Flatten(),
    torch.nn.Linear(12544, 2048),
    torch.nn.ReLU(),    
    torch.nn.Linear(2048, number_classes)
)

model = model.to(device)

learning_rate = 0.001
opt = torch.optim.Adam
loss_fn_standart = torch.nn.CrossEntropyLoss()
cur_epochs = 10
```

Результат обучения

```````
epoch 1: train loss=0.054, train accuracy=0.080, valid loss=0.054, valid acc=0.110
epoch 2: train loss=0.045, train accuracy=0.205, valid loss=0.051, valid acc=0.188
epoch 3: train loss=0.039, train accuracy=0.307, valid loss=0.044, valid acc=0.330
epoch 4: train loss=0.031, train accuracy=0.422, valid loss=0.037, valid acc=0.353
epoch 5: train loss=0.025, train accuracy=0.547, valid loss=0.039, valid acc=0.404
epoch 6: train loss=0.019, train accuracy=0.656, valid loss=0.037, valid acc=0.448
epoch 7: train loss=0.013, train accuracy=0.734, valid loss=0.037, valid acc=0.493
epoch 8: train loss=0.009, train accuracy=0.838, valid loss=0.041, valid acc=0.476
epoch 9: train loss=0.005, train accuracy=0.897, valid loss=0.050, valid acc=0.457
epoch 10: train loss=0.003, train accuracy=0.946, valid loss=0.066, valid acc=0.473
```````

![](https://github.com/rugewit/Convolutional-networks/blob/main/report_images/2.png)

Confusion Matrix

![](https://github.com/rugewit/Convolutional-networks/blob/main/report_images/3.png)

Top-3 Accuracy

0.7185069918632507

#### Задача Кошки против собак

Сделаем преобразования над датасетом

```python
dataset = torchvision.datasets.ImageFolder(
    dataset_folder,
    transform = tr.Compose([
        tr.Resize(image_size),
        tr.CenterCrop(image_size),
        tr.ToTensor()
    ]),
    target_transform = tr.Lambda(lambda pet_class: torch.tensor(1) if pet_classes_names[pet_class].startswith('cat_') else torch.tensor(0))
)

len_dataset = len(dataset)
train_length = int(0.8 * len_dataset)
test_length = len_dataset - train_length
split_size = [train_length, test_length]

train_data, test_data = torch.utils.data.random_split(dataset, split_size)

train_loader = torch.utils.data.DataLoader(train_data, batch_size = 64, shuffle = True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size = 64, shuffle = True)
```

Определим модель

```python
binary_model = torch.nn.Sequential(
    torch.nn.Conv2d(3, 16, (3, 3)),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(2),
    torch.nn.Conv2d(16, 32, (3, 3)),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(2),
    torch.nn.Conv2d(32, 64, (3, 3)),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(2),
    torch.nn.Flatten(),
    torch.nn.Linear(12544, 2048),
    torch.nn.ReLU(),    
    torch.nn.Linear(2048, 2)
)

binary_model = binary_model.to(device)

history = train(binary_model, train_loader, test_loader, epochs = cur_epochs, lr = learning_rate, loss_fn = loss_fn_standart)
```

Результат обучения

``````
epoch 1: train loss=0.011, train accuracy=0.637, valid loss=0.011, valid acc=0.628
epoch 2: train loss=0.008, train accuracy=0.738, valid loss=0.007, valid acc=0.816
epoch 3: train loss=0.006, train accuracy=0.824, valid loss=0.007, valid acc=0.834
epoch 4: train loss=0.006, train accuracy=0.827, valid loss=0.006, valid acc=0.840
epoch 5: train loss=0.005, train accuracy=0.863, valid loss=0.005, valid acc=0.865
epoch 6: train loss=0.004, train accuracy=0.885, valid loss=0.005, valid acc=0.851
epoch 7: train loss=0.004, train accuracy=0.911, valid loss=0.005, valid acc=0.857
epoch 8: train loss=0.003, train accuracy=0.922, valid loss=0.006, valid acc=0.880
epoch 9: train loss=0.003, train accuracy=0.939, valid loss=0.005, valid acc=0.886
epoch 10: train loss=0.002, train accuracy=0.940, valid loss=0.004, valid acc=0.908
``````

![](https://github.com/rugewit/Convolutional-networks/blob/main/report_images/4.png)

### Oxford Pets и Transfer Learing

Используйте оригинальный датасет **[Oxford Pets](https://www.kaggle.com/datasets/tanlikesmath/the-oxfordiiit-pet-dataset)** и предобученные сети VGG-16/VGG-19 и ResNet для построения классификатора пород.

В качестве результата необходимо:

- Обучить три классификатора пород: на основе VGG-16/19 и на основе ResNet.
- Посчитать точность классификатора на тестовом датасете отдельно для каждого из классификаторов, для дальнейших действий выбрать сеть с лучшей точностью
- Посчитать точность двоичной классификации "кошки против собак" такой сетью на тестовом датасете
- Построить confusion matrix
- Посчитать top-3 и top-5 accuracy

Посмотрим на примеры из датасета

![](https://github.com/rugewit/Convolutional-networks/blob/main/report_images/5.png)

Разобьём датасет на тренировочную и тестовую части

``````python
import tensorflow as tf

image_size = 224
batch_size = 32


my_seed = 322

train, test = [
    tf.keras.preprocessing.image_dataset_from_directory(
    folder_name,
    image_size = (image_size,image_size),
    validation_split = 0.2,
    subset = s,
    batch_size = batch_size,
    seed = my_seed)
    for s in ['training','validation']]

class_names = train.class_names
``````

#### VGG-16

``````python
vgg_16 = tf.keras.applications.VGG16(include_top = False, input_shape = (224, 224, 3))

vgg_16_model = tf.keras.models.Sequential()
vgg_16_model.add(tf.keras.layers.Lambda(tf.keras.applications.vgg16.preprocess_input))
vgg_16_model.add(vgg_16)
vgg_16_model.add(tf.keras.layers.Flatten())
vgg_16_model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

vgg_16_model.layers[1].trainable = False
``````

Результат обучения

``````
Epoch 1/3
110/185 [================>.............] - ETA: 5s - loss: 21.4443 - acc: 0.5832
Corrupt JPEG data: premature end of data segment
174/185 [===========================>..] - ETA: 0s - loss: 19.1593 - acc: 0.6347
Corrupt JPEG data: 240 extraneous bytes before marker 0xd9
185/185 [==============================] - 17s 86ms/step - loss: 18.9784 - acc: 0.6404 - val_loss: 13.4948 - val_acc: 0.7585
Epoch 2/3
110/185 [================>.............] - ETA: 5s - loss: 3.8790 - acc: 0.9136
Corrupt JPEG data: premature end of data segment
174/185 [===========================>..] - ETA: 0s - loss: 4.1919 - acc: 0.9097
Corrupt JPEG data: 240 extraneous bytes before marker 0xd9
185/185 [==============================] - 15s 82ms/step - loss: 4.1991 - acc: 0.9095 - val_loss: 14.3398 - val_acc: 0.7869
Epoch 3/3
110/185 [================>.............] - ETA: 5s - loss: 2.1975 - acc: 0.9526
Corrupt JPEG data: premature end of data segment
174/185 [===========================>..] - ETA: 0s - loss: 2.1766 - acc: 0.9517
Corrupt JPEG data: 240 extraneous bytes before marker 0xd9
185/185 [==============================] - 16s 85ms/step - loss: 2.2381 - acc: 0.9513 - val_loss: 16.3463 - val_acc: 0.7788
``````

![](https://github.com/rugewit/Convolutional-networks/blob/main/report_images/6.png)

VGG-16 Top 3 Accuracy

0.9600811907983762



VGG-16 Top 5 Accuracy

0.9844384303112313



#### VGG-19

``````python
vgg_19 = tf.keras.applications.VGG19(include_top=False, input_shape=(224,224,3))
vgg_19_model = tf.keras.models.Sequential()
vgg_19_model.add(tf.keras.layers.Lambda(tf.keras.applications.vgg19.preprocess_input))
vgg_19_model.add(vgg_19)
vgg_19_model.add(tf.keras.layers.Flatten())
vgg_19_model.add(tf.keras.layers.Dense(num_classes,activation='softmax'))

vgg_19_model.layers[1].trainable = False
``````

``````
Epoch 1/3
109/185 [================>.............] - ETA: 5s - loss: 19.8030 - acc: 0.5851
Corrupt JPEG data: premature end of data segment
173/185 [===========================>..] - ETA: 0s - loss: 17.2636 - acc: 0.6432
Corrupt JPEG data: 240 extraneous bytes before marker 0xd9
185/185 [==============================] - 19s 98ms/step - loss: 17.0446 - acc: 0.6499 - val_loss: 13.6410 - val_acc: 0.7618
Epoch 2/3
109/185 [================>.............] - ETA: 5s - loss: 3.4729 - acc: 0.9163
Corrupt JPEG data: premature end of data segment
173/185 [===========================>..] - ETA: 0s - loss: 3.7326 - acc: 0.9137
Corrupt JPEG data: 240 extraneous bytes before marker 0xd9
185/185 [==============================] - 18s 100ms/step - loss: 3.7491 - acc: 0.9139 - val_loss: 13.5914 - val_acc: 0.7950
Epoch 3/3
109/185 [================>.............] - ETA: 6s - loss: 1.7236 - acc: 0.9573
Corrupt JPEG data: premature end of data segment
173/185 [===========================>..] - ETA: 0s - loss: 1.8544 - acc: 0.9556
Corrupt JPEG data: 240 extraneous bytes before marker 0xd9
185/185 [==============================] - 19s 101ms/step - loss: 1.8348 - acc: 0.9555 - val_loss: 17.8820 - val_acc: 0.7909
``````

![](https://github.com/rugewit/Convolutional-networks/blob/main/report_images/7.png)

VGG-19 Top 3 Accuracy

0.9668470906630582



VGG-19 Top 5 Accuracy

0.9939106901217862



#### Resnet

``````python
resnet = tf.keras.applications.ResNet50(include_top = False, input_shape = (224,224,3))
resnet_model = tf.keras.models.Sequential()
resnet_model.add(tf.keras.layers.Lambda(tf.keras.applications.resnet50.preprocess_input))
resnet_model.add(resnet)
resnet_model.add(tf.keras.layers.Flatten())
resnet_model.add(tf.keras.layers.Dense(num_classes,activation='softmax'))

resnet_model.layers[1].trainable = False
``````

Результат обучения

``````
Epoch 1/3
110/185 [================>.............] - ETA: 4s - loss: 7.7648 - acc: 0.6653
Corrupt JPEG data: premature end of data segment
174/185 [===========================>..] - ETA: 0s - loss: 6.9079 - acc: 0.7170
Corrupt JPEG data: 240 extraneous bytes before marker 0xd9
185/185 [==============================] - 18s 81ms/step - loss: 6.9912 - acc: 0.7202 - val_loss: 7.3337 - val_acc: 0.7930
Epoch 2/3
110/185 [================>.............] - ETA: 4s - loss: 1.7018 - acc: 0.9310
Corrupt JPEG data: premature end of data segment
174/185 [===========================>..] - ETA: 0s - loss: 1.8596 - acc: 0.9264
Corrupt JPEG data: 240 extraneous bytes before marker 0xd9
185/185 [==============================] - 14s 73ms/step - loss: 1.8750 - acc: 0.9257 - val_loss: 6.8435 - val_acc: 0.8275
Epoch 3/3
110/185 [================>.............] - ETA: 4s - loss: 0.7436 - acc: 0.9693
Corrupt JPEG data: premature end of data segment
174/185 [===========================>..] - ETA: 0s - loss: 0.7875 - acc: 0.9655
Corrupt JPEG data: 240 extraneous bytes before marker 0xd9
185/185 [==============================] - 14s 76ms/step - loss: 0.7711 - acc: 0.9660 - val_loss: 7.6601 - val_acc: 0.8342
``````

![](https://github.com/rugewit/Convolutional-networks/blob/main/report_images/8.png)

ResNet Top 3 Accuracy

0.9566982408660352



ResNet Top 5 Accuracy

0.9844384303112313



Confusion matrix

![](https://github.com/rugewit/Convolutional-networks/blob/main/report_images/9.png)



#### Задача кошки против собак

Подготовка датасета

```python
image_size = 224
batch_size = 32

my_seed = 322

train_bin, test_bin = [
    tf.keras.preprocessing.image_dataset_from_directory(
    'petfaces',
    image_size=(image_size, image_size),
    validation_split = 0.2,
    subset = s,
    batch_size=batch_size,
    seed = my_seed)
    for s in ['training','validation']]

class_names = train_bin.class_names
```

Определение модели

```python
resnet_binary_model = tf.keras.models.Sequential()
resnet_binary_model.add(tf.keras.layers.Lambda(tf.keras.applications.resnet50.preprocess_input))
resnet_binary_model.add(resnet)
resnet_binary_model.add(tf.keras.layers.Flatten())
resnet_binary_model.add(tf.keras.layers.Dense(1,activation='sigmoid'))

resnet_binary_model.layers[1].trainable = False
```

Результат обучения

``````
Epoch 1/3
81/81 [==============================] - 10s 86ms/step - loss: 0.2097 - acc: 0.9774 - val_loss: 0.0837 - val_acc: 0.9860
Epoch 2/3
81/81 [==============================] - 6s 77ms/step - loss: 0.0064 - acc: 0.9984 - val_loss: 0.0141 - val_acc: 0.9969
Epoch 3/3
81/81 [==============================] - 6s 78ms/step - loss: 8.6846e-04 - acc: 0.9996 - val_loss: 0.0222 - val_acc: 0.9984
``````

![](https://github.com/rugewit/Convolutional-networks/blob/main/report_images/10.png)

Confusion matrix

![](https://github.com/rugewit/Convolutional-networks/blob/main/report_images/11.png)
