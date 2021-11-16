# Amir Hossein Ebrahimnezhad - ECE 740 Course Project - Fall 2020


# Case 1: CNN Model with 4 Convolutional layers

Dat = [100, 1000, 4500, 4500, 4500, 4500] # Considering different training data size for understanidgn the impact of the number of training samples
Epoch = [10, 10, 10, 20, 30, 50]          # Considering different epoch sets for comparing the impact of different epochs 
case = 1






val_loss_data = []
val_acc_data = []

for h in range(len(Epoch)):

  import matplotlib.pyplot as plt
  import tensorflow as tf
  import numpy as np
  import cv2 
  import os
  import time


  from tensorflow.keras import layers, losses, optimizers
  from tensorflow.keras.models import Sequential
  from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, UpSampling2D, Softmax
  from tqdm import tqdm
  from keras import callbacks

  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  session = tf.Session(config=config)
  os.system('clear')

  Datadir = os.getcwd() # The directory of training data
  Datadir = os.path.join(Datadir, "Data") 

     

  CATEGORIES = ["X", "Y"]                      # Defining the categories for input and output data samples


  input1 = []  # Main input without resizing
  output1 = [] # Main output without resizing

  input_main = []  # Main input which will be resized
  output_main = [] # Main output which will be resized

  input_main2 = []  # Main input which will be resized and shuffled 
  output_main2 = [] # Main output which will be resized and shuffled 

  train_input = []  # Training input data  80%
  train_output = [] # Training output data 80%
  
  val_input = []    # Valdiation input data  10%
  val_output = []    # Valdiation output data 10%

  test_input = []   # Test input data  10%
  test_output = []  # Test output data 10%


  limit = Dat[h]  # Limiting the number of input data                           


  # original input size

  800
  1280
  
  # Resized image size
  
  d1 = 80  
  d2 = 128


  a1 = int(800/2-d1/2+20-5-3-2-1-2)
  b1 = int(1260/2-d2/2+20+5+1)

  # Loading the input data

  path = os.path.join(Datadir, "X")
  name = os.listdir(path)
  name.sort()
  i = 0
  for img in name:
    img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
    input_main.append(img_array/255.)
    input1.append(img_array[a1:a1+d1, b1:b1+d2]/255.)

    i = i + 1
    print ("\033[A                             \033[A")
    print("Input data load progress: ", int(i/limit*100), " %")

    if i == limit:
      break
    

  print("\n")


  path = os.path.join(Datadir, "Y")
  name = os.listdir(path)
  name.sort()
  i = 0
  for img in name:
    img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
    output_main.append(img_array/255.)
    output1.append(img_array[a1:a1+d1, b1:b1+d2]/255.)

    i = i + 1
    print ("\033[A                             \033[A")
    print("Output data load progress: ", int(i/limit*100), " %")

    if i == limit:
      break



  # Splitting the data into training 80%, validation 10% and test 10%

  a = np.random.permutation(limit)

  a[0:10] = np.arange(10)

  l1 = int(len(a)/10*1)
  l2 = int(len(a)/10*1)
  l3 = int(len(a)/10*2+1)

  for i in range(l1):
    test_input.append(input1[int(a[i])])
    test_output.append(output1[int(a[i])])


  for i in range(l2, l3):

    val_input.append(input1[int(a[i])])
    val_output.append(output1[int(a[i])])
    input_main2.append(input_main[int(a[i])])
    output_main2.append(output_main[int(a[i])])

  for i in range(l3, len(a)):
    train_input.append(input1[int(a[i])])
    train_output.append(output1[int(a[i])])


  # Reshaping the data for keras configuration

  train_input  =   np.array(train_input).reshape(-1, d1, d2, 1)
  train_output =   np.array(train_output).reshape(-1, d1, d2, 1)

  val_input   =   np.array(test_input).reshape(-1, d1, d2, 1)
  val_output  =   np.array(test_output).reshape(-1, d1, d2, 1)

  test_input   =   np.array(test_input).reshape(-1, d1, d2, 1)
  test_output  =   np.array(test_output).reshape(-1, d1, d2, 1)


  input_main2  =   np.array(input_main2).reshape(-1, 800, 1280,1)
  output_main2 =   np.array(output_main2).reshape(-1, 800, 1280,1)



  # Defining the model

  model = Sequential()

  model.add(Conv2D(32, (3,3), input_shape = (d1, d2, 1), activation = "relu", padding = 'same'))
  model.add(MaxPooling2D(pool_size=(2,2), padding='same'))




  model.add(Conv2D(64, (3,3), activation = "relu", padding = 'same'))
  model.add(MaxPooling2D(pool_size=(1,1), padding='same'))


  model.add(UpSampling2D((2,2)))



  model.add(Conv2D(32, (3,3), activation = "relu", padding = 'same'))

  model.add(Conv2D(1, (3,3), activation = "relu", padding = 'same'))


  model.summary()


  # Saving the training log

  model_checkpoint = callbacks.ModelCheckpoint('/model{}_{}.h5'.format(case, h))

  logger = callbacks.CSVLogger('training{}_{}.log'.format(case, h))

  tensorboard = callbacks.TensorBoard(log_dir='./Models')

  callbacks = [logger, tensorboard]


  model.compile(optimizer=optimizers.Adam(), loss=losses.MeanSquaredError(), metrics=["accuracy"])

  # Training the model

  model.fit(train_input, train_output,
                  epochs=Epoch[h],
                  batch_size=64,
                  shuffle=True,
                  validation_data=(val_input, val_output),
                  callbacks = callbacks)



  val_loss, val_acc = model.evaluate(test_input, test_output) # Evaluating the model
  val_loss = round(val_loss, 4)
  val_acc  = round(val_acc, 4)
  val_loss_data.append(val_loss) 
  val_acc_data.append(val_acc) 

  # Saving the validaiton data

  if h == (len(Epoch)-1):

    f1 =  open('val_loss{}.txt'.format(case), 'w') 
    f2 =  open('val_acc{}.txt'.format(case), 'w')

    f1.write('Model {}, Epoch = {}, Data = {} \n'.format(case, Epoch[h], Dat[h]))
    f2.write('Model {}, Epoch = {}, Data = {} \n'.format(case, Epoch[h], Dat[h]))

    for loss in val_loss_data:
        f1.write(str(loss) + '\n')

    for acc in val_acc_data:
        f2.write(str(acc) + '\n')

  model.save('model{}_{}.h5'.format(case, h))

  decoded_imgs = model.predict(test_input)
  decoded_test = np.zeros([800, 1280])

  model.save('model{}_{}.h5'.format(case, h))

  decoded_imgs = model.predict(test_input)
  decoded_test = np.zeros([800, 1280])


  k = 0

  l = 20

  l1 = int(800/l)
  l2 = int(1280/l)

  for i in range(l-1):
    for j in range(l-1):
      temp = model.predict(input_main2[0:1,l1*i:l1*i+80, l2*j:l2*j+128])
      decoded_test[l1*i:l1*i+80, l2*j:l2*j+128] = (temp[0].reshape(80, 128)+decoded_test[l1*i:l1*i+80, l2*j:l2*j+128])/2
      k = k+1


  test_input    = np.array(test_input).reshape(-1, d1, d2)
  test_output   = np.array(test_output).reshape(-1, d1, d2)
  decoded_imgs  = np.array(decoded_imgs).reshape(-1, d1, d2)

  input_main2  =   np.array(input_main2).reshape(-1, 800, 1280)
  output_main2 =   np.array(output_main2).reshape(-1, 800, 1280)


  n = 3

  for i in range(n):
      # display original input
      if i == 0:
        plt.figure(7+i, figsize=(8, 6))

      ax = plt.subplot(n, 3, 3*i+1)
      plt.imshow(test_input[i]*255., cmap = "gray")
      if i == (n-1):
        plt.xlabel('original input')



      # display original output
      ax = plt.subplot(n, 3, 3*i+2)
      plt.imshow(test_output[i]*255., cmap = "gray")
      if i ==0:
        plt.title('Model {}, Epochs = {}, Input Data = {}'.format(case, Epoch[h], Dat[h]))
      if i == (n-1):
        plt.xlabel('original output')


      # display predicted (decoded) output
      ax = plt.subplot(n, 3, 3*i+3)
      plt.imshow(decoded_imgs[i]*255., cmap = "gray")
      if i == (n-1):
        plt.xlabel('decoded output')




  plt.savefig('I{}_{}.png'.format(case, h))


  plt.show(block=False)
  plt.pause(3)
  plt.close()




# Case 2: CNN Model with 6 Convolutional layers


Dat = [100, 1000, 4500, 4500, 4500, 4500] # Considering different training data size for understanidgn the impact of the number of training samples
Epoch = [10, 10, 10, 20, 30, 50]          # Considering different epoch sets for comparing the impact of different epochs 
case = 2


val_loss_data = []
val_acc_data = []

for h in range(len(Epoch)):

  import matplotlib.pyplot as plt
  import tensorflow as tf
  import numpy as np
  import cv2 
  import os
  import time


  from tensorflow.keras import layers, losses, optimizers
  from tensorflow.keras.models import Sequential
  from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, UpSampling2D, Softmax
  from tqdm import tqdm
  from keras import callbacks

  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  session = tf.Session(config=config)
  os.system('clear')

  Datadir = os.getcwd() # The directory of training data
  Datadir = os.path.join(Datadir, "Data") 


  CATEGORIES = ["X", "Y"]                      # Defining the categories for input and output data samples


  input1 = []  # Main input without resizing
  output1 = [] # Main output without resizing

  input_main = []  # Main input which will be resized
  output_main = [] # Main output which will be resized

  input_main2 = []  # Main input which will be resized and shuffled 
  output_main2 = [] # Main output which will be resized and shuffled 

  train_input = []  # Training input data  80%
  train_output = [] # Training output data 80%
  
  val_input = []    # Valdiation input data  10%
  val_output = []    # Valdiation output data 10%

  test_input = []   # Test input data  10%
  test_output = []  # Test output data 10%


  limit = Dat[h]  # Limiting the number of input data                           


  # original input size

  800
  1280
  
  # Resized image size
  
  d1 = 80  
  d2 = 128


  a1 = int(800/2-d1/2+20-5-3-2-1-2)
  b1 = int(1260/2-d2/2+20+5+1)

  # Loading the input data

  path = os.path.join(Datadir, "X")
  name = os.listdir(path)
  name.sort()
  i = 0
  for img in name:
    img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
    input_main.append(img_array/255.)
    input1.append(img_array[a1:a1+d1, b1:b1+d2]/255.)

    i = i + 1
    print ("\033[A                             \033[A")
    print("Input data load progress: ", int(i/limit*100), " %")

    if i == limit:
      break
    

  print("\n")


  path = os.path.join(Datadir, "Y")
  name = os.listdir(path)
  name.sort()
  i = 0
  for img in name:
    img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
    output_main.append(img_array/255.)
    output1.append(img_array[a1:a1+d1, b1:b1+d2]/255.)

    i = i + 1
    print ("\033[A                             \033[A")
    print("Output data load progress: ", int(i/limit*100), " %")

    if i == limit:
      break



  # Splitting the data into training 80%, validation 10% and test 10%

  a = np.random.permutation(limit)

  a[0:10] = np.arange(10)

  l1 = int(len(a)/10*1)
  l2 = int(len(a)/10*1)
  l3 = int(len(a)/10*2+1)

  for i in range(l1):
    test_input.append(input1[int(a[i])])
    test_output.append(output1[int(a[i])])


  for i in range(l2, l3):

    val_input.append(input1[int(a[i])])
    val_output.append(output1[int(a[i])])
    input_main2.append(input_main[int(a[i])])
    output_main2.append(output_main[int(a[i])])

  for i in range(l3, len(a)):
    train_input.append(input1[int(a[i])])
    train_output.append(output1[int(a[i])])


  # Reshaping the data for keras configuration

  train_input  =   np.array(train_input).reshape(-1, d1, d2, 1)
  train_output =   np.array(train_output).reshape(-1, d1, d2, 1)

  val_input   =   np.array(test_input).reshape(-1, d1, d2, 1)
  val_output  =   np.array(test_output).reshape(-1, d1, d2, 1)

  test_input   =   np.array(test_input).reshape(-1, d1, d2, 1)
  test_output  =   np.array(test_output).reshape(-1, d1, d2, 1)


  input_main2  =   np.array(input_main2).reshape(-1, 800, 1280,1)
  output_main2 =   np.array(output_main2).reshape(-1, 800, 1280,1)



  # Defining the model

  model = Sequential()

  model.add(Conv2D(32, (3,3), input_shape = (d1, d2, 1), activation = "relu", padding = 'same'))
  model.add(MaxPooling2D(pool_size=(2,2), padding='same'))


  model.add(Conv2D(64, (3,3), activation = "relu", padding = 'same'))
  model.add(MaxPooling2D(pool_size=(2,2), padding='same'))


  model.add(Conv2D(64, (3,3), activation = "relu", padding = 'same'))
  model.add(MaxPooling2D(pool_size=(1,1), padding='same'))


  model.add(UpSampling2D((2,2)))

  model.add(Conv2D(64, (3,3), activation = "relu", padding = 'same'))
  model.add(UpSampling2D((2,2)))

  model.add(Conv2D(32, (3,3), activation = "relu", padding = 'same'))

  model.add(Conv2D(1, (3,3), activation = "relu", padding = 'same'))


  model.summary()


  # Saving the training log

  model_checkpoint = callbacks.ModelCheckpoint('/model{}_{}.h5'.format(case, h))

  logger = callbacks.CSVLogger('training{}_{}.log'.format(case, h))

  tensorboard = callbacks.TensorBoard(log_dir='./Models')

  callbacks = [logger, tensorboard]


  model.compile(optimizer=optimizers.Adam(), loss=losses.MeanSquaredError(), metrics=["accuracy"])

  # Training the model

  model.fit(train_input, train_output,
                  epochs=Epoch[h],
                  batch_size=64,
                  shuffle=True,
                  validation_data=(val_input, val_output),
                  callbacks = callbacks)



  val_loss, val_acc = model.evaluate(test_input, test_output) # Evaluating the model
  val_loss = round(val_loss, 4)
  val_acc  = round(val_acc, 4)
  val_loss_data.append(val_loss) 
  val_acc_data.append(val_acc) 


  # Saving the validaiton data

  if h == (len(Epoch)-1):

    f1 =  open('val_loss{}.txt'.format(case), 'w') 
    f2 =  open('val_acc{}.txt'.format(case), 'w')

    f1.write('Model {}, Epoch = {}, Data = {} \n'.format(case, Epoch[h], Dat[h]))
    f2.write('Model {}, Epoch = {}, Data = {} \n'.format(case, Epoch[h], Dat[h]))

    for loss in val_loss_data:
        f1.write(str(loss) + '\n')

    for acc in val_acc_data:
        f2.write(str(acc) + '\n')

  model.save('model{}_{}.h5'.format(case, h))

  decoded_imgs = model.predict(test_input)
  decoded_test = np.zeros([800, 1280])

  model.save('model{}_{}.h5'.format(case, h))

  decoded_imgs = model.predict(test_input)
  decoded_test = np.zeros([800, 1280])


  k = 0

  l = 20

  l1 = int(800/l)
  l2 = int(1280/l)

  for i in range(l-1):
    for j in range(l-1):
      temp = model.predict(input_main2[0:1,l1*i:l1*i+80, l2*j:l2*j+128])
      decoded_test[l1*i:l1*i+80, l2*j:l2*j+128] = (temp[0].reshape(80, 128)+decoded_test[l1*i:l1*i+80, l2*j:l2*j+128])/2
      k = k+1


  test_input    = np.array(test_input).reshape(-1, d1, d2)
  test_output   = np.array(test_output).reshape(-1, d1, d2)
  decoded_imgs  = np.array(decoded_imgs).reshape(-1, d1, d2)

  input_main2  =   np.array(input_main2).reshape(-1, 800, 1280)
  output_main2 =   np.array(output_main2).reshape(-1, 800, 1280)


  n = 3

  for i in range(n):
      # display original input
      if i == 0:
        plt.figure(7+i, figsize=(8, 6))

      ax = plt.subplot(n, 3, 3*i+1)
      plt.imshow(test_input[i]*255., cmap = "gray")
      if i == (n-1):
        plt.xlabel('original input')



      # display original output
      ax = plt.subplot(n, 3, 3*i+2)
      plt.imshow(test_output[i]*255., cmap = "gray")
      if i ==0:
        plt.title('Model {}, Epochs = {}, Input Data = {}'.format(case, Epoch[h], Dat[h]))
      if i == (n-1):
        plt.xlabel('original output')


      # display predicted (decoded) output
      ax = plt.subplot(n, 3, 3*i+3)
      plt.imshow(decoded_imgs[i]*255., cmap = "gray")
      if i == (n-1):
        plt.xlabel('decoded output')




  plt.savefig('I{}_{}.png'.format(case, h))


  plt.show(block=False)
  plt.pause(3)
  plt.close()




# Case 3: CNN Model with 8 Convolutional layers


Dat = [100, 1000, 4500, 4500, 4500, 4500] # Considering different training data size for understanidgn the impact of the number of training samples
Epoch = [10, 10, 10, 20, 30, 50]          # Considering different epoch sets for comparing the impact of different epochs 
case = 3


val_loss_data = []
val_acc_data = []

for h in range(len(Epoch)):

  import matplotlib.pyplot as plt
  import tensorflow as tf
  import numpy as np
  import cv2 
  import os
  import time


  from tensorflow.keras import layers, losses, optimizers
  from tensorflow.keras.models import Sequential
  from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, UpSampling2D, Softmax
  from tqdm import tqdm
  from keras import callbacks

  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  session = tf.Session(config=config)
  os.system('clear')

  Datadir = os.getcwd() # The directory of training data
  Datadir = os.path.join(Datadir, "Data") 


  CATEGORIES = ["X", "Y"]                      # Defining the categories for input and output data samples


  input1 = []  # Main input without resizing
  output1 = [] # Main output without resizing

  input_main = []  # Main input which will be resized
  output_main = [] # Main output which will be resized

  input_main2 = []  # Main input which will be resized and shuffled 
  output_main2 = [] # Main output which will be resized and shuffled 

  train_input = []  # Training input data  80%
  train_output = [] # Training output data 80%
  
  val_input = []    # Valdiation input data  10%
  val_output = []    # Valdiation output data 10%

  test_input = []   # Test input data  10%
  test_output = []  # Test output data 10%


  limit = Dat[h]  # Limiting the number of input data                           


  # original input size

  800
  1280
  
  # Resized image size
  
  d1 = 80  
  d2 = 128


  a1 = int(800/2-d1/2+20-5-3-2-1-2)
  b1 = int(1260/2-d2/2+20+5+1)

  # Loading the input data

  path = os.path.join(Datadir, "X")
  name = os.listdir(path)
  name.sort()
  i = 0
  for img in name:
    img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
    input_main.append(img_array/255.)
    input1.append(img_array[a1:a1+d1, b1:b1+d2]/255.)

    i = i + 1
    print ("\033[A                             \033[A")
    print("Input data load progress: ", int(i/limit*100), " %")

    if i == limit:
      break
    

  print("\n")


  path = os.path.join(Datadir, "Y")
  name = os.listdir(path)
  name.sort()
  i = 0
  for img in name:
    img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
    output_main.append(img_array/255.)
    output1.append(img_array[a1:a1+d1, b1:b1+d2]/255.)

    i = i + 1
    print ("\033[A                             \033[A")
    print("Output data load progress: ", int(i/limit*100), " %")

    if i == limit:
      break



  # Splitting the data into training 80%, validation 10% and test 10%

  a = np.random.permutation(limit)

  a[0:10] = np.arange(10)

  l1 = int(len(a)/10*1)
  l2 = int(len(a)/10*1)
  l3 = int(len(a)/10*2+1)

  for i in range(l1):
    test_input.append(input1[int(a[i])])
    test_output.append(output1[int(a[i])])


  for i in range(l2, l3):

    val_input.append(input1[int(a[i])])
    val_output.append(output1[int(a[i])])
    input_main2.append(input_main[int(a[i])])
    output_main2.append(output_main[int(a[i])])

  for i in range(l3, len(a)):
    train_input.append(input1[int(a[i])])
    train_output.append(output1[int(a[i])])


  # Reshaping the data for keras configuration

  train_input  =   np.array(train_input).reshape(-1, d1, d2, 1)
  train_output =   np.array(train_output).reshape(-1, d1, d2, 1)

  val_input   =   np.array(test_input).reshape(-1, d1, d2, 1)
  val_output  =   np.array(test_output).reshape(-1, d1, d2, 1)

  test_input   =   np.array(test_input).reshape(-1, d1, d2, 1)
  test_output  =   np.array(test_output).reshape(-1, d1, d2, 1)


  input_main2  =   np.array(input_main2).reshape(-1, 800, 1280,1)
  output_main2 =   np.array(output_main2).reshape(-1, 800, 1280,1)



  # Defining the model

  model = Sequential()

  model.add(Conv2D(32, (3,3), input_shape = (d1, d2, 1), activation = "relu", padding = 'same'))
  model.add(MaxPooling2D(pool_size=(2,2), padding='same'))


  model.add(Conv2D(64, (3,3), activation = "relu", padding = 'same'))
  model.add(MaxPooling2D(pool_size=(2,2), padding='same'))


  model.add(Conv2D(64, (3,3), activation = "relu", padding = 'same'))
  model.add(MaxPooling2D(pool_size=(2,2), padding='same'))

  model.add(Conv2D(128, (3,3), activation = "relu", padding = 'same'))
  model.add(MaxPooling2D(pool_size=(1,1), padding='same'))


  model.add(UpSampling2D((2,2)))

  model.add(Conv2D(64, (3,3), activation = "relu", padding = 'same'))
  model.add(UpSampling2D((2,2)))

  model.add(Conv2D(64, (3,3), activation = "relu", padding = 'same'))
  model.add(UpSampling2D((2,2)))

  model.add(Conv2D(32, (3,3), activation = "relu", padding = 'same'))


  model.add(Conv2D(1, (3,3), activation = "relu", padding = 'same'))


  model.summary()


  # Saving the training log

  model_checkpoint = callbacks.ModelCheckpoint('/model{}_{}.h5'.format(case, h))

  logger = callbacks.CSVLogger('training{}_{}.log'.format(case, h))

  tensorboard = callbacks.TensorBoard(log_dir='./Models')

  callbacks = [logger, tensorboard]


  model.compile(optimizer=optimizers.Adam(), loss=losses.MeanSquaredError(), metrics=["accuracy"])

  # Training the model

  model.fit(train_input, train_output,
                  epochs=Epoch[h],
                  batch_size=64,
                  shuffle=True,
                  validation_data=(val_input, val_output),
                  callbacks = callbacks)



  val_loss, val_acc = model.evaluate(test_input, test_output) # Evaluating the model
  val_loss = round(val_loss, 4)
  val_acc  = round(val_acc, 4)
  val_loss_data.append(val_loss) 
  val_acc_data.append(val_acc) 

    
  # Saving the validaiton data

  if h == (len(Epoch)-1):

    f1 =  open('val_loss{}.txt'.format(case), 'w') 
    f2 =  open('val_acc{}.txt'.format(case), 'w')

    f1.write('Model {}, Epoch = {}, Data = {} \n'.format(case, Epoch[h], Dat[h]))
    f2.write('Model {}, Epoch = {}, Data = {} \n'.format(case, Epoch[h], Dat[h]))

    for loss in val_loss_data:
        f1.write(str(loss) + '\n')

    for acc in val_acc_data:
        f2.write(str(acc) + '\n')

  model.save('model{}_{}.h5'.format(case, h))

  decoded_imgs = model.predict(test_input)
  decoded_test = np.zeros([800, 1280])


  k = 0

  l = 20

  l1 = int(800/l)
  l2 = int(1280/l)

  for i in range(l-1):
    for j in range(l-1):
      temp = model.predict(input_main2[0:1,l1*i:l1*i+80, l2*j:l2*j+128])
      decoded_test[l1*i:l1*i+80, l2*j:l2*j+128] = (temp[0].reshape(80, 128)+decoded_test[l1*i:l1*i+80, l2*j:l2*j+128])/2
      k = k+1


  test_input    = np.array(test_input).reshape(-1, d1, d2)
  test_output   = np.array(test_output).reshape(-1, d1, d2)
  decoded_imgs  = np.array(decoded_imgs).reshape(-1, d1, d2)

  input_main2  =   np.array(input_main2).reshape(-1, 800, 1280)
  output_main2 =   np.array(output_main2).reshape(-1, 800, 1280)


  n = 3

  for i in range(n):
      # display original input
      if i == 0:
        plt.figure(7+i, figsize=(8, 6))

      ax = plt.subplot(n, 3, 3*i+1)
      plt.imshow(test_input[i]*255., cmap = "gray")
      if i == (n-1):
        plt.xlabel('original input')



      # display original output
      ax = plt.subplot(n, 3, 3*i+2)
      plt.imshow(test_output[i]*255., cmap = "gray")
      if i ==0:
        plt.title('Model {}, Epochs = {}, Input Data = {}'.format(case, Epoch[h], Dat[h]))
      if i == (n-1):
        plt.xlabel('original output')


      # display predicted (decoded) output
      ax = plt.subplot(n, 3, 3*i+3)
      plt.imshow(decoded_imgs[i]*255., cmap = "gray")
      if i == (n-1):
        plt.xlabel('decoded output')




  plt.savefig('I{}_{}.png'.format(case, h))


  plt.show(block=False)
  plt.pause(3)
  plt.close()


