# Import cac thu vien can thiet
'''
!pip install numpy
!pip install matplotlib
!pip install --upgrade tensorflow
!pip install ipython
'''
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import IPython
from IPython.display import clear_output


#load datashet
# Tải dữ liệu MNIST, reshape dữ liệu để phù hợp với đầu vào của mô hình, và hiển thị kích thước của dữ liệu
print("Load MNIST Database")
mnist = tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test)= mnist.load_data()
x_train=np.reshape(x_train,(60000,784))/255.0
x_test= np.reshape(x_test,(10000,784))/255.0
y_train = np.matrix(np.eye(10)[y_train])
y_test = np.matrix(np.eye(10)[y_test])
print("----------------------------------")
print(x_train.shape)
print(y_train.shape)


#Activation function
# Định nghĩa các hàm kích hoạt: sigmoid, softmax, và ReLU
def sigmoid(x):
    return 1./(1.+np.exp(-x))

def softmax(x):
    return np.divide(np.matrix(np.exp(x)),np.mat(np.sum(np.exp(x),axis=1)))

def relu(x):
    return np.maximum(x, 0)


# Hàm lan truyền xuôi (forward propagation) của mạng nơ-ron, với 2 lớp ẩn sử dụng ReLU và sigmoid làm hàm kích hoạt
def Forwardpass(X,Wh1,bh1,Wh2,bh2, Wo,bo):
    zh1 = X@Wh1.T + bh1
    ah1 = relu(zh1)

    zh2 = ah1@Wh2.T + bh2 
    ah2 = sigmoid(zh2)

    z = ah2@Wo.T + bo
    o = softmax(z)
    return o


# Hàm tính độ chính xác của mô hình dựa trên nhãn thực tế và nhãn dự đoán
def AccTest(label,prediction):    # calculate the matching score
    OutMaxArg=np.argmax(prediction,axis=1)
    LabelMaxArg=np.argmax(label,axis=1)
    Accuracy=np.mean(OutMaxArg==LabelMaxArg)
    return Accuracy



# Thiết lập các tham số cho mô hình: tốc độ học, số epoch, số lượng mẫu huấn luyện/kiểm tra, số lượng đầu vào, số nút ẩn, và số lớp
learningRate = 0.1
Epoch = 50
NumTrainSamples=60000
NumTestSamples=10000

NumInputs=784
NumHiddenUnits=512
NumHiddenUnits2=512
NumClasses=10


#inital weights
#hidden layer 1
# Khởi tạo trọng số và hệ số điều chỉnh cho các lớp của mô hình

Wh1=np.matrix(np.random.uniform(-0.5,0.5,(NumHiddenUnits,NumInputs)))
bh1= np.random.uniform(0,0.5,(1,NumHiddenUnits))
dWh1= np.zeros((NumHiddenUnits,NumInputs))
dbh1= np.zeros((1,NumHiddenUnits))
#hidden layer
Wh2=np.matrix(np.random.uniform(-0.5,0.5,(NumHiddenUnits2,NumHiddenUnits)))
bh2= np.random.uniform(0,0.5,(1,NumHiddenUnits))
dWh2= np.zeros((NumHiddenUnits,NumInputs))
dbh2= np.zeros((1,NumHiddenUnits))
#Output layer
Wo=np.random.uniform(-0.5,0.5,(NumClasses,NumHiddenUnits2))
bo= np.random.uniform(0,0.5,(1,NumClasses))
dWo= np.zeros((NumClasses,NumHiddenUnits))
dbo= np.zeros((1,NumClasses))


# Khởi tạo các mảng để lưu trữ loss và độ chính xác trong quá trình huấn luyện, và thiết lập kích thước batch
loss = []
Acc = []
Batch_size =200
Stochastic_samples = np.arange(NumTrainSamples)

# Vòng lặp huấn luyện mô hình với số epoch đã cho, trong mỗi epoch, dữ liệu được trộn ngẫu nhiên và chia thành các batch
for ep in range (Epoch):
  np.random.shuffle(Stochastic_samples)
  for ite in range (0,NumTrainSamples,Batch_size):


    #feed fordware propagation
    # Lấy dữ liệu huấn luyện cho batch hiện tại
    Batch_samples = Stochastic_samples[ite:ite+Batch_size]
    x = x_train[Batch_samples,:]
    y = y_train[Batch_samples,:]


    # Thực hiện lan truyền xuôi để tính toán đầu ra của mô hình cho batch hiện tại
    zh1 = x@Wh1.T + bh1
    ah1 = relu(zh1)

    zh2 = ah1@Wh2.T + bh2
    ah2 = sigmoid(zh2)

    z = ah2@Wo.T + bo
    o = softmax(z)


    #calculate loss
    # Tính toán giá trị loss cho batch hiện tại và lưu vào mảng loss
    loss.append(-np.sum(np.multiply(y,np.log10(o))))


    #calculate the error for the ouput layer
    d = o-y
    #Back propagate error
    dh = d@Wo
    dhs1 = np.multiply(np.multiply(dh,ah2),(1-ah2))
    dhs0 = np.where(zh1 > 0, np.matmul(dhs1, Wh2), 0)   
    # dhs0 = np.multiply(np.multiply(np.multiply(dh,ah2),(1-ah2)))

    #update weight
    # Tính toán gradient của trọng số và hệ số điều chỉnh cho các lớp
    dWo = np.matmul(np.transpose(d),ah2)
    dbo = np.mean(d)  # consider a is 1 for bias
    dWh2 = np.matmul(np.transpose(dhs1),ah1)
    dbh2 = np.mean(dhs1)  # consider a is 1 for bias
    dWh1 = np.matmul(np.transpose(dhs0),x)
    dbh1 = np.mean(dhs0)  # consider a is 1 for bias


    # Cập nhật trọng số và hệ số điều chỉnh cho các lớp dựa trên gradient
    Wo =Wo - learningRate*dWo/Batch_size
    bo =bo - learningRate*dbo
    Wh2 =Wh2-learningRate*dWh2/Batch_size
    bh2 =bh2-learningRate*dbh2
    Wh1 =Wh1-learningRate*dWh1/Batch_size
    bh1 =bh1-learningRate*dbh1



    #Test accuracy with random innitial weights
    '''
    Tính toán đầu ra của mô hình trên tập dữ liệu kiểm tra, đánh giá độ chính xác, 
    hiển thị đồ thị độ chính xác và in ra độ chính xác, loss cho batch hiện tại
    '''
    prediction = Forwardpass(x_test,Wh1,bh1,Wh2,bh2,Wo,bo)
    Acc.append(AccTest(y_test,prediction))
    clear_output(wait=True)
    plt.plot([i for i, _ in enumerate(Acc)],Acc,'o')

    plt.show()

    print('Accuracy:',AccTest(y_test,prediction) )
    print('Loss:',-np.sum(np.multiply(y,np.log10(o))) )


  '''
  Sau khi kết thúc vòng lặp huấn luyện, in ra epoch hiện tại,
  độ chính xác trên tập kiểm tra và loss trên toàn bộ tập huấn luyện,
  hiển thị đồ thị cuối cùng của độ chính xác
  '''
  print('Epoch:', ep )
  print('Accuracy:',AccTest(y_test,prediction) )
  print('Loss:',-np.sum(np.multiply(y,np.log10(o))) )
  plt.show()