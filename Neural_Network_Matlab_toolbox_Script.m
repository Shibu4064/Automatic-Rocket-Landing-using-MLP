% Read your normalised game csv file Assuming the first row contains column headers
data = csvread('ce889_datacollection.csv', 1); %add file path correctly

% Extract input and output data
input_data = data(:, 1:2); % Columns x1 and x2
output_data = data(:, 3:4); % Columns y1 and y2

%input and out data should be the training subset of your dataset ideal threshold will be 70:30 70%training and 30%testing dataset 
input_data = transpose(input_data) %transpose your input and output data as we need two inputs and two outputs neuron 
output_data = transpose(output_data)

%train your neural network model 
net = feedforwardnet(3,'traingdm'); % parameters: number of neurons in hidden layer, 
                                    %training function: traingdm  (gradient decent with momentum) back propagation neural network

net.trainParam.lr = 0.5; % alpha value (used in sigmoid activation function (feed forward) and calculating the gradients (back propagation))
net.trainParam.mc = 0.9; % momentum value (used in back propagation during weight updation)
net.trainParam.min_grad = 1e-5; %eta value (used in weight updation)

net = train(net,input_data,output_data); %train network on input and output data
y = net(input_data) %get predicted outputs
