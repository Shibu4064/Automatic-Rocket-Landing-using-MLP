data = csvread('Normalize.csv', 1);
X = data(:,1:2).';
Y = data(:,3:4).';

hidden_list = [3 4 5 8];
lr_list     = [0.1 0.2 0.3];
mc_list     = [0.7 0.9];

best_rmse   = Inf;
best_config = struct();

for h = hidden_list
    for lr = lr_list
        for mc = mc_list

            net = feedforwardnet(h, 'traingdm');

            % Data split
            net.divideParam.trainRatio = 0.7;
            net.divideParam.valRatio   = 0.15;
            net.divideParam.testRatio  = 0.15;

            % Hyperparameters
            net.trainParam.lr       = lr;
            net.trainParam.mc       = mc;
            net.trainParam.min_grad = 1e-5;

            % Train
            [net, tr] = train(net, X, Y);

            % Validation predictions
            valX = X(:, tr.valInd);
            valY = Y(:, tr.valInd);
            valYhat = net(valX);

            % RMSE on validation
            mse_val = mean((valYhat(:) - valY(:)).^2);
            rmse_val = sqrt(mse_val);

            fprintf('h=%d, lr=%.3f, mc=%.2f -> val RMSE=%.4f\n', ...
                     h, lr, mc, rmse_val);

            % Keep best
            if rmse_val < best_rmse
                best_rmse = rmse_val;
                best_config.h  = h;
                best_config.lr = lr;
                best_config.mc = mc;
                best_net = net; %#ok<NASGU> % store if you want
            end

        end
    end
end

disp(best_config)
fprintf('Best validation RMSE = %.4f\n', best_rmse);