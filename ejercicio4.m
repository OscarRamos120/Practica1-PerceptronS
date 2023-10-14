% Cargar los datos desde el archivo irisbin.csv
data = readtable('irisbin.csv');

% Convertir la tabla a matrices
X = table2array(data(:, 1:4));
y = table2array(data(:, 5:7));

% Dividir los datos en conjuntos de entrenamiento y generalización (80% entrenamiento, 20% generalización)
n = size(X, 1);
train_percent = 0.8;
train_size = round(train_percent * n);
indices = randperm(n);
X_train = X(indices(1:train_size), :);
y_train = y(indices(1:train_size), :);
X_test = X(indices(train_size+1:end), :);
y_test = y(indices(train_size+1:end), :);

% Crear una red neuronal multicapa con una capa oculta
hiddenLayerSize = 10;
net = patternnet(hiddenLayerSize);

% Entrenar la red neuronal con los datos de entrenamiento
net.trainParam.epochs = 1000; % Número de épocas de entrenamiento
net = train(net, X_train', y_train');

% Validar el rendimiento de la red con los datos de generalización
y_pred = net(X_test');
y_test_rounded = vec2ind(y_test')';  % Convertir etiquetas de clases a índices
y_pred_rounded = vec2ind(y_pred)';   % Convertir las predicciones a índices

% Asegurarse de que las dimensiones sean iguales
assert(isequal(size(y_test_rounded), size(y_pred_rounded)), 'Dimensiones incompatibles.');

% Plot de los resultados
figure;
plotconfusion(y_test_rounded, y_pred_rounded); % Matriz de confusión
title('Matriz de Confusión');

% Leave-k-out Cross-Validation
k = 10; % Puedes ajustar el valor de k según tus necesidades
c = cvpartition(n, 'KFold', k);
cv_errors = zeros(k, 1);

for i = 1:k
    train_idx = training(c, i);
    test_idx = test(c, i);
    X_train_cv = X(train_idx, :);
    y_train_cv = y(train_idx, :);
    X_test_cv = X(test_idx, :);
    y_test_cv = y(test_idx, :);
    
    net_cv = patternnet(hiddenLayerSize);
    net_cv.trainParam.epochs = 1000;
    net_cv = train(net_cv, X_train_cv', y_train_cv');
    y_pred_cv = net_cv(X_test_cv');
    errors_cv = gsubtract(y_test_cv', y_pred_cv);
    classification_error_cv = sum(sum(abs(errors_cv))) / numel(y_test_cv);
    cv_errors(i) = classification_error_cv;
end

% Leave-one-out Cross-Validation
loocv = cvpartition(n, 'LeaveOut');
loocv_errors = zeros(loocv.NumTestSets, 1);

for i = 1:loocv.NumTestSets
    train_idx = training(loocv, i);
    test_idx = test(loocv, i);
    X_train_loocv = X(train_idx, :);
    y_train_loocv = y(train_idx, :);
    X_test_loocv = X(test_idx, :);
    y_test_loocv = y(test_idx, :);
    
    net_loocv = patternnet(hiddenLayerSize);
    net_loocv.trainParam.epochs = 1000;
    net_loocv = train(net_loocv, X_train_loocv', y_train_loocv');
    y_pred_loocv = net_loocv(X_test_loocv');
    errors_loocv = gsubtract(y_test_loocv', y_pred_loocv);
    classification_error_loocv = sum(sum(abs(errors_loocv))) / numel(y_test_loocv);
    loocv_errors(i) = classification_error_loocv;
end

% Calcular promedio y desviación estándar de los errores de ambos métodos
cv_avg_error = mean(cv_errors);
cv_std_error = std(cv_errors);
loocv_avg_error = mean(loocv_errors);
loocv_std_error = std(loocv_errors);

fprintf('Error de clasificación en generalización: %.2f%%\n', classification_error * 100);
fprintf('Leave-k-out Cross-Validation:\n');
fprintf('Promedio del error: %.2f%%\n', cv_avg_error * 100);
fprintf('Desviación estándar del error: %.2f%%\n', cv_std_error * 100);
fprintf('Leave-one-out Cross-Validation:\n');
fprintf('Promedio del error: %.2f%%\n', loocv_avg_error * 100);
fprintf('Desviación estándar del error: %.2f%%\n', loocv_std_error * 100);


% Validar el rendimiento de la red con los datos de generalización
y_pred = net(X_test');
errors = gsubtract(y_test', y_pred);
performance = perform(net, y_test', y_pred);

% Calcular el error esperado de clasificación
classification_error = sum(sum(abs(errors))) / numel(y_test);
