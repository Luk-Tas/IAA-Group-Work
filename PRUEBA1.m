clear; clc; close all;
load('Trainnumbers.mat');
X = double(Trainnumbers.image);   % 784 x N
N = size(X, 2);                   % Número de muestras

% ==========================
% CENTRAR LOS DATOS
% ==========================
X_mean = mean(X, 2);
X_centered = X - X_mean;

% ==========================
% MATRIZ DE COVARIANZA Y AUTOVALORES/VECTORES
% ==========================
C = cov(X_centered');   % 784 x 784
[EigVec, EigVal] = eig(C);
[eigValsSorted, idx] = sort(diag(EigVal), 'descend');
EigVec = EigVec(:, idx);

% ==========================
% VARIANZA ACUMULADA
% ==========================
total_variance = sum(eigValsSorted);
accum_variance = cumsum(eigValsSorted) / total_variance;

% Mostrar % de varianza con 50, 100, 200, 300 componentes
fprintf('Varianza acumulada:\n');
componentes = [50, 100, 200, 300, 400, 784];
for c = componentes
    fprintf('k = %d --> Varianza: %.4f%%\n', c, accum_variance(c)*100);
end

% Buscar k para superar 90% y 95%
k90 = find(accum_variance >= 0.90, 1);
k95 = find(accum_variance >= 0.95, 1);
fprintf('\nPara 90%% de varianza se necesitan: %d componentes\n', k90);
fprintf('Para 95%% de varianza se necesitan: %d componentes\n', k95);

% Graficar Varianza acumulada
figure;
plot(accum_variance, 'LineWidth', 2);
xlabel('Número de componentes');
ylabel('Varianza acumulada');
title('Curva de varianza acumulada');
grid on;

% ==========================
% RECONSTRUCCIÓN Y MSE
% ==========================
dims = [10, 20, 50, 100, 200, 400];
MSE = zeros(size(dims));
for i = 1:length(dims)
    k = dims(i);
    W = EigVec(:, 1:k);
    projected = W' * X_centered;
    reconstructed = W * projected + X_mean;
    error = X - reconstructed;
    MSE(i) = mean(error(:).^2);
end

% Graficar MSE
figure;
plot(dims, MSE, '-o', 'LineWidth', 2);
xlabel('Componentes principales');
ylabel('MSE de reconstrucción');
title('MSE vs Número de componentes');
grid on;

% ==========================
% MOSTRAR 4 DÍGITOS DIFERENTES
% ==========================
% Escoge el k con el que quieres la reconstrucción final
k = 154; 
W = EigVec(:, 1:k);
projected = W' * X_centered;
reconstructed = W * projected + X_mean;

% Índices de 4 imágenes a comparar (puedes cambiarlos)
indices = [1, 100, 500, 1000];  

figure;
for j = 1:length(indices)
    idx_img = indices(j);
    
    % Mostrar original
    subplot(4, 2, (j-1)*2 + 1);
    imshow(reshape(X(:,idx_img), [28 28])', [0 255]);
    title(['Original #' num2str(idx_img)]);
    
    % Mostrar reconstruida
    subplot(4, 2, (j-1)*2 + 2);
    img_rec = reshape(reconstructed(:,idx_img), [28 28])';
    imshow(mat2gray(img_rec));  
    title(['Reconstruida k=' num2str(k)]);
end

% ================================
% k-NN sobre la proyección PCA
% ================================
% Usamos la proyección 'projected' de PCA con k=154 componentes
projected_data = projected;   % Esta es la data reducida con 154 componentes

% Particionamos en entrenamiento y prueba (80% - 20%)
cv = cvpartition(Trainnumbers.label, 'HoldOut', 0.1);
trainIdx = training(cv);
testIdx = test(cv);

% Preparamos los datos
X_train = projected_data(:, trainIdx)';   % Transponer por formato de MATLAB
Y_train = Trainnumbers.label(trainIdx);
X_test = projected_data(:, testIdx)';
Y_test = Trainnumbers.label(testIdx);

% Aplicamos k-NN
k_valor = 3;   % Puedes probar con 1, 3, 5
Mdl_knn = fitcknn(X_train, Y_train, 'NumNeighbors', k_valor);

% Predicción
Y_pred = predict(Mdl_knn, X_test);


% ================================
% Matriz de confusión y análisis detallado
% ================================
figure;
cm = confusionchart(Y_test, Y_pred);
title(['Matriz de Confusión - k-NN (k=' num2str(k_valor) ')']);

% Calcular la matriz numérica
Cmat = confusionmat(Y_test, Y_pred);

% Aciertos = suma de la diagonal
aciertos = sum(diag(Cmat));

% Errores = total de test - aciertos
errores = sum(Cmat(:)) - aciertos;

% Calcular Accuracy
accuracy = (aciertos / sum(Cmat(:))) * 100;

% Mostrar resultados
fprintf('Total de aciertos: %d\n', aciertos);
fprintf('Total de errores: %d\n', errores);
fprintf('Accuracy: %.2f%%\n', accuracy);

% ================================
% Mostrar confusiones relevantes
% ================================
fprintf('\nConfusiones más relevantes (cuando el error > 10):\n');
for i = 1:size(Cmat,1)
    for j = 1:size(Cmat,2)
        if i ~= j && Cmat(i,j) > 10
            fprintf('El número %d se confunde %d veces con el número %d\n', i-1, Cmat(i,j), j-1);
        end
    end
end
% =========================================
% Clasificador Bayesiano usando la proyección PCA
% =========================================
fprintf('\n=========== CLASIFICADOR BAYESIANO ===========\n');

% Reutilizamos los mismos datos de train/test de la parte de k-NN
% X_train y X_test ya están en el espacio reducido PCA (k = 154)
% Y_train y Y_test también ya están preparados

% Entrenar el clasificador Bayesiano
Mdl_bayes = fitcnb(X_train, Y_train, 'DistributionNames', 'kernel');

% Predecir sobre el test set
Y_pred_bayes = predict(Mdl_bayes, X_test);

% Matriz de confusión y análisis
figure;
cm_bayes = confusionchart(Y_test, Y_pred_bayes);
title('Matriz de Confusión - Bayes');

% Calcular la matriz numérica
Cmat_bayes = confusionmat(Y_test, Y_pred_bayes);

% Aciertos y errores
aciertos_bayes = sum(diag(Cmat_bayes));
errores_bayes = sum(Cmat_bayes(:)) - aciertos_bayes;
accuracy_bayes = (aciertos_bayes / sum(Cmat_bayes(:))) * 100;

% Mostrar resultados
fprintf('Total de aciertos (Bayes): %d\n', aciertos_bayes);
fprintf('Total de errores (Bayes): %d\n', errores_bayes);
fprintf('Accuracy (Bayes): %.2f%%\n', accuracy_bayes);

% Mostrar confusiones relevantes
fprintf('\nConfusiones más relevantes en Bayes (cuando el error > 10):\n');
for i = 1:size(Cmat_bayes,1)
    for j = 1:size(Cmat_bayes,2)
        if i ~= j && Cmat_bayes(i,j) > 10
            fprintf('El número %d se confunde %d veces con el número %d\n', i-1, Cmat_bayes(i,j), j-1);
        end
    end
end
