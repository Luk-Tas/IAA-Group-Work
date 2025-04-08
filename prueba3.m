clear; clc; close all;
load('Trainnumbers.mat');
X = double(Trainnumbers.image);   % 784 x N
N = size(X, 2);                   % Número de muestras
labels = Trainnumbers.label;

fprintf('=== INICIO DEL PROCESAMIENTO ===\n');

% ==========================
% CENTRAR LOS DATOS
% ==========================
fprintf('Centrando los datos...\n');
X_mean = mean(X, 2);
X_centered = X - X_mean;

% ==========================
% PCA SIN NORMALIZAR
% ==========================
fprintf('Calculando PCA sin normalizar...\n');
C = cov(X_centered');
[EigVec, EigVal] = eig(C);
[eigValsSorted, idx] = sort(diag(EigVal), 'descend');
EigVec = EigVec(:, idx);

% ==========================
% PCA CON NORMALIZACIÓN PREVIA
% ==========================
fprintf('Normalizando y aplicando PCA...\n');
X_norm = normalize(X, 2, 'center', 'mean', 'scale', 'std');
X_norm(isnan(X_norm)) = 0;
Xn_mean = mean(X_norm, 2);
Xn_centered = X_norm - Xn_mean;
Cn = cov(Xn_centered');
[EigVec_n, EigVal_n] = eig(Cn);
[eigValsSorted_n, idx_n] = sort(diag(EigVal_n), 'descend');
EigVec_n = EigVec_n(:, idx_n);

% ==========================
% Proyección con k componentes
% ==========================
k = 154;
fprintf('Proyectando datos con k = %d componentes...\n', k);
W = EigVec(:, 1:k);
projected = W' * X_centered;

Wn = EigVec_n(:, 1:k);
projected_n = Wn' * Xn_centered;

cv = cvpartition(labels, 'HoldOut', 0.1);
trainIdx = training(cv);
testIdx = test(cv);

X_train_sin = projected(:, trainIdx)';
Y_train_sin = labels(trainIdx);
X_test_sin = projected(:, testIdx)';
Y_test_sin = labels(testIdx);

X_train_n = projected_n(:, trainIdx)';
Y_train_n = labels(trainIdx);
X_test_n = projected_n(:, testIdx)';
Y_test_n = labels(testIdx);

% ================================
% K-NN
% ================================
fprintf('Entrenando k-NN...\n');
Mdl_knn_sin = fitcknn(X_train_sin, Y_train_sin, 'NumNeighbors', 3);
Y_pred_knn_sin = predict(Mdl_knn_sin, X_test_sin);
C_knn_sin = confusionmat(Y_test_sin, Y_pred_knn_sin);
accuracy_knn_sin = 100 * sum(diag(C_knn_sin)) / sum(C_knn_sin(:));

fprintf('Entrenando k-NN normalizado...\n');
Mdl_knn_n = fitcknn(X_train_n, Y_train_n, 'NumNeighbors', 3);
Y_pred_knn_n = predict(Mdl_knn_n, X_test_n);
C_knn_n = confusionmat(Y_test_n, Y_pred_knn_n);
accuracy_knn_n = 100 * sum(diag(C_knn_n)) / sum(C_knn_n(:));

% ================================
% Bayes
% ================================
fprintf('Entrenando clasificador Bayes...\n');
Mdl_bayes_sin = fitcnb(X_train_sin, Y_train_sin, 'DistributionNames', 'kernel');
Y_pred_bayes_sin = predict(Mdl_bayes_sin, X_test_sin);
C_bayes_sin = confusionmat(Y_test_sin, Y_pred_bayes_sin);
accuracy_bayes_sin = 100 * sum(diag(C_bayes_sin)) / sum(C_bayes_sin(:));

fprintf('Entrenando Bayes normalizado...\n');
Mdl_bayes_n = fitcnb(X_train_n, Y_train_n, 'DistributionNames', 'kernel');
Y_pred_bayes_n = predict(Mdl_bayes_n, X_test_n);
C_bayes_n = confusionmat(Y_test_n, Y_pred_bayes_n);
accuracy_bayes_n = 100 * sum(diag(C_bayes_n)) / sum(C_bayes_n(:));

% ================================
% K-means
% ================================
fprintf('Aplicando K-means sin normalizar...\n');
[idx_kmeans_sin, C_sin] = kmeans(projected', 10, 'Replicates', 3);
C_kmeans_sin = confusionmat(labels, idx_kmeans_sin);
accuracy_kmeans_sin = 100 * sum(max(C_kmeans_sin)) / N;

fprintf('Aplicando K-means normalizado...\n');
[idx_kmeans_n, C_n] = kmeans(projected_n', 10, 'Replicates', 3);
C_kmeans_n = confusionmat(labels, idx_kmeans_n);
accuracy_kmeans_n = 100 * sum(max(C_kmeans_n)) / N;

% ================================
% TABLA RESUMEN FINAL
% ================================
accuracy_table = table([accuracy_knn_sin; accuracy_knn_n; accuracy_bayes_sin; accuracy_bayes_n; accuracy_kmeans_sin; accuracy_kmeans_n], ...
    'RowNames', {'kNN sin normalizar','kNN normalizado','Bayes sin normalizar','Bayes normalizado','K-means sin normalizar','K-means normalizado'}, ...
    'VariableNames', {'Accuracy'});

fprintf('================= RESUMEN DE ACCURACY =================\n');
disp(accuracy_table);

% ================================
% Gráficos de matrices de confusión
% ================================
figure;
confusionchart(C_knn_sin);
title('k-NN sin normalizar');

figure;
confusionchart(C_knn_n);
title('k-NN normalizado');

figure;
confusionchart(C_bayes_sin);
title('Bayes sin normalizar');

figure;
confusionchart(C_bayes_n);
title('Bayes normalizado');

% ================================
% Mostrar confusiones relevantes
% ================================
fprintf('\nConfusiones relevantes en kNN sin normalizar (>10):\n');
for i = 1:10
    for j = 1:10
        if i ~= j && C_knn_sin(i,j) > 10
            fprintf('Número %d se confunde %d veces con %d\n', i-1, C_knn_sin(i,j), j-1);
        end
    end
end

fprintf('\nConfusiones relevantes en Bayes sin normalizar (>10):\n');
for i = 1:10
    for j = 1:10
        if i ~= j && C_bayes_sin(i,j) > 10
            fprintf('Número %d se confunde %d veces con %d\n', i-1, C_bayes_sin(i,j), j-1);
        end
    end
end

fprintf('=== PROCESO COMPLETADO ===\n');

