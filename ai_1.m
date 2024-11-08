% Определяем входные данные
data = [
    1.5 3.5;
    3.0 5.5;
    2.0 3.0;
    2.0 5.5;
    1.0 6.0;
    4.0 1.5;
    5.0 2.5
];

% Определяем различные метрики расстояния для анализа
metrics = {'euclidean', 'cityblock', 'correlation', 'cosine'};
metric_names = {'Евклидово', 'Манхэттенское', 'Корреляционное', 'Косинусное'};
n_metrics = length(metrics);

% Создаем фигуру для визуализации
figure('Position', [100 100 1500 1200]);

% Массивы для хранения результатов
Z_all = cell(n_metrics, 1);
silhouette_scores_all = cell(n_metrics, 1);
total_distances_all = cell(n_metrics, 1);
optimal_k_all = zeros(n_metrics, 2); % [silhouette, elbow]

% Анализ для каждой метрики
for m = 1:n_metrics
    % Вычисляем дендрограмму
    Z_all{m} = linkage(pdist(data, metrics{m}), 'complete');
    
    % Вычисляем метрики для разного числа кластеров
    max_k = 6;
    silhouette_scores = zeros(max_k-1, 1);
    total_distances = zeros(max_k, 1);
    
    for k = 1:max_k
        clusters = cluster(Z_all{m}, 'maxclust', k);
        
        % Для метода локтя
        total_dist = 0;
        for i = 1:k
            cluster_points = data(clusters == i, :);
            centroid = mean(cluster_points, 1);
            
            % Вычисляем расстояния в зависимости от метрики
            if strcmp(metrics{m}, 'cityblock')
                distances = sum(abs(cluster_points - centroid), 2);
            elseif strcmp(metrics{m}, 'cosine')
                normalized_points = cluster_points ./ sqrt(sum(cluster_points.^2, 2));
                normalized_centroid = centroid ./ sqrt(sum(centroid.^2));
                distances = 1 - normalized_points * normalized_centroid';
            elseif strcmp(metrics{m}, 'correlation')
                centered_points = cluster_points - mean(cluster_points, 2);
                centered_centroid = centroid - mean(centroid);
                distances = 1 - (centered_points * centered_centroid') ./ ...
                           (sqrt(sum(centered_points.^2, 2)) * sqrt(sum(centered_centroid.^2)));
            else % euclidean
                distances = sqrt(sum((cluster_points - centroid).^2, 2));
            end
            total_dist = total_dist + sum(distances);
        end
        total_distances(k) = total_dist;
        
        % Для силуэтного анализа
        if k >= 2
            silhouette_scores(k-1) = mean(silhouette(data, clusters, metrics{m}));
        end
    end
    
    silhouette_scores_all{m} = silhouette_scores;
    total_distances_all{m} = total_distances;
    
    % Находим оптимальное k по силуэту
    [~, opt_k_sil] = max(silhouette_scores);
    optimal_k_all(m, 1) = opt_k_sil + 1;
    
    % Находим оптимальное k по методу локтя
    diffs = diff(total_distances);
    acceleration = diff(diffs);
    [~, elbow_idx] = max(abs(acceleration));
    optimal_k_all(m, 2) = elbow_idx + 1;
    
    % Визуализация результатов для каждой метрики
    % 1. Дендрограмма
    subplot(n_metrics, 4, (m-1)*4 + 1);
    dendrogram(Z_all{m});
    title(sprintf('Дендрограмма (%s)', metric_names{m}));
    xlabel('Номер точки');
    ylabel('Расстояние');
    
    % 2. Метод локтя
    subplot(n_metrics, 4, (m-1)*4 + 2);
    plot(1:max_k, total_distances, 'bo-', 'LineWidth', 2);
    hold on;
    plot(optimal_k_all(m, 2), total_distances(optimal_k_all(m, 2)), 'ro', 'MarkerSize', 10);
    xlabel('Количество кластеров (k)');
    ylabel('Суммарное расстояние');
    title(sprintf('Метод локтя (%s)', metric_names{m}));
    grid on;
    
    % 3. Силуэтный анализ
    subplot(n_metrics, 4, (m-1)*4 + 3);
    plot(2:max_k, silhouette_scores, 'ro-', 'LineWidth', 2);
    xlabel('Количество кластеров (k)');
    ylabel('Средний силуэтный коэффициент');
    title(sprintf('Силуэтный анализ (%s)', metric_names{m}));
    grid on;
    
    % 4. Результат кластеризации (используем k из метода локтя)
    subplot(n_metrics, 4, (m-1)*4 + 4);
    clusters = cluster(Z_all{m}, 'maxclust', optimal_k_all(m, 2));
    scatter(data(:,1), data(:,2), 100, clusters, 'filled');
    hold on;
    % Добавляем номера точек
    for i = 1:size(data,1)
        text(data(i,1), data(i,2), num2str(i), ...
            'HorizontalAlignment', 'right', ...
            'VerticalAlignment', 'bottom');
    end
    % Добавляем центроиды
    for i = 1:optimal_k_all(m, 2)
        cluster_points = data(clusters == i, :);
        centroid = mean(cluster_points, 1);
        plot(centroid(1), centroid(2), 'kx', 'MarkerSize', 15, 'LineWidth', 3);
    end
    xlabel('X');
    ylabel('Y');
    title(sprintf('Кластеры (%s, k=%d)', metric_names{m}, optimal_k_all(m, 2)));
    grid on;
    hold off;
end

% Вывод результатов анализа
fprintf('\nСравнительный анализ метрик расстояния:\n');
for m = 1:n_metrics
    fprintf('\n%s расстояние:\n', metric_names{m});
    fprintf('Оптимальное число кластеров:\n');
    fprintf('  - По силуэтному методу: %d (коэффициент: %.3f)\n', ...
        optimal_k_all(m, 1), max(silhouette_scores_all{m}));
    fprintf('  - По методу локтя: %d\n', optimal_k_all(m, 2));
    
    % Анализ кластеров для k из метода локтя
    clusters = cluster(Z_all{m}, 'maxclust', optimal_k_all(m, 2));
    fprintf('\nСостав кластеров (k=%d):\n', optimal_k_all(m, 2));
    
    for i = 1:optimal_k_all(m, 2)
        cluster_points = data(clusters == i, :);
        centroid = mean(cluster_points, 1);
        
        % Вычисляем расстояния до центроида
        if strcmp(metrics{m}, 'cityblock')
            distances = sum(abs(cluster_points - centroid), 2);
        elseif strcmp(metrics{m}, 'cosine')
            normalized_points = cluster_points ./ sqrt(sum(cluster_points.^2, 2));
            normalized_centroid = centroid ./ sqrt(sum(centroid.^2));
            distances = 1 - normalized_points * normalized_centroid';
        elseif strcmp(metrics{m}, 'correlation')
            centered_points = cluster_points - mean(cluster_points, 2);
            centered_centroid = centroid - mean(centroid);
            distances = 1 - (centered_points * centered_centroid') ./ ...
                       (sqrt(sum(centered_points.^2, 2)) * sqrt(sum(centered_centroid.^2)));
        else % euclidean
            distances = sqrt(sum((cluster_points - centroid).^2, 2));
        end
        
        fprintf('  Кластер %d:\n', i);
        fprintf('    Точки: %s\n', num2str(find(clusters == i)));
        fprintf('    Центроид: (%.2f, %.2f)\n', centroid(1), centroid(2));
        fprintf('    Среднее расстояние до центроида: %.2f\n', mean(distances));
        fprintf('    Максимальное расстояние до центроида: %.2f\n', max(distances));
    end
end