data = readtable('heart_failure_clinical_records_dataset.csv'); 
% Bağımsız değişkenler ve hedef değişkenler
X = data(:, {'age', 'anaemia', 'creatinine_phosphokinase', 'diabetes', ...
    'ejection_fraction', 'high_blood_pressure', 'platelets', ...
    'serum_creatinine', 'serum_sodium', 'sex', 'smoking', 'time'}); % Bağımsız değişkenler
y = data(:, 'DEATH_EVENT'); % Bağımlı değişken
% K-fold çapraz doğrulama için ayarlar
k = 10; % Kat sayısı
% Doğruluklar için boş bir dizi oluşturma
accuracies_smoking_yes = zeros(k, 1);
accuracies_smoking_no = zeros(k, 1);
f1_scores_smoking_yes = zeros(k, 1);
f1_scores_smoking_no = zeros(k, 1);
% K-fold çapraz doğrulama
cv = cvpartition(size(data, 1), 'KFold', k);
for i = 1:k
    train_idx = training(cv, i);
    test_idx = test(cv, i);
    
    X_train = X(train_idx, :);
    y_train = y(train_idx, :);
    
    X_test = X(test_idx, :);
    y_test = y(test_idx, :);
    
    % Sigara içenler için model oluşturma ve test etme
    mdl_smoking_yes = fitcsvm(X_train{:,:}, y_train.DEATH_EVENT);
    y_pred_smoking_yes = predict(mdl_smoking_yes, X_test{:,:});
    
    % Sigara içmeyenler için model oluşturma ve test etme
    mdl_smoking_no = fitcsvm(X_train{:,:}, y_train.DEATH_EVENT);
    y_pred_smoking_no = predict(mdl_smoking_no, X_test{:,:});
    
    % Doğrulukları kaydetme
    accuracies_smoking_yes(i) = sum(y_pred_smoking_yes == y_test.DEATH_EVENT) / numel(y_test.DEATH_EVENT);
    accuracies_smoking_no(i) = sum(y_pred_smoking_no == y_test.DEATH_EVENT) / numel(y_test.DEATH_EVENT);
    
    % F1 puanlarını hesaplama
    conf_matrix_smoking_yes = confusionmat(y_test.DEATH_EVENT, y_pred_smoking_yes);
    precision_smoking_yes = conf_matrix_smoking_yes(2, 2) / sum(conf_matrix_smoking_yes(:, 2));
    recall_smoking_yes = conf_matrix_smoking_yes(2, 2) / sum(conf_matrix_smoking_yes(2, :));
    f1_scores_smoking_yes(i) = 2 * (precision_smoking_yes * recall_smoking_yes) / (precision_smoking_yes + recall_smoking_yes);
    
    conf_matrix_smoking_no = confusionmat(y_test.DEATH_EVENT, y_pred_smoking_no);
    precision_smoking_no = conf_matrix_smoking_no(2, 2) / sum(conf_matrix_smoking_no(:, 2));
    recall_smoking_no = conf_matrix_smoking_no(2, 2) / sum(conf_matrix_smoking_no(2, :));
    f1_scores_smoking_no(i) = 2 * (precision_smoking_no * recall_smoking_no) / (precision_smoking_no + recall_smoking_no);
end
% Ortalama doğrulukları ve F1 puanlarını hesaplama
mean_accuracy_smoking_yes = mean(accuracies_smoking_yes);
mean_accuracy_smoking_no = mean(accuracies_smoking_no);
mean_f1_score_smoking_yes = mean(f1_scores_smoking_yes);
mean_f1_score_smoking_no = mean(f1_scores_smoking_no);
% Sonuçları ekrana yazdırma (biraz uzun sürüyor çıktı vermesi)
disp(['Sigara içenler için 10-fold cross-validation doğruluğu: ', num2str(mean_accuracy_smoking_yes)]);
disp(['Sigara içmeyenler için 10-fold cross-validation doğruluğu: ', num2str(mean_accuracy_smoking_no)]);
disp(['Sigara içenler için ortalama F1 puanı: ', num2str(mean_f1_score_smoking_yes)]);
disp(['Sigara içmeyenler için ortalama F1 puanı: ', num2str(mean_f1_score_smoking_no)]);