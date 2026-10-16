%% ========================================================================
%  小样本代谢组学四分类完整分析（最终版）
%  ------------------------------------------------------------------------
%  类别映射：1=HBLD, 2=DZLD, 3=CYLD, 4=BYLD
%
%  数据：positive_ions.csv（正离子）、negative_ions.csv（负离子）
%     第1列：样本名；第2列：类别号；第3列起：特征值
%     两文件样本顺序必须一致
%
%  输出：
%    图（11 张，PNG + FIG）：
%      fig1  性能箱线图（5 子图：Acc/Sens/Spec/AUC/nFeat）
%      fig2  特征稳定性条形图
%      fig3  4 算法宏平均 ROC + 95% CI
%      fig4  SVM 三核对比
%      fig5  每算法 4 类 ROC 点估计（Okabe-Ito）
%      fig6  每算法 4 类 ROC + CI 阴影（Okabe-Ito）
%      fig7  每算法 4 类 ROC + CI 虚线（Okabe-Ito）
%      fig8  混淆矩阵
%      fig9  AUC 分布直方图
%      fig10 特征数 vs 选择频率阈值
%      fig11 各分类器参数/样本比
%    Excel（13 个）：
%      performance_summary.xlsx / performance_summary_wide.xlsx
%      bootstrap_AUC_CI.xlsx / AUC_with_CI.xlsx
%      all_metrics_summary.xlsx
%      SVM_kernel_comparison.xlsx
%      stable_features.xlsx
%      per_class_metrics.xlsx
%      per_class_all_metrics_long.xlsx / per_class_all_metrics_wide.xlsx
%      repeat_level_metrics.xlsx
%      confusion_matrices.xlsx
%      feature_selection_history.xlsx
%    TXT：
%      best_SVM_kernel.txt
%
%  环境要求：
%    MATLAB R2020a 或更新
%    Statistics and Machine Learning Toolbox（必需）
%    Deep Learning Toolbox（BPNN 使用；缺失时自动移除）
% ========================================================================

clear; clc; close all;

%% ======================== 参数配置 ========================
cfg.dataPos   = 'positive_ions.csv';
cfg.dataNeg   = 'negative_ions.csv';
cfg.outDir    = 'ML_results';
cfg.nRep      = 100;                      % ★ 正式运行改为 100
cfg.kOuter    = 5;
cfg.kInner    = 4;
cfg.nBoot     = 1000;
cfg.nBootROC  = 500;
cfg.alphaEN   = [0.1 0.3 0.5 0.7 0.9 1.0];
cfg.seed      = 2024;
cfg.maxFeat   = 8;

cfg.classNames  = {'HBLD', 'DZLD', 'CYLD', 'BYLD'};
cfg.classifiers = {'SVM_linear', 'SVM_rbf', 'SVM_poly', 'KNN', 'RF', 'BPNN'};
cfg.dispNames   = cellfun(@(s) strrep(s, '_', '-'), cfg.classifiers, ...
                          'UniformOutput', false);

nRep = cfg.nRep; kOuter = cfg.kOuter; kInner = cfg.kInner;
nBoot = cfg.nBoot; nBootROC = cfg.nBootROC;

if ~exist(cfg.outDir, 'dir'), mkdir(cfg.outDir); end
rng(cfg.seed, 'twister');

fprintf('========================================================\n');
fprintf('   小样本代谢组学四分类分析（最终版）\n');
fprintf('   类别: %s\n', strjoin(cfg.classNames, ' / '));
fprintf('   输出目录: %s\n', cfg.outDir);
fprintf('========================================================\n\n');

%% ==================== 步骤 0：工具箱检查 ====================
fprintf('[步骤 0/15] 检查 MATLAB 工具箱...\n');

hasStatsToolbox = license('test', 'Statistics_Toolbox') && ~isempty(ver('stats'));
if ~hasStatsToolbox
    error('缺少 Statistics and Machine Learning Toolbox，无法继续。');
end
fprintf('  ✓ Statistics and Machine Learning Toolbox（已安装）\n');

hasDLToolbox = license('test', 'Neural_Network_Toolbox') && ~isempty(ver('nnet'));
if hasDLToolbox
    fprintf('  ✓ Deep Learning Toolbox（已安装）\n');
    fprintf('  → BPNN 将参与分析（作为探索性对比）\n');
else
    fprintf('  ⚠ 未检测到 Deep Learning Toolbox，BPNN 将移除\n');
    keepMask = ~strcmp(cfg.classifiers, 'BPNN');
    cfg.classifiers = cfg.classifiers(keepMask);
    cfg.dispNames   = cfg.dispNames(keepMask);
end
fprintf('  参与分析的分类器: %s\n\n', strjoin(cfg.dispNames, ', '));

%% ==================== 步骤 1：数据加载 ====================
fprintf('[步骤 1/15] 加载数据...\n');
[X_pos, X_neg, Y, ~, ~, featName_pos, featName_neg] = ...
    loadData(cfg.dataPos, cfg.dataNeg);

nSamples    = numel(Y);
nClasses    = numel(unique(Y));
classCounts = histcounts(Y, 1:(nClasses+1));

fprintf('  样本总数: %d | 类别数: %d\n', nSamples, nClasses);
for c = 1:nClasses
    fprintf('    类别 %d (%s): %d 样本\n', c, cfg.classNames{c}, classCounts(c));
end
fprintf('  正离子特征数: %d | 负离子特征数: %d\n', ...
    size(X_pos,2), size(X_neg,2));
if nSamples < 10, error('样本数过少 (%d)', nSamples); end

%% ==================== 步骤 2：特征预处理 ====================
fprintf('\n[步骤 2/15] 特征预处理...\n');
X_all         = [X_pos, X_neg];
featureSource = [ones(1, size(X_pos,2)), 2*ones(1, size(X_neg,2))];
featureNames  = [featName_pos(:)', featName_neg(:)'];

varVals  = var(X_all, 0, 1);
keepMask = varVals > 1e-10 & ~isnan(varVals);
X_all         = X_all(:, keepMask);
featureSource = featureSource(keepMask);
featureNames  = featureNames(keepMask);

fprintf('  去零方差后保留: %d (正离子 %d, 负离子 %d)\n', ...
    size(X_all,2), sum(featureSource==1), sum(featureSource==2));
fprintf('  特征/样本比: %.1f : 1\n', size(X_all,2)/nSamples);

%% ==================== 步骤 3：嵌套交叉验证 ====================
fprintf('\n[步骤 3/15] 嵌套交叉验证 (%d次重复 × %d折)...\n', nRep, kOuter);

R = struct();
for ci = 1:numel(cfg.classifiers)
    cname = cfg.classifiers{ci};
    R.(cname).accuracy    = nan(nRep, 1);
    R.(cname).sensitivity = nan(nRep, 1);
    R.(cname).specificity = nan(nRep, 1);
    R.(cname).precision   = nan(nRep, 1);
    R.(cname).f1          = nan(nRep, 1);
    R.(cname).auc         = nan(nRep, 1);
    R.(cname).foldAUC     = cell(nRep, 1);
    R.(cname).allScores   = cell(nRep, 1);
    R.(cname).allLabels   = cell(nRep, 1);
    R.(cname).allPred     = cell(nRep, 1);
    R.(cname).nFeatures   = nan(nRep, 1);
    R.(cname).perClass    = cell(nRep, 1);
end
featureFreq = zeros(1, size(X_all, 2));
totalFolds  = nRep * kOuter;

% ★ 新增：记录每次重复每次折选出的特征
featureSelectionHistory = cell(nRep, kOuter);
featureSelectionRep     = cell(nRep, 1);

tStart = tic;
for rep = 1:nRep
    rng(cfg.seed + rep * 137, 'twister');
    cvOuter = cvpartition(Y, 'KFold', kOuter, 'Stratify', true);
    
    repSelectedIdx = [];   % ★ 本重复所有折的选中特征
    
    tmp = struct();
    for ci = 1:numel(cfg.classifiers)
        cname = cfg.classifiers{ci};
        tmp.(cname).acc = nan(kOuter,1); tmp.(cname).sens = nan(kOuter,1);
        tmp.(cname).spec = nan(kOuter,1); tmp.(cname).prec = nan(kOuter,1);
        tmp.(cname).f1 = nan(kOuter,1); tmp.(cname).auc = nan(kOuter,1);
        tmp.(cname).scores = []; tmp.(cname).labels = []; tmp.(cname).preds = [];
        tmp.(cname).nFeat = nan(kOuter,1);
    end
    
    for fold = 1:kOuter
        trIdx = training(cvOuter, fold);
        teIdx = test(cvOuter, fold);
        Xtr = X_all(trIdx, :); Ytr = Y(trIdx);
        Xte = X_all(teIdx, :); Yte = Y(teIdx);
        
        mu = mean(Xtr, 1); sd = std(Xtr, 0, 1); sd(sd < 1e-10) = 1;
        Xtr = (Xtr - mu) ./ sd; Xte = (Xte - mu) ./ sd;
        
        [selIdx, ~] = featureSelectLASSO(Xtr, Ytr, cfg.alphaEN, kInner, ...
            nClasses, cfg.maxFeat);
        featureFreq(selIdx) = featureFreq(selIdx) + 1;
        XtrS = Xtr(:, selIdx); XteS = Xte(:, selIdx);
        
        % ★ 记录本折选出的特征
        featureSelectionHistory{rep, fold} = selIdx;
        repSelectedIdx = [repSelectedIdx; selIdx(:)]; %#ok<AGROW>
        
        for ci = 1:numel(cfg.classifiers)
            cname = cfg.classifiers{ci};
            try
                [pred, score] = trainOneClassifier(cname, XtrS, Ytr, XteS, ...
                    nClasses, kInner);
                if any(isnan(pred)) || any(isnan(score(:))), continue; end
                
                [acc, sens, spec, prec, f1] = computeMetrics(Yte, pred, nClasses);
                tmp.(cname).acc(fold)  = acc;
                tmp.(cname).sens(fold) = sens;
                tmp.(cname).spec(fold) = spec;
                tmp.(cname).prec(fold) = prec;
                tmp.(cname).f1(fold)   = f1;
                tmp.(cname).scores     = [tmp.(cname).scores; score];
                tmp.(cname).labels     = [tmp.(cname).labels; Yte];
                tmp.(cname).preds      = [tmp.(cname).preds;  pred];
                tmp.(cname).nFeat(fold) = numel(selIdx);
                tmp.(cname).auc(fold)   = multiClassAUC(Yte, score, nClasses);
            catch ME
                fprintf('    [警告] %s fold %d: %s\n', cname, fold, ME.message);
            end
        end
    end
    
    % ★ 本重复的并集特征（去重）
    featureSelectionRep{rep} = unique(repSelectedIdx);
    
    for ci = 1:numel(cfg.classifiers)
        cname = cfg.classifiers{ci};
        R.(cname).accuracy(rep)    = mean(tmp.(cname).acc,  'omitnan');
        R.(cname).sensitivity(rep) = mean(tmp.(cname).sens, 'omitnan');
        R.(cname).specificity(rep) = mean(tmp.(cname).spec, 'omitnan');
        R.(cname).precision(rep)   = mean(tmp.(cname).prec, 'omitnan');
        R.(cname).f1(rep)          = mean(tmp.(cname).f1,   'omitnan');
        R.(cname).auc(rep)         = mean(tmp.(cname).auc,  'omitnan');
        R.(cname).foldAUC{rep}     = tmp.(cname).auc;
        R.(cname).allScores{rep}   = tmp.(cname).scores;
        R.(cname).allLabels{rep}   = tmp.(cname).labels;
        R.(cname).allPred{rep}     = tmp.(cname).preds;
        R.(cname).nFeatures(rep)   = mean(tmp.(cname).nFeat, 'omitnan');
        
        % 保存 per-class 指标
        allYr = tmp.(cname).labels;
        allPr = tmp.(cname).preds;
        allSr = tmp.(cname).scores;
        
        pc = struct('acc', nan(nClasses,1), 'sens', nan(nClasses,1), ...
                    'spec', nan(nClasses,1), 'prec', nan(nClasses,1), ...
                    'f1', nan(nClasses,1), 'auc', nan(nClasses,1));
        
        if ~isempty(allYr)
            Cr = confusionmat(allYr, allPr, 'Order', 1:nClasses);
            for k = 1:nClasses
                TP = Cr(k,k); FN = sum(Cr(k,:))-TP;
                FP = sum(Cr(:,k))-TP; TN = sum(Cr(:))-TP-FN-FP;
                if (TP+TN+FP+FN)>0, pc.acc(k)  = (TP+TN)/(TP+TN+FP+FN); end
                if (TP+FN)>0,       pc.sens(k) = TP/(TP+FN); end
                if (TN+FP)>0,       pc.spec(k) = TN/(TN+FP); end
                if (TP+FP)>0,       pc.prec(k) = TP/(TP+FP); end
                if (pc.prec(k)+pc.sens(k))>0
                    pc.f1(k) = 2*pc.prec(k)*pc.sens(k)/(pc.prec(k)+pc.sens(k));
                end
                if size(allSr,2) >= k && ~any(isnan(allSr(:,k)))
                    yBin = double(allYr == k);
                    s = allSr(:,k);
                    if numel(unique(yBin)) >= 2 && std(s) > 1e-10
                        try
                            [~,~,~,a] = perfcurve(yBin, s, 1);
                            if ~isnan(a), pc.auc(k) = a; end
                        catch, end
                    end
                end
            end
        end
        R.(cname).perClass{rep} = pc;
    end
    
    if mod(rep, 10) == 0
        fprintf('  重复 %3d/%d | SVM-RBF-Acc=%.3f | 已用 %.1fs\n', ...
            rep, nRep, R.SVM_rbf.accuracy(rep), toc(tStart));
    end
end
fprintf('\n嵌套交叉验证完成，总用时 %.1f 秒\n', toc(tStart));

%% ==================== 步骤 4：选择最佳 SVM 核 ====================
fprintf('\n[步骤 4/15] 选择最佳 SVM 核函数...\n');
[bestSVMField, bestSVMName, svmComp] = ...
    selectBestSVMKernel(R, cfg.classifiers, cfg.dispNames);

fprintf('\n  +--------------------------------------------------+\n');
fprintf('  |           SVM 三种核函数对比结果                  |\n');
fprintf('  +--------------------------------------------------+\n');
for i = 1:height(svmComp)
    mark = '  ';
    if strcmp(svmComp.SVM_Kernel{i}, bestSVMName), mark = '* '; end
    fprintf('  | %s%-15s  AUC=%.4f   Acc=%.4f           |\n', ...
        mark, svmComp.SVM_Kernel{i}, svmComp.AUC_mean(i), svmComp.Acc_mean(i));
end
fprintf('  +--------------------------------------------------+\n');
fprintf('\n  *** 最佳 SVM 核 = 【 %s 】 ***\n\n', bestSVMName);

writetable(svmComp, fullfile(cfg.outDir, 'SVM_kernel_comparison.xlsx'), ...
    'FileType', 'spreadsheet', 'Sheet', 'SVM_kernels');

plotClf   = {bestSVMField, 'KNN', 'RF'};
plotNames = {bestSVMName,   'KNN', 'RF'};
if any(strcmp(cfg.classifiers, 'BPNN'))
    plotClf{end+1}   = 'BPNN';
    plotNames{end+1} = 'BPNN (Exploratory)';
end

fid = fopen(fullfile(cfg.outDir, 'best_SVM_kernel.txt'), 'w');
fprintf(fid, 'Best SVM kernel: %s\n', bestSVMName);
fprintf(fid, 'Selection rule: highest mean AUC; tie broken by mean Accuracy\n');
fclose(fid);

%% ==================== 步骤 5：总体性能汇总 ====================
fprintf('\n[步骤 5/15] 结果汇总（6 分类器 × 6 指标 × 95%% CI）...\n');
reportResults(R, cfg.classifiers, cfg.dispNames, nRep, cfg.outDir, nBoot);

%% ==================== 步骤 5.5：保存重复级原始指标 ====================
fprintf('\n[步骤 5.5/15] 保存 100 次重复的原始指标...\n');
saveRepeatLevelMetrics(R, cfg.classifiers, cfg.dispNames, nRep, cfg.outDir);

%% ==================== 步骤 5.6：保存特征选择历史 ====================
fprintf('\n[步骤 5.6/15] 保存每次重复选出的特征...\n');
saveFeatureSelectionHistory(featureSelectionHistory, featureSelectionRep, ...
    featureNames, featureSource, nRep, kOuter, cfg.outDir);

%% ==================== 步骤 6：稳定性选择 ====================
fprintf('\n[步骤 6/15] 稳定性选择（简约特征面板）...\n');
selectFreq = featureFreq / totalFolds;
[sortedFreq, sortedIdx] = sort(selectFreq, 'descend');

topK = min(10, sum(sortedFreq >= 0.5));
if topK < 5, topK = min(8, sum(sortedFreq > 0.1)); end
if topK < 1, topK = min(8, numel(sortedFreq)); end

fprintf('  Top-%d 稳定特征:\n', topK);
for k = 1:topK
    srcStr = '正离子';
    if featureSource(sortedIdx(k)) == 2, srcStr = '负离子'; end
    fprintf('    #%2d  %-15s  %-4s  %.1f%%\n', ...
        k, featureNames{sortedIdx(k)}, srcStr, sortedFreq(k)*100);
end

stableTable = table((1:topK)', featureNames(sortedIdx(1:topK))', ...
    sortedIdx(1:topK)', featureSource(sortedIdx(1:topK))', ...
    sortedFreq(1:topK)', ...
    'VariableNames', {'Rank','FeatureName','FeatureIdx','IonMode','SelectionFreq'});
writetable(stableTable, fullfile(cfg.outDir, 'stable_features.xlsx'), ...
    'FileType', 'spreadsheet', 'Sheet', 'Stable_features');

nPos = sum(featureSource(sortedIdx(1:topK)) == 1);
nNeg = sum(featureSource(sortedIdx(1:topK)) == 2);
fprintf('\n  ★ 正离子特征: %d | 负离子特征: %d | 合计: %d\n', nPos, nNeg, topK);

%% ==================== 步骤 7：总体绘图（fig1-4）====================
fprintf('\n[步骤 7/15] 生成总体图表 fig1-fig4...\n');
plotAllResults(R, plotClf, plotNames, cfg.classifiers, cfg.dispNames, ...
    bestSVMName, selectFreq, featureNames, nRep, nClasses, nBootROC, cfg.outDir);

%% ==================== 步骤 8：每类性能点估计 ====================
fprintf('\n[步骤 8/15] 每类性能点估计...\n');
perClass = computePerClassResults(R, plotClf, nRep, nClasses);
writePerClassResults(perClass, plotClf, plotNames, cfg.classNames, cfg.outDir);

%% ==================== 步骤 8.5：每类 × 每分类器完整指标 ====================
fprintf('\n[步骤 8.5/15] 每类 × 每分类器 6 指标（mean±SD + CI）...\n');
reportPerClassMetrics(R, plotClf, plotNames, cfg.classNames, ...
    nRep, nClasses, nBoot, cfg.outDir);

%% ==================== 步骤 9：每算法 ROC（fig5）====================
fprintf('\n[步骤 9/15] 生成 fig5（Okabe-Ito 配色）...\n');
plotPerAlgoROC(R, plotClf, plotNames, cfg.classNames, nRep, nClasses, cfg.outDir);

%% ==================== 步骤 10：每算法 ROC + 阴影 CI（fig6）====================
fprintf('\n[步骤 10/15] 生成 fig6（Okabe-Ito + 阴影 CI）...\n');
plotPerAlgoROCWithCI(R, plotClf, plotNames, cfg.classNames, ...
    nRep, nClasses, nBootROC, cfg.outDir);

%% ==================== 步骤 11：每算法 ROC + 虚线 CI（fig7）====================
fprintf('\n[步骤 11/15] 生成 fig7（Okabe-Ito + 虚线 CI）...\n');
plotPerAlgoROCWithCILines(R, plotClf, plotNames, cfg.classNames, ...
    nRep, nClasses, nBootROC, cfg.outDir);

%% ==================== 步骤 12：混淆矩阵（fig8）====================
fprintf('\n[步骤 12/15] 生成 fig8（混淆矩阵）...\n');
plotConfusionMatrices(R, plotClf, plotNames, cfg.classNames, nRep, cfg.outDir);

%% ==================== 步骤 12.5：保存混淆矩阵数值 ====================
fprintf('\n[步骤 12.5/15] 保存混淆矩阵数值表...\n');
saveConfusionTables(R, plotClf, plotNames, cfg.classNames, nRep, cfg.outDir);

%% ==================== 步骤 13：AUC 分布（fig9）====================
fprintf('\n[步骤 13/15] 生成 fig9（AUC 分布直方图）...\n');
plotAUCDistribution(R, plotClf, plotNames, nRep, nBoot, cfg.outDir);

%% ==================== 步骤 14：特征阈值曲线（fig10）====================
fprintf('\n[步骤 14/15] 生成 fig10（特征数 vs 阈值）...\n');
plotFeatureSelectionCurve(selectFreq, cfg.outDir);

%% ==================== 步骤 15：参数量/样本比（fig11）====================
fprintf('\n[步骤 15/15] 生成 fig11（参数/样本比）...\n');
plotBPNNParamRatio(plotClf, plotNames, cfg.maxFeat, nClasses, kOuter, cfg.outDir);

fprintf('\n========================================================\n');
fprintf('  全部分析完成！结果目录: %s\n', cfg.outDir);
fprintf('  共生成 11 张图 + 13 个 Excel + 1 个 TXT\n');
fprintf('========================================================\n');


%% ========================================================================
%                          局部函数定义
%% ========================================================================

function [X_pos, X_neg, Y, id_pos, id_neg, featName_pos, featName_neg] = ...
        loadData(fPos, fNeg)
    fprintf('  读取正离子文件: %s\n', fPos);
    T_pos = readCSVSafe(fPos);
    fprintf('  读取负离子文件: %s\n', fNeg);
    T_neg = readCSVSafe(fNeg);
    if width(T_pos) < 3 || width(T_neg) < 3
        error('文件列数不足（至少：样本名、类别、1个特征）');
    end
    if height(T_pos) ~= height(T_neg)
        error('两文件样本数不一致: %d vs %d', height(T_pos), height(T_neg));
    end
    id_pos = T_pos{:, 1}; id_neg = T_neg{:, 1};
    Y_pos  = T_pos{:, 2}; Y_neg  = T_neg{:, 2};
    if ~isnumeric(Y_pos)
        [~, ~, Y_pos] = unique(Y_pos);
        [~, ~, Y_neg] = unique(Y_neg);
    end
    Y_pos = double(Y_pos(:)); Y_neg = double(Y_neg(:));
    if ~isequal(Y_pos, Y_neg), error('两文件类别标签不一致！'); end
    Y = Y_pos;
    X_pos = T_pos{:, 3:end}; X_neg = T_neg{:, 3:end};
    if ~isnumeric(X_pos), X_pos = convertFeatureMatrix(X_pos); end
    if ~isnumeric(X_neg), X_neg = convertFeatureMatrix(X_neg); end
    featName_pos = T_pos.Properties.VariableNames(3:end);
    featName_neg = T_neg.Properties.VariableNames(3:end);
    validMask = ~all(isnan(X_pos), 2) & ~all(isnan(X_neg), 2) & ~isnan(Y);
    X_pos = X_pos(validMask, :); X_neg = X_neg(validMask, :);
    Y = Y(validMask); id_pos = id_pos(validMask); id_neg = id_neg(validMask);
    X_pos = fillMissingByMedian(X_pos);
    X_neg = fillMissingByMedian(X_neg);
    fprintf('  数据加载完成: %d样本, %d+%d特征\n', ...
        numel(Y), size(X_pos,2), size(X_neg,2));
end

function T = readCSVSafe(fname)
    try
        T = readtable(fname, 'VariableNamingRule', 'preserve');
        return;
    catch, end
    try
        T = readtable(fname);
        return;
    catch ME
        error('无法读取 %s: %s', fname, ME.message);
    end
end

function Xn = convertFeatureMatrix(X)
    if istable(X)
        n = height(X); p = width(X); Xn = nan(n, p);
        for j = 1:p
            col = X{:, j};
            if isnumeric(col), Xn(:, j) = col;
            elseif iscell(col) || isstring(col), Xn(:, j) = str2double(string(col));
            elseif iscategorical(col), Xn(:, j) = double(col); end
        end
    elseif iscell(X)
        Xn = str2double(string(X));
    else
        Xn = double(X);
    end
end

function Xf = fillMissingByMedian(X)
    Xf = X;
    for j = 1:size(Xf, 2)
        col = Xf(:, j); nanMask = isnan(col);
        if any(nanMask)
            medVal = median(col(~nanMask), 'omitnan');
            if isnan(medVal), medVal = 0; end
            col(nanMask) = medVal; Xf(:, j) = col;
        end
    end
end

function [selIdx, bestAlpha] = featureSelectLASSO(X, Y, alphaRange, kFolds, ...
        nClasses, maxFeat)
    if nargin < 6, maxFeat = 8; end
    p = size(X, 2); selUnion = false(1, p); bestAlpha = 1.0;
    minClassCount = min(histcounts(Y, 1:(nClasses+1)));
    kInnerActual = min(kFolds, minClassCount);
    if kInnerActual < 2, kInnerActual = 2; end
    alphaMSE = zeros(numel(alphaRange), 1);
    for ai = 1:numel(alphaRange)
        a = alphaRange(ai); mseC = zeros(nClasses, 1);
        for c = 1:nClasses
            yBin = double(Y == c);
            try
                [~, fitInfo] = lasso(X, yBin, 'Alpha', a, 'CV', kInnerActual, ...
                    'Standardize', false, 'Intercept', true, 'NumLambda', 30);
                mseC(c) = min(fitInfo.MSE);
            catch
                mseC(c) = inf;
            end
        end
        alphaMSE(ai) = mean(mseC);
    end
    [~, bestAi] = min(alphaMSE);
    bestAlpha = alphaRange(bestAi);
    for c = 1:nClasses
        yBin = double(Y == c);
        try
            [B, fitInfo] = lasso(X, yBin, 'Alpha', bestAlpha, 'CV', kInnerActual, ...
                'Standardize', false, 'Intercept', true, 'NumLambda', 30);
            [~, minIdx] = min(fitInfo.MSE);
            selUnion = selUnion | (abs(B(:, minIdx)) > 1e-8)';
        catch
        end
    end
    selIdx = find(selUnion);
    mf = min(maxFeat, p);
    if numel(selIdx) > mf
        v = var(X(:, selIdx), 0, 1);
        [~, ord] = sort(v, 'descend');
        selIdx = selIdx(ord(1:mf));
    end
    if isempty(selIdx)
        [~, order] = sort(var(X, 0, 1), 'descend');
        selIdx = order(1:min(5, p));
    end
end

function [pred, score] = trainOneClassifier(name, Xtr, Ytr, Xte, nClasses, kFolds)
    Xtr = double(Xtr); Xte = double(Xte); Ytr = double(Ytr(:));
    switch upper(name)
        case 'SVM_LINEAR', [pred, score] = trainSVM(Xtr, Ytr, Xte, nClasses, 'linear');
        case 'SVM_RBF',    [pred, score] = trainSVM(Xtr, Ytr, Xte, nClasses, 'rbf');
        case 'SVM_POLY',   [pred, score] = trainSVM(Xtr, Ytr, Xte, nClasses, 'polynomial');
        case 'KNN',        [pred, score] = trainKNN(Xtr, Ytr, Xte, nClasses, kFolds);
        case 'RF',         [pred, score] = trainRF(Xtr, Ytr, Xte, nClasses);
        case 'BPNN',       [pred, score] = trainBPNN(Xtr, Ytr, Xte, nClasses);
        otherwise,         error('未知分类器: %s', name);
    end
    if isempty(pred) || any(isnan(pred)) || isempty(score) || any(isnan(score(:)))
        return;
    end
    score = padScore(score, nClasses);
end

function [pred, score] = trainSVM(Xtr, Ytr, Xte, nClasses, kernelName)
    minClassCount = min(histcounts(Ytr, 1:(nClasses+1)));
    kf = min(3, minClassCount); if kf < 2, kf = 2; end
    cList = [0.1, 1, 10];
    if strcmpi(kernelName, 'rbf') || strcmpi(kernelName, 'polynomial')
        gammaList = [0.01, 0.1, 1];
    else
        gammaList = NaN;
    end
    bestAcc = -inf; bestC = 1; bestG = NaN;
    for c = cList
        if isnan(gammaList)
            try
                t = templateSVM('KernelFunction', kernelName, ...
                    'Standardize', false, 'BoxConstraint', c);
                mdl = fitcecoc(Xtr, Ytr, 'Learners', t, 'Coding', 'onevsall');
                cvM = crossval(mdl, 'KFold', kf);
                a = 1 - kfoldLoss(cvM);
                if a > bestAcc, bestAcc = a; bestC = c; end
            catch, end
        else
            for g = gammaList
                try
                    t = templateSVM('KernelFunction', kernelName, ...
                        'Standardize', false, 'BoxConstraint', c, ...
                        'KernelScale', 1/sqrt(2*g));
                    mdl = fitcecoc(Xtr, Ytr, 'Learners', t, 'Coding', 'onevsall');
                    cvM = crossval(mdl, 'KFold', kf);
                    a = 1 - kfoldLoss(cvM);
                    if a > bestAcc, bestAcc = a; bestC = c; bestG = g; end
                catch, end
            end
        end
    end
    if isnan(bestG)
        tBest = templateSVM('KernelFunction', kernelName, ...
            'Standardize', false, 'BoxConstraint', bestC);
    else
        tBest = templateSVM('KernelFunction', kernelName, ...
            'Standardize', false, 'BoxConstraint', bestC, ...
            'KernelScale', 1/sqrt(2*bestG));
    end
    mdl = fitcecoc(Xtr, Ytr, 'Learners', tBest, 'Coding', 'onevsall');
    [pred, score] = predict(mdl, Xte);
    pred = double(pred);
end

function [pred, score] = trainKNN(Xtr, Ytr, Xte, nClasses, kFolds)
    maxK = min(5, size(Xtr, 1) - 1); if maxK < 1, maxK = 1; end
    kRange = 1:maxK; cvAcc = zeros(1, numel(kRange));
    minClassCount = min(histcounts(Ytr, 1:(nClasses+1)));
    kf = min(kFolds, minClassCount); if kf < 2, kf = 2; end
    for ki = 1:numel(kRange)
        try
            mdl = fitcknn(Xtr, Ytr, 'NumNeighbors', kRange(ki), 'Distance', 'euclidean');
            cvM = crossval(mdl, 'KFold', kf);
            cvAcc(ki) = 1 - kfoldLoss(cvM);
        catch
            cvAcc(ki) = 0;
        end
    end
    [~, bestKi] = max(cvAcc);
    mdl = fitcknn(Xtr, Ytr, 'NumNeighbors', kRange(bestKi), 'Distance', 'euclidean');
    [pred, score] = predict(mdl, Xte);
    pred = double(pred);
end

function [pred, score] = trainRF(Xtr, Ytr, Xte, nClasses)
    try
        mdl = TreeBagger(100, Xtr, Ytr, 'Method', 'classification', ...
            'OOBPrediction', 'on', 'MinLeafSize', 1);
        [predCell, scoreRaw] = predict(mdl, Xte);
        pred = str2double(predCell);
        scoreRaw = double(scoreRaw);
        clsNum  = cellfun(@str2double, mdl.ClassNames);
        [clsSorted, sortIdx] = sort(clsNum);
        scoreSorted = scoreRaw(:, sortIdx);
        score = zeros(size(scoreRaw, 1), nClasses);
        for k = 1:numel(clsSorted)
            score(:, clsSorted(k)) = scoreSorted(:, k);
        end
    catch ME
        fprintf('    RF 失败: %s\n', ME.message);
        pred = NaN(size(Xte, 1), 1);
        score = NaN(size(Xte, 1), nClasses);
    end
end

function [pred, score] = trainBPNN(Xtr, Ytr, Xte, nClasses)
    [nTrain, nFeat] = size(Xtr);
    nHidden = 2;
    nParams = nFeat*nHidden + nHidden + nHidden*nClasses + nClasses;
    fprintf('    [BPNN] 特征=%d, 隐层=%d, 参数=%d, 参数/样本比=%.1f:1\n', ...
        nFeat, nHidden, nParams, nParams/nTrain);
    try
        Xtr_t = Xtr'; Xte_t = Xte';
        YtrOneHot = full(ind2vec(Ytr', nClasses));
        net = patternnet(nHidden);
        net.trainFcn = 'trainscg';
        net.trainParam.epochs = 50;
        net.trainParam.showWindow = false;
        net.trainParam.showCommandLine = false;
        net.performFcn = 'crossentropy';
        net.performParam.regularization = 0.1;
        net.divideFcn = 'dividetrain';
        net = train(net, Xtr_t, YtrOneHot);
        score = net(Xte_t)';
        [~, pred] = max(score, [], 2);
    catch ME
        fprintf('    [BPNN] 训练失败（%s），返回 NaN\n', ME.message);
        pred  = NaN(size(Xte, 1), 1);
        score = NaN(size(Xte, 1), nClasses);
    end
end

function s = padScore(s, nClasses)
    if size(s, 2) < nClasses
        s = [s, zeros(size(s,1), nClasses - size(s,2))];
    elseif size(s, 2) > nClasses
        s = s(:, 1:nClasses);
    end
end

function [acc, sens, spec, prec, f1] = computeMetrics(Ytrue, Ypred, nClasses)
    C = confusionmat(Ytrue, Ypred, 'Order', 1:nClasses);
    acc = sum(diag(C)) / max(sum(C(:)), 1);
    sensC = zeros(nClasses,1); specC = zeros(nClasses,1);
    precC = zeros(nClasses,1); f1C = zeros(nClasses,1);
    for c = 1:nClasses
        TP = C(c,c); FN = sum(C(c,:))-TP;
        FP = sum(C(:,c))-TP; TN = sum(C(:))-TP-FN-FP;
        if (TP+FN)>0, sensC(c) = TP/(TP+FN); end
        if (TN+FP)>0, specC(c) = TN/(TN+FP); end
        if (TP+FP)>0, precC(c) = TP/(TP+FP); end
        if (precC(c)+sensC(c))>0
            f1C(c) = 2*precC(c)*sensC(c)/(precC(c)+sensC(c));
        end
    end
    sens = mean(sensC); spec = mean(specC);
    prec = mean(precC); f1 = mean(f1C);
end

function aucVal = multiClassAUC(Ytrue, score, nClasses)
    if size(score, 2) < nClasses, aucVal = NaN; return; end
    if any(isnan(score(:))), aucVal = NaN; return; end
    aucs = [];
    for c = 1:nClasses
        yBin = double(Ytrue == c);
        if numel(unique(yBin)) < 2, continue; end
        s = score(:, c);
        if std(s) < 1e-10, continue; end
        try
            [~, ~, ~, a] = perfcurve(yBin, s, 1);
            if ~isnan(a), aucs(end+1) = a; end %#ok<AGROW>
        catch, end
    end
    if isempty(aucs), aucVal = NaN; else, aucVal = mean(aucs); end
end

function [bestField, bestName, compTable] = selectBestSVMKernel(R, clfList, dispList)
    svmIdx = find(startsWith(clfList, 'SVM_'));
    nSVM = numel(svmIdx);
    aucM = zeros(nSVM, 1); accM = zeros(nSVM, 1);
    for i = 1:nSVM
        a = R.(clfList{svmIdx(i)}).auc; a = a(~isnan(a));
        b = R.(clfList{svmIdx(i)}).accuracy; b = b(~isnan(b));
        if isempty(a), aucM(i) = 0; else, aucM(i) = mean(a); end
        if isempty(b), accM(i) = 0; else, accM(i) = mean(b); end
    end
    [~, ord] = sortrows([aucM, accM], [-1 -2]);
    bestIdx = ord(1);
    bestField = clfList{svmIdx(bestIdx)};
    bestName  = dispList{svmIdx(bestIdx)};
    compTable = table(dispList(svmIdx)', aucM, accM, ...
        'VariableNames', {'SVM_Kernel','AUC_mean','Acc_mean'});
end

function perClass = computePerClassResults(R, plotClf, nRep, nClasses)
    perClass = struct();
    for ci = 1:numel(plotClf)
        cname = plotClf{ci};
        allY = []; allPred = [];
        for r = 1:nRep
            if isempty(R.(cname).allLabels{r}), continue; end
            allY    = [allY;    R.(cname).allLabels{r}]; %#ok<AGROW>
            allPred = [allPred; R.(cname).allPred{r}];   %#ok<AGROW>
        end
        C = confusionmat(allY, allPred, 'Order', 1:nClasses);
        sensK = zeros(nClasses,1); specK = zeros(nClasses,1);
        precK = zeros(nClasses,1); f1K = zeros(nClasses,1);
        aucK = zeros(nClasses,1); accK = zeros(nClasses,1);
        for k = 1:nClasses
            TP = C(k,k); FN = sum(C(k,:))-TP;
            FP = sum(C(:,k))-TP; TN = sum(C(:))-TP-FN-FP;
            if (TP+TN+FP+FN)>0, accK(k) = (TP+TN)/(TP+TN+FP+FN); end
            if (TP+FN)>0, sensK(k) = TP/(TP+FN); end
            if (TN+FP)>0, specK(k) = TN/(TN+FP); end
            if (TP+FP)>0, precK(k) = TP/(TP+FP); end
            if (precK(k)+sensK(k))>0
                f1K(k) = 2*precK(k)*sensK(k)/(precK(k)+sensK(k));
            end
            allS = [];
            for r = 1:nRep
                if isempty(R.(cname).allScores{r}), continue; end
                allS = [allS; R.(cname).allScores{r}(:,k)]; %#ok<AGROW>
            end
            yBin = double(allY == k);
            if numel(unique(yBin)) >= 2 && numel(allS) == numel(yBin) ...
                    && ~any(isnan(allS)) && std(allS) > 1e-10
                try
                    [~, ~, ~, a] = perfcurve(yBin, allS, 1);
                    if ~isnan(a), aucK(k) = a; end
                catch, end
            end
        end
        perClass.(cname).acc  = accK;
        perClass.(cname).sens = sensK;
        perClass.(cname).spec = specK;
        perClass.(cname).prec = precK;
        perClass.(cname).f1   = f1K;
        perClass.(cname).auc  = aucK;
    end
end

function writePerClassResults(perClass, plotClf, plotNames, classNames, outDir)
    nClf = numel(plotClf); nCls = numel(classNames);
    ClfCol = {}; ClassCol = {};
    Acc = []; Sens = []; Spec = []; Prec = []; F1 = []; AUC = []; AUC_Status = {};
    for ci = 1:nClf
        cname = plotClf{ci}; pname = plotNames{ci};
        for k = 1:nCls
            ClfCol{end+1,1} = pname; %#ok<AGROW>
            ClassCol{end+1,1} = classNames{k}; %#ok<AGROW>
            Acc(end+1,1)  = perClass.(cname).acc(k); %#ok<AGROW>
            Sens(end+1,1) = perClass.(cname).sens(k); %#ok<AGROW>
            Spec(end+1,1) = perClass.(cname).spec(k); %#ok<AGROW>
            Prec(end+1,1) = perClass.(cname).prec(k); %#ok<AGROW>
            F1(end+1,1)   = perClass.(cname).f1(k);   %#ok<AGROW>
            AUC(end+1,1)  = perClass.(cname).auc(k);  %#ok<AGROW>
            if isnan(perClass.(cname).auc(k))
                AUC_Status{end+1,1} = 'N/A'; %#ok<AGROW>
            else
                AUC_Status{end+1,1} = 'Valid'; %#ok<AGROW>
            end
        end
    end
    T = table(ClfCol, ClassCol, Acc, Sens, Spec, Prec, F1, AUC, AUC_Status, ...
        'VariableNames', {'Classifier','Class','Overall_Accuracy','Sensitivity', ...
                          'Specificity','Precision','F1_Score','AUC','AUC_Status'});
    writetable(T, fullfile(outDir, 'per_class_metrics.xlsx'), ...
        'FileType', 'spreadsheet', 'Sheet', 'PerClass_Metrics');
    fprintf('  每类点估计结果已保存: per_class_metrics.xlsx\n');
end

function reportPerClassMetrics(R, plotClf, plotNames, classNames, ...
        nRep, nClasses, nBoot, outDir)
    metricsList = {'acc', 'sens', 'spec', 'prec', 'f1', 'auc'};
    metricNames = {'Overall_Accuracy', 'Sensitivity', 'Specificity', ...
                   'Precision', 'F1_Score', 'AUC'};
    rng(8888, 'twister');
    rowsLong = {}; rowsWide = {};
    for ci = 1:numel(plotClf)
        cname = plotClf{ci}; dname = plotNames{ci};
        perClassVals = struct();
        for mi = 1:numel(metricsList)
            perClassVals.(metricsList{mi}) = nan(nRep, nClasses);
        end
        for r = 1:nRep
            if isempty(R.(cname).perClass{r}), continue; end
            pc = R.(cname).perClass{r};
            for mi = 1:numel(metricsList)
                mn = metricsList{mi};
                if isfield(pc, mn)
                    perClassVals.(mn)(r, :) = pc.(mn);
                end
            end
        end
        fprintf('\n  ===== [%s] 每类指标 =====\n', dname);
        fprintf('  %-6s | %-18s | %-18s | %-18s | %-18s | %-18s | %-18s\n', ...
            'Class', 'Overall_Acc', 'Sensitivity', 'Specificity', ...
            'Precision', 'F1_Score', 'AUC');
        fprintf('  %s\n', repmat('-', 1, 130));
        for k = 1:nClasses
            rowWide = {dname, classNames{k}};
            vals = cell(1, numel(metricsList));
            for mi = 1:numel(metricsList)
                mn = metricsList{mi};
                v = perClassVals.(mn)(:, k); v = v(~isnan(v));
                if isempty(v)
                    vals{mi} = 'N/A';
                    rowsLong(end+1, :) = {dname, classNames{k}, metricNames{mi}, ...
                        NaN, NaN, NaN, NaN}; %#ok<AGROW>
                    rowWide{end+1} = 'N/A'; %#ok<AGROW>
                    rowWide{end+1} = NaN;    %#ok<AGROW>
                    rowWide{end+1} = NaN;    %#ok<AGROW>
                else
                    m = mean(v); s = std(v);
                    if numel(v) < 5
                        ciLo = NaN; ciHi = NaN;
                    else
                        bootVals = zeros(nBoot, 1);
                        for b = 1:nBoot
                            idx = randi(numel(v), numel(v), 1);
                            bootVals(b) = mean(v(idx));
                        end
                        ciLo = quantile(bootVals, 0.025);
                        ciHi = quantile(bootVals, 0.975);
                    end
                    vals{mi} = sprintf('%.3f±%.3f', m, s);
                    rowsLong(end+1, :) = {dname, classNames{k}, metricNames{mi}, ...
                        m, s, ciLo, ciHi}; %#ok<AGROW>
                    rowWide{end+1} = sprintf('%.4f ± %.4f [%.4f, %.4f]', ...
                        m, s, ciLo, ciHi); %#ok<AGROW>
                    rowWide{end+1} = m; %#ok<AGROW>
                    rowWide{end+1} = s; %#ok<AGROW>
                end
            end
            fprintf('  %-6s | %-18s | %-18s | %-18s | %-18s | %-18s | %-18s\n', ...
                classNames{k}, vals{1}, vals{2}, vals{3}, vals{4}, vals{5}, vals{6});
            rowsWide(end+1, :) = rowWide; %#ok<AGROW>
        end
    end
    Tlong = cell2table(rowsLong, 'VariableNames', ...
        {'Classifier','Class','Metric','Mean','Std','CI_low','CI_high'});
    writetable(Tlong, fullfile(outDir, 'per_class_all_metrics_long.xlsx'), ...
        'FileType', 'spreadsheet', 'Sheet', 'PerClass_Long');
    varNames = {'Classifier', 'Class'};
    for mi = 1:numel(metricNames)
        varNames{end+1} = [metricNames{mi} '_Str'];   %#ok<AGROW>
        varNames{end+1} = [metricNames{mi} '_Mean'];  %#ok<AGROW>
        varNames{end+1} = [metricNames{mi} '_SD'];    %#ok<AGROW>
    end
    Twide = cell2table(rowsWide, 'VariableNames', varNames);
    writetable(Twide, fullfile(outDir, 'per_class_all_metrics_wide.xlsx'), ...
        'FileType', 'spreadsheet', 'Sheet', 'PerClass_Wide');
    fprintf('\n  ✓ 每类完整指标已保存:\n');
    fprintf('      - per_class_all_metrics_long.xlsx (%d 行)\n', height(Tlong));
    fprintf('      - per_class_all_metrics_wide.xlsx (%d 行)\n', height(Twide));
end

function reportResults(R, classifiers, dispNames, nRep, outDir, nBoot)
    fprintf('\n  ============================================================================\n');
    fprintf('  完整性能汇总（%d 分类器 × 6 指标 × 95%% Bootstrap CI）\n', numel(classifiers));
    fprintf('  ============================================================================\n');
    metricsList = {'accuracy', 'sensitivity', 'specificity', ...
                   'precision', 'f1', 'auc'};
    metricNames = {'Overall Accuracy', 'Sensitivity', 'Specificity', ...
                   'Precision', 'F1-Score', 'AUC'};
    rng(9999, 'twister');
    rows = {};
    for ci = 1:numel(classifiers)
        cname = classifiers{ci}; dname = dispNames{ci};
        if strcmpi(cname, 'BPNN')
            noteStr = 'Exploratory only; parameter-to-sample ratio > 2:1';
            dispLabel = [dname ' [Exploratory]'];
        else
            noteStr = ''; dispLabel = dname;
        end
        fprintf('\n  [%s]\n', dispLabel);
        fprintf('  %-18s | %-9s | %-9s | %-22s\n', ...
            'Metric', 'Mean', 'SD', '95% CI');
        fprintf('  %s\n', repmat('-', 1, 66));
        for mi = 1:numel(metricsList)
            mn = metricsList{mi};
            vals = R.(cname).(mn); vals = vals(~isnan(vals));
            if isempty(vals)
                m = NaN; s = NaN; ciLo = NaN; ciHi = NaN;
                mStr = 'N/A'; sStr = 'N/A'; ciStr = 'N/A';
            else
                m = mean(vals); s = std(vals);
                bootVals = zeros(nBoot, 1);
                for b = 1:nBoot
                    idx = randi(numel(vals), numel(vals), 1);
                    bootVals(b) = mean(vals(idx));
                end
                ciLo = quantile(bootVals, 0.025);
                ciHi = quantile(bootVals, 0.975);
                mStr = sprintf('%.4f', m); sStr = sprintf('%.4f', s);
                ciStr = sprintf('[%.4f, %.4f]', ciLo, ciHi);
            end
            fprintf('  %-18s | %-9s | %-9s | %-22s\n', ...
                metricNames{mi}, mStr, sStr, ciStr);
            rows(end+1, :) = {dispLabel, metricNames{mi}, ...
                m, s, ciLo, ciHi, noteStr}; %#ok<AGROW>
        end
    end
    T = cell2table(rows, 'VariableNames', ...
        {'Classifier', 'Metric', 'Mean', 'Std', 'CI_low', 'CI_high', 'Note'});
    writetable(T, fullfile(outDir, 'all_metrics_summary.xlsx'), ...
        'FileType', 'spreadsheet', 'Sheet', 'All_Metrics');
    fprintf('\n  ✓ all_metrics_summary.xlsx 已保存\n');
    aucRows = {};
    for ci = 1:numel(classifiers)
        cname = classifiers{ci}; dname = dispNames{ci};
        au = R.(cname).auc; au = au(~isnan(au));
        if isempty(au), continue; end
        m = mean(au); s = std(au);
        bootVals = zeros(nBoot, 1);
        for b = 1:nBoot
            idx = randi(numel(au), numel(au), 1);
            bootVals(b) = mean(au(idx));
        end
        aucRows(end+1, :) = {dname, m, s, quantile(bootVals,0.025), ...
            quantile(bootVals,0.975)}; %#ok<AGROW>
    end
    if ~isempty(aucRows)
        Ta = cell2table(aucRows, 'VariableNames', ...
            {'Classifier', 'AUC_mean', 'AUC_std', 'AUC_CI_low', 'AUC_CI_high'});
        writetable(Ta, fullfile(outDir, 'AUC_with_CI.xlsx'), ...
            'FileType', 'spreadsheet', 'Sheet', 'AUC_CI');
        writetable(Ta, fullfile(outDir, 'bootstrap_AUC_CI.xlsx'), ...
            'FileType', 'spreadsheet', 'Sheet', 'Bootstrap_AUC');
        fprintf('  ✓ AUC_with_CI.xlsx / bootstrap_AUC_CI.xlsx 已保存\n');
    end
    wideRows = {};
    for ci = 1:numel(classifiers)
        cname = classifiers{ci}; dname = dispNames{ci};
        row = {dname};
        for mi = 1:numel(metricsList)
            v = R.(cname).(metricsList{mi}); v = v(~isnan(v));
            if isempty(v), row{end+1} = 'N/A'; %#ok<AGROW>
            else, row{end+1} = sprintf('%.4f ± %.4f', mean(v), std(v)); end %#ok<AGROW>
        end
        au = R.(cname).auc; au = au(~isnan(au));
        if ~isempty(au)
            bootVals = zeros(nBoot, 1);
            for b = 1:nBoot
                idx = randi(numel(au), numel(au), 1);
                bootVals(b) = mean(au(idx));
            end
            row{end+1} = sprintf('[%.4f, %.4f]', ...
                quantile(bootVals,0.025), quantile(bootVals,0.975)); %#ok<AGROW>
        else
            row{end+1} = 'N/A'; %#ok<AGROW>
        end
        if strcmpi(cname, 'BPNN')
            row{end+1} = 'Exploratory only; parameter-to-sample ratio > 2:1'; %#ok<AGROW>
        else
            row{end+1} = ''; %#ok<AGROW>
        end
        wideRows(end+1, :) = row; %#ok<AGROW>
    end
    varNames = {'Classifier', metricNames{:}, 'AUC_95CI', 'Note'};
    Tw = cell2table(wideRows, 'VariableNames', varNames);
    writetable(Tw, fullfile(outDir, 'performance_summary.xlsx'), ...
        'FileType', 'spreadsheet', 'Sheet', 'Performance');
    writetable(Tw, fullfile(outDir, 'performance_summary_wide.xlsx'), ...
        'FileType', 'spreadsheet', 'Sheet', 'Performance_Wide');
    fprintf('  ✓ performance_summary.xlsx / performance_summary_wide.xlsx 已保存\n');
end

% ------------------------------------------------------------------------
% ★ 新增 1：保存 100 次重复的原始指标
% ------------------------------------------------------------------------
function saveRepeatLevelMetrics(R, classifiers, dispNames, nRep, outDir)
    metricsList = {'accuracy', 'sensitivity', 'specificity', ...
                   'precision', 'f1', 'auc'};
    metricLabels = {'Overall_Accuracy', 'Sensitivity', 'Specificity', ...
                    'Precision', 'F1_Score', 'AUC'};
    T = table();
    T.Repeat = (1:nRep)';
    for ci = 1:numel(classifiers)
        cname = classifiers{ci}; dname = dispNames{ci};
        prefix = strrep(dname, ' ', '_');
        prefix = strrep(prefix, '-', '_');
        prefix = strrep(prefix, '(', '');
        prefix = strrep(prefix, ')', '');
        for mi = 1:numel(metricsList)
            mn = metricsList{mi};
            ml = metricLabels{mi};
            colName = sprintf('%s_%s', prefix, ml);
            colName = matlab.lang.makeValidName(colName);
            v = R.(cname).(mn);
            T.(colName) = v;
        end
    end
    writetable(T, fullfile(outDir, 'repeat_level_metrics.xlsx'), ...
        'FileType', 'spreadsheet', 'Sheet', 'Repeat_Metrics');
    fprintf('  100 次重复的原始指标已保存: repeat_level_metrics.xlsx (%d 行 × %d 列)\n', ...
        height(T), width(T));
end

% ------------------------------------------------------------------------
% ★ 新增 2：保存混淆矩阵数值表
% ------------------------------------------------------------------------
function saveConfusionTables(R, plotClf, plotNames, classNames, nRep, outDir)
    nCls = numel(classNames);
    xlsxFile = fullfile(outDir, 'confusion_matrices.xlsx');
    if exist(xlsxFile, 'file'), delete(xlsxFile); end
    for ci = 1:numel(plotClf)
        cname = plotClf{ci}; pname = plotNames{ci};
        allY = []; allPred = [];
        for r = 1:nRep
            if isempty(R.(cname).allLabels{r}), continue; end
            allY    = [allY;    R.(cname).allLabels{r}]; %#ok<AGROW>
            allPred = [allPred; R.(cname).allPred{r}];   %#ok<AGROW>
        end
        if isempty(allY), continue; end
        C = confusionmat(allY, allPred, 'Order', 1:nCls);
        predNames = cellfun(@(x) ['Pred_' x], classNames, 'UniformOutput', false);
        rowNames  = cellfun(@(x) ['True_' x], classNames, 'UniformOutput', false);
        T = array2table(C, 'VariableNames', predNames, 'RowNames', rowNames);
        sheetName = strrep(pname, ' ', '_');
        sheetName = strrep(sheetName, '-', '_');
        sheetName = strrep(sheetName, '(', '');
        sheetName = strrep(sheetName, ')', '');
        sheetName = matlab.lang.makeValidName(sheetName);
        if length(sheetName) > 31, sheetName = sheetName(1:31); end
        if ci == 1
            writetable(T, xlsxFile, 'FileType', 'spreadsheet', ...
                'Sheet', sheetName, 'WriteRowNames', true);
        else
            writetable(T, xlsxFile, 'FileType', 'spreadsheet', ...
                'Sheet', sheetName, 'WriteRowNames', true, ...
                'WriteMode', 'append');
        end
    end
    fprintf('  混淆矩阵数值已保存: confusion_matrices.xlsx\n');
end

% ------------------------------------------------------------------------
% ★ 新增 3：保存每折选出的特征历史
% ------------------------------------------------------------------------
function saveFeatureSelectionHistory(histCell, repUnionCell, ...
        featureNames, featureSource, nRep, kOuter, outDir)
    Rep = []; Fold = []; FeatureIdx = []; FeatureName = {}; IonMode = [];
    for rep = 1:nRep
        for fold = 1:kOuter
            selIdx = histCell{rep, fold};
            if isempty(selIdx), continue; end
            for k = 1:numel(selIdx)
                Rep(end+1, 1)         = rep;                       %#ok<AGROW>
                Fold(end+1, 1)        = fold;                      %#ok<AGROW>
                FeatureIdx(end+1, 1)  = selIdx(k);                 %#ok<AGROW>
                FeatureName{end+1, 1} = featureNames{selIdx(k)};   %#ok<AGROW>
                IonMode(end+1, 1)     = featureSource(selIdx(k));  %#ok<AGROW>
            end
        end
    end
    T1 = table(Rep, Fold, FeatureIdx, FeatureName, IonMode);
    xlsxFile = fullfile(outDir, 'feature_selection_history.xlsx');
    if exist(xlsxFile, 'file'), delete(xlsxFile); end
    writetable(T1, xlsxFile, ...
        'FileType', 'spreadsheet', 'Sheet', 'Per_Fold');
    Rep2 = (1:nRep)';
    NFeatures = zeros(nRep, 1);
    FeatureList = cell(nRep, 1);
    for rep = 1:nRep
        u = repUnionCell{rep};
        NFeatures(rep) = numel(u);
        if isempty(u)
            FeatureList{rep} = '';
        else
            names = featureNames(u);
            FeatureList{rep} = strjoin(names, '; ');
        end
    end
    T2 = table(Rep2, NFeatures, FeatureList, ...
        'VariableNames', {'Repeat', 'N_Union_Features', 'Union_Feature_Names'});
    writetable(T2, xlsxFile, ...
        'FileType', 'spreadsheet', 'Sheet', 'Per_Repeat', ...
        'WriteMode', 'append');
    fprintf('  特征选择历史已保存: feature_selection_history.xlsx\n');
    fprintf('      - Per_Fold 工作表: %d 行（每折选出的特征）\n', height(T1));
    fprintf('      - Per_Repeat 工作表: %d 行（每次重复的并集特征数）\n', height(T2));
end

function plotPerAlgoROC(R, plotClf, plotNames, classNames, nRep, nClasses, outDir)
    nAlgo = numel(plotClf);
    if nAlgo == 0, return; end
    if nAlgo <= 2, nRow = 1; nCol = nAlgo;
    elseif nAlgo == 3, nRow = 1; nCol = 3;
    else, nRow = 2; nCol = 2; end
    f = figure('Position', [30 30 1300 1150], 'Visible', 'off', 'Color', 'w');
    for ci = 1:nAlgo
        cname = plotClf{ci}; pname = plotNames{ci};
        allS = []; allL = [];
        for r = 1:nRep
            if isempty(R.(cname).allScores{r}), continue; end
            allS = [allS; R.(cname).allScores{r}]; %#ok<AGROW>
            allL = [allL; R.(cname).allLabels{r}]; %#ok<AGROW>
        end
        subplot(nRow, nCol, ci); hold on; set(gca, 'Color', 'w');
        if isempty(allS) || any(isnan(allS(:)))
            text(0.5, 0.5, sprintf('%s\nNo valid scores', pname), ...
                'HorizontalAlignment', 'center', 'Units', 'normalized');
            axis off; continue;
        end
        classColors = [ 0.000 0.447 0.698; 0.835 0.369 0.000; ...
                        0.000 0.620 0.451; 0.800 0.475 0.655 ];
        colors = classColors(1:nClasses, :);
        aucVals = nan(nClasses, 1); nanClasses = {};
        for k = 1:nClasses
            yBin = double(allL == k);
            if numel(unique(yBin)) < 2, nanClasses{end+1} = classNames{k}; continue; end
            s = allS(:, k);
            if any(isnan(s)) || std(s) < 1e-10
                nanClasses{end+1} = classNames{k}; continue;
            end
            try
                [fpr, tpr, ~, a] = perfcurve(yBin, s, 1);
                plot(fpr, tpr, 'Color', colors(k,:), 'LineWidth', 1.8, ...
                    'DisplayName', sprintf('%s (AUC = %.3f)', classNames{k}, a));
                aucVals(k) = a;
            catch, nanClasses{end+1} = classNames{k}; end
        end
        plot([0 1], [0 1], 'k--', 'LineWidth', 1, 'HandleVisibility', 'off');
        xlabel('1 - Specificity', 'FontSize', 10);
        ylabel('Sensitivity', 'FontSize', 10);
        title(sprintf('%s  (macro AUC = %.3f)', pname, mean(aucVals, 'omitnan')), 'FontSize', 11);
        if ~isempty(nanClasses)
            text(0.02, 0.98, sprintf('N/A: %s', strjoin(nanClasses, ', ')), ...
                'Units', 'normalized', 'FontSize', 8, 'Color', [0.6 0 0], ...
                'VerticalAlignment', 'top', 'BackgroundColor', [1 1 0.9], ...
                'EdgeColor', [0.6 0 0], 'Margin', 2);
        end
        legend('Location', 'southeast', 'FontSize', 8);
        grid on; axis square; xlim([0 1]); ylim([0 1]);
    end
    sgtitle(sprintf('Per-Class ROC Curves (point estimates, %d repeats × 5 folds)', nRep), ...
        'FontSize', 13, 'FontWeight', 'bold');
    saveas(f, fullfile(outDir, 'fig5_ROC_all.png'));
    savefig(f, fullfile(outDir, 'fig5_ROC_all.fig'));
    close(f);
    fprintf('  图5 已保存: fig5_ROC_all.png + .fig\n');
end

function plotPerAlgoROCWithCI(R, plotClf, plotNames, classNames, ...
        nRep, nClasses, nBootROC, outDir)
    nGrid = 200;
    fprGrid = linspace(0, 1, nGrid);
    nAlgo = numel(plotClf);
    if nAlgo == 0, return; end
    if nAlgo <= 2, nRow = 1; nCol = nAlgo;
    elseif nAlgo == 3, nRow = 1; nCol = 3;
    else, nRow = 2; nCol = 2; end
    f = figure('Position', [30 30 1300 1150], 'Visible', 'off', 'Color', 'w');
    for ci = 1:nAlgo
        cname = plotClf{ci}; pname = plotNames{ci};
        allS = []; allL = [];
        for r = 1:nRep
            if isempty(R.(cname).allScores{r}), continue; end
            allS = [allS; R.(cname).allScores{r}]; %#ok<AGROW>
            allL = [allL; R.(cname).allLabels{r}]; %#ok<AGROW>
        end
        subplot(nRow, nCol, ci); hold on; set(gca, 'Color', 'w');
        if isempty(allS) || any(isnan(allS(:)))
            text(0.5, 0.5, sprintf('%s\nNo valid scores', pname), ...
                'HorizontalAlignment', 'center', 'Units', 'normalized');
            axis off; continue;
        end
        fprintf('  [%s] 计算 Bootstrap 置信带（fig6 阴影）...\n', pname);
        classColors = [ 0.000 0.447 0.698; 0.835 0.369 0.000; ...
                        0.000 0.620 0.451; 0.800 0.475 0.655 ];
        colors = classColors(1:nClasses, :);
        aucVals = nan(nClasses, 1); nanClasses = {};
        for k = 1:nClasses
            yBin = double(allL == k);
            s = allS(:, k);
            if numel(unique(yBin)) < 2, nanClasses{end+1} = classNames{k}; continue; end
            if any(isnan(s)) || std(s) < 1e-10
                nanClasses{end+1} = classNames{k}; continue;
            end
            try
                [fpr0, tpr0, ~, auc0] = perfcurve(yBin, s, 1);
            catch, nanClasses{end+1} = classNames{k}; continue; end
            aucVals(k) = auc0;
            [tprLo, tprHi] = bootstrapROCBand(yBin, s, fprGrid, nBootROC);
            xBand = [fprGrid, fliplr(fprGrid)];
            yBand = [tprLo(:)', fliplr(tprHi(:)')];
            fill(xBand, yBand, colors(k,:), 'FaceAlpha', 0.15, ...
                 'EdgeColor', 'none', 'HandleVisibility', 'off');
            plot(fpr0, tpr0, 'Color', colors(k,:), 'LineWidth', 1.8, ...
                'DisplayName', sprintf('%s (AUC = %.3f)', classNames{k}, auc0));
        end
        plot([0 1], [0 1], 'k--', 'LineWidth', 1, 'HandleVisibility', 'off');
        xlabel('1 - Specificity', 'FontSize', 10);
        ylabel('Sensitivity', 'FontSize', 10);
        title(sprintf('%s  (macro AUC = %.3f)', pname, mean(aucVals, 'omitnan')), 'FontSize', 11);
        if ~isempty(nanClasses)
            text(0.02, 0.98, sprintf('N/A: %s', strjoin(nanClasses, ', ')), ...
                'Units', 'normalized', 'FontSize', 8, 'Color', [0.6 0 0], ...
                'VerticalAlignment', 'top', 'BackgroundColor', [1 1 0.9], ...
                'EdgeColor', [0.6 0 0], 'Margin', 2);
        end
        legend('Location', 'southeast', 'FontSize', 8);
        grid on; axis square; xlim([0 1]); ylim([0 1]);
    end
    sgtitle(sprintf(['Per-Class ROC Curves with 95%% Bootstrap CI Band\n', ...
        '(%d repeats × 5 folds, %d Bootstrap resamples)'], nRep, nBootROC), ...
        'FontSize', 12, 'FontWeight', 'bold');
    saveas(f, fullfile(outDir, 'fig6_ROC_CI_all.png'));
    savefig(f, fullfile(outDir, 'fig6_ROC_CI_all.fig'));
    close(f);
    fprintf('  图6 已保存: fig6_ROC_CI_all.png + .fig\n');
end

function plotPerAlgoROCWithCILines(R, plotClf, plotNames, classNames, ...
        nRep, nClasses, nBootROC, outDir)
    nGrid = 200;
    fprGrid = linspace(0, 1, nGrid);
    nAlgo = numel(plotClf);
    if nAlgo == 0, return; end
    if nAlgo <= 2, nRow = 1; nCol = nAlgo;
    elseif nAlgo == 3, nRow = 1; nCol = 3;
    else, nRow = 2; nCol = 2; end
    classColors = [ 0.000 0.447 0.698; 0.835 0.369 0.000; ...
                    0.000 0.620 0.451; 0.800 0.475 0.655 ];
    colors = classColors(1:nClasses, :);
    f = figure('Position', [30 30 1300 1150], 'Visible', 'off', 'Color', 'w');
    for ci = 1:nAlgo
        cname = plotClf{ci}; pname = plotNames{ci};
        allS = []; allL = [];
        for r = 1:nRep
            if isempty(R.(cname).allScores{r}), continue; end
            allS = [allS; R.(cname).allScores{r}]; %#ok<AGROW>
            allL = [allL; R.(cname).allLabels{r}]; %#ok<AGROW>
        end
        subplot(nRow, nCol, ci); hold on; set(gca, 'Color', 'w');
        if isempty(allS) || any(isnan(allS(:)))
            text(0.5, 0.5, sprintf('%s\nNo valid scores', pname), ...
                'HorizontalAlignment', 'center', 'Units', 'normalized');
            axis off; continue;
        end
        fprintf('  [%s] 计算 Bootstrap CI（fig7 虚线版）...\n', pname);
        aucVals = nan(nClasses, 1); nanClasses = {};
        for k = 1:nClasses
            yBin = double(allL == k);
            s = allS(:, k);
            if numel(unique(yBin)) < 2, nanClasses{end+1} = classNames{k}; continue; end
            if any(isnan(s)) || std(s) < 1e-10
                nanClasses{end+1} = classNames{k}; continue;
            end
            try
                [fpr0, tpr0, ~, auc0] = perfcurve(yBin, s, 1);
            catch, nanClasses{end+1} = classNames{k}; continue; end
            aucVals(k) = auc0;
            [tprLo, tprHi] = bootstrapROCBand(yBin, s, fprGrid, nBootROC);
            plot(fprGrid, tprHi, 'Color', colors(k,:), ...
                'LineStyle', ':', 'LineWidth', 1.2, 'HandleVisibility', 'off');
            plot(fprGrid, tprLo, 'Color', colors(k,:), ...
                'LineStyle', ':', 'LineWidth', 1.2, 'HandleVisibility', 'off');
            plot(fpr0, tpr0, 'Color', colors(k,:), 'LineWidth', 1.8, ...
                'DisplayName', sprintf('%s (AUC = %.3f)', classNames{k}, auc0));
        end
        plot([0 1], [0 1], 'k--', 'LineWidth', 1, 'HandleVisibility', 'off');
        xlabel('1 - Specificity', 'FontSize', 10);
        ylabel('Sensitivity', 'FontSize', 10);
        title(sprintf('%s  (macro AUC = %.3f)', pname, mean(aucVals, 'omitnan')), 'FontSize', 11);
        if ~isempty(nanClasses)
            text(0.02, 0.98, sprintf('N/A: %s', strjoin(nanClasses, ', ')), ...
                'Units', 'normalized', 'FontSize', 8, 'Color', [0.6 0 0], ...
                'VerticalAlignment', 'top', 'BackgroundColor', [1 1 0.9], ...
                'EdgeColor', [0.6 0 0], 'Margin', 2);
        end
        legend('Location', 'southeast', 'FontSize', 8);
        text(0.02, 0.02, 'Dotted = 95% Bootstrap CI', ...
            'FontSize', 8, 'Color', [0.4 0.4 0.4], 'Units', 'normalized');
        grid on; axis square; xlim([0 1]); ylim([0 1]);
    end
    sgtitle(sprintf(['Per-Class ROC Curves with 95%% Bootstrap CI (dotted boundaries)\n', ...
        '(%d repeats × 5 folds, %d Bootstrap resamples)'], nRep, nBootROC), ...
        'FontSize', 12, 'FontWeight', 'bold');
    saveas(f, fullfile(outDir, 'fig7_ROC_CI_lines.png'));
    savefig(f, fullfile(outDir, 'fig7_ROC_CI_lines.fig'));
    close(f);
    fprintf('  图7 已保存: fig7_ROC_CI_lines.png + .fig\n');
end

function plotConfusionMatrices(R, plotClf, plotNames, classNames, nRep, outDir)
    nAlgo = numel(plotClf); nCls = numel(classNames);
    if nAlgo == 0, return; end
    if nAlgo <= 2, nRow = 1; nCol = nAlgo;
    elseif nAlgo == 3, nRow = 1; nCol = 3;
    else, nRow = 2; nCol = 2; end
    f = figure('Position', [30 30 1250 1150], 'Visible', 'off', 'Color', 'w');
    for ci = 1:nAlgo
        cname = plotClf{ci}; pname = plotNames{ci};
        allY = []; allPred = [];
        for r = 1:nRep
            if isempty(R.(cname).allLabels{r}), continue; end
            allY    = [allY;    R.(cname).allLabels{r}]; %#ok<AGROW>
            allPred = [allPred; R.(cname).allPred{r}];   %#ok<AGROW>
        end
        subplot(nRow, nCol, ci);
        if isempty(allY)
            axis off;
            text(0.5, 0.5, sprintf('%s\nNo valid predictions', pname), ...
                'HorizontalAlignment', 'center', 'Units', 'normalized');
            continue;
        end
        C = confusionmat(allY, allPred, 'Order', 1:nCls);
        rowSum = sum(C, 2); rowSum(rowSum == 0) = 1;
        Cnorm = C ./ rowSum * 100;
        imagesc(Cnorm);
        colormap(flipud(gray)); caxis([0 100]);
        cb = colorbar; cb.Label.String = 'Row-normalized (%)';
        set(gca, 'XTick', 1:nCls, 'XTickLabel', classNames, ...
                 'YTick', 1:nCls, 'YTickLabel', classNames, ...
                 'FontSize', 10, 'LineWidth', 1.0);
        xlabel('Predicted breed', 'FontSize', 11);
        ylabel('True breed', 'FontSize', 11);
        accOverall = sum(diag(C)) / max(sum(C(:)), 1);
        title(sprintf('%s  (OA = %.3f)', pname, accOverall), 'FontSize', 11);
        for i = 1:nCls
            for j = 1:nCls
                if Cnorm(i,j) > 50, txtColor = 'w'; else, txtColor = 'r'; end
                text(j, i, sprintf('%.1f', Cnorm(i,j)), ...
                    'HorizontalAlignment', 'center', ...
                    'Color', txtColor, 'FontSize', 10, 'FontWeight', 'bold');
            end
        end
    end
    sgtitle(sprintf('Confusion Matrices (row-normalized, %d repeats × 5 folds pooled)', nRep), ...
        'FontSize', 13, 'FontWeight', 'bold');
    saveas(f, fullfile(outDir, 'fig8_confusion_matrices.png'));
    savefig(f, fullfile(outDir, 'fig8_confusion_matrices.fig'));
    close(f);
    fprintf('  图8 混淆矩阵已保存: fig8_confusion_matrices.png + .fig\n');
end

function plotAUCDistribution(R, plotClf, plotNames, nRep, nBoot, outDir)
    nAlgo = numel(plotClf);
    if nAlgo == 0, return; end
    if nAlgo <= 2, nRow = 1; nCol = nAlgo;
    elseif nAlgo == 3, nRow = 1; nCol = 3;
    else, nRow = 2; nCol = 2; end
    f = figure('Position', [30 30 1250 1150], 'Visible', 'off', 'Color', 'w');
    rng(5555, 'twister');
    for ci = 1:nAlgo
        cname = plotClf{ci}; pname = plotNames{ci};
        aucVals = R.(cname).auc; aucVals = aucVals(~isnan(aucVals));
        subplot(nRow, nCol, ci); hold on; set(gca, 'Color', 'w');
        if isempty(aucVals)
            axis off;
            text(0.5, 0.5, sprintf('%s\nNo valid AUC', pname), ...
                'HorizontalAlignment', 'center', 'Units', 'normalized');
            continue;
        end
        edges = linspace(max(0, min(aucVals) - 0.05), min(1, max(aucVals) + 0.05), 15);
        histogram(aucVals, edges, 'FaceColor', [0.3 0.5 0.75], ...
            'EdgeColor', 'w', 'FaceAlpha', 0.85);
        m = mean(aucVals); s = std(aucVals);
        bootVals = zeros(nBoot, 1);
        for b = 1:nBoot
            idx = randi(numel(aucVals), numel(aucVals), 1);
            bootVals(b) = mean(aucVals(idx));
        end
        ciLo = quantile(bootVals, 0.025);
        ciHi = quantile(bootVals, 0.975);
        yl = ylim;
        xline(m,    'k-',  'LineWidth', 2, 'Label', sprintf('Mean = %.3f', m), ...
            'LabelOrientation', 'horizontal', 'FontSize', 9);
        xline(ciLo, 'r--', 'LineWidth', 1.5, 'Label', sprintf('%.3f', ciLo), ...
            'FontSize', 8, 'LabelVerticalAlignment', 'bottom');
        xline(ciHi, 'r--', 'LineWidth', 1.5, 'Label', sprintf('%.3f', ciHi), ...
            'FontSize', 8, 'LabelVerticalAlignment', 'bottom');
        ylim(yl);
        xlabel('Macro-averaged AUC', 'FontSize', 11);
        ylabel('Frequency', 'FontSize', 10);
        xlim([0 1]);
        title(sprintf('%s  (n = %d, mean ± SD = %.3f ± %.3f)', ...
            pname, numel(aucVals), m, s), 'FontSize', 10);
        grid on;
    end
    sgtitle(sprintf('Distribution of Macro-Averaged AUC across %d Repeats', nRep), ...
        'FontSize', 13, 'FontWeight', 'bold');
    saveas(f, fullfile(outDir, 'fig9_AUC_distribution.png'));
    savefig(f, fullfile(outDir, 'fig9_AUC_distribution.fig'));
    close(f);
    fprintf('  图9 AUC 分布直方图已保存: fig9_AUC_distribution.png + .fig\n');
end

function plotFeatureSelectionCurve(selectFreq, outDir)
    f = figure('Position', [50 50 800 600], 'Visible', 'off', 'Color', 'w');
    hold on;
    thresholds = 0.05:0.05:1.0;
    nKept = zeros(size(thresholds));
    for i = 1:numel(thresholds)
        nKept(i) = sum(selectFreq >= thresholds(i));
    end
    plot(thresholds * 100, nKept, 'o-', ...
        'Color', [0.0 0.447 0.698], 'LineWidth', 2, 'MarkerSize', 8, ...
        'MarkerFaceColor', [0.0 0.447 0.698]);
    idx50 = find(abs(thresholds - 0.5) < 1e-6);
    if ~isempty(idx50)
        plot(50, nKept(idx50), 'rp', 'MarkerSize', 16, 'MarkerFaceColor', 'r');
        text(52, nKept(idx50), sprintf('50%% threshold: %d features', nKept(idx50)), ...
            'FontSize', 11, 'Color', 'r', 'FontWeight', 'bold');
    end
    xlabel('Selection frequency threshold (%)', 'FontSize', 11);
    ylabel('Number of retained features', 'FontSize', 11);
    title('Feature Count vs Selection Frequency Threshold', ...
        'FontSize', 12, 'FontWeight', 'bold');
    grid on;
    xlim([0 105]); ylim([0, max(nKept) * 1.2 + 2]);
    saveas(f, fullfile(outDir, 'fig10_feature_threshold_curve.png'));
    savefig(f, fullfile(outDir, 'fig10_feature_threshold_curve.fig'));
    close(f);
    fprintf('  图10 特征阈值曲线已保存: fig10_feature_threshold_curve.png + .fig\n');
end

function plotBPNNParamRatio(plotClf, plotNames, maxFeat, nClasses, kOuter, outDir)
    nSamples = 20;
    nTrainPerFold = round(nSamples * (1 - 1/kOuter));
    f = figure('Position', [50 50 1000 600], 'Visible', 'off', 'Color', 'w');
    hold on;
    nAlgo = numel(plotClf);
    ratioVals = nan(1, nAlgo);
    for ci = 1:nAlgo
        cname = plotClf{ci};
        if strcmpi(cname, 'BPNN')
            nHidden = 2;
            nP = maxFeat * nHidden + nHidden + nHidden * nClasses + nClasses;
        elseif startsWith(cname, 'SVM')
            nP = maxFeat * 2;
        elseif strcmpi(cname, 'KNN')
            nP = maxFeat;
        elseif strcmpi(cname, 'RF')
            nP = 100;
        else
            nP = NaN;
        end
        ratioVals(ci) = nP / nTrainPerFold;
    end
    barH = bar(1:nAlgo, ratioVals, 0.6, 'FaceColor', 'flat');
    colMatrix = [ 0.000 0.447 0.698; 0.200 0.200 0.200; 0.000 0.620 0.451; ...
                  0.600 0.600 0.600; 0.835 0.369 0.000; 0.700 0.700 0.700 ];
    for ci = 1:nAlgo
        barH.CData(ci, :) = colMatrix(mod(ci-1, size(colMatrix,1)) + 1, :);
    end
    for ci = 1:nAlgo
        text(ci, ratioVals(ci) + 0.05, sprintf('%.2f:1', ratioVals(ci)), ...
            'HorizontalAlignment', 'center', 'FontSize', 11, 'FontWeight', 'bold');
    end
    yline(1/5, 'r--', 'LineWidth', 1.5, ...
        'Label', 'Recommended upper bound (1:5)', 'FontSize', 10, ...
        'LabelHorizontalAlignment', 'left');
    yline(1/10, 'b--', 'LineWidth', 1.5, ...
        'Label', 'Ideal (1:10)', 'FontSize', 10, ...
        'LabelHorizontalAlignment', 'left');
    set(gca, 'XTick', 1:nAlgo, 'XTickLabel', plotNames, ...
        'FontSize', 11, 'XTickLabelRotation', 20);
    ylabel('Parameter-to-sample ratio', 'FontSize', 11);
    title(sprintf('Parameter-to-Sample Ratio of Each Classifier (n_{train} = %d per fold)', ...
        nTrainPerFold), 'FontSize', 12, 'FontWeight', 'bold');
    grid on;
    saveas(f, fullfile(outDir, 'fig11_param_sample_ratio.png'));
    savefig(f, fullfile(outDir, 'fig11_param_sample_ratio.fig'));
    close(f);
    fprintf('  图11 参数/样本比图已保存: fig11_param_sample_ratio.png + .fig\n');
end

function [tprLo, tprHi] = bootstrapROCBand(yBin, score, fprGrid, nBoot)
    yBin = yBin(:); score = score(:);
    posIdx = find(yBin == 1); negIdx = find(yBin == 0);
    nPos = numel(posIdx); nNeg = numel(negIdx);
    nGrid = numel(fprGrid);
    if nPos < 2 || nNeg < 2
        tprLo = zeros(nGrid, 1); tprHi = ones(nGrid, 1); return;
    end
    tprMat = nan(nBoot, nGrid);
    for b = 1:nBoot
        bootPos = posIdx(randi(nPos, nPos, 1));
        bootNeg = negIdx(randi(nNeg, nNeg, 1));
        bootIdx = [bootPos; bootNeg];
        yb = yBin(bootIdx); sb = score(bootIdx);
        if numel(unique(yb)) < 2, continue; end
        if std(sb) < 1e-10, continue; end
        try
            [fpr, tpr] = perfcurve(yb, sb, 1);
            tprMat(b, :) = interp1(fpr, tpr, fprGrid, 'linear', NaN);
        catch, end
    end
    tprLo = quantile(tprMat, 0.025, 1)';
    tprHi = quantile(tprMat, 0.975, 1)';
    tprLo(isnan(tprLo)) = 0;
    tprHi(isnan(tprHi)) = 1;
    tprLo = max(0, min(1, tprLo));
    tprHi = max(0, min(1, tprHi));
end

function [fprGrid, tprMean, tprLo, tprHi, aucMean, nValid] = ...
        macroAvgROCWithCI(Y, score, nClasses, nGrid, nBoot)
    fprGrid = linspace(0, 1, nGrid);
    tprs    = nan(nClasses, nGrid);
    aucs    = nan(nClasses, 1);
    for c = 1:nClasses
        yBin = double(Y == c);
        if numel(unique(yBin)) < 2, continue; end
        s = score(:, c);
        if any(isnan(s)) || std(s) < 1e-10, continue; end
        try
            [fpr, tpr, ~, a] = perfcurve(yBin, s, 1);
            tprs(c, :) = interp1(fpr, tpr, fprGrid, 'linear', NaN);
            aucs(c) = a;
        catch, end
    end
    tprMean = mean(tprs, 1, 'omitnan');
    validAUC = aucs(~isnan(aucs));
    if isempty(validAUC)
        aucMean = NaN; nValid = 0; tprLo = tprMean; tprHi = tprMean; return;
    end
    aucMean = mean(validAUC); nValid = numel(validAUC);
    uniqC = unique(Y);
    clsIdx = cell(numel(uniqC), 1);
    for u = 1:numel(uniqC)
        clsIdx{u} = find(Y == uniqC(u));
    end
    bootTPR = nan(nBoot, nGrid);
    for b = 1:nBoot
        bootIdx = [];
        for u = 1:numel(uniqC)
            idx_u = clsIdx{u};
            bootIdx = [bootIdx; idx_u(randi(numel(idx_u), numel(idx_u), 1))]; %#ok<AGROW>
        end
        Yb = Y(bootIdx); Sb = score(bootIdx, :);
        if any(isnan(Sb(:))), continue; end
        tprsB = nan(nClasses, nGrid);
        for c = 1:nClasses
            yBin = double(Yb == c);
            if numel(unique(yBin)) < 2, continue; end
            s = Sb(:, c);
            if std(s) < 1e-10, continue; end
            try
                [fpr, tpr] = perfcurve(yBin, s, 1);
                tprsB(c, :) = interp1(fpr, tpr, fprGrid, 'linear', NaN);
            catch, end
        end
        bootTPR(b, :) = mean(tprsB, 1, 'omitnan');
    end
    tprLo = quantile(bootTPR, 0.025, 1)';
    tprHi = quantile(bootTPR, 0.975, 1)';
    tprLo(isnan(tprLo)) = 0;
    tprHi(isnan(tprHi)) = 1;
    tprLo = max(0, min(1, tprLo));
    tprHi = max(0, min(1, tprHi));
end

function plotAllResults(R, plotClf, plotNames, clfAll, dispAll, bestSVMName, ...
        selectFreq, featureNames, nRep, nClasses, nBootROC, outDir)
    nPlot = numel(plotClf);
    f1 = figure('Position', [30 30 1600 430], 'Visible', 'off', 'Color', 'w');
    metrics     = {'accuracy','sensitivity','specificity','auc'};
    metricNames = {'Overall Accuracy','Sensitivity','Specificity','AUC'};
    for mi = 1:numel(metrics)
        subplot(1, 5, mi); set(gca, 'Color', 'w');
        dataMat = nan(nRep, nPlot);
        for ci = 1:nPlot
            v = R.(plotClf{ci}).(metrics{mi});
            dataMat(1:numel(v), ci) = v;
        end
        boxplot(dataMat, 'Labels', plotNames);
        ylabel(metricNames{mi}); title(metricNames{mi});
        ylim([0 1.05]); grid on; set(gca, 'XTickLabelRotation', 25);
    end
    subplot(1, 5, 5); set(gca, 'Color', 'w');
    dataMat = nan(nRep, nPlot);
    for ci = 1:nPlot
        v = R.(plotClf{ci}).nFeatures;
        dataMat(1:numel(v), ci) = v;
    end
    boxplot(dataMat, 'Labels', plotNames);
    ylabel('Number of Selected Features'); title('Feature Count'); grid on;
    set(gca, 'XTickLabelRotation', 25);
    sgtitle(sprintf('Nested CV Performance (%d repeats x 5 folds) | Best SVM = %s', ...
        nRep, bestSVMName));
    saveas(f1, fullfile(outDir, 'fig1_performance_boxplot.png'));
    savefig(f1, fullfile(outDir, 'fig1_performance_boxplot.fig'));
    close(f1);
    fprintf('  图1 已保存: fig1_performance_boxplot.png + .fig\n');
    
    f2 = figure('Position', [50 50 900 700], 'Visible', 'off', 'Color', 'w');
    set(gca, 'Color', 'w');
    [sortedF, sortedI] = sort(selectFreq, 'descend');
    topK = min(20, sum(sortedF > 0));
    if topK < 1, topK = min(20, numel(sortedF)); end
    barh(topK:-1:1, sortedF(1:topK), 'FaceColor', [0.2 0.5 0.8]);
    set(gca, 'YTick', 1:topK, 'YTickLabel', featureNames(sortedI(topK:-1:1)));
    xlabel('Selection Frequency');
    title(sprintf('Top-%d Feature Stability Selection', topK));
    grid on; xlim([0 1]); hold on;
    plot([0.5 0.5], [0 topK+1], 'r--', 'LineWidth', 1.2);
    text(0.52, topK, 'Frequency = 50%', 'Color', 'r');
    saveas(f2, fullfile(outDir, 'fig2_feature_stability.png'));
    savefig(f2, fullfile(outDir, 'fig2_feature_stability.fig'));
    close(f2);
    fprintf('  图2 已保存: fig2_feature_stability.png + .fig\n');
    
    f3 = figure('Position', [50 50 750 700], 'Visible', 'off', 'Color', 'w');
    set(gca, 'Color', 'w'); hold on;
    colors = lines(nPlot); nGrid  = 200;
    for ci = 1:nPlot
        cname = plotClf{ci};
        allS = []; allL = [];
        for r = 1:nRep
            if isempty(R.(cname).allScores{r}), continue; end
            allS = [allS; R.(cname).allScores{r}]; %#ok<AGROW>
            allL = [allL; R.(cname).allLabels{r}]; %#ok<AGROW>
        end
        if isempty(allS) || any(isnan(allS(:))), continue; end
        fprintf('  [fig3] 计算 %s 宏平均 ROC 的 95%% CI...\n', cname);
        [fprGrid, tprMean, tprLo, tprHi, aucMean, nValid] = ...
            macroAvgROCWithCI(allL, allS, nClasses, nGrid, nBootROC);
        if isnan(aucMean), continue; end
        xBand = [fprGrid, fliplr(fprGrid)];
        yBand = [tprLo(:)', fliplr(tprHi(:)')];
        if strcmpi(cname, 'BPNN')
            fill(xBand, yBand, [0.7 0 0], 'FaceAlpha', 0.12, ...
                'EdgeColor', 'none', 'HandleVisibility', 'off');
            plot(fprGrid, tprMean, 'Color', [0.7 0 0], 'LineWidth', 2, ...
                'LineStyle', '--', ...
                'DisplayName', sprintf('%s (AUC = %.3f, n=%d/%d, Exploratory)', ...
                    plotNames{ci}, aucMean, nValid, nClasses));
        else
            fill(xBand, yBand, colors(ci,:), 'FaceAlpha', 0.12, ...
                'EdgeColor', 'none', 'HandleVisibility', 'off');
            plot(fprGrid, tprMean, 'Color', colors(ci,:), 'LineWidth', 2, ...
                'DisplayName', sprintf('%s (AUC = %.3f, n=%d/%d)', ...
                    plotNames{ci}, aucMean, nValid, nClasses));
        end
    end
    plot([0 1], [0 1], 'k--', 'LineWidth', 1, 'HandleVisibility', 'off');
    xlabel('1 - Specificity', 'FontSize', 11);
    ylabel('Sensitivity', 'FontSize', 11);
    title(sprintf(['Macro-Averaged ROC with 95%% Bootstrap CI\n', ...
        '(Best SVM = %s, %d resamples)'], bestSVMName, nBootROC), 'FontSize', 11);
    legend('Location', 'southeast', 'FontSize', 9);
    grid on; axis square; xlim([0 1]); ylim([0 1]);
    text(0.02, 0.02, 'Shaded = 95% Bootstrap CI', ...
        'FontSize', 8, 'Color', [0.4 0.4 0.4], 'Units', 'normalized');
    saveas(f3, fullfile(outDir, 'fig3_ROC_comparison_CI.png'));
    savefig(f3, fullfile(outDir, 'fig3_ROC_comparison_CI.fig'));
    close(f3);
    fprintf('  图3 已保存: fig3_ROC_comparison_CI.png + .fig\n');
    
    f4 = figure('Position', [50 50 1200 400], 'Visible', 'off', 'Color', 'w');
    svmIdx = find(startsWith(clfAll, 'SVM_'));
    nSVM = numel(svmIdx); svmNames = dispAll(svmIdx);
    metricsPlot = {'accuracy','auc','sensitivity','specificity'};
    titlesPlot  = {'Overall Accuracy','AUC','Sensitivity','Specificity'};
    for mi = 1:4
        subplot(1, 4, mi); set(gca, 'Color', 'w');
        dataMat = nan(nRep, nSVM);
        for si = 1:nSVM
            v = R.(clfAll{svmIdx(si)}).(metricsPlot{mi});
            dataMat(1:numel(v), si) = v;
        end
        boxplot(dataMat, 'Labels', svmNames);
        title(titlesPlot{mi}); ylim([0 1.05]); grid on;
        set(gca, 'XTickLabelRotation', 25);
        for si = 1:nSVM
            if strcmp(svmNames{si}, bestSVMName)
                text(si, 1.02, '*', 'HorizontalAlignment', 'center', ...
                    'Color', [0.85 0 0], 'FontSize', 18, 'FontWeight', 'bold');
            end
        end
    end
    sgtitle(sprintf('Three SVM Kernels (* = Best: %s)', bestSVMName));
    saveas(f4, fullfile(outDir, 'fig4_SVM_kernel_comparison.png'));
    savefig(f4, fullfile(outDir, 'fig4_SVM_kernel_comparison.fig'));
    close(f4);
    fprintf('  图4 已保存: fig4_SVM_kernel_comparison.png + .fig\n');
end