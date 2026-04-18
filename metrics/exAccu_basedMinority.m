function exAccu = exAccu_basedMinority(label, pred)
% label/pred: M×Q, 元素取值为 +1/-1
% example-based Accuracy（Jaccard）按“每个标签列的少数类作为正类”计算
% 约定：当某样本真实集合和预测集合都为空时，该样本 accuracy 记为 1

    if ~isequal(size(label), size(pred))
        error('label/pred 尺寸不一致');
    end
    if any(label(:) ~= 1 & label(:) ~= -1) || any(pred(:) ~= 1 & pred(:) ~= -1)
        error('label/pred 必须为 +1/-1 编码');
    end

    [M, Q] = size(label); %#ok<ASGLU>

    % 每个标签列中，选择少数类作为“正类”
    posCount = sum(label == 1, 1);
    negCount = sum(label == -1, 1);

    posValVec = ones(1, Q);
    posValVec(posCount > negCount) = -1;   % 若 -1 更少，则以 -1 为正
    % 平票时默认 +1 为正

    % 构造“少数类为正类”下的真实集合与预测集合
    Y  = bsxfun(@eq, label, posValVec);
    Yh = bsxfun(@eq, pred,  posValVec);

    % 每个样本的交集与并集大小
    tp  = sum(Y & Yh, 2);          % |Y ∩ Ŷ|
    py  = sum(Yh, 2);              % |Ŷ|
    ty  = sum(Y,  2);              % |Y|
    uni = py + ty - tp;            % |Y ∪ Ŷ|

    % example-based Accuracy / Jaccard
    Ji = zeros(M, 1);
    idxNonEmpty = (uni > 0);
    Ji(idxNonEmpty) = tp(idxNonEmpty) ./ uni(idxNonEmpty);

    % 空集-空集样本记为 1
    Ji(py == 0 & ty == 0) = 1;

    % 最终 exAccu
    exAccu = mean(Ji);
end