function run_glm_reference(run_dir, author_dir)
% Run the AUTHORS' functions on Python's shared fixture, without MC_Reg/DGP.
% author_dir must point to the downloaded ML_Codes/Simu_Matlab directory.
% Reference: Gu, Kelly and Xiu, Empirical Asset Pricing via Machine Learning.

assert(isfile(fullfile(run_dir, 'fixture.mat')), 'Prepare the Python fixture first.');
output = fullfile(run_dir, 'matlab_results.mat');
assert(~isfile(output), 'MATLAB output already exists. Prepare a fresh comparison directory.');
[ok, attributes] = fileattrib(author_dir);
assert(ok && attributes.directory, 'author_dir must be an existing Simu_Matlab directory.');
author_dir = attributes.Name;
old_path = path;
cleanup = onCleanup(@() path(old_path)); %#ok<NASGU>
addpath(author_dir, '-begin');
names = {'cut_knots_degree2', 'proximal', 'proximalH', 'f_grad', ...
         'f_gradh', 'soft_threshodg', 'lossh', 'fw1'};
r = struct();
for j = 1:numel(names)
    name = names{j};
    expected = fullfile(author_dir, [name '.m']);
    assert(isfile(expected), ['Missing author function: ' expected]);
    assert(strcmpi(which(name), expected), ['Another function shadows ' name]);
    r.author_sources.(name) = fileread(expected);
end
r.matlab_version = version;
r.fixture = load(fullfile(run_dir, 'fixture.mat'));
f = r.fixture;
r.raw_scale = std(f.X_train, 0, 1);
r.raw_scale(r.raw_scale <= 0) = 1;
train = f.X_train ./ r.raw_scale;
validation = f.X_validation ./ r.raw_scale;
test = f.X_test ./ r.raw_scale;
r.knots = zeros(4, size(train, 2));
for j = 1:size(train, 2)
    r.knots(:, j) = quantile(train(:, j), f.knot_probabilities);
end
% Preserve the original helper's separate centering of all three datasets.
r.basis_train = cut_knots_degree2(train, 4, r.knots);
r.basis_validation = cut_knots_degree2(validation, 4, r.knots);
r.basis_test = cut_knots_degree2(test, 4, r.knots);
r.basis_scale = std(r.basis_train, 0, 1);
r.basis_scale(r.basis_scale <= 0) = 1;
r.Z_train = r.basis_train ./ r.basis_scale;
r.Z_validation = r.basis_validation ./ r.basis_scale;
r.Z_test = r.basis_test ./ r.basis_scale;
r.target_mean = mean(f.y_train);
y = f.y_train - r.target_mean;
XX = r.Z_train' * r.Z_train;
XY = r.Z_train' * y;
[~, S, ~] = svd(XX);
r.lipschitz = S(1);
nc = size(train, 2);
groups = repelem(1:nc, 5);
r.coefficient_path = zeros(size(XX, 1), numel(f.lambdas));
r.validation_r2_path = zeros(numel(f.lambdas), 1);
for j = 1:numel(f.lambdas)
    b = proximal(groups, nc, XX, XY, f.tol, r.lipschitz, f.lambdas(j), @soft_threshodg);
    r.coefficient_path(:, j) = b;
    prediction = r.Z_validation * b + r.target_mean;
    r.validation_r2_path(j) = 1 - sum((prediction - f.y_validation).^2) / sum((f.y_validation - r.target_mean).^2);
end
r.selected_index = fw1(r.validation_r2_path);
r.selected_lambda = f.lambdas(r.selected_index);
r.beta_squared = r.coefficient_path(:, r.selected_index);
r.beta_huber = proximalH(groups, nc, r.Z_validation, r.target_mean, f.y_validation, ...
    r.beta_squared, r.Z_train, y, f.delta, f.tol, r.lipschitz, r.selected_lambda, @soft_threshodg);
r.prox_probe = soft_threshodg(repelem(1:2, 5), 2, f.probe_values, f.probe_threshold);
methods = {'squared', 'huber'};
splits = {'train', 'validation', 'test'};
for m = 1:numel(methods)
    method = methods{m};
    b = r.(['beta_' method]);
    norms = sqrt(sum(reshape(b, 5, nc).^2, 1))';
    r.(['group_norms_' method]) = norms;
    residual = r.Z_train * b - y;
    if strcmp(method, 'squared')
        smooth = .5 * sum(residual.^2);
        gradient = f_grad(XX, XY, b);
    else
        smooth = .5 * numel(y) * lossh(r.Z_train * b, y, f.delta);
        gradient = f_gradh(b, r.Z_train, y, f.delta);
    end
    r.(['objective_' method]) = smooth + r.selected_lambda * sum(norms);
    mapping = r.lipschitz * (b - soft_threshodg(groups, nc, ...
        b - gradient / r.lipschitz, r.selected_lambda / r.lipschitz));
    r.(['stationarity_' method]) = max(abs(mapping));
    for s = 1:numel(splits)
        split = splits{s};
        prediction = r.(['Z_' split]) * b + r.target_mean;
        target = f.(['y_' split]);
        r.(['prediction_' method '_' split]) = prediction;
        r.(['r2_' method '_' split]) = 1 - sum((prediction - target).^2) / sum((target - r.target_mean).^2);
    end
end
save(output, '-struct', 'r', '-v7');
fprintf('Saved MATLAB reference: %s\nSelected lambda: %.8g\n', output, r.selected_lambda);
end
