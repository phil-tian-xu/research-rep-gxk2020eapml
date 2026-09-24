1. Hyperparameter searches: exactly what is missing
Table A.5, appendix p. 13, is the primary source.
Method	What is specified	What is missing practically	Status
Elastic net	$\rho=0.5$; $\lambda\in(10^{-4},10^{-1})$	Exact candidate values, number of candidates, spacing, and endpoint inclusion	Partial
GLM/group lasso	Three knots; $\lambda\in(10^{-4},10^{-1})$	Exact penalty grid	Partial
PCR	Select number of components $K$	Candidate range and grid; whether $K=0$ is allowed and what it predicts	Missing
PLS	Select number of components $K$	Candidate range and grid; whether $K=0$ is allowed and what it predicts	Missing
Random forest	Depth 1–6; 300 trees; features per split $\{3,5,10,20,30,50,\ldots\}$	The remainder of the feature-count grid after the ellipsis	Partial
GBRT	Depth 1–2; 1–1,000 trees; learning rates $\{0.01,0.1\}$	Whether every tree count is evaluated or checkpoints are used	Partial, relatively minor
NN1–NN5	$\lambda_1\in(10^{-5},10^{-3})$; learning rates $\{0.001,0.01\}$	Exact $L_1$-penalty grid	Partial
Standalone lasso and ridge	Methods and penalties are described; both appear in simulations	Their own search grids are not separately supplied in Table A.5	Missing for simulation replication


For example, the ENet range is provided. What is missing is whether the search uses something like 4, 10, or 100 logarithmically spaced candidates. Those are different experiments even though they share the same endpoints.
The standalone lasso/ridge gaps matter if we replicate the Monte Carlo appendix. They do not add separate model columns to the main 13-model empirical exercise.
2. Shared preprocessing and estimation gaps
These affect several model families simultaneously.
Item	What the documents provide	What remains unresolved	Why it matters
Feature scaling	Monthly rank transformation of stock characteristics; 920-feature construction	Whether macro variables or completed interaction columns receive additional scaling; the exact order and estimation window of any such transformations	Changes penalties, component extraction, gradients, and NN optimization
Return centering and intercepts	Linear prediction equations; explicit NN biases	Complete operational treatment of return demeaning, intercept recovery, and intercept penalization for linear/component models	Can change the level of predicted returns and effective regularization
Industry indicators	74 SIC indicators	Treatment of missing/unseen industries, redundant indicators, and any separate intercept	Affects the design matrix and identification
Ranking ties	Rank transformation	Tie-breaking convention, especially for discrete characteristics	Different rank conventions produce different predictors
Imputation order	Monthly cross-sectional median replacement	Whether imputation precedes ranking; how all-missing characteristic-months are handled	Imputing before ranking can alter the ranks of observed values
Unbalanced-panel weighting	Pooled loss in the main text; monthly-average formulation in Appendix B.3	Whether each stock-month or each month receives equal aggregate influence in the actual NN implementation	Large and small cross-sections receive different weights


There is an important qualification about ranking. Appendix A.1 supplies
$$
c_{ij,t}=\frac{2}{N+1}\operatorname{CSrank}(\bar c_{ij,t})-1
$$and says this matches the empirical cleaning procedure. Thus, the basic rank-scaling formula is available. The remaining gaps concern ties, missing observations, and the practical definition of the ranking cross-section.
The weighting ambiguity can be written explicitly:
$$
L_{\text{observations}}
=
\frac{1}{\sum_tN_t}\sum_t\sum_i\ell_{i,t},
\qquad
L_{\text{months}}
=
\frac{1}{T}\sum_t\frac{1}{N_t}\sum_i\ell_{i,t}.
$$These coincide when $N_t$ is constant, but generally differ otherwise. The documents do not fully reconcile the two formulations for the actual unbalanced panel.
References: main Sections 1.2 and 2.1, pp. 2233 and 2248–2249; Appendix A.1 and B.3.
3. Huber loss: the threshold is only partially specified
This affects OLS+H, OLS-3+H, ENet+H, GLM+H, and GBRT+H.
Provided: The Huber formula and the statement that $\xi$ is the 99.9% quantile.
Missing practically:
Question	Why an answer is needed
Quantile of what—returns, absolute returns, residuals, or absolute residuals?	These produce different thresholds
If residuals, from which preliminary or current model?	The threshold then depends on another estimation step
Computed from which observations?	Determines the information set and potential leakage
Fixed during optimization or updated?	Changes the estimation algorithm
Recomputed for each annual fit, hyperparameter candidate, or boosting iteration?	Changes robustness across models and years
Does validation use the same threshold and loss?	Can change which configuration wins


It would be too strong to say “the Huber threshold is unspecified.” The quantile level is specified; its operational construction is not.
References: main Section 1.2.1, pp. 2233–2234; Table A.5.
4. OLS, lasso, ridge, and elastic net
Beyond the shared issues and grids, the main gaps concern numerical optimization.
Item	Provided	Missing or ambiguous
APG starting point	Algorithm 1 gives $\theta_0=0$	This is specified and should not be listed as missing
APG step size $\gamma$	Appears in Algorithm 1	Numerical value or rule for choosing it; whether backtracking is used
Convergence	Iterate until convergence	Criterion, tolerance, and iteration limit
Intercept handling	Prediction and penalty equations	How a separate intercept, if used, is excluded from or included in penalization
Singular design matrices	Full linear predictor set	Numerical handling of collinearity and redundant columns
Huber linear solver	Huber objective	Detailed numerical implementation for the unpenalized robust benchmarks


A mathematical inconsistency also needs attention. Appendix p. 7 prints the lasso proximal operator as
$$
\lambda S(\theta,\lambda\gamma).
$$Given the penalty and proximal definition on the preceding page, the corresponding operator is
$$
S(\theta,\lambda\gamma).
$$The extra leading $\lambda$ appears to be a typographical error. This conclusion follows directly from the stated optimization problem. A replication should resolve that inconsistency rather than copy the printed expression mechanically.
The ENet mixing parameter does not need to be invented: use the empirical setting $\rho=0.5$ reported in Table A.5, while noting that the main discussion describes the more general possibility of tuning it.
References: main Sections 1.2–1.3; Appendix B.1, Algorithm 1, and Table A.5.
5. PCR and PLS
Item	Provided	Missing practically
Component count	$K$ selected using validation	Exact grid
Predictor standardization	Covariance-based component objectives	Whether features are additionally standardized to unit variance
Centering and prediction reconstruction	Covariance definitions involve centering	Complete sequence for centering $X$ and $y$, fitting the final regression, and recovering the prediction intercept
PLS implementation	Conceptual construction and mention of SIMPLS	Exact algorithm implementation and normalization/deflation conventions
Degenerate components	General component optimization	Handling of zero variance, rank deficiency, and requests for more components than the usable rank


The population/sample objectives are described. The gap is the complete numerical recipe and search space.
For example, running PCA on a covariance matrix and running PCA after unit-variance standardization need not produce the same components. Rank-transforming the underlying stock characteristics does not automatically make all 920 columns equally scaled.
References: main Section 1.4, pp. 2235–2237; Table A.5.
6. GLM with splines and group lasso
This has one of the largest specification gaps.
Item	Provided	Missing practically
Spline family	Order-two spline description	Fully explicit piecewise basis
Knots	Three knots	Locations; evenly spaced versus quantile-based; treatment of boundary knots
Knot estimation	No complete operational rule	Which sample determines knots and whether they change at annual refits
Out-of-range inputs	No explicit rule	Extrapolation outside the training support
Basis preprocessing	Group-lasso penalty on basis coefficients	Centering/scaling or orthogonalization of basis columns
Constants	Printed basis includes a constant	How repeated constants across feature groups and a global intercept are handled
Indicator inputs	General expansion applies to predictors	Whether SIC and binary characteristics are expanded or treated separately
Penalty search and optimization	Range and APG	Exact grid and numerical controls


The printed basis is
$$
\left(1,z,(z-c_1)^2,\ldots,(z-c_{K-2})^2\right).
$$Taken literally, these shifted squares span global quadratic functions; they do not introduce knot-specific piecewise behavior. A truncation or other piecewise convention therefore needs clarification to implement the intended spline model.
This is more consequential than choosing a solver tolerance: different interpretations can produce different model classes.
References: main Section 1.5, p. 2238; Appendix B.1 and Table A.5.
7. Random forests
The core forest design is well described. The remaining issues are narrower.
Item	Provided	Missing practically
Feature subsampling	Random subset at every split; partial candidate grid	Complete feature-count grid
Bootstrap	Bootstrap samples of training observations	Sample size and fully explicit sampling protocol
Tree stopping	Depth restriction	Minimum leaf/split sizes and handling of nodes with no valid split
Split search	Minimize squared-error impurity	Exact versus approximate threshold search; handling of tied optimal splits
Randomness	Bootstrap and feature subsampling	Seeds and random-number implementation


300 trees, depth 1–6, squared-error impurity, and averaging predictions are specified.
Some missing details have conventional defaults. For instance, ordinary bootstrap sampling suggests sampling observations with replacement. These are smaller reproducibility gaps than the incomplete feature-count grid; they do not justify assuming an undocumented time-block bootstrap.
References: main Section 1.6; Appendix Algorithms 2–3 and Table A.5.
8. Gradient-boosted regression trees
Item	Provided	Missing practically
Huber threshold	99.9% quantile	Its definition and update schedule
Weak-tree implementation	Fit a shallow tree to negative gradients	Leaf-size restrictions, invalid-split handling, and threshold-search details
Tree-count selection	1–1,000 trees	Exact validation-check schedule
Numerical reproducibility	Sequential boosting algorithm	Tie-breaking and implementation details


Several potentially important choices are already specified:
- Initial prediction is zero.
- Trees fit the negative loss gradient.
- Updates use the reported learning-rate multiplier.
- Depth is 1 or 2.
- Learning rates are 0.01 or 0.1.
Algorithm 4 does not describe an additional leafwise line search. That should not be treated as a missing mandatory step: adding one would change the stated algorithm.
Similarly, evaluating every integer tree count from 1 to 1,000 is a natural reading of the table. The absence of a detailed checking schedule is a relatively minor gap.
References: Appendix Algorithms 2 and 4; Table A.5.
9. Neural networks
The architecture is relatively complete. The largest gaps are in the training procedure.
Item	Provided	Missing practically	Likely consequence
$L_1$ search	Range $10^{-5}$–$10^{-3}$	Exact grid	Different selected regularization
Penalty coverage	Penalization of weight parameters	Whether it includes biases, output-layer parameters, and BN scale/shift	Different optimization objective
Initialization	Different random initializations across ensemble members	Distribution, scale, bias initialization, and seeds	Different optimization paths
Adam constants	“Default” parameters	Explicit numerical $\beta_1,\beta_2,\epsilon$ and implementation version	Defaults may differ across implementations
Batch formation	Random sampling; batch size 10,000	With/without replacement, epoch shuffling, final incomplete batch, observation-versus-month sampling	Changes gradients and batch statistics
Early-stopping frequency	Patience 5; Algorithm 6 checks after $h$ updates	Value of $h$	Five checks could correspond to very different training lengths
Early-stopping score	Validation prediction error	Exact aggregation and whether regularization is included	Different retained checkpoint
BN numerical settings	Normalization formula	Numerical $\epsilon$, initialization details	Different activations and optimization
BN inference statistics	Training-batch normalization	How population/running moments are estimated and used for validation/test predictions	Predictions can depend on batch composition
Ensemble tuning	Ten networks; average predictions	Whether hyperparameters are shared or selected separately by seed	Different final ensemble
Ensemble selection score	Validation-based tuning generally	Score averaged predictions or average member losses?	Different hyperparameter selection
Annual NN refits	Refit each year	Whether previous-year weights/optimizer state are reused or each fit starts fresh	Different optimization history


Two of these are especially consequential.
Batch normalization at prediction time. Appendix Algorithm 7 describes the training-batch transform. It does not fully specify how the authors obtain the statistics used for validation and testing. Using stored training statistics and using the current prediction batch’s statistics are different procedures.
Ensemble selection. These are generally different validation criteria:
$$
\operatorname{MSE}\!\left(y,\frac{1}{10}\sum_{s=1}^{10}\hat y_s\right)
\quad\text{and}\quad
\frac{1}{10}\sum_{s=1}^{10}\operatorname{MSE}(y,\hat y_s).
$$The first evaluates the ensemble forecast. The second evaluates average individual-network performance. The paper does not fully specify which criterion governs hyperparameter selection.
Conversely, the patience comparison itself is specified: Algorithm 6 uses a strict improvement in validation error and returns the best recorded parameters. There is no need to invent a missing positive min_delta.
References: main Section 1.7, pp. 2242–2246; Appendix B.3, Algorithms 5–7, and Table A.5.
10. Training, validation, and testing
The broad chronology is clear. These operational details remain:
Item	Provided	Missing practically
Validation loss	Select hyperparameters using validation forecast errors	Exact scoring rule for each method, particularly Huber-trained models
Validation weighting	General objective descriptions	Stock-month weighting versus equal-month weighting in the unbalanced panel
Search execution	Compare hyperparameter configurations	Complete grids and whether all combinations are evaluated
Ties in validation performance	No explicit rule	Prefer simpler model, first candidate, or another rule
Month labels at boundaries	$z_{i,t}$ predicts $r_{i,t+1}$; year ranges given	Fully explicit indexing convention for assigning predictor–target pairs to calendar windows
Annual-horizon experiments	Annual prediction results	Target construction, overlapping versus nonoverlapping observations, and handling targets crossing split boundaries


The month-label issue is an implementation detail, not permission to shift dates arbitrarily. A December predictor paired with a January return must be assigned consistently with the intended forecast origin and the information available when the model is fitted.
These parts are already specified:
- Initial training: 1957–1974.
- Initial validation: 1975–1986.
- Out-of-sample evaluation: 1987–2016.
- Expanding training window.
- Rolling 12-year validation window.
- Annual refitting.
- Parameter estimation from training data alone.
Thus, “whether to refit on training plus validation” is not an open choice in the stated procedure. Doing so would depart from Section 1.1.
References: main Sections 1.1 and 2.1; Appendix D.
11. Additional gaps if we replicate the diagnostic tables and figures
These are separate from fitting the prediction models.
Output	Provided	Missing practically
Diebold–Mariano tests	Monthly cross-sectional loss differences; Newey–West standard errors	Lag/bandwidth choice and finite-sample correction settings
Characteristic importance	Zero a predictor while holding model estimates fixed	Exact aggregation from 920 features to 94 characteristics and industry importance
Interaction importance	Separate interaction analysis	Whether characteristic-level importance jointly removes all associated interactions or aggregates featurewise scores
Normalized importance	Normalize importance to sum to one	Treatment of negative importance values and precise normalization/averaging order
NN ensemble importance	Models use ten-member ensembles	Importance of the averaged prediction versus average member importance
Annual results	Reported annual $R^2$ values	Full target and split-boundary implementation, as noted above


For importance, jointly zeroing a characteristic and all its macro interactions need not give the same result as summing the effects of zeroing each feature separately.
References: main Sections 1.8–1.9 and 2.3, pp. 2246–2247 and 2254 onward.
12. Things we should not incorrectly classify as missing
Item	Correct reading
NN dropout rate	NN dropout is not reported as part of the method; there is no dropout rate to recover
NN depth and width	Explicitly specified for NN1–NN5
Hidden activation	ReLU
Output activation	Linear scalar output
BN placement	After ReLU
NN ensemble size	10
NN batch size	10,000
NN patience	5 validation checks; checking frequency remains unspecified
Additional learning-rate scheduler	No separate scheduler is established beyond the described Adam procedure
Elastic-net mixing weight	Empirical table gives $\rho=0.5$
Random-forest size	300 trees
GBRT initialization	Zero
Stock-level test $R^2$ benchmark	Zero excess return, without demeaning


The practical order I would resolve these gaps in is:
1. Preprocessing, centering, and panel weighting, because they affect almost every model.
2. Exact search grids and validation scores, because they determine model selection.
3. Huber-threshold construction, because it affects several headline specifications.
4. Spline basis and knots, because the GLM is not uniquely implementable as printed.
5. NN penalty coverage, initialization, BN inference, and ensemble tuning.
6. Numerical tolerances, random seeds, and smaller tree implementation details.
7. Annual-horizon and diagnostic details, when those outputs enter the replication scope.
A setting inherited from a software default should be recorded as our replication choice unless it can be traced to the authors’ implementation.
- Choose defaults for the gaps
- Check the authors’ implementation
- Prioritize sensitivity checks