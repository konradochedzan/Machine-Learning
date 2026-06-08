# Machine Learning

This repository contains three student projects completed for the Machine Learning course at ETH. Each project is implemented as a Jupyter notebook and combines mathematical modelling, numerical implementation, model evaluation, and interpretation of results. The notebooks cover three separate parts of applied machine learning: linear regression and regularization, probabilistic credit-risk classification, and neural-network-based option hedging.

The repository is not structured as a software package. It is a compact collection of course notebooks where the main value is the modelling logic: how the statistical or financial problem is written mathematically, how the corresponding estimator is implemented in Python, and how the numerical output is interpreted.

## `Linear model.ipynb`

This notebook studies supervised regression for house-price prediction. The response variable is `SalePrice`, loaded from `sample_data/Housing.csv`, and the main mathematical task is to approximate a transformed price variable by a linear function of housing features. The notebook first checks whether the raw target is compatible with the Gaussian-error assumptions commonly used in linear regression. This is done by comparing the empirical distribution of `SalePrice` with a normal distribution using histograms and QQ-plots.

The target transformation is based on the Box-Cox family

$$
y^{(\lambda)} =
\begin{cases}
\dfrac{y^\lambda - 1}{\lambda}, & \lambda \neq 0, \\
\log(y), & \lambda = 0.
\end{cases}
$$

The fitted Box-Cox parameter is approximately `lambda = -0.077`, which is close to zero, so the notebook proceeds with the simpler log transformation

$$
Y = \log(\mathrm{SalePrice}).
$$

The purpose of this step is not cosmetic. In the usual linear model

$$
Y_i = \beta_0 + \beta_1 X_{i1} + \cdots + \beta_d X_{id} + \varepsilon_i,
$$

one typically wants the residuals `epsilon_i` to be closer to centered, homoscedastic, approximately Gaussian noise. The log transformation reduces the right skew of house prices and makes squared-error regression more appropriate. A Shapiro-Wilk test is also computed after the transformation; the p-value remains small, so the notebook treats normality as an approximation rather than an exact assumption.

The preprocessing has two branches. Numerical missing values are filled with column means. Categorical missing values are separated into two cases: for variables where `"NA"` represents a real category, such as absence of a basement, garage, pool, fence, or alley access, `"NA"` is kept as a categorical value; for ordinary missing categorical observations, the mode is used. Categorical variables are then one-hot encoded, producing a full feature matrix for the later regularized models. A numerical-only dataframe `Housing2` is also created for the first regression experiments.

The first estimator is ordinary least squares on the numerical features. With an augmented design matrix

$$
A = [\mathbf{1}, X],
$$

where the first column is the intercept, the least-squares estimator is

$$
\hat{\beta}
= \arg\min_{\beta} \lVert y - A\beta \rVert_2^2
= (A^\top A)^{-1} A^\top y,
$$

when `A^T A` is invertible and numerically stable. Predictions are

$$
\hat{y} = A\hat{\beta}.
$$

The notebook evaluates models with mean squared error

$$
\mathrm{MSE}
= \frac{1}{m}\sum_{i=1}^{m}(y_i - \hat{y}_i)^2
$$

and coefficient of determination

$$
R^2
= 1 -
\frac{\sum_i (y_i - \hat{y}_i)^2}
{\sum_i (y_i - \bar{y})^2}.
$$

A 70/30 train-test split is used with a fixed random seed. For the numerical-only linear model, the notebook obtains approximately `MSE = 0.0225`, `R^2 = 0.8549` in sample and `MSE = 0.0188`, `R^2 = 0.8892` out of sample.

The notebook also manually computes standard errors for the estimated coefficients. In the homoscedastic linear model, if

$$
\varepsilon \sim \mathcal{N}(0, \sigma^2 I),
$$

then

$$
\operatorname{Var}(\hat{\beta})
= \sigma^2 (A^\top A)^{-1}.
$$

The residual variance is estimated by

$$
\hat{\sigma}^2
= \frac{\mathrm{RSS}}{m - d - 1},
\qquad
\mathrm{RSS}
= \sum_i (y_i - \hat{y}_i)^2,
$$

so the standard error of coefficient `beta_j` is

$$
\operatorname{SE}(\hat{\beta}_j)
= \sqrt{\hat{\sigma}^2 \left[(A^\top A)^{-1}\right]_{jj}}.
$$

This part of the notebook exposes a numerical issue: directly inverting `A^T A` is unstable because the matrix has an eigenvalue close to zero. This means the condition number is large and small numerical perturbations can produce a poor estimate. The notebook therefore switches to the Moore-Penrose pseudoinverse. In singular value form,

$$
A = U\Sigma V^\top,
\qquad
A^+ = V\Sigma^+ U^\top,
$$

where `Sigma^+` reciprocates nonzero singular values. The implemented thresholded version discards singular values below a cutoff `t`, which regularizes the inversion by suppressing nearly singular directions. Using the pseudoinverse restores the expected numerical behaviour and gives the same in-sample quality as the stable linear model.

The notebook then examines polynomial feature expansion. For degree `k`, the original feature vector `x` is mapped into a larger vector `phi_k(x)` containing monomials and interaction terms up to degree `k`. The fitted model becomes

$$
Y_i = \beta^\top \phi_k(X_i) + \varepsilon_i.
$$

The quadratic model fits the training data much better but performs much worse on the test set, and the cubic model essentially memorizes the training data while losing predictive power out of sample. This is a direct demonstration of overfitting: the hypothesis class becomes expressive enough to reduce training error but too unstable to generalize.

The final part uses the full one-hot-encoded feature space and compares unregularized and regularized linear models. Ridge regression solves

$$
\min_{\beta}
\left\{
\lVert y - X\beta \rVert_2^2
+ \alpha \lVert \beta \rVert_2^2
\right\},
$$

which shrinks coefficients continuously and stabilizes correlated features. LASSO solves

$$
\min_{\beta}
\left\{
\lVert y - X\beta \rVert_2^2
+ \alpha \lVert \beta \rVert_1
\right\},
$$

which uses the `L^1` penalty to promote exact sparsity, often setting many coefficients to zero. The custom pseudoinverse estimator also has a hyperparameter `t`, the singular-value threshold. The notebook tunes `alpha` or `t` over a logarithmic grid using 8-fold cross-validation, minimizing average test-fold MSE. The final comparison shows that the categorical variables carry useful signal: full-feature models outperform the numerical-only baseline. LASSO gives the best out-of-sample result in the notebook, approximately `MSE = 0.0140`, `R^2 = 0.9173`, while using only 88 nonzero coefficients. This is the main modelling conclusion: the best practical model is not the most flexible unregularized model, but a sparse regularized linear model that balances prediction accuracy and interpretability.

## `Credit Analytics.ipynb`

This notebook studies binary classification and credit-risk modelling. The data are synthetic, so the true data-generating mechanism is known. There are `m = 20000` training observations and `n = 10000` test observations. The explanatory variables are generated as

$$
X_1 \sim \operatorname{Uniform}(18, 80),
\qquad
X_2 \sim \operatorname{Uniform}(1, 15),
\qquad
X_3 \sim \operatorname{Bernoulli}(0.1).
$$

Qualitatively, these represent applicant characteristics such as age-like information, financial capacity, and a binary risk flag. The notebook constructs two repayment datasets. In both cases, a borrower repays when a uniform random variable `ksi` falls below a feature-dependent repayment probability. Equivalently,

$$
Y = \mathbf{1}\{\xi < p(X)\},
\qquad
\xi \sim \operatorname{Uniform}(0, 1).
$$

The first probability model is logistic and essentially linear in the features:

$$
p_1(x)
= \operatorname{sigmoid}(13.3 - 0.33x_1 + 3.5x_2 - 3x_3),
$$

where

$$
\operatorname{sigmoid}(z)
= \frac{1}{1 + \exp(-z)}.
$$

The second probability model is more nonlinear because it penalizes extreme ages:

$$
p_2(x)
= \operatorname{sigmoid}\left(
5
- 10\left(\mathbf{1}\{x_1 < 25\} + \mathbf{1}\{x_1 > 75\}\right)
+ 1.1x_2
- x_3
\right).
$$

This construction creates a useful contrast. Dataset 1 is naturally well matched to logistic regression because the log-odds are linear in `x`. Dataset 2 contains threshold effects through indicator functions, so a more flexible nonlinear classifier should have an advantage.

The first model class is logistic regression. It estimates

$$
q_\beta(x)
= \mathbb{P}(Y = 1 \mid X = x)
= \operatorname{sigmoid}(\beta_0 + \beta^\top x)
$$

by minimizing negative conditional log-likelihood, also called cross-entropy loss:

$$
L(\beta)
= -\sum_i
\left[
y_i \log(q_\beta(x_i))
+ (1-y_i)\log(1-q_\beta(x_i))
\right].
$$

The notebook fits separate logistic models for the two datasets and evaluates the predicted probabilities on train and test samples. As expected, logistic regression performs well on the first dataset and less well on the second, because the second has a nonlinear structure that is not directly representable by a linear logit.

The second model class is an RBF-kernel support vector classifier. Before fitting, the features are normalized by their training-sample standard deviations, and the same scaling is applied to the test set. This is important because the RBF kernel depends on Euclidean distance:

$$
K(x, x')
= \exp\left(-\gamma \lVert x - x' \rVert_2^2\right).
$$

The SVM is trained with hinge-loss geometry in the reproducing kernel Hilbert space associated with `K`, and `probability=True` is used to obtain estimated repayment probabilities from the classifier. The notebook uses `gamma = 1/10`; for the first SVM it also derives `C` from the regularization parameter used in the course notation. The RKHS model is more flexible than logistic regression because nonlinear decision boundaries in the original feature space become linear boundaries in the implicit feature space.

Model quality is assessed not only by cross-entropy, but also by threshold-based classification curves. For a threshold `c`, applicants are predicted as repayers when

$$
\hat{p}(x) > c.
$$

The notebook computes

$$
\operatorname{TP}(c)
= \#\{i : y_i = 1 \ \mathrm{and}\ \hat{p}_i > c\},
\qquad
\operatorname{FP}(c)
= \#\{i : y_i = 0 \ \mathrm{and}\ \hat{p}_i > c\},
$$

then

$$
\operatorname{TPR}(c)
= \frac{\operatorname{TP}(c)}{\#\{i : y_i = 1\}},
\qquad
\operatorname{FDR}(c)
= \frac{\operatorname{FP}(c)}
{\operatorname{FP}(c) + \operatorname{TP}(c)}.
$$

The curve is plotted as false discovery rate against true positive rate across 100 thresholds. Its area is approximated by the trapezoidal rule. In this notebook, smaller area is better because the desired curve has high true-positive rate while keeping false discoveries low. The output reflects the data-generating mechanisms: on dataset 1, logistic regression and RKHS SVM are very close; on dataset 2, the nonlinear RKHS model is substantially better. The reported AUC values are approximately `0.00810` for logistic regression on dataset 2 and `0.000865` for the RKHS SVM on dataset 2.

The last part turns predicted repayment probabilities into a lending strategy. A loan is granted only if the estimated repayment probability exceeds 95 percent:

$$
\text{accept applicant } i
\quad \Longleftrightarrow \quad
\hat{p}_i \geq 0.95.
$$

The notebook compares three strategies. Strategy 1 accepts all applicants and charges a higher interest rate. Strategy 2 accepts applicants selected by logistic regression and charges a lower rate. Strategy 3 accepts applicants selected by the SVM model and also charges the lower rate. For portfolio simulation, the notebook generates a repayment matrix

$$
D_{ij}
= \mathbf{1}\{\xi_{ij} \leq p_2(x_i)\},
$$

where each column is one market scenario. If accepted borrower `i` repays, the lender earns interest; if they default, the lender loses the loan principal. For a set of accepted borrowers `B`, the scenario balance is of the form

$$
\operatorname{Balance}_j
= \#\{\text{repayers in }B\text{ under scenario }j\}
\cdot \mathrm{loan}\cdot \mathrm{interest}
- \#\{\text{defaulters in }B\text{ under scenario }j\}
\cdot \mathrm{loan}.
$$

The notebook reports expected profit/loss

$$
\mathbb{E}[\operatorname{Balance}]
$$

and 95 percent Value at Risk, implemented as

$$
\operatorname{VaR}_{95}
= -\operatorname{percentile}_{5}(\operatorname{Balance}).
$$

The main mathematical lesson is that classification probabilities are not only labels; they can be decision variables in a risk-return optimization problem. The SVM strategy has the strongest reported performance in the simulation because the nonlinear model better identifies high-quality applicants under the nonlinear repayment mechanism.

## `Deep Hedging.ipynb`

This notebook studies dynamic hedging of a European call option using both analytical Black-Scholes delta hedging and neural-network hedging. The market model is a zero-drift geometric Brownian motion with initial price `S_0 = 1`, volatility `sigma = 0.5`, maturity `T = 30/365`, and `N = 30` discrete hedging dates. The simulated price process is

$$
S_t
= S_0 \exp\left(\sigma W_t - \frac{1}{2}\sigma^2 t\right).
$$

Using Ito's formula, the notebook verifies that this solves

$$
dS_t = \sigma S_t\,dW_t,
$$

which is the Black-Scholes SDE with interest rate `r = 0`. The cancellation of the drift term follows from the Ito correction:

$$
d\exp\left(\sigma W_t - \frac{1}{2}\sigma^2 t\right)
= \exp\left(\sigma W_t - \frac{1}{2}\sigma^2 t\right)\sigma\,dW_t.
$$

The dataset consists of simulated paths. In discrete time, increments are generated by

$$
\Delta \log S_j
= -\frac{1}{2}\sigma^2\Delta t
+ \sigma\sqrt{\Delta t}\,Z_j,
\qquad
Z_j \sim \mathcal{N}(0,1),
$$

and cumulative sums are exponentiated to obtain paths. The payoff is that of a European call,

$$
G = (S_T - K)^+,
$$

with strike `K = 1`. The training set has `10^5` simulated paths and the test set has `10^4` paths.

The core deep-hedging objective is to learn a predictable hedging strategy

$$
H = (H_0, H_1, \ldots, H_{N-1}),
$$

where each `H_j` is the number of shares held between time `t_j` and `t_{j+1}`. Given initial option price `p`, terminal payoff `G`, and stock path `S`, the discrete hedging PnL error is

$$
G - p - \sum_{j=0}^{N-1} H_j(S_{j+1} - S_j).
$$

The neural network is trained by minimizing empirical squared hedging error:

$$
\operatorname{Loss}(H)
= \widehat{\mathbb{E}}
\left[
\left(G - p - \sum_j H_j\Delta S_j\right)^2
\right].
$$

This is the finite-sample version of a mean-square hedging objective. If the learned strategy perfectly replicated the option in the discrete model, this loss would be zero. In practice it is minimized over the neural-network function class and evaluated by the distribution of terminal PnL errors.

The first neural architecture uses one separate network per time step. Each network receives `log(S_j)` and outputs the hedge ratio `H_j`. The architecture is `1 -> 64 -> 128 -> 32 -> 1`, with batch normalization, SiLU activations, and dropout. The complete model is a `ModuleList` of 30 such networks:

$$
H_j = f_j(\log S_j),
\qquad
j = 0,\ldots,29.
$$

This is expressive but parameter-heavy because it learns 30 related functions independently. The model is trained with Adam, mini-batches of size 100, and a linear learning-rate schedule.

The analytical benchmark is Black-Scholes delta hedging. The Black-Scholes call price is

$$
C(s,t)
= \Phi(d_+)s - \Phi(d_-)K\exp(-r(T-t)),
$$

where

$$
d_+
= \frac{\log(s/K) + \left(r + \frac{1}{2}\sigma^2\right)(T-t)}
{\sigma\sqrt{T-t}},
\qquad
d_- = d_+ - \sigma\sqrt{T-t}.
$$

The hedge ratio is the derivative of the option price with respect to the stock price:

$$
H_t^{\mathrm{BS}}(s)
= \frac{\partial C(s,t)}{\partial s}
= \Phi(d_+).
$$

The notebook derives this explicitly by differentiating `C(s,t)`. The terms involving derivatives of `Phi(d_+)` and `Phi(d_-)` cancel after substituting the definitions of `d_+` and `d_-`, leaving only `Phi(d_+)`. This gives the standard delta hedge. Since the notebook rebalances only on 30 discrete dates, the analytical delta hedge is not perfectly replicating, but its PnL should be tightly centered around zero.

The notebook compares the learned deep hedge with the analytical delta hedge using terminal PnL histograms, mean PnL, and standard deviation. The first neural model reports mean PnL approximately `2.21e-05` and standard deviation approximately `0.00983`; the analytical delta hedge reports mean PnL approximately `2.69e-05` and standard deviation approximately `0.00894`. The conclusion is that the neural strategy learns a hedge close to the analytical one, with a slightly wider PnL distribution under the reported training setup. The notebook also plots cross-sections of `f_j(log s)` against the Black-Scholes delta curve for several time steps.

The second neural architecture addresses overparameterization by learning a single function of both time-to-maturity and log-price:

$$
H_j = f\left(\sqrt{T-t_j}, \log S_j\right).
$$

Its architecture is `2 -> 32 -> 64 -> 32 -> 1`, again with batch normalization and SiLU activations. This model is mathematically better aligned with the Black-Scholes delta function, because the analytical hedge is itself a function of current price and time-to-maturity:

$$
\Delta(t,s) = \Phi(d_+(t,s)).
$$

Rather than estimating 30 unrelated maps, the second model estimates a single surface over `(time, price)`. The notebook reports that this model has far fewer parameters, about `4289` instead of `378270`, and faster inference, about `0.1048s` instead of `0.4178s` over the timing experiment. The reported PnL statistics are also competitive. The central conclusion is that the time-aware architecture is a more natural approximation to the mathematical structure of the hedging problem: it captures the same functional dependence as the Black-Scholes delta while using fewer parameters and less computation.

## Technical Stack

The notebooks use Python with NumPy, pandas, Matplotlib, SciPy, scikit-learn, statsmodels, and PyTorch. The project emphasizes explicit formulas and numerical verification over software abstraction: regression objectives are written as least-squares or penalized least-squares problems, classification is treated through conditional probability estimation and threshold decision rules, and hedging is formulated as minimization of terminal replication error.
