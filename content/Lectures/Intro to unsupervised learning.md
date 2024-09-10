# Supervised learning

Objects are described by
- $x$ — input variables,
- $y$ — target variable, e.g. $y \in \{0, 1, ..., K\}$ (classification) or $y \in \mathbb{R}$ (regression).

Examples:
- object — patient, $x$ — CT scan, $y$ — diagnosis ($1$, if patient has a certain disease, else $0$).
- object — email, $x$ — its text, $y$ — spam label ($1$ if spam, else $0$).
- object — molecule, $x$ — relative positions of atoms in a molecule, $y$ — its physical or clinical properties.

Goal is to learn how to predict $y$ based on $x$:
$$
y \approx f(x).
$$

For this we collect train data $\{(x_i, y_i)\}_{i = 1}^N \overset{\text{i.i.d.}}{\sim} p(x, y)$, and find $f$, using *empirical risk minimization* principle:
$$
\sum_{i = 1}^N l(f(x_i), y_i) \to \min_{f \in \mathcal{F}}.
$$

There are mathematical guarantees that it works on test data if
- training error is small,
- complexity of $\mathcal{F}$ (number of functions, number of parameters, VC dimension) is not too large,
- training dataset is sufficiently large.

In practice, it usually works even better than math can prove.

# Unsupervised learning

Objects are described by
- $x, y$ — observable variables.
- $z$ — latent (hidden) variables.

Examples:
- $x$ — an image (pixel values), $z$ — its content, i.e. objects, stuff, etc.
- $y$ — a text (in a certain language), $z$ — its meaning (independent from language).
- $(x, y)$ — an image and its text description, $z$ — content of image described in the text.

Goals:
- learn to generate new data $x \sim p(x)$, $x \sim p(x | y)$,
- learn to estimate $p(x)$ for a given data point $x$, e.g. for anomaly detection,
- learn latent variables, i.e. data representations $z = f(x)$, which are useful for downstream tasks, e.g. classification, retrieval, etc.

## Generative modeling

See https://education.yandex.ru/handbook/ml/article/vvedenie-v-generativnoe-modelirovanie.

## Representation learning

See https://education.yandex.ru/handbook/ml/article/obuchenie-predstavlenij.

# Additional resources

1. [Ilya Sutskever's talk on unsupervised learning, 2023](https://www.youtube.com/watch?v=AKMuA_TVz3A)