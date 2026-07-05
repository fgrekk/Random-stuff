from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import roc_auc_score

from tqdm.auto import tqdm

from sklearn.base import ClassifierMixin

class TargetEncoder:
    def __init__(self, cat_features: Iterable[int] | None = None):
        self.cat_features = list(cat_features) if cat_features is not None else []
        self.map = {}
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        y_bin  = np.where(y == -1, 0, y)
        self.global_mean = np.mean(y_bin)
        self.map = {}

        for col in self.cat_features:
            categories = X[:, col]
            unique_cats = np.unique(categories)
            col_map = {}
            
            for cat in unique_cats:
                msk = (categories == cat)
                col_map[cat] = np.mean(y_bin[msk])
                
            self.map[col] = col_map
        return self
    
    def transform(self, X: np.ndarray):
        X_trans = X.copy()
        
        for col in self.cat_features:
            col_mapping = self.map.get(col, {})
            categories = X[:, col]
            transformed_col = np.full(X.shape[0], self.global_mean, dtype=float)
            
            for cat, val in col_mapping.items():
                transformed_col[categories == cat] = val
                
            X_trans[:, col] = transformed_col
            
        return X_trans.astype(float)

    def fit_transform(self, X: np.ndarray, y: np.ndarray):
        return self.fit(X, y).transform(X)

class OrderedBoostingEncoder(TargetEncoder):
    def fit_transform(self, X: np.ndarray, y: np.ndarray, random_state: int | None = 69) -> np.ndarray:
        rng = np.random.default_rng(random_state)
        perm = rng.permutation(len(X))
        
        X_shuffled = X[perm]
        y_shuffled = np.where(y[perm] == -1, 0, y[perm])
        
        X_encoded = X.copy()
        self.global_mean = np.mean(y_shuffled)

        for col in self.cat_features:
            cumsum = defaultdict(float)
            cumcount = defaultdict(float)
            
            encoded_col = np.zeros(len(X))
            for i in range(len(X)):
                cat = X_shuffled[i, col]
                encoded_col[i] = (cumsum[cat] / cumcount[cat]) if cumcount[cat] > 0 else self.global_mean
                
                cumsum[cat] += y_shuffled[i]
                cumcount[cat] += 1
            
            X_encoded[perm, col] = encoded_col
            self.map[col] = {cat: cumsum[cat] / cumcount[cat] for cat in cumcount}
            
        return X_encoded.astype(float)

class Quantizer:
    def __init__(self, quantization_type: str | None = None, nbins: int = 255):
        self.quantization_type = quantization_type
        self.nbins = nbins
        self.bins_ = {}
        
    def fit(self, X: np.ndarray, y: np.ndarray = None):
        if self.quantization_type is None:
            return self
            
        for col in range(X.shape[1]):
            col_data = X[:, col]

            if self.quantization_type == 'uniform':
                min_val, max_val = np.min(col_data), np.max(col_data)
                self.bins_[col] = np.linspace(min_val, max_val, self.nbins + 1)[1:-1]
            elif self.quantization_type == 'quantile':
                quantiles = np.linspace(0, 1, self.nbins + 1)[1:-1]
                bins = np.quantile(col_data, quantiles)
                self.bins_[col] = np.unique(bins)
            elif self.quantization_type == 'piecewise':
                clf = DecisionTreeRegressor(max_leaf_nodes=self.nbins, random_state=42)
                clf.fit(col_data.reshape(-1, 1), y)
                
                msk = (clf.tree_.children_left != -1) | (clf.tree_.children_right != -1) 
                thresholds = clf.tree_.threshold[msk]
                self.bins_[col] = np.sort(np.unique(thresholds))
            else:
                raise ValueError(f"Неизвестный тип квантизации: {self.quantization_type}")
                
        return self
        
    def transform(self, X: np.ndarray) -> np.ndarray:
        X_trans = X.copy()
        for col in range(X.shape[1]):
            col_data = X_trans[:, col]
            quantized_col = np.digitize(col_data, self.bins_[col]).astype(float)
            X_trans[:, col] = quantized_col
            
        return X_trans

    def fit_transform(self, X: np.ndarray, y: np.ndarray = None) -> np.ndarray:
        return self.fit(X, y).transform(X)


class BoostingClassifier(ClassifierMixin):

    def __init__(
        self,
        base_model_class = DecisionTreeRegressor,
        base_model_params: dict | None = None,
        n_estimators: int = 20,
        learning_rate: float = 0.05,
        l2: float | None = 0.0,
        random_state: int | None = None,
        early_stopping_rounds: int | None = 0,
        eval_metric: str | None = None,
        goss: bool = False,
        goss_k: float = 0.2,
        subsample: float = 0.3,
        quantization_type: str | None = None,
        nbins: int = 255,
        verbose: bool = True
    ):
        super().__init__()

        self.base_model_class = base_model_class
        self.base_model_params = {} if base_model_params is None else base_model_params

        self.n_estimators = n_estimators
        self.learning_rate = learning_rate

        self.models = [0] * (n_estimators)
        self.gammas = [0] * (n_estimators)
        self._train_predictions = None
        self.l2 = l2
        self.goss = goss
        self.goss_k = goss_k
        self.subsample = subsample
        self.quantization_type = quantization_type
        self.nbins = nbins

        self.early_stopping_rounds = early_stopping_rounds
        self.eval_metric = eval_metric

        self.random_state = random_state  # не забудьте вставить его везде, где у вас возникает рандом
        self.verbose = verbose

        self.history = defaultdict(list)  # {"train_roc_auc": [], "train_loss": [], ...}

        self.sigmoid = lambda x: 1 / (1 + np.exp(-x))
        self.loss_fn = lambda y, z: -np.log(self.sigmoid(y * z)).mean()
        self.grad_fn = lambda y, z: y / (1 + np.exp(y * z))
        self.hess_fn = lambda y, z: np.clip(np.exp(y * z) / (1 + np.exp(y * z)) ** 2, 1e-6, None)

    def plot_history(self, keys: str | Iterable[str]):
        if isinstance(keys, str):
            keys = [keys]
        
        plt.figure(figsize=(10, 5))
        for key in keys:
            if key in self.history:
                plt.plot(self.history[key], label=key, lw=2)
        plt.xlabel("Итерация")
        plt.ylabel("Значение")
        plt.title("History")
        plt.legend()
        plt.show()
    
    def _goss_sample(self, X: np.ndarray, grads: np.ndarray, hess: np.ndarray):
        if not self.goss:
            return X, grads, hess
            
        n = len(X)
        top_k_count = int(n * self.goss_k)
        
        abs_grads = np.abs(grads)
        sorted_indices = np.argsort(abs_grads)[::-1]
        top_indices = sorted_indices[:top_k_count]
        small_indices = sorted_indices[top_k_count:]
        rng = np.random.default_rng(self.random_state)
        sample_size = int(len(small_indices) * self.subsample)
        
        if sample_size == 0:
            return X[top_indices], grads[top_indices], hess[top_indices]
            
        sampled_small_indices = rng.choice(small_indices, size=sample_size, replace=False)
        weight = (1.0 - self.goss_k) / self.subsample
        grads_goss = grads.copy()
        hess_goss = hess.copy()
        grads_goss[sampled_small_indices] *= weight
        hess_goss[sampled_small_indices] *= weight
        
        final_indices = np.concatenate([top_indices, sampled_small_indices])
        
        return X[final_indices], grads_goss[final_indices], hess_goss[final_indices]

    def partial_fit(self, X: np.ndarray, y: np.ndarray) -> None:
        shf = self.grad_fn(y, self._train_predictions)
        grads = shf
        hess = self.hess_fn(y, self._train_predictions)
        X_it, grad_it, hess_it = self._goss_sample(X, grads, hess)
        if isinstance(self.base_model_class, DecisionTreeRegressor):
            model = self.base_model_class(**self.base_model_params, random_state=self.random_state)
            model.fit(X_it, grad_it)
        else:
            model = self.base_model_class(**self.base_model_params, l2 = max(1e-8, self.l2))
            model.fit(X_it, grad_it, hess_it)
        y_pred = model.predict(X)
        gamma = 1 if self.l2 != 0 else self._find_optimal_gamma(y, self._train_predictions, y_pred)
        self.models.append(model)
        self.gammas.append(gamma)

        self._train_predictions += self.learning_rate * gamma * y_pred

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, eval_set: tuple[np.ndarray] | None = None, use_best_model: bool = False) -> None:
        self._train_predictions = np.zeros(X_train.shape[0]).astype(float)
        self.classes_ = np.unique(y_train)  # не рекомендуется убирать, нужно для калибровки
        estimator_range = range(self.n_estimators)
        quantizer = None if self.quantization_type is None else Quantizer(quantization_type=self.quantization_type, nbins=self.nbins)
        if self.verbose:
            estimator_range = tqdm(estimator_range)

        y_train1 = np.where(y_train == 0, -1, y_train)
        if eval_set is not None:
            X_val, y_val = eval_set
            y_val1 = np.where(y_val == 0, -1, y_val)
            val_predictions = np.zeros(X_val.shape[0]).astype(float)

        if (quantizer is not None):
            X_train = quantizer.fit_transform(X_train, y_train1)
            if eval_set is not None:
                X_val = quantizer.transform(X_val)

        best_score = None
        best_it = 0
        no_improve = 0
        is_loss = self.eval_metric is not None and "loss" in self.eval_metric

        self.models = []
        self.gammas = []
        for i in estimator_range:
            self.partial_fit(X_train, y_train1)
            cur_loss = self.loss_fn(y_train1, self._train_predictions)
            cur_proba = self.sigmoid(self._train_predictions)
            cur_auc = roc_auc_score(y_train1 == 1, cur_proba)
            self.history["train_loss"].append(cur_loss)
            self.history["train_roc_auc"].append(cur_auc)
            if eval_set is not None:
                val_predictions += self.learning_rate * self.gammas[-1] * self.models[-1].predict(X_val)
                val_loss = self.loss_fn(y_val1, val_predictions)
                val_proba = self.sigmoid(val_predictions)
                val_auc = roc_auc_score(y_val1 == 1, val_proba)
                self.history["val_loss"].append(val_loss)
                self.history["val_roc_auc"].append(val_auc)
                if self.early_stopping_rounds and self.eval_metric:
                    cur_score = self.history[self.eval_metric][-1]
                    if not is_loss:
                        cur_score *= -1
                    
                    if best_score is None:
                        best_score = cur_score
                        best_it = i
                    elif cur_score < best_score:
                        best_score = cur_score
                        best_it = i
                        no_improve = 0
                    else:
                        no_improve += 1
                    
                    if no_improve >= self.early_stopping_rounds or (i >= 5 and self.eval_metric == 'val_roc_auc' and cur_loss >= -0.6):
                        break
        
        if use_best_model and self.early_stopping_rounds and eval_set is not None:
            self.models = self.models[:best_it + 1]
            self.gammas = self.gammas[:best_it + 1]

        # чтобы было удобнее смотреть
        for key in self.history:
            self.history[key] = np.array(self.history[key])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        preds = np.zeros(X.shape[0])
        for model, gamma in zip(self.models, self.gammas):
            preds += self.learning_rate * gamma * model.predict(X)
        proba = self.sigmoid(preds)
        return np.column_stack((1 - proba, proba))

    def _find_optimal_gamma(
        self,
        y: np.ndarray,
        old_predictions: np.ndarray, 
        new_predictions: np.ndarray
    ) -> float:
        gammas = np.linspace(start=0, stop=1, num=100)
        losses = [
            self.loss_fn(y, old_predictions + gamma * new_predictions)
            for gamma in gammas
        ]
        return gammas[np.argmin(losses)]

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return roc_auc_score(y == 1, self.predict_proba(X)[:, 1])
