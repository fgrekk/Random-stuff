import numpy as np
from collections import Counter


def find_best_split(feature_vector, target_vector):
    """
    Указания:
    * Пороги, приводящие к попаданию в одно из поддеревьев пустого множества объектов, не рассматриваются.
    * В качестве порогов нужно брать среднее двух соседних при сортировке значений признака
    * Поведение функции в случае константного признака может быть любым
    * При одинаковых приростах критерия Джини для нескольких порогов нужно выбирать сплит, у которого значение порога минимально
    * Достаточно поддерживать только бинарную классификацию.
    * За наличие в функции циклов балл будет снижен. Векторизуйте! :)

    :param feature_vector: вещественнозначный вектор значений признака
    :param target_vector: вектор классов объектов, len(feature_vector) == len(target_vector)

    :return thresholds: отсортированный по возрастанию вектор со всеми возможными порогами, по которым объекты можно разделить на две различные подвыборки или поддерева
    :return ginis: вектор со значениями критерия Джини для каждого из порогов в thresholds, len(ginis) == len(thresholds)
    :return threshold_best: оптимальный порог (число)
    :return gini_best: оптимальное значение критерия Джини (число)
    """
    # ╰( ͡° ͜ʖ ͡° )つ──☆*:・ﾟ
    """
    Находит оптимальный порог для разбиения.
    """
    # Сортируем признаки и таргет
    sort_idx = np.argsort(feature_vector)
    X_sorted = feature_vector[sort_idx]
    y_sorted = target_vector[sort_idx]

    # Ищем индексы, где значение признака меняется
    diffs = X_sorted[1:] != X_sorted[:-1]
    split_indices = np.where(diffs)[0] + 1  # Индексы = количество элементов в левом поддереве

    # Если признак константный или нет возможных сплитов
    if len(split_indices) == 0:
        return np.array([]), np.array([]), None, None

    # Вычисляем пороги (среднее между соседними различными значениями)
    thresholds = (X_sorted[split_indices - 1] + X_sorted[split_indices]) / 2.0

    N = len(target_vector)
    # Кумулятивная сумма единичек в таргете
    S_cum = np.cumsum(y_sorted)

    # Количество единиц слева и справа для каждого порога
    S_L = S_cum[split_indices - 1]
    S_R = S_cum[-1] - S_L

    # Размеры левого и правого поддеревьев
    N_L = split_indices
    N_R = N - N_L

    # Критерий Джини для левых и правых поддеревьев: 1 - p_1^2 - p_0^2
    H_L = 1.0 - (S_L / N_L)**2 - ((N_L - S_L) / N_L)**2
    H_R = 1.0 - (S_R / N_R)**2 - ((N_R - S_R) / N_R)**2

    # Джини исходного узла (родителя)
    S_tot = S_cum[-1]
    H_parent = 1.0 - (S_tot / N)**2 - ((N - S_tot) / N)**2

    # Прирост информации (Information Gain по критерию Джини)
    ginis = H_parent - (N_L / N) * H_L - (N_R / N) * H_R
    
    # Решаем проблему точности float для одинаковых сплитов и берем первый максимум
    # np.argmax возвращает первое вхождение (т.е. минимальный порог из-за сортировки)
    best_idx = np.argmax(np.round(ginis, 9))
    threshold_best = thresholds[best_idx]
    gini_best = ginis[best_idx]

    return thresholds, ginis, threshold_best, gini_best


class DecisionTree:
    """
    Простое классификационное дерево, поддерживающее:
    * real / categorical признаки
    * binary цели (метки могут быть числами или строками)
    * ограничения max_depth, min_samples_split, min_samples_leaf (как в sklearn по смыслу)

    ВНИМАНИЕ: в методе _fit_node ниже могут быть намеренно оставлены некоторые ошибки.
    Их нужно исправить в рамках задания.
    """
    def __init__(self, feature_types, max_depth=None, min_samples_split=None, min_samples_leaf=None):
        if np.any(list(map(lambda x: x != "real" and x != "categorical", feature_types))):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf

    def _fit_node(self, sub_X, sub_y, node, depth=0):
        if np.all(sub_y == sub_y[0]):
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return
        
        if (self._max_depth is not None and depth >= self._max_depth):
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return
        
        if (self._min_samples_split is not None and len(sub_y) < self._min_samples_split):
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        feature_best, threshold_best, gini_best, split = None, None, None, None
        for feature in range(sub_X.shape[1]):
            feature_type = self._feature_types[feature]
            categories_map = {}

            if feature_type == "real":
                feature_vector = sub_X[:, feature]
            elif feature_type == "categorical":
                counts = Counter(sub_X[:, feature])
                pos_class = np.unique(sub_y)[1]
                clicks = Counter(sub_X[sub_y == pos_class, feature])
                ratio = {}
                for key, current_count in counts.items():
                    if key in clicks:
                        current_click = clicks[key]
                    else:
                        current_click = 0
                    ratio[key] = current_click / current_count
                sorted_categories = list(map(lambda x: x[0], sorted(ratio.items(), key=lambda x: x[1])))
                categories_map = dict(zip(sorted_categories, list(range(len(sorted_categories)))))

                feature_vector = np.array(list(map(lambda x: categories_map[x], sub_X[:, feature])))
            else:
                raise ValueError
            
            if np.all(feature_vector == feature_vector[0]):
                continue

            _, _, threshold, gini = find_best_split(feature_vector, sub_y)
            if threshold is None:
                continue

            if (self._min_samples_leaf is not None):
                cnt = feature_vector < threshold
                if (cnt.sum() < self._min_samples_leaf or (~cnt).sum() < self._min_samples_leaf):
                    continue

            if gini_best is None or gini > gini_best:
                feature_best = feature
                gini_best = gini
                split = feature_vector < threshold

                if feature_type == "real":
                    threshold_best = threshold
                elif feature_type == "categorical":
                    threshold_best = list(map(lambda x: x[0],
                                              filter(lambda x: x[1] < threshold, categories_map.items())))
                else:
                    raise ValueError

        if feature_best is None:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        node["type"] = "nonterminal"

        node["feature_split"] = feature_best
        if self._feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        elif self._feature_types[feature_best] == "categorical":
            node["categories_split"] = threshold_best
        else:
            raise ValueError
        node["left_child"], node["right_child"] = {}, {}
        self._fit_node(sub_X[split], sub_y[split], node["left_child"], depth=depth+1)
        self._fit_node(sub_X[np.logical_not(split)], sub_y[np.logical_not(split)], node["right_child"], depth=depth+1)

    def _predict_node(self, x, node):
        if (node["type"] == "terminal"):
            return node["class"]
        
        feature = node["feature_split"]
        if self._feature_types[feature] == "real":
            if x[feature] < node["threshold"]:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])
        elif self._feature_types[feature] == "categorical":
            if x[feature] in node["categories_split"]:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])
        else:
            raise ValueError

    def fit(self, X, y):
        self._fit_node(X, y, self._tree)

    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)
