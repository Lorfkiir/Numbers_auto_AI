"""Остановка обучения при отсутствии улучшения метрики."""


class EarlyStopping:
    def __init__(
        self,
        mode="min",
        patience=10,
        threshold=0.0001,
        threshold_mode="rel",
    ):
        if mode not in {"min", "max"}:
            raise ValueError("mode должен быть 'min' или 'max'")
        if threshold_mode not in {"rel", "abs"}:
            raise ValueError("threshold_mode должен быть 'rel' или 'abs'")
        if not isinstance(patience, int) or patience < 1:
            raise TypeError("patience должен быть положительным целым числом")
        if not isinstance(threshold, (int, float)) or not 0 <= threshold < 1:
            raise ValueError("threshold должен находиться в диапазоне [0, 1)")

        self.mode = mode
        self.patience = patience
        self.threshold = float(threshold)
        self.threshold_mode = threshold_mode
        self.count = 0
        self.best = None

    def __call__(self, tracked_parameter):
        current = float(tracked_parameter)
        if self.best is None:
            self.best = current
            return False

        if self._is_better(current, self.best):
            self.best = current
            self.count = 0
        else:
            self.count += 1

        if self.count >= self.patience:
            self.count = 0
            return True
        return False

    def _is_better(self, current, best):
        if self.mode == "min":
            limit = best - best * self.threshold if self.threshold_mode == "rel" else best - self.threshold
            return current < limit

        limit = best + best * self.threshold if self.threshold_mode == "rel" else best + self.threshold
        return current > limit
