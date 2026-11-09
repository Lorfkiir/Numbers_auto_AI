class EarlyStopping():
    def __init__(self, mode='min', patience=10, threshold=0.0001, threshold_mode='rel'):
        if mode not in {'min', 'max'}:
            raise ValueError('Параметр mode может принимать только значения min и max')
        if threshold_mode not in {'rel', 'abs'}:
            raise ValueError('Параметр threshold_mode может принимать только значения rel и abs')
        if not isinstance(patience, int):
            raise TypeError('Параметр patience должен быть целым числом')
        if not isinstance(threshold, float):
            raise TypeError('Параметр threshold должен быть типа float и меньше 1.0')
        if threshold >= 1.0:
            raise ValueError('Параметр threshold должен быть меньше 1.0')


        self.mode = mode
        self.patience = patience
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.count = 0
        self.best = None

    def __call__(self, tracked_parametr):
        current = float(tracked_parametr)
        if self.best is None:
            self.best = current
            return False

        if self.changed_better(current, self.best):
            self.best = current
            self.count = 0
        else:
            self.count += 1

        if self.count >= self.patience:
            self.count = 0
            return True

        return False

    def changed_better(self, current, best):
        if self.mode == 'min' and self.threshold_mode == 'rel':
            return current < best - best * self.threshold

        elif self.mode == 'min' and self.threshold_mode == 'abs':
            return current < best - self.threshold

        elif self.mode == 'max' and self.threshold_mode == 'rel':
            return current > best + best * self.threshold
        else:
            return current > best + self.threshold