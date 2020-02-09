
class TrainSupporter(object):

    def __init__(self, model, threshold=1.0E-30):
        self.model = model
        self.threshold = threshold

    def train_until(self, *argv, **kwargs):
        last_loss = self.model.train(*argv, **kwargs)
        train_count = 0

        while True:
            train_count += 1
            loss = self.model.train(*argv, **kwargs)
            # print(loss)
            if abs(last_loss - loss) < self.threshold:
                break
            last_loss = loss

        return train_count, loss
