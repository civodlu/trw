from trw.train import callback


class CallbackSkipEpoch(callback.Callback):
    """
    Run its callbacks every few epochs
    """
    def __init__(self, nb_epochs, callbacks):
        """
        :param nb_epochs: the number of epochs to skip
        :param callbacks: the callbacks to be called
        """
        self.nb_epochs = nb_epochs
        self.callbacks = callbacks

    def __call__(self, options, history, model, losses, outputs, datasets, datasets_infos, callbacks_per_batch, **kwargs):
        if len(history) % self.nb_epochs == 0:
            for callback in self.callbacks:
                callback(options, history, model, losses, outputs, datasets, datasets_infos, callbacks_per_batch)
