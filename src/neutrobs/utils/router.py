# neutrobs/utils/router.py

class MetricRouter:
    def __init__(self, store=None, logger=None, tensor_processor=None):
        self.store = store
        self.logger = logger
        self.tensor_processor = tensor_processor

    def route(self, metric):
        # tensor → expand
        if metric.kind == "tensor":
            processed = self.tensor_processor.process(metric)
            for m in processed:
                self.route(m)
            return

        # store
        if metric.log_to_store and self.store:
            self.store.log(metric)

        # logger
        if metric.log_to_logger and self.logger:
            self.logger.log(metric)