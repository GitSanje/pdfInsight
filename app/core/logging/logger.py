import atexit
import sys
import time
import logging.config
import pathlib
import json
sys.path.append(str(pathlib.Path(__file__).parent))



def setup_logging(logger_name: str = "papermind") -> logging.Logger :
    logger = logging.getLogger(logger_name)
    config_file = pathlib.Path(__file__).parent / "config.json"
    with open(config_file) as f:
        config = json.load(f)

    for handler in config.get("handlers", {}).values():
        if "filename" in handler:
            log_path = pathlib.Path(handler["filename"])
            if not log_path.parent.exists():
                log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.config.dictConfig(config)
    queue_handler = logging.getHandlerByName("queue_handler")
    if queue_handler is not None:
        queue_handler.listener.start()
        atexit.register(queue_handler.listener.stop)

    return logger




class LoggedEmbeddings:
    def __init__(self, embeddings, settings):
        self.embeddings = embeddings
        self.logger = settings.logger 

    def embed_documents(self, texts):
        self.logger.info("Embedding documents with Gemini", extra={"num_texts": len(texts)})
        vectors = self.embeddings.embed_documents(texts)
        self.logger.info("Embedding completed", extra={"num_vectors": len(vectors)})
        return vectors

    def embed_query(self, query):
        self.logger.info("Embedding query with Gemini", extra={"query": query})
        vector = self.embeddings.embed_query(query)
        self.logger.info("Query embedding completed", extra={"vector_length": len(vector)})
        return vector


def main():
    logger = setup_logging()
    logging.basicConfig(level="INFO")
    logger.debug("debug message", extra={"x": "hello"})
    logger.info("info message")
    logger.warning("warning message")
    logger.error("error message")
    logger.critical("critical message")
    try:
        1 / 0
    except ZeroDivisionError:
        logger.exception("exception message")


# if __name__ == "__main__":
#     main()