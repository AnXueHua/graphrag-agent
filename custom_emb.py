from typing import List
from langchain_core.embeddings import Embeddings
from config import Config

API_KEY = Config.LLM_API_KEY
EMB_MODEL = Config.EMB_MODEL

class CustomEmbeddings(Embeddings):
    """
    自定义的 Embeddings 类。
    """
    def __init__(self):
        """
        实例化千问为values["client"]

        Args:

            values (Dict): 包含配置信息的字典，必须包含 client 的字段.
        Returns:

            values (Dict): 包含配置信息的字典。
        """
        self.embclient = DashScopeEmbeddings(
            model=EMB_MODEL,
            dashscope_api_key=API_KEY
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        嵌入文档。

        Args:
            texts (List[str]): 要嵌入的文本列表。

        Returns:
            List[List[float]]: 输入列表中每个文档的 embedding 列表。每个 embedding 都表示为一个浮点值列表。
        """
        return self.embclient.embed_documents(texts)
    
    def embed_query(self, text: str) -> List[float]:
        """
        嵌入查询文本。

        Args:
            text (str): 要嵌入的查询文本。

        Returns:
            List[float]: 输入查询文本的 embedding，表示为一个浮点值列表。
        """
        return self.embclient.embed_query(self.embed_documents([text])[0])