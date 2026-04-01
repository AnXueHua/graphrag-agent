"""
pydantic 对象结构化输出(推荐)
typeddict 字典结构化输出
jsonSchema json结构化输出(用于动态输入)
"""
from pydantic import BaseModel, Field
from typing import TypedDict, Annotated
from init_llm import google_llm

class FileInfo(BaseModel):
    title: str = Field(default=None, description="文件标题")
    date: str = Field(default=None, description="文件日期")
    writer: str = Field(default=None, description="文件作者")

class FileInfo_typing(TypedDict):
    title: Annotated[str, "文件标题"]
    date: Annotated[str, "文件日期"]
    writer: Annotated[str, "文件作者"]

FileInfo_json = {
    "title": "FileInfo",
    "description": "文件信息",
    "type": "object",
    "properties": {
        "title": {
            "type": "string",
            "description": "文件标题"
        },
        "date": {
            "type": "string",
            "description": "文件日期"
        },
        "writer": {
            "type": "string",
            "description": "文件作者"
        }
    },
    "required": ["title", "date", "writer"]
}

llm_struct_output = google_llm.with_structured_output(FileInfo)
# response = llm_struct_output.invoke("帮我介绍一下《红楼梦》")
# print(response)
