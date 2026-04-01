import os
import time
import uuid
import requests
import zipfile
import shutil
from config import Config
'''
MinerU解析文件
'''
class MinerUBatchParser:
    def __init__(self, MINERU_API_KEY):
        self.token = MINERU_API_KEY
        self.base_url = "https://mineru.net/api/v4"
        self.header = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.token}"
        }
        self.output_dir = Config.GRAPHRAG_INPUT_DIR

    # =======================================================
    # 1. 上传函数
    # =======================================================
    def upload_files(self, file_paths):
        """
        步骤一：批量上传文件
        :return: (batch_id, id_map) 
                 batch_id: 用于后续查询
                 id_map: {data_id: {"name": "文件名", "path": "本地路径"}} 用于关联结果
        """
        if not file_paths:
            print("⚠️ [上传] 没有文件需要处理")
            return None, None

        print(f"🚀 [上传] 准备处理 {len(file_paths)} 个文件...")
        
        # 1.1 准备 Payload 和本地映射
        files_payload = []
        id_map = {}
        
        for path in file_paths:
            file_name = os.path.basename(path)
            data_id = str(uuid.uuid4()) # 生成唯一ID
            
            files_payload.append({"name": file_name, "data_id": data_id})
            id_map[data_id] = {"name": file_name, "path": path}

        # 1.2 申请上传链接
        url_batch = f"{self.base_url}/file-urls/batch"
        data = {"files": files_payload, "model_version": "vlm"}
        
        try:
            res = requests.post(url_batch, headers=self.header, json=data)
            if res.status_code != 200 or res.json().get("code") != 0:
                print(f"❌ [上传] 申请链接失败: {res.text}")
                return None, None
            
            res_data = res.json()["data"]
            batch_id = res_data["batch_id"]
            urls = res_data["file_urls"]
            
            print(f"✅ [上传] 获取 Batch ID: {batch_id}")

            # 1.3 执行 PUT 上传
            # API 保证返回的 urls 顺序与 files_payload 一致
            for i, upload_url in enumerate(urls):
                local_file = file_paths[i]
                print(f"   ⬆️  正在上传: {os.path.basename(local_file)}")
                with open(local_file, 'rb') as f:
                    put_res = requests.put(upload_url, data=f)
                    if put_res.status_code != 200:
                        print(f"   ❌ 上传失败: {local_file}")
            
            return batch_id, id_map

        except Exception as e:
            print(f"❌ [上传] 异常: {e}")
            return None, None

    # =======================================================
    # 2. 轮询状态函数
    # =======================================================
    def wait_for_completion(self, batch_id):
        """
        步骤二：阻塞式轮询，直到 Batch 内所有文件处理完毕
        :return: extract_result_list (包含所有文件的最终状态列表)
        """
        if not batch_id: return []
        
        url_result = f"{self.base_url}/extract-results/batch/{batch_id}"
        print(f"\n⏳ [查询] 开始轮询任务状态 (Batch: {batch_id})...")
        
        while True:
            try:
                res = requests.get(url_result, headers=self.header)
                if res.status_code != 200:
                    print(f"❌ [查询] HTTP错误: {res.status_code}")
                    break
                
                res_json = res.json()
                if res_json["code"] != 0:
                    print(f"❌ [查询] API错误: {res_json['msg']}")
                    break
                
                extract_results = res_json["data"]["extract_result"]
                
                # 检查是否全部完成 (API状态不包含 pending/running/waiting 即视为结束)
                # 常见中间状态: waiting-file, pending, running, converting
                # 终态: done, failed
                ongoing_count = 0
                for item in extract_results:
                    if item["state"] not in ["done", "failed"]:
                        ongoing_count += 1
                
                if ongoing_count == 0:
                    print("\n🎉 [查询] 所有任务处理完毕！")
                    return extract_results
                else:
                    # 还有任务在运行，打印进度并等待
                    print(f"\r   ...剩余 {ongoing_count} 个文件正在处理", end="", flush=True)
                    time.sleep(3) # 每3秒轮询一次
                    
            except Exception as e:
                print(f"❌ [查询] 轮询异常: {e}")
                time.sleep(5)
        
        return []

    # =======================================================
    # 3. 下载解压函数
    # =======================================================
    def download_and_extract(self, extract_results, id_map):
        """
        步骤三：下载成功的文件并解压 Markdown
        """
        if not extract_results: return

        print(f"\n📥 [下载] 开始下载并处理结果...")
        
        for item in extract_results:
            data_id = item["data_id"]
            state = item["state"]
            
            # 找到原始文件信息
            file_info = id_map.get(data_id)
            if not file_info: continue
            
            file_name = file_info["name"]

            if state == "done":
                download_url = item["full_zip_url"]
                print(f"   ✅ 处理成功: {file_name} -> 正在下载...")
                self._process_single_zip(download_url, file_name)
            else:
                err_msg = item.get("err_msg", "未知错误")
                print(f"   ❌ 处理失败: {file_name}, 原因: {err_msg}")

    def _process_single_zip(self, url, original_filename):
        """内部辅助：下载 Zip -> 解压 -> 移动 .md"""
        try:
            # 1. 下载
            r = requests.get(url)
            zip_name = f"temp_{uuid.uuid4().hex[:6]}.zip"
            zip_path = os.path.join(self.output_dir, zip_name)
            
            with open(zip_path, 'wb') as f:
                f.write(r.content)
            
            # 2. 解压
            extract_temp = os.path.join(self.output_dir, f"temp_{uuid.uuid4().hex[:6]}")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_temp)
            
            # 3. 提取 MD 并重命名
            # GraphRAG 需要文件名清晰的 .md
            target_md_name = os.path.splitext(original_filename)[0] + ".md"
            target_path = os.path.join(self.output_dir, target_md_name)
            
            found = False
            for root, dirs, files in os.walk(extract_temp):
                for file in files:
                    if file.endswith(".md"):
                        src = os.path.join(root, file)
                        # 覆盖移动
                        shutil.move(src, target_path)
                        found = True
                        break 
                if found: break

            # 4. 清理临时文件
            if os.path.exists(zip_path): os.remove(zip_path)
            if os.path.exists(extract_temp): shutil.rmtree(extract_temp)

        except Exception as e:
            print(f"      ⚠️ 下载解压异常: {e}")

## MinerU解析工具(补充增量解析)
from langchain_core.tools import tool
from typing import Optional, List
@tool("parser_file",description="专业的文件解析工具，支持批量解析多种格式的文件(PDF、图片(png/jpg/jpeg/jp2/webp/gif/bmp)、Doc、Docx、Ppt、PPTx)")
def minerU_parser(files: Optional[List[str]] = None, mode: str = "all") -> str:
    """
    解析文件。
    Args:
        files: 指定要解析的文件名列表。若为空，则扫描工作区。
        mode: 'all' (解析所有), 'incremental' (仅解析新增或修改过的文件)。
    Returns:
        解析结果。
    """
    parser = MinerUBatchParser(Config.MINERU_API_KEY)
    if mode == "all":
        # 解析所有文件
        raw_dir = Config.RAW_FILES_DIR
        all_files = []
        for root, dirs, file in os.walk(raw_dir):
            for f in file:
                all_files.append(os.path.join(root, f))
        if all_files:
            # 1. 执行上传
            batch_id, id_map = parser.upload_files(all_files)
            if batch_id and id_map:
                # 2. 执行轮询 (阻塞直到完成)
                results = parser.wait_for_completion(batch_id)
                # 3. 执行下载和解压
                parser.download_and_extract(results, id_map)
            return f"成功处理 {len(all_files)} 个文件。解析结果已保存至输出目录。"
    elif mode == "incremental":
        pass # TODO 增量解析逻辑

# ==========================================
# 🚀 主程序执行流
# ==========================================
if __name__ == "__main__":
    parser = MinerUBatchParser(Config.MINERU_API_KEY)
    
    # 0. 准备文件列表
    raw_dir = Config.RAW_FILES_DIR
    pdf_files = []
    for root, dirs, files in os.walk(raw_dir):
        for f in files:
            if f.lower().endswith('.pdf'):
                pdf_files.append(os.path.join(root, f))
    if pdf_files:
        # 1. 执行上传
        batch_id, id_map = parser.upload_files(pdf_files)
        
        if batch_id and id_map:
            # 2. 执行轮询 (阻塞直到完成)
            results = parser.wait_for_completion(batch_id)
            
            # 3. 执行下载和解压
            parser.download_and_extract(results, id_map)
            
        print("\n✨ 流程结束。请检查 graphrag/input 目录。")
    else:
        print("没有找到 PDF 文件。")