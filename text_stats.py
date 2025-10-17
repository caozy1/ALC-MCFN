import os


def calculate_document_stats(folder_path):
    """
    统计每个文档的总字符数，并计算所有文档的全局平均字符数
    """
    document_stats = []  # 存储每个文档的总字符数

    # 遍历文件夹中的所有文件
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if not file.endswith(".txt"):
                continue  # 跳过非文本文件

            file_path = os.path.join(root, file)
            try:
                with open(file_path, "r", encoding="GBK") as f:
                    content = f.read()
                    total_chars = len(content)  # 文档的总字符数
                    document_stats.append(total_chars)
                    print(f"文件: {file} | 总字符数: {total_chars}")
            except Exception as e:
                print(f"读取文件 {file} 时出错: {str(e)}")
                continue

    # 全局统计
    if not document_stats:
        print("未找到任何 .txt 文件")
        return

    global_avg = sum(document_stats) / len(document_stats)
    max_chars = max(document_stats)
    min_chars = min(document_stats)

    print("\n=== 全局统计 ===")
    print(f"文档总数: {len(document_stats)}")
    print(f"最长文档字符数: {max_chars}")
    print(f"最短文档字符数: {min_chars}")
    print(f"全局平均字符数: {global_avg:.2f}")


# 使用示例
if __name__ == "__main__":
    folder_path = input("请输入文件夹路径: ").strip()
    calculate_document_stats(folder_path)