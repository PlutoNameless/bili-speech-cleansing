import os
import time
import xml.etree.ElementTree as ET

# 文件夹路径，包含你所有的.xml文件
folder_path = r"/path/to/your/xml/files"
# .txt文件保存路径，如果这个路径为空，则.txt文件将被保存在xml文件所在的目录下
txt_folder_path = r""  # 或者你可以指定一个路径，例如 '/path/to/your/txt/files'

# 如果没有指定.txt文件保存路径，那么就使用xml文件所在的目录作为保存路径
if txt_folder_path == "":
    txt_folder_path = folder_path

# 定义统计变量
xml_files_count = 0
txt_files_count = 0
lines_count = 0
error_files = []  # 用来记录出错的文件
processing_times = []  # 用来记录每个文件的处理时间

for filename in os.listdir(folder_path):
    if filename.endswith('.xml'):
        xml_files_count += 1
        start_time = time.time()  # 记录开始处理文件的时间

        try:
            tree = ET.parse(os.path.join(folder_path, filename))
            root = tree.getroot()

            total_texts = len(root.findall('.//d'))  # 计算该文件的元素总数
            seen_texts = set()  # 用来保存已经遇到过的文本内容

            # 在指定的.txt文件保存路径下创建一个新的.txt文件
            with open(os.path.join(txt_folder_path, filename.replace('.xml', '.txt')), 'w', encoding='utf-8') as f:
                for d in root.findall('.//d'):
                    text = d.text
                    if text not in seen_texts:  # 如果这个文本内容还没有遇到过
                        seen_texts.add(text)  # 添加到集合中
                        # 写入'd'标签的文本内容
                        f.write(text + '\n')
                        lines_count += 1
            txt_files_count += 1

            unique_texts = len(seen_texts)  # 计算该文件的唯一元素总数
            removed_duplicates = total_texts - unique_texts  # 计算去除的重复元素的数量

            print(f"在 {filename} 中，去除了 {removed_duplicates} 条重复的内容。")

        except Exception as e:
            print(f"在处理文件 {filename} 时发生错误: {str(e)}")
            error_files.append(filename)

        processing_time = time.time() - start_time  # 计算处理这个文件所用的时间
        processing_times.append(processing_time)

        # 计算预估的剩余处理时间
        avg_processing_time = sum(processing_times) / len(processing_times)
        remaining_files = xml_files_count - txt_files_count
        estimated_remaining_time = remaining_files * avg_processing_time

        print(f"预计处理完本次任务还需要 {estimated_remaining_time} 秒")

print(f"本次脚本一共处理了 {xml_files_count} 个xml文件，成功保存了 {txt_files_count} 个txt文件。")
print(f"在成功保存的txt文件中，一共处理了 {lines_count} 行有效数据。")
print(f"txt文件保存的文件夹路径是: {txt_folder_path}")

if error_files:
    print(f"在处理过程中遇到错误的文件有：")
    for error_file in error_files:
        print(error_file)
