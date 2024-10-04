import pickle
import pprint

# 读取 pkl 文件
with open('jhmdb_test3.pkl', 'rb') as file:
    data = pickle.load(file)

print('len', len(data['video_name']))

# 使用 pprint 格式化数据
#formatted_data = pprint.pformat(data, indent=4)

# 将格式化后的数据保存为 txt 文件
#with open('output.txt', 'w', encoding='utf-8') as txt_file:
#    txt_file.write(formatted_data)