gt_filename = 'dict.txt'
jieba_filename = 'dict_jieba_out.txt'
out_filename = 'dict_jieba_incorrect_out.txt'

medical_tag = ['疾病和诊断', '解刨部位', '影像检查', '实验室检验', '药物', '手术']
dict_file = open(gt_filename, 'r', encoding='utf-8-sig')
dict_lines = dict_file.readlines()
jieba_file = open(jieba_filename, 'r', encoding='utf-8-sig')
medical_words = jieba_file.readlines()
out_file = open(out_filename, 'w', encoding='utf-8-sig')
seperator = ' / '
for i in range(len(medical_words)):
    # print(i)
    dict_line = dict_lines[i]
    source_word, freq, tag = dict_line.strip().split('@@')
    medical_jieba_line = medical_words[i].strip()
    if seperator in medical_jieba_line:
        print_line = source_word + '_' + tag + '###' + medical_jieba_line
    else:
        if i == 1184:
            jieba_word = medical_jieba_line[:5]
            jieba_tag = medical_jieba_line[6:]
        else:
            jieba_word, jieba_tag = medical_jieba_line.split('_')
        if jieba_tag == tag:
            continue
        else:
            print_line = source_word + '_' + tag + '###' + jieba_word + '_' + jieba_tag
    print(print_line, file=out_file)
out_file.close()
dict_file.close()
