in_filename = 'dict.txt'
out_filename = 'dict_medical_source_words.txt'

medical_tag = ['疾病和诊断', '解刨部位', '影像检查', '实验室检验', '药物', '手术']
dict_file = open(in_filename, 'r', encoding='utf-8-sig')
out_file = open(out_filename, 'w', encoding='utf-8-sig')
lines = dict_file.readlines()
for line in lines:
    source_word, freq, tag = line.strip().split('@@')
    if source_word and tag in medical_tag:
        print(source_word, file=out_file)
out_file.close()
dict_file.close()
