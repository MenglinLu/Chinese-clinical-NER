"""
Created on Wed May  22 13:18:00 2019

@author: zeyuyou
"""
import os
import json
from colorama import Fore


class PrintLabel(object):
    def __init__(self):
        self.dir_path = os.getcwd()
        self.doc_path = os.path.join(self.dir_path, 'data/subtask1_training_part1.txt')
        self.class_map = {'疾病和诊断': 0, '影像检查': 1, '实验室检验': 2, '手术': 3, '药物': 4, '解剖部位': 5}
        self.color_map = {0: Fore.RED, 1: Fore.GREEN, 2: Fore.YELLOW, 3: Fore.BLUE, 4: Fore.MAGENTA, 5: Fore.CYAN}

    def print_text_with_groundtruth(self, write_file):
        text_file = open(self.doc_path, 'r', encoding='utf-8-sig')
        write_file = open(write_file, 'w', encoding='utf-8-sig')
        lines = text_file.readlines()
        for i in range(len(lines)):
            line = lines[i]
            print(Fore.WHITE, 'doc-id=', i)
            print(Fore.WHITE, 'doc-id=', i, file=write_file)
            if line.strip():
                line_dict = json.loads(line)
                origin_text = line_dict['originalText']
                entities = line_dict['entities']
                last_start = 0
                for entity in entities:
                    label_type = entity['label_type']
                    start_pos = entity['start_pos']
                    end_pos = entity['end_pos']
                    overlap = entity['overlap']
                    if overlap:
                        print(overlap)
                        print(overlap, file=write_file)
                    else:
                        print(Fore.WHITE, origin_text[last_start:start_pos])
                        print(Fore.WHITE, origin_text[last_start:start_pos], file=write_file)
                        print(self.color_map[self.class_map[label_type]], origin_text[start_pos:end_pos], '-',
                              label_type)
                        print(self.color_map[self.class_map[label_type]], origin_text[start_pos:end_pos], '-',
                              label_type, file=write_file)

                        last_start = end_pos
            print('\n')
            print('\n', file=write_file)
        write_file.close()
        text_file.close()

    def print_text_with_prediction(self, pred_filename, write_filename):
        text_file = open(self.doc_path, 'r', encoding='utf-8-sig')
        ori_lines = text_file.readlines()
        # print('original doc length=', len(ori_lines))
        pred_file = open(pred_filename, 'r', encoding='utf-8-sig')
        pred_lines = pred_file.readlines()
        # print('prediction doc length=', len(pred_lines))
        write_file = open(write_filename, 'w', encoding='utf-8-sig')
        for i in range(len(ori_lines)):
            print(Fore.WHITE, 'doc-id=', i)
            print(Fore.WHITE, 'doc-id=', i, file=write_file)
            ori_line = ori_lines[i]
            ori_line = ori_line.strip()
            line_dict = json.loads(ori_line)
            origin_text = line_dict['originalText']
            pred_doc = pred_lines[i]
            pred_doc = pred_doc.strip()
            doc_id, pred_line = pred_doc.split(',')
            pred_line = pred_line.strip()
            pred_entities = pred_line.split(';')
            last_start = 0
            for e in pred_entities:
                if e:
                    # blanks are parsed
                    if len(e.split()) < 4:
                        e_source = ' '
                        e_begin, e_end, label_type = e.split()
                    else:
                        e_source, e_begin, e_end, label_type = e.split()

                    e_begin = int(e_begin)
                    e_end = int(e_end)
                    if e_source != origin_text[e_begin:e_end]:
                        aa = origin_text[e_begin:e_end]
                        print('pred=', e_source, 'ori=', origin_text[e_begin:e_end])
                    else:
                        if last_start <= e_begin:
                            print(Fore.WHITE, origin_text[last_start:e_begin])
                            print(Fore.WHITE, origin_text[last_start:e_begin], file=write_file)
                            print(self.color_map[self.class_map[label_type]], e_source, '-',
                                  label_type)
                            print(self.color_map[self.class_map[label_type]], e_source, '-',
                                  label_type, file=write_file)
                        else:
                            print('there is an overlap in the prediction at doc ', doc_id, 'at index ', e_begin,
                                  'and last start index is ', last_start)
                        last_start = e_end
            print('\n')
            print('\n', file=write_file)

        write_file.close()
        pred_file.close()
        text_file.close()


if __name__ == '__main__':
    P = PrintLabel()
    result_file = 'res/ground_truth_printout.txt'
    # P.print_text_with_groundtruth(result_file)
    pred_file = 'res/predict_res/predict_label.txt'
    pred_result_file = 'res/predicted_text_printout.txt'
    P.print_text_with_prediction(pred_file, pred_result_file)
