import json

dataset_dir = '/Users/sean/Seans Mac Pro/Programming_Projects/AI/ChestXrayReportGen/dataxray/openi/'
section_tgt = 'FINDINGS'

reports = json.load(open(dataset_dir + 'Full_openi_data.json', 'r'))
sentence_extnum = {}
for file_name, report in reports.items():
    report = report['report']
    if section_tgt in report and report[section_tgt] != '':
        paragraph = report[section_tgt]
        sentences = paragraph.split('.')
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence not in sentence_extnum:
                sentence_extnum[sentence] = 1
            else:
                sentence_extnum[sentence] += 1

json.dump(sentence_extnum, open(dataset_dir + 'sentence_extnum.json', 'w'))