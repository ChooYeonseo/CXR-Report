import csv
import json

with open('/Users/sean/Seans Mac Pro/Programming_Projects/AI/ChestXrayReportGen/dataxray/openi/Full_openi_data.json') as f:
    data = json.load(f)
L = []
with open('labeled_reports.csv', 'w',newline='') as f:
    for key, value in data.items():
        text = value['abstract']['FINDINGS']
        if text == "none":
            text = value['abstract']['IMPRESSION']
         
        if text == "none":
            text = value['abstract']['INDICATION']
        
        if text == "none":
            text = value['abstract']['COMPARISON']
        if text == "none":
            print(key)
        L.append(text)
    write = csv.writer(f)
    for element in L:
        write.writerow([element])