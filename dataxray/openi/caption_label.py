import csv
import json


with open("/Users/sean/Seans Mac Pro/Programming_Projects/AI/ChestXrayReportGen/dataxray/openi/captions.json", "r") as f:
    data = json.load(f)

with open("/Users/sean/Seans Mac Pro/Programming_Projects/AI/xray_report_generation/open-i/file2label.json", "r") as f1:
    databeta = json.load(f1)

rows = []
with open("/Users/sean/Seans Mac Pro/Programming_Projects/AI/ChestXrayReportGen/dataxray/openi/chexpert-labeler/output_reports.csv", 'r') as file:
    csvreader = csv.reader(file)
    header = next(csvreader)
    for row in csvreader:
        rows.append(row)

L = []
for i in rows:
    temp = []
    for j in i[1:]:
        if j == '1.0':
            temp.append(1)
        if j == '':
            temp.append(0)
        if j == '0.0':
            temp.append(0)
        if j == '-1.0':
            temp.append(1)
    assert len(temp) == 14

    L.append(temp)

result = {}
c = 0
for k, v in data.items():
    title = list(k.split("_"))
    title[0] = title[0].replace("CXR", "")
    result["ecgen-radiology/" + title[0] + ".xml"] = L[c]
    c += 1

print("Converting to Json file...")

json_string = json.dumps(result)
with open('file2lable.json', 'w') as f:
    f.write(json_string)
print("Json file process complete!")