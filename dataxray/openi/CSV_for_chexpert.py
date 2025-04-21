import json
import csv

def read_json_data(dir):
    with open(dir + "captions.json", "r") as f:
        data = json.load(f)
    
    return data

def write_csv(data):
    L = []
    with open('/Users/sean/Seans Mac Pro/Programming_Projects/AI/ChestXrayReportGen/dataxray/openi/input_report.csv', 'w', newline='') as f:
        for key, value in data.items():
            L.append(value)
        write = csv.writer(f)
        for element in L:
            write.writerow([element])

#######################################################################
####################### Main Work Space ###############################
#######################################################################
data_dir = '/Users/sean/Seans Mac Pro/Programming_Projects/AI/ChestXrayReportGen/dataxray/openi/'
data = read_json_data(data_dir)
print(data)
write_csv(data)

