import os
import json
import xml.etree.ElementTree as ET

directory = "/Users/sean/Seans Mac Pro/Programming_Projects/AI/ChestXrayReportGen/dataxray/OPENI/ecgen-radiology"
xml_files = os.listdir(directory)

report_json = {}

print("Reading Data...")
for filename in xml_files:
    with open(os.path.join(directory, filename), 'r') as f:
        tree = ET.parse(f)
        root = tree.getroot()
    xmldir = "/ecgen-radiology/" + filename

    Comparison = str(root.find("MedlineCitation").find("Article").find("Abstract")[0].text).lower()
    Indication = str(root.find("MedlineCitation").find("Article").find("Abstract")[1].text).lower()
    Findings = str(root.find("MedlineCitation").find("Article").find("Abstract")[2].text).lower()
    Impression = str(root.find("MedlineCitation").find("Article").find("Abstract")[3].text).lower()
    
    temp_img_l = []
    for img in root.findall("parentImage"):
        temp_img_l.append(img.get("id"))      

    report_json[xmldir] = {"image": temp_img_l, "abstract": {"COMPARISON":Comparison, "INDICATION":Indication, "FINDINGS":Findings, "IMPRESSION":Impression}}
print("Dictionary process complete!")
print("Converting to Json file...")

json_string = json.dumps(report_json)
with open('Full_openi_data.json', 'w') as f:
    f.write(json_string)
print("Json file process complete!")