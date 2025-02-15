import csv
from pytube import YouTube

input_file = './ll_dataset - Sheet6.csv'
output_file = 'updated_csv_file.csv'

with open(input_file, mode='r', newline='', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    rows = list(reader)
    
    for row in rows:
        yt_link = row['yt_link']
        try:
            yt = YouTube(yt_link)
            duration = yt.length
            row['duration'] = duration
        except Exception as e:
            print(f"Error processing {yt_link}: {e}")
            row['duration'] = 'Error'
    
    fieldnames = reader.fieldnames

with open(output_file, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Save CSV to {output_file}")