import csv

def change_delimiter_in_a_file(csv_file_to_read: str, csv_file_to_write: str) -> None:
    
    counter = 0

    with open(csv_file_to_read, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter = ";")
        for row in reader:
            new_row_list = row
            #print(f"new_row_list = {new_row_list}")
            with open(csv_file_to_write, 'a', newline='', encoding="utf-8") as fw:
                writer = csv.writer(fw)
                #print(f"new_row_list len = {len(new_row_list)}")
                writer.writerow(new_row_list)
                fw.close()

            print(f"converted {counter} rows")
            counter += 1
