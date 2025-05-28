import datasets

new_data = datasets.load_from_disk('./m2d2_filtered_long_clean_texts')


print(new_data[22]["text"])
