import re
import json
import random

from torch.utils.data import Dataset

from testing.utils import count_tokens

class PileDataset(Dataset):
    def __init__(self, file):
        self.raw_data = self.parse_file(file)

    def parse_file(self, file):
        ret = []
        record = set()
        with open(file, 'r') as f:
            for line in f:
                data = json.loads(line)['text']
                if data not in record:
                    record.add(data)
                    ret.append(data)
        return ret

    def __getitem__(self, index):
        data = self.raw_data[index]
        return data

    def __len__(self):
        return len(self.raw_data)
    
    def shuffle(self):
        random.shuffle(self.raw_data)

class PwCDataset(Dataset):
    def __init__(self, file):
        self.raw_data = self.parse_file(file)

    def parse_file(self, file):
        ret = []
        answer_lengths = [] # To store lengths of 'answer'
        max_answer_length = 0 # To store the maximum 'answer' length

        with open(file, 'r', encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)

                if count_tokens(data["input"]) > 80_000:
                    print("skipping too long input!")
                    continue
                # if self.not_english(data['input']): continue
                
                # Calculate token length of 'answer' and update stats
                answer_len = count_tokens(data["answer"])
                answer_lengths.append(answer_len)
                if answer_len > max_answer_length:
                    max_answer_length = answer_len

                ret.append(data)

        # Print distribution of answer lengths (you might want to refine this for larger datasets)
        if answer_lengths:
            print("\nDistribution of 'answer' lengths (in tokens):")
            # A simple way to show distribution for small datasets is a sorted list or bins
            # For a more detailed distribution, you might consider using collections.Counter
            # or calculating percentiles.
            answer_lengths.sort()
            
            # You can print a few quantiles or a simple list if the number of answers is small.
            if len(answer_lengths) < 20: # Just an arbitrary threshold for printing all
                print(answer_lengths)
            else:
                # Example: print min, max, and a few percentile values
                print(f"Min: {answer_lengths[0]}")
                print(f"25th percentile: {answer_lengths[len(answer_lengths) // 4]}")
                print(f"Median (50th percentile): {answer_lengths[len(answer_lengths) // 2]}")
                print(f"75th percentile: {answer_lengths[3 * len(answer_lengths) // 4]}")
                print(f"Max: {answer_lengths[-1]}")

        # Print the maximum token length of the longest 'answer'
        print(f"\nMaximum token length of 'answer': {max_answer_length}")

        return ret
    
    def __len__(self):
        return len(self.raw_data)
    
    def __getitem__(self, index):
        data = self.raw_data[index]
        context = data['input']
        prompt = data['prompt']
        answer = data['answer']
        return (context, prompt, answer)
    
    def shuffle(self):
        random.shuffle(self.raw_data)

    def not_english(self, text):
        pattern = re.compile(r'[\u4E00-\u9FFF\u3040-\u30FF\uFF00-\uFFEF]+')
        match = pattern.search(text)
        return bool(match)
    
class PwCWithTemplate(PwCDataset):
    def __getitem__(self, index):
        context, prompt, answer = super().__getitem__(index)
        prompt = "\n\nPrompt: " + prompt
        return (context, prompt, answer)
    
class PwCForTest(PwCDataset):
    # def parse_file(self, file):
    #     ret = []
    #     with open(file, 'r') as f:
    #         for line in f:
    #             data = json.loads(line)
    #             # For evaluation convinience, we only select first 10 questions
    #             if data['prompt'] == "Write a paragraph (i.e., continuation) that follows the above text.":
    #                 continue
    #             if data['prompt'] == "Rephrase the above text.":
    #                 continue
    #             if data['prompt'] == "Summarize the above text.":
    #                 continue
    #             if data['prompt'] == "Write a title for the above text.":
    #                 continue
    #             if data['prompt'] == "Extract a few keywords for the above text.":
    #                 continue
    #             if self.not_english(data['input']):
    #                 continue
    #             ret.append(data)
    #     return ret

    def __getitem__(self, index):
        context, prompt, answer = super().__getitem__(index)
        prompt = "\n\nPrompt: " + prompt
        return (context, prompt, answer)
