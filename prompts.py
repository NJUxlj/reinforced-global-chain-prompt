from dataclasses import dataclass
import random


@dataclass
class Question:
    sentences: list # 用于存储问题的不同部分。例如，这可以是问题文本分成的多个段落
    choices: list  # 存储问题的多个选项
    answer_idx: int # 一个整数，表示正确答案在 choices 列表中的索引位置
    task: str = None # 存储任务的描述

    def get_choices_len(self):
        '''
            返回的是选项列表 choices 的长度，即问题有多少个选项
        '''
        return len(self.choices)

    def get_answer_str(self):
        return self.choices[self.answer_idx]

    def _get_prompt(self, include_choices):
        prompt = ""
        for sentence in self.sentences:
            prompt += f"{str(sentence)}\n"
        if include_choices:
            for i, choice in enumerate(self.choices):
                prompt += f"{idx_to_ltr(i)}. {choice}\n"
        return prompt + "Answer:"

    def get_prompt_with_choices(self):
        '''    
         prompt同时包含选项和问题
        '''
        return self._get_prompt(include_choices=True)

    def get_prompt_no_choices(self):
        '''
          CP    
          
          prompt 只包含问题，不包含选项
        '''
        return self._get_prompt(include_choices=False)

    def strong_shuffle(self):
        # This method shuffles choices such that choosing
        # the answer at the originally correct
        # index will mean getting the question wrong

        # For degenerate questions where all choices are the same
        if len(set(self.choices)) == 1:
            return

        answer_idx = self.answer_idx
        answer_str = self.get_answer_str()
        while self.choices[answer_idx] == answer_str:
            random.shuffle(self.choices)
            self.answer_idx = self.choices.index(answer_str)

    def permute_choices(self, perm):
        '''
            根据指定的顺序perm，对选项进行重排列
        '''
        self.choices = [self.choices[i] for i in perm]
        self.answer_idx = perm.index(self.answer_idx)
        
        
        

@dataclass
class QuestionFewShot:
    '''
        A question prompt that includes few-shot examples
    '''
    def __init__(self, question, num_shots):
        self.question = question





if __name__ == '__main__':
    q = Question(sentences=["This is a question"], choices=["A", "B", "C"], answer_idx=0)
    
    prompt = q.get_prompt_with_choices()
    print(prompt)