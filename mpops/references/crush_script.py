from dataclasses import dataclass
import datetime
import re
from typing import List, Optional, Union

from data_list import data_list


@dataclass
class TaskObject:
    task: str = ""
    assignee: str = ""
    term: Union[datetime.datetime, None] = None

    def __repr__(self):
        return f"Задача: {self.task} " * bool(self.task) + \
            f"Ответственный: {self.assignee} " * bool(self.assignee) + \
            f"Срок: {self.term}\n" * bool(self.term)

    @staticmethod
    def get_regex_pattern():
        return re.compile(r"""
            \d+\.\s*(?P<task>.*?);?\s*
            Ответственный: *\s*(?P<assignee>.*?);?\s*
            (?:Срок: *\s*(?P<term>.*?);?)?$
            """, re.VERBOSE)

    @staticmethod
    def convert_term(term: Optional[str]):
        if term:
            try:
                return datetime.datetime.strptime(term, "%d.%m.%Y")
            except ValueError:
                return None
        return None


def make_tasks(data: List[str]) -> List[TaskObject]:
    """Создание списка задач из данных"""
    pattern = TaskObject.get_regex_pattern()
    tasks = []
    for elem in data:
        matches = re.match(pattern, elem)
        if matches:
            task_data = matches.groupdict()
            task_object = TaskObject(
                task=task_data['task'],
                assignee=task_data['assignee'],
                term=TaskObject.convert_term(task_data['term'])
            )
            tasks.append(task_object)
    return tasks


def div_numbers(a: int, b: int) -> int:
    """Делит два целых числа."""
    if b == 0:
        raise ValueError("b cannot be zero")
    return a // b


def summ_numbers(a: int, b: int) -> int:
    """Складывает два целых числа."""
    return a + b


def main():
    tasks = make_tasks(data_list)
    Eva_tasks = [task for task in tasks if task.assignee == "Ева"]
    Sasha_tasks = [task for task in tasks if task.assignee == "Саша"]
    result_div = div_numbers(len(Sasha_tasks), len(Eva_tasks))
    print(f"Список задач: \n{tasks}",
          f"Количество задач Саши к количеству задач Евы: {result_div}",
          sep="\n\n")


if __name__ == "__main__":
    main()
