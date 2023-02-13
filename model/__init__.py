import re
from typing import Iterable, List, Optional, Union
from unittest import TestCase

import pyperclip

from util import ensure_dir


def parse_module_name(module_file_name: str):
    delim = '/' if '/' in module_file_name else '\\'
    return module_file_name.split(delim)[-2]


def read_lines_from(iterable: Iterable[str], path_prefix: str):
    all_lines = []
    for file in iterable:
        with open(f'{path_prefix}/{file}', 'r') as f:
            all_lines.extend((_.rstrip() for _ in f.readlines()))
    return all_lines


def export_to(i: int, content: List[str], export_prefix: str):
    with open(f'{export_prefix}/cell_{i}.py', 'w') as f:
        f.write('\n'.join(content))


class AbstractFilter:
    def __call__(self, content: List[str], debug: bool) -> List[str]:
        """
        Modify the contents according to the pre-defined actions.

        :param content: list of lines to be modified
        :param debug: if `True`, insert comments to indicate which line is being modified and printed to stdout
        :return: list of modified lines
        """
        raise NotImplementedError('The abstract base class shall not be invoked!')


class RegexModifierFilter(AbstractFilter):
    def __init__(self, pattern: str, repl: str):
        self.pattern = pattern
        self.compiled_pattern = re.compile(pattern)
        self.repl = repl

    def __call__(self, content: List[str], debug: bool):
        collected = []
        for i, line in enumerate(content):
            if self.compiled_pattern.search(line):
                modified = re.sub(self.pattern, self.repl, line)
                if debug:
                    print(f'We have a match at line {i}. The change is as follows:')
                    print(f'>>> {line}')
                    if len(modified) > 0:
                        print(f'<<< {modified}')
                    else:
                        print('(Modified line is empty)')
                    print()
                    collected.append(f'# line@{i} ({line}) filtered by {self}')
                if len(modified) > 0:
                    collected.append(modified)
            else:
                collected.append(line)
        return collected

    def __str__(self):
        return f'RegexModifierFilter(pattern={self.pattern}, repl={self.repl})'


class InsertionFilter(AbstractFilter):
    def __init__(self, lines: List[str], apply_when: Optional[Union[int, List[int]]] = None):
        self.lines = lines
        self.n_invoked = 0
        if apply_when is None:
            self.apply_when = [0]
        elif isinstance(apply_when, int):
            self.apply_when = [apply_when]
        else:
            self.apple_when = apply_when

    def __call__(self, content: List[str], debug: bool):
        collected = []
        if self.n_invoked in self.apply_when:
            if debug:
                print(f'{self} is invoked.')
                print()
                collected.append(f'# {self} is invoked.')
            collected.extend(self.lines)
        else:
            if debug:
                print(f'{self} is not invoked, skipping...')
                print()
                collected.append(f'# {self} is not invoked, skipping...')

        collected.extend(content)
        self.n_invoked += 1
        return collected

    def __str__(self):
        return f'InsertionFilter(lines=(omitted), n_invoked={self.n_invoked}, apply_when={self.apply_when})'


def export_notebook(module_file_name: str, files_to_concat: Iterable[str], filters: Optional[Iterable[AbstractFilter]],
                    path_prefix: str = '../..', filter_debug: bool = False, clip: bool = True):
    name = parse_module_name(module_file_name)
    model_file = f'model/{name}/__init__.py'

    export_prefix = f'{path_prefix}/model/{name}/export'
    ensure_dir(export_prefix)
    files_to_iterate = files_to_concat, (model_file,)
    for i, paths in enumerate(files_to_iterate):
        content = read_lines_from(paths, path_prefix)
        if filters is not None:
            for filter in filters:
                content = filter(content, filter_debug)
        if clip:
            pyperclip.copy('\n'.join(content))
            pyperclip.paste()
            input(f'The {i + 1}-th cell has been copied to clipboard...'
                  'Press enter ' + ('for the next one:' if i + 1 < len(files_to_iterate) else 'to end.'))
        export_to(i + 1, content, export_prefix)


class Test(TestCase):
    def test_name_split(self):
        self.assertEqual(parse_module_name(r'/qids-2023-comp/model/linear/config.py'), 'linear')
        self.assertEqual(parse_module_name(r'C:\qids-2023-comp\model\linear\config.py'), 'linear')
