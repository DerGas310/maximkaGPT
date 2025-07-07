import re

# Паттерн для строки, содержащей только цифры и пробелы (с начала до конца строки)
only_numbers_pattern = re.compile(r'^[ХМIVXLCDM1234567890\s]+$')

def remove_lines_with_only_numerals(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f_in:
        lines = f_in.readlines()

    filtered_lines = [line for line in lines if not only_numbers_pattern.match(line.strip())]

    with open(output_file, 'w', encoding='utf-8') as f_out:
        f_out.writelines(filtered_lines)

    print(f'Удалено {len(lines) - len(filtered_lines)} строк, содержащих только цифры.')

def remove_empty_lines_in_range(input_file, output_file, start_line, end_line):
    with open(input_file, 'r', encoding='utf-8') as f_in:
        lines = f_in.readlines()

    new_lines = []
    for i, line in enumerate(lines, start=1):
        if start_line <= i <= end_line:
            # если строка пустая (после удаления пробелов), пропускаем её
            if line.strip() == '':
                continue
        new_lines.append(line)

    with open(output_file, 'w', encoding='utf-8') as f_out:
        f_out.writelines(new_lines)

    print(f'Удалено {len(lines) - len(new_lines)} пустых строк в диапазоне с {start_line} по {end_line}.')


input_path = 'dataset.txt'  # исходный файл
intermediate_path = 'dataset.txt'  # промежуточный файл после удаления цифр
output_path = 'dataset.txt'  # итоговый файл после удаления пустых строк в диапазоне

# Удаляем строки с цифрами
remove_lines_with_only_numerals(input_path, intermediate_path)

# Удаляем пустые строки в заданном диапазоне строк
remove_empty_lines_in_range(intermediate_path, output_path, 0, 0)

